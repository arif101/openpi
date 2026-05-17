"""Convert Phase A refined-rollout NPZs into a LeRobot dataset.

Phase B distillation needs the (image, wrist_image, state, prompt, refined_action)
pairs we logged during MPPI rollouts to be loadable through openpi's existing
training pipeline. The pipeline expects a LeRobot-format dataset under a HF
repo_id; this script converts the NPZ corpus into that format.

What each NPZ contains (per-decision arrays, one entry per MPPI call within a
successful episode):
  image:          [N, 224, 224, 3] uint8
  wrist_image:    [N, 224, 224, 3] uint8
  state:          [N, 8]           float32 (xyz + axisangle + gripper qpos)
  prompt:         [N]              object (str)
  prior_action:   [N, H, 7]        float32  (Pi0.5's chunk, ignored for training)
  refined_action: [N, H, 7]        float32  (MPPI's chunk, the distill target)
  ...cost diagnostics (ignored)

One NPZ per successful episode. The whole corpus glob is the training set.

The output dataset schema matches openpi's pi05_libero training expectation:
  image, wrist_image, state, actions, prompt   (one row per decision point)

We discard the prior_action (Phase B teaches the model to skip MPPI by emitting
refined_action natively given the same obs). The action_horizon (chunk length)
that openpi's pi05_libero uses is 10; if the logged refined_action is longer
we truncate to 10. If shorter we pad with the last action.

Usage:
    # Local save only:
    uv run python3 scripts/convert_refined_rollouts_to_lerobot.py \\
        --rollouts-dir data/contact_mpc/refined_rollouts_libero10 \\
        --repo-id arif101/openpi_refined_rollouts_libero10_phase_b

    # Push to HF:
    uv run python3 scripts/convert_refined_rollouts_to_lerobot.py \\
        --rollouts-dir data/contact_mpc/refined_rollouts_libero10 \\
        --repo-id arif101/openpi_refined_rollouts_libero10_phase_b \\
        --push-to-hub
"""

from __future__ import annotations

import argparse
import pathlib
import shutil
import sys
import time

import numpy as np


REQUIRED_KEYS = {"image", "wrist_image", "state", "refined_action"}
ACTION_HORIZON = 10  # matches pi05_libero training config
IMG_SIZE = 224        # matches build_obs_element in run_reason_v3_mppi.py


def _fit_chunk(chunk: np.ndarray, horizon: int) -> np.ndarray:
    """Truncate or last-pad an action chunk to the target horizon."""
    if chunk.shape[0] >= horizon:
        return chunk[:horizon]
    pad = np.repeat(chunk[-1:], horizon - chunk.shape[0], axis=0)
    return np.concatenate([chunk, pad], axis=0)


def _iter_npz(rollouts_dir: pathlib.Path):
    paths = sorted(rollouts_dir.glob("*.npz"))
    if not paths:
        raise FileNotFoundError(
            f"No .npz files in {rollouts_dir}. Did Phase A successfully log any "
            f"refined rollouts? Check --log-refined-rollouts on run_reason_v3_mppi.py."
        )
    for p in paths:
        data = np.load(p, allow_pickle=True)
        missing = REQUIRED_KEYS - set(data.files)
        if missing:
            print(f"[warn] {p.name} missing keys {missing}; skipping", file=sys.stderr)
            continue
        yield p, data


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts-dir", required=True,
                    help="Directory of per-episode refined-rollout NPZ files.")
    ap.add_argument("--repo-id", required=True,
                    help="LeRobot/HF repo_id (e.g. arif101/openpi_refined_rollouts_libero10_phase_b).")
    ap.add_argument("--push-to-hub", action="store_true")
    ap.add_argument("--fps", type=int, default=20,
                    help="Nominal fps; LIBERO uses 20Hz physics with ~5-step replan.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Skip LeRobot dataset write; print what would happen.")
    args = ap.parse_args()

    rollouts_dir = pathlib.Path(args.rollouts_dir).resolve()
    print(f"Reading rollouts from: {rollouts_dir}", flush=True)

    # Pre-pass: count frames and verify schema
    total_frames = 0
    n_eps = 0
    for p, data in _iter_npz(rollouts_dir):
        n_eps += 1
        total_frames += int(data["image"].shape[0])
    print(f"Found {n_eps} episodes, {total_frames} total decision frames.")

    if args.dry_run:
        print("[dry-run] would create LeRobot dataset, then add frames per episode.")
        return 0

    # Defer LeRobot import — it's heavy and pulls in torch/datasets
    from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    out_path = HF_LEROBOT_HOME / args.repo_id
    if out_path.exists():
        print(f"Removing existing dataset at {out_path}")
        shutil.rmtree(out_path)

    # Match pi05_libero LeRobot schema. action_horizon == 10 chunk dim.
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        robot_type="panda",
        fps=args.fps,
        features={
            "image": {
                "dtype": "image",
                "shape": (IMG_SIZE, IMG_SIZE, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": (IMG_SIZE, IMG_SIZE, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (ACTION_HORIZON, 7),
                "names": ["action_horizon", "action_dim"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    t0 = time.time()
    added = 0
    for ep_idx, (p, data) in enumerate(_iter_npz(rollouts_dir)):
        images = data["image"]                 # [N, 224, 224, 3] uint8
        wrist = data["wrist_image"]
        states = data["state"].astype(np.float32)
        refined = data["refined_action"].astype(np.float32)  # [N, H, 7]
        prompts = data["prompt"]
        N = images.shape[0]
        task_text = str(data["task_description"]) if "task_description" in data.files else str(prompts[0])

        for i in range(N):
            chunk = _fit_chunk(refined[i], ACTION_HORIZON)  # [10, 7]
            dataset.add_frame(
                {
                    "image": images[i],
                    "wrist_image": wrist[i],
                    "state": states[i],
                    "actions": chunk,
                },
                task=task_text,
            )
            added += 1
        dataset.save_episode()
        if (ep_idx + 1) % 5 == 0:
            print(f"  episode {ep_idx+1}/{n_eps}, frames so far: {added}", flush=True)

    elapsed = time.time() - t0
    print(f"\nWrote {added} frames across {n_eps} episodes in {elapsed:.1f}s.")
    print(f"Local: {out_path}")

    if args.push_to_hub:
        print(f"\nPushing to HF Hub as {args.repo_id} ...")
        dataset.push_to_hub(tags=["openpi", "libero", "physics-mppi-refined", "phase-b"])
        print("Pushed.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
