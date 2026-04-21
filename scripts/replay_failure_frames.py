"""Replay failure rollouts deterministically in LIBERO to extract frames.

The rollouts_libero_90.npz file from run_collect_rollouts.py stores hidden
states and action chunks but NOT image frames. This script replays each
failure episode by re-executing the stored action sequences from the same
LIBERO init states, capturing agentview frames at each decision point.

LIBERO is deterministic given (init_state, action_sequence), so the
replayed frames reproduce the original rollout pixel-for-pixel.

Usage:
    PYTHONPATH=/workspace/openpi/src:/workspace/openpi/third_party/libero \
    /workspace/openpi/.venv/bin/python -u scripts/replay_failure_frames.py \
        --rollouts data/contact_mpc/rollouts/rollouts_libero_90.npz \
        --output-dir data/contact_mpc/frames \
        --task-suite libero_90

Smoke test (first 5 failures):
    ... --max-episodes 5
"""

import argparse
import collections
import io
import json
import math
import pathlib
import time

import numpy as np
import torch
from PIL import Image

_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--rollouts", default="data/contact_mpc/rollouts/rollouts_libero_90.npz")
    p.add_argument("--task-suite", default="libero_90")
    p.add_argument("--output-dir", default="data/contact_mpc/frames")
    p.add_argument("--num-steps-wait", type=int, default=10)
    p.add_argument("--replan-steps", type=int, default=5)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--max-episodes", type=int, default=None,
                   help="Cap on number of failure episodes to replay (for smoke testing).")
    p.add_argument("--jpeg-quality", type=int, default=80)
    p.add_argument("--only-failures", action="store_true", default=True,
                   help="Default: only replay is_success==False episodes.")
    p.add_argument("--include-successes", action="store_true",
                   help="Also replay successful episodes (e.g., for matched-pair LoRA training).")
    p.add_argument("--no-save-observations", action="store_true",
                   help=("Skip saving observations.npz. Default is to save per-decision "
                         "(agentview, wrist, state) tensors needed for DPO training."))
    return p.parse_args()


def load_episode_records(rollouts_path: str) -> dict:
    """Group decision records by episode_id.

    Returns:
        {episode_id: {"task_id": int, "is_success": bool,
                      "decisions": [(timestep, action_chunk), ...]}}
    """
    data = np.load(rollouts_path)
    hidden_states = data["hidden_states"]
    action_chunks = data["action_chunks"]
    episode_ids = data["episode_ids"]
    task_ids = data["task_ids"]
    timesteps = data["timesteps"]
    is_success = data["is_success"]

    episodes: dict[int, dict] = {}
    for i in range(len(episode_ids)):
        eid = int(episode_ids[i])
        if eid not in episodes:
            episodes[eid] = {
                "task_id": int(task_ids[i]),
                "is_success": bool(is_success[i]),
                "decisions": [],
            }
        else:
            # Sanity check: task_id and is_success should be consistent within an episode
            assert episodes[eid]["task_id"] == int(task_ids[i]), \
                f"Inconsistent task_id in episode {eid}"
            assert episodes[eid]["is_success"] == bool(is_success[i]), \
                f"Inconsistent is_success in episode {eid}"

        episodes[eid]["decisions"].append(
            (int(timesteps[i]), action_chunks[i])
        )

    # Sort decisions within each episode by timestep
    for eid in episodes:
        episodes[eid]["decisions"].sort(key=lambda x: x[0])

    return episodes


def derive_trial_indices(episodes: dict) -> dict:
    """For each episode_id, derive (task_id, trial_idx_within_task).

    Assumes collection ordering: episodes within a task were processed
    in increasing episode_id. Verifies by checking sorted episode_ids
    per task form a contiguous run.
    """
    task_to_episodes: dict[int, list[int]] = collections.defaultdict(list)
    for eid, info in episodes.items():
        task_to_episodes[info["task_id"]].append(eid)

    episode_to_trial: dict[int, int] = {}
    for task_id, eids in task_to_episodes.items():
        eids.sort()
        for trial_idx, eid in enumerate(eids):
            episode_to_trial[eid] = trial_idx

    return episode_to_trial


def extract_agentview_frame(obs) -> np.ndarray:
    """Match the flip used in run_collect_rollouts.py."""
    return np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])


def extract_wrist_frame(obs) -> np.ndarray:
    return np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])


def _quat2axisangle(quat):
    """Match the helper in run_collect_rollouts.py so state vectors align."""
    quat = quat.copy()
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def extract_state_vector(obs) -> np.ndarray:
    """7-dim: EEF pos (3) + axis-angle EEF rot (3) + gripper qpos (1)."""
    return np.concatenate((
        obs["robot0_eef_pos"],
        _quat2axisangle(obs["robot0_eef_quat"]),
        obs["robot0_gripper_qpos"],
    )).astype(np.float32)


def save_jpeg(arr: np.ndarray, path: pathlib.Path, quality: int) -> int:
    """Save a uint8 HxWx3 array as JPEG. Returns bytes written."""
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    data = buf.getvalue()
    path.write_bytes(data)
    return len(data)


def replay_episode(
    env,
    init_state,
    decisions: list,
    num_steps_wait: int,
    out_dir: pathlib.Path,
    jpeg_quality: int,
    save_observations: bool = True,
) -> dict:
    """Replay one episode, saving per-decision observations + frames.

    When save_observations=True (default), saves a single observations.npz
    per episode containing, per decision point:
      - agentview_rgb: [N, 256, 256, 3] uint8   (pre-flip matches training)
      - wrist_rgb:     [N, 256, 256, 3] uint8
      - state:         [N, 7] float32           (EEF pos + axis-angle + gripper)
      - timestep:      [N] int64

    These observations are what Pi0.5 needs to recompute its flow-matching
    loss during DPO training. The separate per-decision .jpg files remain
    for human-readable inspection and the attribution judge.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    env.reset()
    obs = env.set_init_state(init_state)

    # num_steps_wait dummy actions (t: 0 -> num_steps_wait)
    for _ in range(num_steps_wait):
        obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
        if done:
            break

    frames_saved = []
    total_bytes = 0
    done = False
    final_t = num_steps_wait

    obs_agentview: list[np.ndarray] = []
    obs_wrist: list[np.ndarray] = []
    obs_state: list[np.ndarray] = []
    obs_timesteps: list[int] = []

    for decision_timestep, action_chunk in decisions:
        # Save the JPEG for human inspection / VLM judge
        frame = extract_agentview_frame(obs)
        fpath = out_dir / f"t{decision_timestep:04d}.jpg"
        total_bytes += save_jpeg(frame, fpath, jpeg_quality)
        frames_saved.append({"timestep": decision_timestep, "path": fpath.name})

        # Save the full observation tensors (needed for DPO training)
        if save_observations:
            obs_agentview.append(frame)
            obs_wrist.append(extract_wrist_frame(obs))
            obs_state.append(extract_state_vector(obs))
            obs_timesteps.append(decision_timestep)

        # Execute the stored action chunk
        for action in action_chunk:
            obs, _, done, _ = env.step(action.tolist())
            final_t += 1
            if done:
                break
        if done:
            break

    # Save a final frame to capture terminal state
    final_frame = extract_agentview_frame(obs)
    fpath = out_dir / f"t{final_t:04d}_final.jpg"
    total_bytes += save_jpeg(final_frame, fpath, jpeg_quality)
    frames_saved.append({"timestep": final_t, "path": fpath.name, "is_final": True})

    obs_bytes = 0
    if save_observations and obs_timesteps:
        obs_path = out_dir / "observations.npz"
        np.savez_compressed(
            obs_path,
            agentview_rgb=np.stack(obs_agentview).astype(np.uint8),
            wrist_rgb=np.stack(obs_wrist).astype(np.uint8),
            state=np.stack(obs_state).astype(np.float32),
            timestep=np.array(obs_timesteps, dtype=np.int64),
        )
        obs_bytes = obs_path.stat().st_size
        total_bytes += obs_bytes

    return {
        "num_frames": len(frames_saved),
        "frames": frames_saved,
        "final_t": final_t,
        "replay_done": bool(done),
        "bytes_written": total_bytes,
        "obs_bytes": obs_bytes,
        "has_observations": save_observations and bool(obs_timesteps),
    }


def main():
    args = parse_args()

    out_root = pathlib.Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"Loading rollouts from {args.rollouts}...", flush=True)
    episodes = load_episode_records(args.rollouts)
    print(f"  Loaded {len(episodes)} episodes "
          f"({sum(1 for e in episodes.values() if e['is_success'])} success, "
          f"{sum(1 for e in episodes.values() if not e['is_success'])} failure)",
          flush=True)

    episode_to_trial = derive_trial_indices(episodes)

    # Select episodes to replay
    target_ids = sorted([
        eid for eid, info in episodes.items()
        if (not info["is_success"]) or args.include_successes
    ])
    if args.max_episodes is not None:
        target_ids = target_ids[: args.max_episodes]
    print(f"Replaying {len(target_ids)} episodes "
          f"(is_success filter: {'all' if args.include_successes else 'failures only'})",
          flush=True)

    # Set up LIBERO benchmark
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite]()

    # Cache task-level resources (envs are expensive to create)
    task_cache: dict[int, dict] = {}

    def get_task_resources(task_id: int) -> dict:
        if task_id not in task_cache:
            task = task_suite.get_task(task_id)
            init_states = task_suite.get_task_init_states(task_id)
            task_cache[task_id] = {
                "task": task,
                "init_states": init_states,
                "language": task.language,
                "bddl": str(pathlib.Path(get_libero_path("bddl_files"))
                             / task.problem_folder / task.bddl_file),
            }
        return task_cache[task_id]

    metadata = {}
    total_bytes = 0
    t_start = time.time()

    for i, eid in enumerate(target_ids):
        info = episodes[eid]
        task_id = info["task_id"]
        trial_idx = episode_to_trial[eid]

        resources = get_task_resources(task_id)
        init_states = resources["init_states"]
        if trial_idx >= len(init_states):
            print(f"  [SKIP] ep={eid} task={task_id} trial={trial_idx} "
                  f"out of range (only {len(init_states)} init states)",
                  flush=True)
            continue

        env = OffScreenRenderEnv(
            bddl_file_name=resources["bddl"],
            camera_heights=LIBERO_ENV_RESOLUTION,
            camera_widths=LIBERO_ENV_RESOLUTION,
        )
        env.seed(args.seed)

        ep_dir = out_root / f"ep_{eid:04d}"
        result = replay_episode(
            env=env,
            init_state=init_states[trial_idx],
            decisions=info["decisions"],
            num_steps_wait=args.num_steps_wait,
            out_dir=ep_dir,
            jpeg_quality=args.jpeg_quality,
            save_observations=not args.no_save_observations,
        )
        env.close()

        metadata[eid] = {
            "task_id": task_id,
            "trial_idx": trial_idx,
            "task_language": resources["language"],
            "is_success": info["is_success"],
            "num_decisions": len(info["decisions"]),
            "num_frames": result["num_frames"],
            "final_t": result["final_t"],
            "frames": result["frames"],
        }
        total_bytes += result["bytes_written"]

        if (i + 1) % 10 == 0 or i == len(target_ids) - 1:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed
            eta_min = (len(target_ids) - i - 1) / rate / 60 if rate > 0 else 0
            print(f"  [{i+1}/{len(target_ids)}] ep={eid:04d} task={task_id} "
                  f"trial={trial_idx} frames={result['num_frames']} "
                  f"| elapsed={elapsed/60:.1f}min eta={eta_min:.1f}min "
                  f"| data={total_bytes/1e6:.1f}MB",
                  flush=True)

    meta_path = out_root / "metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2))
    print(f"\nWrote metadata for {len(metadata)} episodes to {meta_path}", flush=True)
    print(f"Total frame data: {total_bytes/1e6:.1f}MB", flush=True)


if __name__ == "__main__":
    main()
