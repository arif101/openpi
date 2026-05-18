"""Backward-reachability recovery data generator.

Premise (per the cross-field-transfer analysis):
  Forward recovery search is hard — sparse reward, large action space, cold-start.
  Backward generation is trivial — take a successful trajectory, perturb a
  mid-state, and the *rest of the recorded trajectory* is the recovery action
  sequence by construction. We never search; we exploit the asymmetry of
  many-to-one dynamics (many perturbed states recover under the same actions)
  plus the simulator's determinism + state-restore.

Mechanism:
  For each successful baseline trace (with init_sim_state):
    For each of K perturbation points t_p ∈ {sampled timesteps}:
      For each of N perturbation samples:
        1. Restore env to t_p via init_sim_state + 10 wait + replay(0..t_p)
        2. Apply a state perturbation:
             - EE offset (1-3cm), OR
             - object xyz offset (1-3cm), OR
             - small qvel kick
        3. Roll the recorded action sequence (t_p..T-1) forward in MuJoCo
        4. Check BDDL goal (env.step's done flag) at the end
        5. If satisfied: save (obs_at_perturbed_state, recorded_actions[t_p:T])
           as a (failure_state, recovery_action_chunk) supervised pair

Info-bottleneck:
  The saved obs is ONLY image + wrist_image + EE pose + gripper_qpos — exactly
  the modalities a deployed model conditioned on observations would have. The
  privileged physics state (full qpos, qvel, contact wrenches) is used ONLY by
  the simulator during rollout verification. It never enters the supervision
  target. So the trained recovery model cannot learn to depend on information
  it won't have at inference.

Output:
  One NPZ per source trace, with arrays of shape [N_valid, ...] containing
  the successful (perturbed_obs, recovery_action) pairs. Filter rate is
  expected to be moderate — small perturbations often recover under the
  original actions; large ones don't. That's intentional: the perturbations
  that DON'T recover are out-of-tube and aren't valid recovery demonstrations.

Usage:
  PYTHONPATH=src:third_party/libero MUJOCO_GL=egl uv run python3 -u \\
      scripts/generate_recovery_pairs.py \\
      --traces-dir data/contact_mpc/physvla_traces_diag \\
      --output-dir data/contact_mpc/recovery_pairs \\
      --perturb-points 4 \\
      --perturb-samples-per-point 8 \\
      --max-traces 20
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time
from typing import Any

import numpy as np
import torch

_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv


LIBERO_DUMMY = [0.0] * 6 + [-1.0]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--traces-dir", required=True,
                   help="Directory of PHYS_OK_*.npz traces with init_sim_state.")
    p.add_argument("--output-dir", required=True,
                   help="Where to write recovery-pair NPZs (one per source trace).")
    p.add_argument("--perturb-points", type=int, default=4,
                   help="Number of perturbation points sampled along each trajectory.")
    p.add_argument("--perturb-samples-per-point", type=int, default=8,
                   help="Number of perturbation samples generated at each point.")
    p.add_argument("--ee-perturb-cm", type=float, default=2.5,
                   help="Std of EE position perturbation (cm).")
    p.add_argument("--obj-perturb-cm", type=float, default=2.0,
                   help="Std of object position perturbation (cm).")
    p.add_argument("--max-traces", type=int, default=999999,
                   help="Cap on number of source traces to process.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--task-suite", default="libero_10")
    p.add_argument("--task-filter", default=None,
                   help="Filename substring filter (e.g. 'task3').")
    p.add_argument("--baseline-only", action="store_true", default=True,
                   help="Only use baseline-mode traces (mppi-mode has replay drift).")
    return p.parse_args()


def list_source_traces(traces_dir: pathlib.Path, task_filter: str | None,
                        baseline_only: bool, max_traces: int) -> list[pathlib.Path]:
    """Find PHYS_OK_*.npz traces eligible as backward-reachability sources."""
    paths = []
    for p in sorted(traces_dir.glob("PHYS_OK_*.npz")):
        if task_filter and task_filter not in p.name:
            continue
        if baseline_only and "_baseline_" not in p.name:
            continue
        # Quick check: has init_sim_state?
        d = np.load(p, allow_pickle=True)
        if "init_sim_state" not in d.files or d["init_sim_state"].size == 0:
            continue
        paths.append(p)
        if len(paths) >= max_traces:
            break
    return paths


def restore_to_step(env, trace_npz: dict, target_step: int) -> bool:
    """Restore env to trace's state at given step. Returns True on success.

    Mechanism: env.reset() -> set_state_from_flattened(init_sim_state) ->
    do 10 wait steps -> replay recorded actions 0..target_step.
    """
    env.reset()
    env.env.sim.set_state_from_flattened(trace_npz["init_sim_state"])
    env.env.sim.forward()
    for _ in range(10):
        env.step(LIBERO_DUMMY)
    actions = trace_npz["action"]
    for i in range(min(target_step, actions.shape[0])):
        try:
            env.step(actions[i].tolist())
        except Exception:
            return False
    return True


def apply_perturbation(env, kind: str, rng: np.random.Generator,
                        ee_perturb_m: float, obj_perturb_m: float,
                        obj_body_ids: list[int]):
    """Apply a small perturbation to the current sim state."""
    sim = env.env.sim
    state = sim.get_state()
    qpos = state.qpos.copy()
    qvel = state.qvel.copy()

    if kind == "ee_offset":
        # Perturb the robot's joint 4-6 slightly to displace EE without
        # breaking task feasibility. Apply to qpos directly.
        for j in range(4, 7):
            qpos[j] += rng.normal(0, 0.05)  # ~3 degree std
    elif kind == "obj_offset":
        # Find object qpos addresses and perturb xyz of one random object.
        if not obj_body_ids:
            return
        mdl = sim.model
        free_joint_qpos_addrs = []
        for j in range(mdl.njnt):
            if mdl.jnt_type[j] == 0:  # free joint
                bid = int(mdl.jnt_bodyid[j])
                if bid in obj_body_ids:
                    free_joint_qpos_addrs.append(int(mdl.jnt_qposadr[j]))
        if not free_joint_qpos_addrs:
            return
        addr = rng.choice(free_joint_qpos_addrs)
        qpos[addr:addr+3] += rng.normal(0, obj_perturb_m, size=3)
    elif kind == "ee_velocity_kick":
        # Small qvel kick on the robot's joints (transient)
        for j in range(7):
            qvel[j] += rng.normal(0, 0.1)
    else:
        raise ValueError(kind)

    # Apply
    sim.set_state_from_flattened(np.concatenate([[state.time], qpos, qvel]))
    sim.forward()


def roll_recorded_actions(env, actions: np.ndarray, start_step: int) -> bool:
    """Roll the recorded actions from start_step to end, return done flag."""
    done = False
    for i in range(start_step, actions.shape[0]):
        try:
            _, _, done, _ = env.step(actions[i].tolist())
        except Exception:
            return False
        if done:
            break
    return bool(done)


def capture_obs(env) -> dict:
    """Capture deployable-modality observation: images + EE pose + gripper qpos.

    Critical info-bottleneck: this is what the deployed recovery model will see
    at inference. Everything else (full qpos, qvel, wrench, object_pos) is
    privileged and intentionally excluded so the trained model can't depend on it.
    """
    obs = env.env._get_observations()
    # LIBERO images are upside-down + mirrored; the original pipeline flips.
    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]).copy()
    wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1]).copy()
    return {
        "image": img,
        "wrist_image": wrist,
        "ee_pos": np.asarray(obs["robot0_eef_pos"], dtype=np.float32),
        "ee_quat": np.asarray(obs["robot0_eef_quat"], dtype=np.float32),
        "gripper_qpos": np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32),
    }


def main() -> int:
    args = parse_args()
    traces_dir = pathlib.Path(args.traces_dir)
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sources = list_source_traces(traces_dir, args.task_filter,
                                  args.baseline_only, args.max_traces)
    print(f"Found {len(sources)} eligible source traces.")
    if not sources:
        return 1

    rng = np.random.default_rng(args.seed)
    perturbation_kinds = ["ee_offset", "obj_offset", "ee_velocity_kick"]
    bm = benchmark.get_benchmark_dict()[args.task_suite]()

    total_attempted = 0
    total_valid = 0
    t_start = time.time()

    import re
    for src_path in sources:
        d = np.load(src_path, allow_pickle=True)
        actions = d["action"]
        T = actions.shape[0]
        if T < 50:
            continue

        # Resolve task index + env
        m = re.search(r"task(\d+)", src_path.name)
        if not m: continue
        tidx = int(m.group(1))
        task = bm.get_task(tidx)
        bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
        env.seed(args.seed)

        # Resolve object body ids for perturbation kind "obj_offset"
        env.reset()
        obj_body_ids = []
        if "object_names" in d.files:
            for nm in d["object_names"]:
                try:
                    obj_body_ids.append(int(env.env.sim.model.body_name2id(str(nm))))
                except Exception:
                    pass

        # Sample perturbation points: evenly spaced fractions of the trajectory.
        # Avoid the very beginning (no time for failure to matter) and very end
        # (no recorded actions left to recover with).
        frac_min, frac_max = 0.15, 0.65
        fracs = np.linspace(frac_min, frac_max, args.perturb_points)
        perturb_points = [max(20, int(T * f)) for f in fracs]

        valid_pairs: list[dict] = []
        per_trace_attempts = 0
        per_trace_valid = 0
        per_trace_t0 = time.time()

        for t_p in perturb_points:
            for sample_i in range(args.perturb_samples_per_point):
                per_trace_attempts += 1
                total_attempted += 1
                # Restore env to t_p
                if not restore_to_step(env, d, t_p):
                    continue
                # Capture pre-perturbation obs (this becomes the "perturbed state"
                # AFTER we apply the perturbation, so capture after instead).
                # Apply perturbation
                kind = perturbation_kinds[sample_i % len(perturbation_kinds)]
                try:
                    apply_perturbation(env, kind, rng,
                                       args.ee_perturb_cm/100,
                                       args.obj_perturb_cm/100, obj_body_ids)
                except Exception as e:
                    continue
                # Capture obs at the perturbed state
                try:
                    obs_at_perturb = capture_obs(env)
                except Exception:
                    continue
                # Roll the recorded actions forward and check goal
                ok = roll_recorded_actions(env, actions, t_p)
                if not ok:
                    continue
                # Valid: save (obs, recovery action chunk)
                recovery_chunk = actions[t_p:].astype(np.float32)
                valid_pairs.append({
                    "obs": obs_at_perturb,
                    "recovery_action_chunk": recovery_chunk,
                    "source_trace": src_path.name,
                    "perturb_step": int(t_p),
                    "perturb_kind": kind,
                    "trajectory_length": int(T),
                })
                per_trace_valid += 1
                total_valid += 1

        env.close()
        elapsed = time.time() - per_trace_t0
        accept_rate = per_trace_valid / max(per_trace_attempts, 1)
        print(f"  {src_path.name[:60]}  attempts={per_trace_attempts}  "
              f"valid={per_trace_valid}  rate={accept_rate*100:.0f}%  "
              f"wall={elapsed:.1f}s", flush=True)

        if valid_pairs:
            # Save per source-trace NPZ. action chunks have variable length
            # (depends on which t_p they started at), so we store as a list
            # of object arrays.
            chunk_lens = np.array([p["recovery_action_chunk"].shape[0] for p in valid_pairs], dtype=np.int32)
            max_chunk = int(chunk_lens.max())
            padded_chunks = np.zeros((len(valid_pairs), max_chunk, 7), dtype=np.float32)
            for i, p in enumerate(valid_pairs):
                ch = p["recovery_action_chunk"]
                padded_chunks[i, :ch.shape[0]] = ch
            images = np.stack([p["obs"]["image"] for p in valid_pairs])
            wrist_images = np.stack([p["obs"]["wrist_image"] for p in valid_pairs])
            ee_pos = np.stack([p["obs"]["ee_pos"] for p in valid_pairs])
            ee_quat = np.stack([p["obs"]["ee_quat"] for p in valid_pairs])
            gripper_qpos = np.stack([p["obs"]["gripper_qpos"] for p in valid_pairs])
            perturb_steps = np.array([p["perturb_step"] for p in valid_pairs], dtype=np.int32)
            perturb_kinds = np.array([p["perturb_kind"] for p in valid_pairs], dtype=object)
            out_path = output_dir / f"RECOVERY_{src_path.stem}.npz"
            np.savez_compressed(
                out_path,
                image=images, wrist_image=wrist_images,
                ee_pos=ee_pos, ee_quat=ee_quat, gripper_qpos=gripper_qpos,
                recovery_action_chunk=padded_chunks,
                recovery_action_chunk_lengths=chunk_lens,
                perturb_step=perturb_steps, perturb_kind=perturb_kinds,
                source_trace=src_path.name,
                trajectory_length=int(T),
            )

    elapsed = time.time() - t_start
    print(f"\nDone. {total_valid} valid pairs / {total_attempted} attempts "
          f"({total_valid/max(total_attempted,1)*100:.0f}% rate) in {elapsed:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
