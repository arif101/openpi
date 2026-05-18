"""Real-P3: privileged planner recovery rate on injected failure states.

This is the test we should have run on day 1 instead of the hand-scripted
primitive diagnostic. The question:

  Given a failure state where Pi0.5's prior gets stuck, can a planner
  with full simulator privilege (state save/restore, ground-truth physics,
  large search budget) find recoveries at rate ≥40%?

  Yes → perception-limited (right action exists, prior can't see it).
        Backward-reachability + (later) a learned-proposal planner can
        unlock real capability beyond Phase A.

  No  → repertoire-limited at the proposal-distribution level. Even with
        infinite simulator access, MPPI-style search around Pi0.5's prior
        can't recover. The wedge has to be elsewhere.

The "privilege" we exploit:
  - sim.set_state_from_flattened to restore failures exactly
  - K=128 candidates per MPPI call (vs Phase A's K=16)
  - σ=0.3 noise (vs Phase A's 0.1) — much wider exploration
  - mppi_iterations=3 (vs Phase A's 1) — multiple refinement passes
  - Goal-aware cost using ground-truth body_xpos (already what Phase A does)

The planner action proposal is still centered on Pi0.5's prior (perturbed
Gaussian), so this isn't unconstrained search — it's "heavy MPPI." If
heavy MPPI doesn't recover, no MPPI variant will, and the proposal
distribution itself is the binding constraint.

Test design:
  1. Find N=10 PHYS_FAIL_*_baseline_*.npz traces (Pi0.5 baseline failures)
  2. For each: restore env via init_sim_state + 10 wait + replay 0..t_p
     where t_p = 50% through trajectory (mid-failure, after policy has
     committed to a bad trajectory but before timeout)
  3. From that state, run heavy MPPI planning loop until done=True OR
     timeout (200 additional steps)
  4. Recovery rate = fraction with done=True

Usage:
  PYTHONPATH=src:third_party/libero MUJOCO_GL=egl uv run python3 -u \\
      scripts/real_p3_privileged_planner.py \\
      --traces-dir data/contact_mpc/recovery_source_traces \\
      --n-failures 10
"""

from __future__ import annotations

import argparse
import collections
import math
import pathlib
import re
import sys
import time

import numpy as np
import torch

_original_torch_load = torch.load
def _patched(*args, **kwargs):
    if "weights_only" not in kwargs: kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import image_tools

from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config


LIBERO_DUMMY = [0.0] * 6 + [-1.0]
RESIZE = 224


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--traces-dir", required=True)
    p.add_argument("--n-failures", type=int, default=10)
    p.add_argument("--task-suite", default="libero_10")
    p.add_argument("--task-filter", default=None)
    p.add_argument("--config-name", default="pi05_libero")
    p.add_argument("--checkpoint", default="gs://openpi-assets/checkpoints/pi05_libero")
    # Heavy MPPI args — the privileged-search regime
    p.add_argument("--mppi-K", type=int, default=128)
    p.add_argument("--mppi-sigma", type=float, default=0.3)
    p.add_argument("--mppi-lambda", type=float, default=0.05)
    p.add_argument("--mppi-iterations", type=int, default=3)
    p.add_argument("--replan-steps", type=int, default=5)
    # Where to restore: fraction through trajectory
    p.add_argument("--restore-frac", type=float, default=0.5,
                   help="Restore to this fraction of the original (failed) "
                        "trajectory. 0.5 = mid-failure.")
    p.add_argument("--max-recovery-steps", type=int, default=300)
    p.add_argument("--targets-yaml", default="scripts/reason_v3_targets.yaml")
    return p.parse_args()


def quat2axisangle(quat):
    q = quat.copy()
    q[3] = max(-1.0, min(1.0, q[3]))
    den = np.sqrt(1.0 - q[3]*q[3])
    if math.isclose(den, 0.0): return np.zeros(3)
    return (q[:3] * 2.0 * math.acos(q[3])) / den


def build_obs(obs, prompt):
    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, RESIZE, RESIZE))
    wrist = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist, RESIZE, RESIZE))
    state = np.concatenate([
        obs["robot0_eef_pos"],
        quat2axisangle(obs["robot0_eef_quat"]),
        obs["robot0_gripper_qpos"],
    ])
    return {
        "observation/image": img, "observation/wrist_image": wrist,
        "observation/state": state, "prompt": str(prompt),
    }


def _is_robot_geom_mask(model):
    """Identify robot-owned geoms (for filtering contacts)."""
    robot_prefixes = ("robot0_", "gripper0_", "panda")
    is_robot_body = np.zeros(model.nbody, dtype=bool)
    for b in range(model.nbody):
        name = model.body_id2name(b) if hasattr(model, "body_id2name") else ""
        if any((name or "").startswith(p) for p in robot_prefixes):
            is_robot_body[b] = True
    for b in range(model.nbody):
        parent = b
        while parent > 0:
            if is_robot_body[parent]:
                is_robot_body[b] = True
                break
            parent = model.body_parentid[parent]
    is_robot_geom = np.zeros(model.ngeom, dtype=bool)
    for g in range(model.ngeom):
        is_robot_geom[g] = is_robot_body[model.geom_bodyid[g]]
    return is_robot_geom


def step_cost(sim, is_robot_geom):
    pen = 0.0
    for i in range(sim.data.ncon):
        c = sim.data.contact[i]
        depth = max(0.0, -float(c.dist))
        if depth == 0: continue
        if is_robot_geom[c.geom1] or is_robot_geom[c.geom2]:
            pen += depth
    return 100.0 * pen  # w_collision=100


def heavy_mppi_step(env, prior_action, goal_xyz, tracked_body_id, is_robot_geom,
                     K, sigma, lam, num_iterations, replan_steps, rng):
    """Heavy MPPI: many candidates, wide noise, multi-iteration. Returns a
    refined action chunk. Same shape as Phase A's mppi_refine but with
    bigger budget."""
    H, action_dim = prior_action.shape
    sim = env.env.sim
    saved_state = sim.get_state().flatten()
    saved_timestep = getattr(env.env, "timestep", None)
    saved_cur_time = getattr(env.env, "cur_time", None)
    saved_done = getattr(env.env, "done", False)
    noised_horizon = min(replan_steps, H)

    def restore():
        sim.set_state_from_flattened(saved_state); sim.forward()
        if saved_timestep is not None: env.env.timestep = saved_timestep
        if saved_cur_time is not None: env.env.cur_time = saved_cur_time
        env.env.done = saved_done

    def evaluate(candidate):
        restore()
        step_cost_sum = 0.0
        last_ee = None
        for action in candidate[:noised_horizon]:
            try:
                obs_local, _, d_flag, _ = env.step(action.tolist())
            except ValueError:
                break
            last_ee = np.asarray(obs_local["robot0_eef_pos"], dtype=np.float64)
            step_cost_sum += step_cost(sim, is_robot_geom)
            if d_flag:
                # Big negative score for hitting goal — softmin makes it dominate
                return -1000.0
        final_tracked = sim.data.body_xpos[tracked_body_id]
        target_cost = float(np.linalg.norm(final_tracked - goal_xyz))
        approach_cost = float(np.linalg.norm(last_ee - final_tracked)) if last_ee is not None else 0.0
        anchor_cost = float(np.sum((candidate[:noised_horizon] - prior_action[:noised_horizon])**2))
        return step_cost_sum + 10.0*target_cost + 5.0*approach_cost + 0.05*anchor_cost

    nominal = prior_action.copy()
    for it in range(num_iterations):
        noise = np.zeros((K, H, action_dim), dtype=prior_action.dtype)
        noise[:, :noised_horizon] = (rng.standard_normal((K, noised_horizon, action_dim)) * sigma).astype(prior_action.dtype)
        candidates = np.clip(nominal[None] + noise, -1.0, 1.0)
        costs = np.zeros(K)
        for k in range(K):
            costs[k] = evaluate(candidates[k])
        w = np.exp(-(costs - costs.min()) / lam); w /= w.sum()
        weighted = np.einsum("k,khd->hd", w, candidates - nominal[None])
        nominal = np.clip(nominal + weighted, -1.0, 1.0).astype(prior_action.dtype)

    restore()  # leave env at pre-MPPI state for caller's execution
    return nominal


def main() -> int:
    args = parse_args()
    traces_dir = pathlib.Path(args.traces_dir)
    failures = []
    for p in sorted(traces_dir.glob("PHYS_FAIL_*_baseline_*.npz")):
        if args.task_filter and args.task_filter not in p.name: continue
        d = np.load(p, allow_pickle=True)
        if "init_sim_state" not in d.files or d["init_sim_state"].size == 0:
            continue
        failures.append(p)
        if len(failures) >= args.n_failures: break

    if not failures:
        print("No eligible failure traces with init_sim_state.")
        return 1
    print(f"Real-P3: heavy-MPPI recovery on {len(failures)} failure states")
    print(f"  K={args.mppi_K}  σ={args.mppi_sigma}  λ={args.mppi_lambda}  iters={args.mppi_iterations}")
    print()

    # Load policy once
    print("Loading Pi0.5...", flush=True)
    from openpi.shared import download
    download.maybe_download(args.checkpoint + "/assets")
    download.maybe_download(args.checkpoint + "/params")
    train_cfg = _config.get_config(args.config_name)
    policy = _policy_config.create_trained_policy(train_cfg, args.checkpoint)
    print("Pi0.5 loaded.\n", flush=True)

    import yaml
    targets = yaml.safe_load(pathlib.Path(args.targets_yaml).read_text())

    bm = benchmark.get_benchmark_dict()[args.task_suite]()
    rng = np.random.default_rng(0)
    n_recovered = 0
    results = []

    for fi, src_path in enumerate(failures):
        d = np.load(src_path, allow_pickle=True)
        m = re.search(r"task(\d+)", src_path.name)
        tidx = int(m.group(1))
        entry = targets.get("libero_10", {}).get(tidx)
        goal_xyz = np.array(entry["goal_xyz"], dtype=np.float64)
        track_names = entry.get("track_bodies", [])

        task = bm.get_task(tidx)
        bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
        env.seed(0)
        env.reset()
        env.env.sim.set_state_from_flattened(d["init_sim_state"])
        env.env.sim.forward()
        for _ in range(10): env.step(LIBERO_DUMMY)

        # Replay to restore_frac × T to reach mid-failure
        actions = d["action"]
        restore_step = int(actions.shape[0] * args.restore_frac)
        for i in range(restore_step):
            try: env.step(actions[i].tolist())
            except: break

        # Resolve tracked body ids
        is_robot_geom = _is_robot_geom_mask(env.env.sim.model)
        tracked_body_ids = []
        for nm in track_names:
            try: tracked_body_ids.append(int(env.env.sim.model.body_name2id(nm)))
            except: pass
        if not tracked_body_ids:
            print(f"  [skip] {src_path.name[:60]}: no track_bodies"); env.close(); continue

        # Heavy-MPPI planning loop
        t0 = time.time()
        plan = collections.deque()
        done = False; step_count = 0
        while step_count < args.max_recovery_steps:
            if not plan:
                # Get Pi0.5 proposal
                obs = env.env._get_observations()
                element = build_obs(obs, task.language)
                result = policy.infer(dict(element))
                a_prior = np.asarray(result["actions"], dtype=np.float32)
                # Pick tracked body: farthest from goal
                pos = env.env.sim.data.body_xpos
                tracked_id = max(tracked_body_ids,
                                 key=lambda b: np.linalg.norm(pos[b] - goal_xyz))
                # Heavy MPPI refinement
                refined = heavy_mppi_step(
                    env, a_prior, goal_xyz, tracked_id, is_robot_geom,
                    args.mppi_K, args.mppi_sigma, args.mppi_lambda,
                    args.mppi_iterations, args.replan_steps, rng,
                )
                plan.extend(refined[:args.replan_steps])
            action = plan.popleft()
            try:
                obs, _, done, _ = env.step(action.tolist())
            except ValueError:
                break
            step_count += 1
            if done: break

        wall = time.time() - t0
        env.close()
        recovered = bool(done)
        if recovered: n_recovered += 1
        results.append({
            "trace": src_path.name, "task": tidx, "restore_step": restore_step,
            "trajectory_length": int(actions.shape[0]), "recovered": recovered,
            "steps_planned": step_count, "wall_seconds": round(wall, 1),
        })
        print(f"  [{fi+1}/{len(failures)}] {src_path.name[:55]}  "
              f"task={tidx}  rec={'OK ' if recovered else 'FAIL'}  "
              f"steps={step_count}  wall={wall:.0f}s", flush=True)

    print(f"\n=== REAL-P3 VERDICT ===")
    print(f"Recovery rate: {n_recovered}/{len(failures)} = {n_recovered/len(failures)*100:.0f}%")
    print()
    if n_recovered / len(failures) >= 0.40:
        print("  ≥40% → PERCEPTION-LIMITED regime confirmed.")
        print("  Privileged simulator search finds recoveries Pi0.5 can't.")
        print("  Backward-reachability is cheap path; learned proposal is")
        print("  the v2 capability ceiling. Project thesis holds.")
    elif n_recovered / len(failures) <= 0.10:
        print("  ≤10% → REPERTOIRE-LIMITED at proposal-distribution level.")
        print("  Heavy MPPI around Pi0.5's prior can't recover. The proposal")
        print("  distribution centered on Pi0.5 is the binding constraint —")
        print("  no MPPI variant will unlock capability. We need either:")
        print("  - a different prior (RL fine-tune of Pi0.5)")
        print("  - a multimodal proposal distribution")
        print("  - acceptance that LIBERO failures of this type are unrecoverable")
        print("    via inference-time methods on top of frozen Pi0.5.")
    else:
        print("  10-40% → marginal regime. Heavy search recovers SOME but not most.")
        print("  Investigate per-task patterns. Likely worth running a larger N.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
