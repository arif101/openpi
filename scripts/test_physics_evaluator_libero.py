"""Integration test: PhysicsEvaluator on a real LIBERO scene.

Verifies the Week-1 foundation works end-to-end on the actual benchmark:
  1. Loads LIBERO scene (BDDL → MJCF)
  2. Sets initial state from a LIBERO init state
  3. Forward-simulates a synthetic action chunk
  4. Computes named cost components

Runs on GPU box (which has libero/robosuite/mujoco==2.3.7 set up). Bypasses
the mujoco-version conflict (LIBERO pins 2.3.7; PhysicsEvaluator wants 3.x)
by extracting the XML through LIBERO's env, then loading a fresh mujoco
model from that XML.

Usage:
    PYTHONPATH=src:third_party/libero uv run python3 -u \\
        scripts/test_physics_evaluator_libero.py \\
        --task-suite libero_10 --task-idx 0
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

import numpy as np
import torch

# PyTorch 2.6 changed torch.load default to weights_only=True. LIBERO's
# get_task_init_states pickles a numpy-containing dict, so we patch torch.load
# back to the pre-2.6 behavior. Mirror of run_libero_pro_mcts.py's patch.
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task-suite", default="libero_10")
    p.add_argument("--task-idx", type=int, default=0)
    p.add_argument("--horizon", type=int, default=10, help="Action chunk length H")
    p.add_argument("--num-samples", type=int, default=8,
                   help="Random action samples to score (MPPI preview)")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    print("=" * 70, flush=True)
    print("PhysicsEvaluator integration test on LIBERO", flush=True)
    print("=" * 70, flush=True)

    # Step 1: Get the LIBERO scene XML
    print(f"\n[1] Load LIBERO scene: {args.task_suite}/task_{args.task_idx}", flush=True)
    try:
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv
        bm = benchmark.get_benchmark_dict()[args.task_suite]()
        task = bm.get_task(args.task_idx)
        bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        env = OffScreenRenderEnv(
            bddl_file_name=str(bddl), camera_heights=128, camera_widths=128,
        )
        xml = env.env.sim.model.get_xml()
        init_states = bm.get_task_init_states(args.task_idx)
        env.reset()
        env.set_init_state(init_states[0])
        init_qpos = env.env.sim.data.qpos.copy()
        init_qvel = env.env.sim.data.qvel.copy()
        env.close()
        print(f"    task: {task.language}", flush=True)
        print(f"    qpos shape: {init_qpos.shape}, qvel shape: {init_qvel.shape}", flush=True)
    except Exception as e:
        print(f"    FAIL: {e}", flush=True)
        import traceback; traceback.print_exc()
        return 1

    # Step 2: Build PhysicsEvaluator from the extracted XML
    print("\n[2] Build PhysicsEvaluator from extracted MJCF...", flush=True)
    try:
        # We need mujoco>=3.0 for PhysicsEvaluator. LIBERO pins 2.3.7.
        # Most LIBERO scenes load fine under 3.x.
        from openpi.contact_mpc.refinement.physics_evaluator import (
            PhysicsEvaluator, CostWeights,
        )
        # Robosuite's Franka uses "robot0_eef" as the end-effector body
        ev = PhysicsEvaluator.from_xml_string(
            xml,
            ee_body_name="robot0_eef",
            weights=CostWeights(),
        )
        print(f"    model: nq={ev.model.nq}, nv={ev.model.nv}, "
              f"nu={ev.model.nu}, nbody={ev.model.nbody}", flush=True)
        print(f"    ee_body_id: {ev.ee_body_id}", flush=True)
        print(f"    limited joints: {ev._jnt_limited.sum()} / {ev.model.njnt}", flush=True)
    except Exception as e:
        print(f"    FAIL: {e}", flush=True)
        import traceback; traceback.print_exc()
        return 2

    # Step 3: Forward-sim a zero action chunk
    print("\n[3] Rollout zero action chunk...", flush=True)
    try:
        H = args.horizon
        zero_action = np.zeros((H, ev.model.nu), dtype=np.float32)
        t0 = time.time()
        roll = ev.rollout(init_qpos, init_qvel, zero_action)
        dt_ms = (time.time() - t0) * 1000
        print(f"    rollout time: {dt_ms:.1f}ms ({dt_ms/H:.2f}ms per step)", flush=True)
        print(f"    initial EE pose: {roll.ee_pose_traj[0, :3]}", flush=True)
        print(f"    final EE pose:   {roll.ee_pose_traj[-1, :3]}", flush=True)
        print(f"    contact counts (raw): {roll.contact_counts.tolist()}", flush=True)
        print(f"    penetration total: {roll.penetration_total.sum():.4f}m  "
              f"(robot-only: {roll.robot_penetration.sum():.4f}m)", flush=True)
    except Exception as e:
        print(f"    FAIL: {e}", flush=True)
        import traceback; traceback.print_exc()
        return 3

    # Step 4: Cost computation
    print("\n[4] Cost breakdown for zero action vs zero prior...", flush=True)
    try:
        cost = ev.cost(roll, zero_action, prior_action=zero_action)
        print(f"    joint_limit:           {cost.joint_limit:.4f}", flush=True)
        print(f"    collision_penetration: {cost.collision_penetration:.6f}m", flush=True)
        print(f"    end_effector:          {cost.end_effector:.4f}", flush=True)
        print(f"    anchor:                {cost.anchor:.4f}", flush=True)
        print(f"    raw_contact_count:     {cost.raw_contact_count:.0f}  (diagnostic only)",
              flush=True)
        print(f"    total:                 {cost.total:.4f}", flush=True)
    except Exception as e:
        print(f"    FAIL: {e}", flush=True)
        import traceback; traceback.print_exc()
        return 4

    # Step 5: Score K random samples around zero — preview of MPPI ranking
    print(f"\n[5] Score K={args.num_samples} random action samples (MPPI preview)...",
          flush=True)
    try:
        sample_noise_std = 0.2
        nominal_target = roll.ee_pose_traj[-1, :3]  # target = where zero-action ended up
        sample_costs = []
        t0 = time.time()
        for k in range(args.num_samples):
            noise = rng.standard_normal((H, ev.model.nu)) * sample_noise_std
            sample = zero_action + noise.astype(np.float32)
            r_k = ev.rollout(init_qpos, init_qvel, sample)
            c_k = ev.cost(r_k, sample, prior_action=zero_action, ee_target_xyz=nominal_target)
            sample_costs.append((k, c_k.total, c_k))
        total_dt_ms = (time.time() - t0) * 1000
        print(f"    {args.num_samples} samples × H={H}: "
              f"{total_dt_ms:.1f}ms total ({total_dt_ms/args.num_samples:.1f}ms per sample)",
              flush=True)
        sample_costs.sort(key=lambda x: x[1])
        print(f"    cost range: {sample_costs[0][1]:.3f} (best) — {sample_costs[-1][1]:.3f} (worst)",
              flush=True)
        print(f"    spread: {sample_costs[-1][1] - sample_costs[0][1]:.3f}  "
              f"(if spread ≈ 0, MPPI won't discriminate — re-tune weights)",
              flush=True)
        print(f"    best sample components: jl={sample_costs[0][2].joint_limit:.3f} "
              f"coll={sample_costs[0][2].collision_penetration:.4f}m "
              f"ee={sample_costs[0][2].end_effector:.4f} "
              f"anchor={sample_costs[0][2].anchor:.3f}", flush=True)
    except Exception as e:
        print(f"    FAIL: {e}", flush=True)
        import traceback; traceback.print_exc()
        return 5

    # Step 6: extrapolate to full MPPI batch
    print("\n[6] Compute budget extrapolation for full MPPI:", flush=True)
    per_sample_ms = total_dt_ms / args.num_samples
    for K in (8, 32, 64, 128, 256):
        est_ms = per_sample_ms * K
        print(f"    K={K:>3}: ~{est_ms:.1f}ms per decision "
              f"({'real-time-safe' if est_ms < 100 else 'tier-2' if est_ms < 1000 else 'tier-3'})",
              flush=True)

    print("\n" + "=" * 70, flush=True)
    print("PASS. PhysicsEvaluator works end-to-end on LIBERO.", flush=True)
    print("=" * 70, flush=True)
    print("\nNext: implement MPPI refinement on top of this evaluator (Week 2).",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
