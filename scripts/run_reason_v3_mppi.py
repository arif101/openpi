"""REASON-VLA v3 — MPPI refinement integration with LIBERO.

Each decision point:
  1. Pi0.5 emits initial action chunk a_0 (in EE-delta space, 7 dims).
  2. MPPI: save env state, sample K perturbations, step each through env,
     score by (EE-to-target, joint_limit, robot_collision, anchor), restore.
  3. Take softmin-weighted average → refined chunk a*.
  4. Execute a*.

Uses LIBERO's env directly for forward sim — guarantees identical
action-space semantics to the eval execution path. State save/restore
via robosuite's sim.get_state() / set_state_from_flattened().

Hardcoded EE targets per task in scripts/reason_v3_targets.yaml. Build
this out by hand for now (Week 3 work: parse BDDL).

Kill criterion: if MPPI-refined success < Pi0.5 baseline on the gated
task, MPPI mechanism doesn't help and we tune (K, sigma, lambda, weights).

Usage:
    PYTHONPATH=src:third_party/libero uv run python3 -u \\
        scripts/run_reason_v3_mppi.py \\
        --task-suite libero_10 --task-idx 0 \\
        --num-trials 10 --perturbation-cm 5.0 --seed 7 \\
        --mppi-K 16 --mppi-sigma 0.1 --mppi-lambda 1.0 \\
        --output-dir data/contact_mpc/reason_v3_mppi
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import pathlib
import time

import numpy as np
import torch
import yaml

_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import image_tools

from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config


LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256
RESIZE_SIZE = 224

MAX_STEPS = {
    "libero_spatial": 220, "libero_object": 280, "libero_goal": 300,
    "libero_10": 520, "libero_90": 400,
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config-name", default="pi05_libero")
    p.add_argument("--checkpoint", default="gs://openpi-assets/checkpoints/pi05_libero")
    p.add_argument("--task-suite", default="libero_10")
    p.add_argument("--task-idx", type=int, default=0, help="Single task to evaluate")
    p.add_argument("--num-trials", type=int, default=10)
    p.add_argument("--num-steps-wait", type=int, default=10)
    p.add_argument("--replan-steps", type=int, default=5)
    p.add_argument("--perturbation-cm", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--episode-timeout", type=int, default=300)
    p.add_argument("--output-dir", default="data/contact_mpc/reason_v3_mppi")
    p.add_argument("--targets-yaml", default="scripts/reason_v3_targets.yaml")
    # MPPI knobs
    p.add_argument("--mppi-K", type=int, default=16, help="Number of MPPI samples")
    p.add_argument("--mppi-sigma", type=float, default=0.1, help="Sample noise std")
    p.add_argument("--mppi-lambda", type=float, default=1.0, help="Softmin temperature")
    p.add_argument("--mppi-iterations", type=int, default=1)
    p.add_argument("--mppi-trigger", choices=["always", "contact"], default="always")
    p.add_argument("--w-ee", type=float, default=10.0,
                   help="Weight for end-effector → target distance term")
    p.add_argument("--w-anchor", type=float, default=0.05,
                   help="Weight for ||action - prior||² anchor term")
    p.add_argument("--w-collision", type=float, default=100.0,
                   help="Weight for robot-involved penetration depth (in meters)")
    p.add_argument("--w-joint-limit", type=float, default=10.0,
                   help="Weight for joint-limit violations")
    p.add_argument("--verbose-mppi", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


# -------------------- LIBERO helpers --------------------


def _quat2axisangle(quat):
    quat = quat.copy()
    if quat[3] > 1.0: quat[3] = 1.0
    elif quat[3] < -1.0: quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0): return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def perturb_object_positions(env, init_state, perturbation_m, rng):
    state = init_state.clone() if hasattr(init_state, "clone") else init_state.copy()
    sim = env.env.sim
    model = sim.model
    for joint_idx in range(model.njnt):
        if model.jnt_type[joint_idx] == 0:
            pos_start = model.jnt_qposadr[joint_idx]
            state[pos_start:pos_start + 3] += rng.normal(0, perturbation_m, size=3)
    return state


def build_obs_element(obs, task_description):
    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, RESIZE_SIZE, RESIZE_SIZE))
    wrist = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist, RESIZE_SIZE, RESIZE_SIZE))
    state = np.concatenate((
        obs["robot0_eef_pos"], _quat2axisangle(obs["robot0_eef_quat"]),
        obs["robot0_gripper_qpos"],
    ))
    return {
        "observation/image": img, "observation/wrist_image": wrist,
        "observation/state": state, "prompt": str(task_description),
    }


def detect_contact_in_chunk(action_chunk: np.ndarray) -> bool:
    gripper_cmds = action_chunk[:, -1] if action_chunk.ndim == 2 else action_chunk[-1:]
    for i in range(1, len(gripper_cmds)):
        if (gripper_cmds[i] > 0) != (gripper_cmds[i - 1] > 0):
            return True
    return False


# -------------------- MPPI cost using LIBERO env directly --------------------


def _robot_geom_mask(model) -> np.ndarray:
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


def compute_step_cost(sim, is_robot_geom, ee_target_xyz, w_collision, w_joint_limit):
    """Compute per-step physics cost terms from current sim state."""
    # Robot-involved penetration depth
    pen = 0.0
    for i in range(sim.data.ncon):
        c = sim.data.contact[i]
        depth = max(0.0, -float(c.dist))
        if depth == 0:
            continue
        if is_robot_geom[c.geom1] or is_robot_geom[c.geom2]:
            pen += depth

    # Joint-limit violations (limited joints only)
    jl_viol = 0.0
    model = sim.model
    for j in range(model.njnt):
        if not model.jnt_limited[j]:
            continue
        jt = model.jnt_type[j]
        # Hinge / slide joints have a single qpos value at jnt_qposadr[j]
        if jt in (2, 3):  # mjJNT_SLIDE=2, mjJNT_HINGE=3
            q = sim.data.qpos[model.jnt_qposadr[j]]
            lo, hi = model.jnt_range[j]
            margin = min(q - lo, hi - q)
            if margin < 0:
                jl_viol += -margin

    return w_collision * pen + w_joint_limit * jl_viol


def mppi_refine(
    env,
    prior_action: np.ndarray,    # [H, 7] from Pi0.5
    ee_target_xyz: np.ndarray,
    K: int,
    sigma: float,
    lam: float,
    num_iterations: int,
    w_ee: float,
    w_anchor: float,
    w_collision: float,
    w_joint_limit: float,
    is_robot_geom: np.ndarray,
    replan_steps: int,
    rng: np.random.Generator,
):
    """Run MPPI refinement on top of Pi0.5's action chunk using LIBERO env.

    Returns:
        refined: [H, 7] same shape as prior_action
        diag: dict with mppi diagnostics
    """
    H, action_dim = prior_action.shape
    sim = env.env.sim
    saved_state = sim.get_state().flatten()
    saved_obs_state = None  # we restore via sim state only; observation is rebuilt

    def evaluate(candidate: np.ndarray) -> float:
        """Step env forward replan_steps using candidate; compute scalar cost; restore."""
        sim.set_state_from_flattened(saved_state)
        sim.forward()

        step_cost_sum = 0.0
        for action in candidate[:replan_steps]:
            _ = env.step(action.tolist())
            step_cost_sum += compute_step_cost(
                sim, is_robot_geom, ee_target_xyz, w_collision, w_joint_limit,
            )

        final_ee = sim.data.body_xpos[sim.model.body_name2id("robot0_eef")
                                       if hasattr(sim.model, "body_name2id")
                                       else sim.model.nbody - 1]
        ee_cost = float(np.linalg.norm(final_ee - ee_target_xyz))
        anchor_cost = float(np.sum((candidate - prior_action) ** 2))
        return step_cost_sum + w_ee * ee_cost + w_anchor * anchor_cost

    nominal = prior_action.copy()
    sample_costs = np.zeros(K)
    nominal_cost = evaluate(nominal)
    final_weights = None

    t0 = time.time()
    for it in range(num_iterations):
        noise = rng.standard_normal((K, H, action_dim)) * sigma
        candidates = np.clip(nominal[None, ...] + noise, -1.0, 1.0)
        for k in range(K):
            sample_costs[k] = evaluate(candidates[k])

        costs_shifted = sample_costs - sample_costs.min()
        weights = np.exp(-costs_shifted / lam)
        weights = weights / weights.sum()
        final_weights = weights

        weighted_perturbation = np.einsum(
            "k,khd->hd", weights, candidates - nominal[None, ...],
        )
        nominal = np.clip(nominal + weighted_perturbation, -1.0, 1.0).astype(prior_action.dtype)

    refined_cost = evaluate(nominal)

    # Restore env to pre-MPPI state so the outer execution proceeds normally
    sim.set_state_from_flattened(saved_state)
    sim.forward()

    diag = {
        "K": K,
        "iterations": num_iterations,
        "nominal_cost": float(nominal_cost),
        "refined_cost": float(refined_cost),
        "best_sample_cost": float(sample_costs.min()),
        "worst_sample_cost": float(sample_costs.max()),
        "cost_spread": float(sample_costs.max() - sample_costs.min()),
        "weight_entropy": float(-np.sum(final_weights * np.log(final_weights + 1e-12))),
        "wall_seconds": time.time() - t0,
    }
    return nominal, diag


# -------------------- Episode + suite eval --------------------


def run_episode(
    *, policy, env, init_state, task_description, ee_target_xyz, is_robot_geom,
    perturbation_m, perturb_rng, max_steps, args, mode, mppi_rng,
):
    env.reset()
    init_state_np = init_state.clone() if hasattr(init_state, "clone") else init_state.copy()
    if perturbation_m > 0:
        init_state_np = perturb_object_positions(env, init_state_np, perturbation_m, perturb_rng)
    obs = env.set_init_state(init_state_np)

    plan = collections.deque()
    done = False
    t = 0
    episode_start = time.time()
    last_action_chunk = None
    refinements_run = 0
    total_decisions = 0
    cost_improvements: list[tuple[float, float]] = []

    while t < max_steps + args.num_steps_wait:
        if time.time() - episode_start > args.episode_timeout:
            break

        if t < args.num_steps_wait:
            obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
            t += 1
            continue

        if not plan:
            total_decisions += 1
            obs_element = build_obs_element(obs, task_description)
            result = policy.infer(dict(obs_element))
            initial_action = np.asarray(result["actions"], dtype=np.float32)

            use_mppi = False
            if mode == "mppi":
                if args.mppi_trigger == "always":
                    use_mppi = True
                elif (args.mppi_trigger == "contact" and last_action_chunk is not None
                        and detect_contact_in_chunk(last_action_chunk)):
                    use_mppi = True

            if use_mppi:
                refined, diag = mppi_refine(
                    env, initial_action, ee_target_xyz,
                    K=args.mppi_K, sigma=args.mppi_sigma, lam=args.mppi_lambda,
                    num_iterations=args.mppi_iterations,
                    w_ee=args.w_ee, w_anchor=args.w_anchor,
                    w_collision=args.w_collision, w_joint_limit=args.w_joint_limit,
                    is_robot_geom=is_robot_geom, replan_steps=args.replan_steps,
                    rng=mppi_rng,
                )
                action_chunk = refined
                refinements_run += 1
                cost_improvements.append((diag["nominal_cost"], diag["refined_cost"]))
                if args.verbose_mppi:
                    print(
                        f"    [MPPI t={t}] nominal={diag['nominal_cost']:.3f} "
                        f"refined={diag['refined_cost']:.3f} "
                        f"best_sample={diag['best_sample_cost']:.3f} "
                        f"spread={diag['cost_spread']:.3f} "
                        f"entropy={diag['weight_entropy']:.2f} "
                        f"wall={diag['wall_seconds']*1000:.0f}ms",
                        flush=True,
                    )
            else:
                action_chunk = initial_action

            last_action_chunk = action_chunk
            plan.extend(action_chunk[:args.replan_steps])

        action = plan.popleft()
        obs, _, done, _ = env.step(action.tolist())
        if done:
            break
        t += 1

    return {
        "success": bool(done),
        "steps": t,
        "refinements_run": refinements_run,
        "total_decisions": total_decisions,
        "wall_seconds": time.time() - episode_start,
        "cost_improvements": cost_improvements,
    }


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.task_suite not in MAX_STEPS:
        raise ValueError(f"Unknown task suite: {args.task_suite}")

    # Load hardcoded EE targets from YAML
    targets_path = pathlib.Path(args.targets_yaml)
    if not targets_path.exists():
        raise FileNotFoundError(
            f"Targets YAML not found: {targets_path}\n"
            f"Expected entries like:\n"
            f"  libero_10:\n"
            f"    0:\n"
            f"      target_xyz: [x, y, z]\n"
            f"      description: 'put both ... in basket'"
        )
    targets = yaml.safe_load(targets_path.read_text())
    suite_targets = (targets or {}).get(args.task_suite, {}) or {}
    entry = suite_targets.get(args.task_idx) or suite_targets.get(str(args.task_idx))
    if entry is None:
        raise KeyError(
            f"No EE target for {args.task_suite}/task_{args.task_idx} in {targets_path}.\n"
            f"Add an entry under {args.task_suite}: {args.task_idx}: target_xyz: [x, y, z]"
        )
    ee_target_xyz = np.array(entry["target_xyz"], dtype=np.float64)
    print(f"EE target for {args.task_suite}/task_{args.task_idx}: {ee_target_xyz} "
          f"({entry.get('description', 'no description')})", flush=True)

    # Load Pi0.5
    print("Loading Pi0.5 policy...", flush=True)
    from openpi.shared import download
    download.maybe_download(args.checkpoint + "/assets")
    download.maybe_download(args.checkpoint + "/params")
    train_cfg = _config.get_config(args.config_name)
    policy = _policy_config.create_trained_policy(train_cfg, args.checkpoint)
    print("Policy loaded.", flush=True)

    bm = benchmark.get_benchmark_dict()[args.task_suite]()
    task = bm.get_task(args.task_idx)
    init_states = bm.get_task_init_states(args.task_idx)
    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl),
        camera_heights=LIBERO_ENV_RESOLUTION, camera_widths=LIBERO_ENV_RESOLUTION,
    )
    env.seed(args.seed)
    is_robot_geom = _robot_geom_mask(env.env.sim.model)
    print(f"Robot geoms identified: {is_robot_geom.sum()} / {env.env.sim.model.ngeom}",
          flush=True)

    max_steps = MAX_STEPS[args.task_suite]
    perturbation_m = args.perturbation_cm / 100.0
    trials = min(args.num_trials, len(init_states))
    mppi_rng = np.random.default_rng(args.seed + 99)

    print(f"\n{'='*70}", flush=True)
    print(f"Task: {task.language}", flush=True)
    print(f"{'='*70}", flush=True)

    results = {}
    for mode in ("baseline", "mppi"):
        perturb_rng = np.random.default_rng(args.seed + 1000)
        successes = 0
        per_trial = []
        print(f"\n--- mode={mode} (K={args.mppi_K}, σ={args.mppi_sigma}, λ={args.mppi_lambda}) ---",
              flush=True)
        for ti in range(trials):
            r = run_episode(
                policy=policy, env=env, init_state=init_states[ti],
                task_description=task.language, ee_target_xyz=ee_target_xyz,
                is_robot_geom=is_robot_geom, perturbation_m=perturbation_m,
                perturb_rng=perturb_rng, max_steps=max_steps, args=args,
                mode=mode, mppi_rng=mppi_rng,
            )
            successes += int(r["success"])
            per_trial.append(r)
            print(f"  trial {ti+1}/{trials}: {'OK' if r['success'] else 'FAIL'} "
                  f"({r['steps']} steps, {r['refinements_run']} refinements, "
                  f"{r['wall_seconds']:.1f}s)",
                  flush=True)
        results[mode] = {
            "successes": successes,
            "trials": trials,
            "success_rate": successes / trials,
            "per_trial": per_trial,
        }
        print(f"  → {mode}: {successes}/{trials} = {successes/trials*100:.1f}%", flush=True)

    env.close()

    base = results["baseline"]["success_rate"]
    mp = results["mppi"]["success_rate"]
    delta_pp = (mp - base) * 100
    print(f"\n{'='*70}", flush=True)
    print(f"baseline:   {base*100:5.1f}% ({results['baseline']['successes']}/{trials})", flush=True)
    print(f"mppi:       {mp*100:5.1f}% ({results['mppi']['successes']}/{trials})", flush=True)
    print(f"delta:      {delta_pp:+.1f} pp", flush=True)
    print(f"{'='*70}", flush=True)
    if delta_pp >= 3:
        print("✓ PASS: MPPI mechanism validates (≥+3pp).", flush=True)
    elif delta_pp >= -2:
        print("≈ FLAT: tune K / σ / λ / weights, OR target may be wrong.", flush=True)
    else:
        print("✗ FAIL: MPPI actively hurting — diagnose cost function.", flush=True)

    payload = {
        "args": vars(args),
        "task_description": task.language,
        "ee_target_xyz": ee_target_xyz.tolist(),
        "results": {
            mode: {k: v for k, v in r.items() if k != "per_trial"}
            for mode, r in results.items()
        },
        "delta_pp": delta_pp,
    }
    out = (output_dir
           / f"v3mppi_{args.task_suite}_task{args.task_idx}_"
             f"{args.perturbation_cm}cm_seed{args.seed}_K{args.mppi_K}.json")
    out.write_text(json.dumps(payload, indent=2, default=str))
    print(f"\nSaved to {out}", flush=True)


if __name__ == "__main__":
    main()
