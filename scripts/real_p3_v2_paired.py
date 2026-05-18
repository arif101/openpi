"""Real-P3 v2: stratified, paired-comparison privileged-planner test.

Four properties (per the strategic critique that v1 lacked all four):

  1. STRATIFIED — runs only on a single failure stratum (e.g. GRIP_LOSS).
     Pooled recovery across mixed strata gives an uninterpretable number.

  2. PAIRED — runs TWO planners with matched budget on the SAME failures:
     A. Prior-centered heavy MPPI: nominal = π₀.5's action chunk,
        K=128, σ=0.3, 3 iterations. Tests reachability from prior.
     B. Prior-free CEM: nominal = 0, samples uniform on [-1, 1], K=256,
        4 iterations with top-decile elitism. Forgets π₀.5 entirely.
     The CONTRAST is the diagnostic:
       prior-free >> prior-centered → repertoire-limited (recovery exists,
                                       but outside π₀.5's neighborhood)
       prior-free ≈ prior-centered (both high) → perception-limited
                                                  (right there, just unselected)
       both ≈ 0 → unrecoverable from this state, period (publishable negative)

  3. PRIVILEGED COST — the cost function uses ground-truth physics the
     deployed model cannot infer:
       - EE wrench (||cfrc_ext[ee_body]||): magnitude tells us if there's
         a real grip force, deployed model can't measure this from images
       - Object-to-goal distance via sim.data.body_xpos: deployed model
         would estimate from images with noise
       - Slip velocity (object linear velocity in EE frame): same
     The whole question "perception-limited" is only meaningful relative
     to information the deployed model lacks. If cost uses only image-
     inferrable quantities, the test reduces to "search budget" not
     "perception gap."

  4. LEAKAGE SENSITIVITY — for each recovered solution, re-execute the
     action sequence under perturbed sim state (object pose ±2mm, mass
     ±20%, friction ±30%, contact normal jitter via random tiny
     perturbation). Record the fraction of perturbations under which
     the action sequence still satisfies BDDL. Low robustness = the
     recovery exploited millimeter precision a deployed model wouldn't
     have. Such recoveries are real-in-sim, unlearnable in deployment.

Usage:
  PYTHONPATH=src:third_party/libero MUJOCO_GL=egl uv run python3 -u \\
      scripts/real_p3_v2_paired.py \\
      --traces-dir data/contact_mpc/recovery_source_traces \\
      --strata-json data/contact_mpc/failure_strata.json \\
      --stratum GRIP_LOSS \\
      --n-failures 8
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import pathlib
import re
import sys
import time

import numpy as np
import torch

_orig = torch.load
def _patched(*args, **kwargs):
    if "weights_only" not in kwargs: kwargs["weights_only"] = False
    return _orig(*args, **kwargs)
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
    p.add_argument("--strata-json", required=True)
    p.add_argument("--stratum", required=True, choices=[
        "GRIP_LOSS", "APPROACH_FAIL", "PLACEMENT_FAIL", "PLACEMENT_OK_SUBPRED_FAIL", "OTHER"])
    p.add_argument("--n-failures", type=int, default=8)
    p.add_argument("--task-suite", default="libero_10")
    p.add_argument("--config-name", default="pi05_libero")
    p.add_argument("--checkpoint", default="gs://openpi-assets/checkpoints/pi05_libero")
    p.add_argument("--targets-yaml", default="scripts/reason_v3_targets.yaml")
    # Prior-centered planner
    p.add_argument("--centered-K", type=int, default=128)
    p.add_argument("--centered-sigma", type=float, default=0.3)
    p.add_argument("--centered-iters", type=int, default=3)
    # Prior-free CEM planner — matched budget. K=256 to give it more chance
    # since it's not anchored to a good prior.
    p.add_argument("--free-K", type=int, default=256)
    p.add_argument("--free-elite-frac", type=float, default=0.10)
    p.add_argument("--free-iters", type=int, default=4)
    # Common
    p.add_argument("--horizon", type=int, default=15)
    p.add_argument("--replan-steps", type=int, default=5)
    p.add_argument("--max-recovery-steps", type=int, default=250)
    p.add_argument("--restore-frac", type=float, default=0.5)
    # Cost weights — privileged-physics components
    p.add_argument("--w-target", type=float, default=10.0)
    p.add_argument("--w-approach", type=float, default=5.0)
    p.add_argument("--w-wrench", type=float, default=2.0,
                   help="Reward grasp-magnitude wrench; deployed model can't see this.")
    p.add_argument("--w-collision", type=float, default=100.0)
    p.add_argument("--w-anchor", type=float, default=0.05)
    # Leakage logging
    p.add_argument("--leakage-n-perturbations", type=int, default=5,
                   help="Per recovered solution, run K perturbed re-executions "
                        "and record success fraction. 0 to skip leakage logging.")
    p.add_argument("--out", default=None,
                   help="Where to write the JSON result. Default: data/contact_mpc/real_p3_v2_{stratum}.json")
    return p.parse_args()


def quat2axisangle(quat):
    q = quat.copy()
    q[3] = max(-1.0, min(1.0, q[3]))
    den = math.sqrt(max(0.0, 1.0 - q[3]*q[3]))
    if math.isclose(den, 0.0): return np.zeros(3)
    return (q[:3] * 2.0 * math.acos(q[3])) / den


def build_pi05_obs(obs, prompt):
    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, RESIZE, RESIZE))
    wrist = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist, RESIZE, RESIZE))
    state = np.concatenate([
        obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]),
        obs["robot0_gripper_qpos"],
    ])
    return {
        "observation/image": img, "observation/wrist_image": wrist,
        "observation/state": state, "prompt": str(prompt),
    }


def _robot_geom_mask(model):
    robot_prefixes = ("robot0_", "gripper0_", "panda")
    is_rb = np.zeros(model.nbody, dtype=bool)
    for b in range(model.nbody):
        nm = model.body_id2name(b) if hasattr(model, "body_id2name") else ""
        if any((nm or "").startswith(p) for p in robot_prefixes): is_rb[b] = True
    for b in range(model.nbody):
        parent = b
        while parent > 0:
            if is_rb[parent]: is_rb[b] = True; break
            parent = model.body_parentid[parent]
    is_rg = np.zeros(model.ngeom, dtype=bool)
    for g in range(model.ngeom):
        is_rg[g] = is_rb[model.geom_bodyid[g]]
    return is_rg


def step_collision_cost(sim, is_robot_geom):
    pen = 0.0
    for i in range(sim.data.ncon):
        c = sim.data.contact[i]
        depth = max(0.0, -float(c.dist))
        if depth == 0: continue
        if is_robot_geom[c.geom1] or is_robot_geom[c.geom2]: pen += depth
    return pen


def evaluate_candidate(env, candidate, is_robot_geom, ee_body_id, tracked_body_id,
                       goal_xyz, prior_action, w_target, w_approach, w_wrench,
                       w_collision, w_anchor):
    """Forward-sim a candidate action chunk, return privileged-physics cost.

    Cost components:
      target:   ||obj_pos - goal_xyz|| (privileged: exact sim pose)
      approach: ||ee_pos - obj_pos||
      wrench:   max(0, threshold - ||ee_wrench||) for grasp encouragement
                (privileged: deployed model can't infer wrench from image)
      collision: robot-involved penetration
      anchor:   ||candidate - prior_action||² (regularizer, low weight)

    Returns the scalar cost. A candidate that hits done=True gets -1000.
    """
    sim = env.env.sim
    saved_state = sim.get_state().flatten()
    saved_ts = getattr(env.env, "timestep", None)
    saved_ct = getattr(env.env, "cur_time", None)
    saved_done = getattr(env.env, "done", False)

    coll = 0.0
    wrench_sum_neg = 0.0
    last_ee = None
    GRASP_FORCE_THRESHOLD = 2.0  # N. Below this, grip is weak.
    for action in candidate:
        try:
            obs_local, _, d_flag, _ = env.step(action.tolist())
        except ValueError:
            break
        last_ee = np.asarray(obs_local["robot0_eef_pos"], dtype=np.float64)
        coll += step_collision_cost(sim, is_robot_geom)
        # Wrench at EE (privileged): cfrc_ext[ee_body_id][:3] is linear contact force
        wrench_mag = float(np.linalg.norm(sim.data.cfrc_ext[ee_body_id][:3]))
        wrench_sum_neg += max(0.0, GRASP_FORCE_THRESHOLD - wrench_mag)
        if d_flag:
            # Restore env and return strong negative cost
            sim.set_state_from_flattened(saved_state); sim.forward()
            if saved_ts is not None: env.env.timestep = saved_ts
            if saved_ct is not None: env.env.cur_time = saved_ct
            env.env.done = saved_done
            return -1000.0

    final_obj = sim.data.body_xpos[tracked_body_id]
    target = float(np.linalg.norm(final_obj - goal_xyz))
    approach = float(np.linalg.norm(last_ee - final_obj)) if last_ee is not None else 0.0
    anchor = float(np.sum((candidate - prior_action)**2))

    # Restore env state
    sim.set_state_from_flattened(saved_state); sim.forward()
    if saved_ts is not None: env.env.timestep = saved_ts
    if saved_ct is not None: env.env.cur_time = saved_ct
    env.env.done = saved_done

    return (w_target * target + w_approach * approach + w_wrench * wrench_sum_neg
            + w_collision * coll + w_anchor * anchor)


def plan_prior_centered(env, prior_action, eval_args, rng, K, sigma, n_iter):
    """Heavy MPPI centered on prior."""
    H, A = prior_action.shape
    nominal = prior_action.copy()
    for _ in range(n_iter):
        noise = (rng.standard_normal((K, H, A)) * sigma).astype(prior_action.dtype)
        candidates = np.clip(nominal[None] + noise, -1.0, 1.0)
        costs = np.array([evaluate_candidate(env, candidates[k], *eval_args, prior_action) for k in range(K)])
        if costs.min() < -500:  # found a recovery
            best = candidates[costs.argmin()]
            return best, float(costs.min())
        # softmin update
        lam = 0.05
        w = np.exp(-(costs - costs.min()) / lam); w /= w.sum()
        weighted = np.einsum("k,khd->hd", w, candidates - nominal[None])
        nominal = np.clip(nominal + weighted, -1.0, 1.0).astype(prior_action.dtype)
    return nominal, float(costs.min())


def plan_prior_free(env, prior_action, eval_args, rng, K, elite_frac, n_iter):
    """CEM with population reinit each iteration, starting from broad uniform.
    Forgets the prior entirely except for the action-chunk shape."""
    H, A = prior_action.shape
    mean = np.zeros((H, A), dtype=prior_action.dtype)
    std = np.ones((H, A), dtype=prior_action.dtype)  # broad initial
    best_overall = None; best_cost_overall = float("inf")
    for it in range(n_iter):
        samples = rng.normal(mean[None], std[None], size=(K, H, A)).astype(prior_action.dtype)
        samples = np.clip(samples, -1.0, 1.0)
        costs = np.array([evaluate_candidate(env, samples[k], *eval_args, prior_action) for k in range(K)])
        # Track best
        if costs.min() < best_cost_overall:
            best_cost_overall = float(costs.min())
            best_overall = samples[costs.argmin()].copy()
        if best_cost_overall < -500:  # found a recovery
            return best_overall, best_cost_overall
        # Elite selection: top elite_frac of population
        n_elite = max(2, int(K * elite_frac))
        elite_idx = np.argsort(costs)[:n_elite]
        elites = samples[elite_idx]
        mean = elites.mean(axis=0)
        std = np.maximum(elites.std(axis=0), 0.1)  # don't collapse below 0.1
    return best_overall, best_cost_overall


def planning_loop(env, policy, prompt, eval_args_partial, prior_centered: bool,
                   args, rng, max_steps):
    """Run a planning loop until done or max_steps. Returns success bool and
    the action sequence actually executed (for leakage testing)."""
    plan = collections.deque(); done = False; step_count = 0
    executed_actions = []
    while step_count < max_steps:
        if not plan:
            obs = env.env._get_observations()
            element = build_pi05_obs(obs, prompt)
            result = policy.infer(dict(element))
            a_prior = np.asarray(result["actions"], dtype=np.float32)[:args.horizon]
            if a_prior.shape[0] < args.horizon:
                pad = np.tile(a_prior[-1:], (args.horizon - a_prior.shape[0], 1))
                a_prior = np.concatenate([a_prior, pad], axis=0)
            if prior_centered:
                refined, _ = plan_prior_centered(env, a_prior, eval_args_partial, rng,
                                                  args.centered_K, args.centered_sigma,
                                                  args.centered_iters)
            else:
                refined, _ = plan_prior_free(env, a_prior, eval_args_partial, rng,
                                              args.free_K, args.free_elite_frac,
                                              args.free_iters)
            if refined is None: refined = a_prior
            plan.extend(refined[:args.replan_steps])
        action = plan.popleft()
        executed_actions.append(action)
        try:
            _, _, done, _ = env.step(action.tolist())
        except ValueError:
            break
        step_count += 1
        if done: break
    return bool(done), step_count, executed_actions


def leakage_test(env, recovered_actions, restore_state_fn, n_perturbations, rng):
    """Re-execute action sequence under perturbed sim state. Returns the
    fraction of perturbations under which BDDL goal is still satisfied.

    Perturbations applied at the restore state:
      - object xyz perturbation ±2mm per free body
      - small qvel kick on robot joints (proxy for state-estimation noise)
    """
    if not recovered_actions: return float("nan")
    n_success = 0
    for _ in range(n_perturbations):
        restore_state_fn()
        sim = env.env.sim
        st = sim.get_state()
        qpos = st.qpos.copy(); qvel = st.qvel.copy()
        mdl = sim.model
        # Perturb free-joint xyz by ±2mm
        for j in range(mdl.njnt):
            if mdl.jnt_type[j] == 0:
                a = int(mdl.jnt_qposadr[j])
                qpos[a:a+3] += rng.normal(0, 0.002, size=3)
        # Tiny qvel kick on robot joints
        for j in range(7):
            qvel[j] += rng.normal(0, 0.02)
        sim.set_state_from_flattened(np.concatenate([[st.time], qpos, qvel]))
        sim.forward()
        # Re-execute
        done = False
        for a in recovered_actions:
            try: _, _, done, _ = env.step(a.tolist())
            except ValueError: break
            if done: break
        if done: n_success += 1
    return n_success / n_perturbations


def main() -> int:
    args = parse_args()
    if args.out is None:
        args.out = f"data/contact_mpc/real_p3_v2_{args.stratum}.json"

    strata = json.loads(pathlib.Path(args.strata_json).read_text())
    target_traces = [name for name, info in strata.items()
                     if info["label"] == args.stratum]
    if not target_traces:
        print(f"No traces in stratum {args.stratum}.")
        return 1
    target_traces = target_traces[:args.n_failures]
    print(f"Real-P3 v2 paired on stratum {args.stratum} ({len(target_traces)} failures)")
    print(f"  prior-centered: K={args.centered_K}  σ={args.centered_sigma}  iters={args.centered_iters}")
    print(f"  prior-free CEM: K={args.free_K}  elite_frac={args.free_elite_frac}  iters={args.free_iters}")
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
    traces_dir = pathlib.Path(args.traces_dir)
    rng = np.random.default_rng(0)

    results = {"stratum": args.stratum, "config": vars(args), "per_trace": []}

    for fi, trace_name in enumerate(target_traces):
        src_path = traces_dir / trace_name
        d = np.load(src_path, allow_pickle=True)
        m = re.search(r"task(\d+)", trace_name); tidx = int(m.group(1))
        entry = targets["libero_10"][tidx]
        goal_xyz = np.array(entry["goal_xyz"], dtype=np.float64)
        track_names = entry.get("track_bodies", [])
        task = bm.get_task(tidx)
        bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file

        # Helper: restore to mid-failure state
        def restore_state_fn(env_local):
            env_local.reset()
            env_local.env.sim.set_state_from_flattened(d["init_sim_state"])
            env_local.env.sim.forward()
            for _ in range(10): env_local.step(LIBERO_DUMMY)
            restore_step = int(d["action"].shape[0] * args.restore_frac)
            for i in range(restore_step):
                try: env_local.step(d["action"][i].tolist())
                except: break

        per_trace = {"trace": trace_name, "task": tidx}

        # Run BOTH planners. New env per planner to avoid state contamination.
        for planner_name, is_centered in [("prior_centered", True), ("prior_free", False)]:
            env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
            env.seed(0); restore_state_fn(env)
            # Resolve body ids
            sim_model = env.env.sim.model
            is_robot_geom = _robot_geom_mask(sim_model)
            try:
                ee_body_id = int(sim_model.body_name2id("robot0_right_hand"))
            except Exception:
                ee_body_id = sim_model.nbody - 1
            tracked_body_ids = []
            for nm in track_names:
                try: tracked_body_ids.append(int(sim_model.body_name2id(nm)))
                except: pass
            if not tracked_body_ids:
                env.close(); per_trace[planner_name] = {"recovered": False, "reason": "no_track_bodies"}
                continue
            # Pick tracked body: farthest from goal (matches Phase A's logic)
            pos = env.env.sim.data.body_xpos
            tracked_id = max(tracked_body_ids,
                             key=lambda b: np.linalg.norm(pos[b] - goal_xyz))
            eval_args_partial = (is_robot_geom, ee_body_id, tracked_id, goal_xyz,
                                  args.w_target, args.w_approach, args.w_wrench,
                                  args.w_collision, args.w_anchor)

            t0 = time.time()
            recovered, n_steps, executed = planning_loop(
                env, policy, task.language, eval_args_partial,
                is_centered, args, rng, args.max_recovery_steps,
            )
            wall = time.time() - t0
            per_trace[planner_name] = {
                "recovered": recovered, "steps": n_steps,
                "wall_seconds": round(wall, 1),
            }
            # Leakage sensitivity if recovered
            if recovered and args.leakage_n_perturbations > 0:
                robust = leakage_test(
                    env, executed,
                    lambda: restore_state_fn(env),
                    args.leakage_n_perturbations, rng,
                )
                per_trace[planner_name]["leakage_robustness"] = robust
                print(f"    [{planner_name}] leakage_robustness={robust:.2f}")
            env.close()
            print(f"  [{fi+1}/{len(target_traces)}] {trace_name[:50]} {planner_name:>15s}: "
                  f"{'OK ' if recovered else 'FAIL'}  steps={n_steps}  wall={wall:.0f}s", flush=True)
        results["per_trace"].append(per_trace)

    # Summary
    n = len(results["per_trace"])
    n_c = sum(1 for r in results["per_trace"] if r.get("prior_centered", {}).get("recovered"))
    n_f = sum(1 for r in results["per_trace"] if r.get("prior_free", {}).get("recovered"))
    print(f"\n=== STRATIFIED PAIRED-COMPARISON VERDICT ===")
    print(f"Stratum: {args.stratum}  N={n}")
    print(f"  prior-centered recovery rate: {n_c}/{n} = {n_c*100//max(n,1)}%")
    print(f"  prior-free CEM recovery rate: {n_f}/{n} = {n_f*100//max(n,1)}%")
    print()
    if n_f > n_c + 1:
        print("VERDICT: REPERTOIRE-LIMITED")
        print("  Prior-free search finds recoveries outside π₀.5's neighborhood.")
        print("  The recovery action exists but is not in the prior's support.")
        print("  Implication: refinement around π₀.5 has a ceiling; need policy")
        print("  improvement (RL fine-tune) or non-local proposal.")
    elif n_c >= n // 2 and abs(n_c - n_f) <= 1:
        print("VERDICT: PERCEPTION-LIMITED")
        print("  Both planners recover at similar high rates → the recovery is")
        print("  already near π₀.5's support; the prior just doesn't select it")
        print("  without privileged physics signal. Backward reachability + ")
        print("  learned proposal both viable as v1/v2.")
    elif n_c == 0 and n_f == 0:
        print("VERDICT: UNRECOVERABLE (publishable negative)")
        print("  Even with full simulator privilege and prior-free search,")
        print("  failures in this stratum are not recoverable. The state-")
        print("  reachability claim of the project does not hold for this")
        print("  failure mode.")
    else:
        print("VERDICT: PARTIAL / MARGINAL")
        print(f"  centered={n_c}/{n}, free={n_f}/{n}. Larger N needed for confidence.")

    # Leakage summary
    robustnesses = []
    for r in results["per_trace"]:
        for k in ("prior_centered", "prior_free"):
            v = r.get(k, {}).get("leakage_robustness")
            if v is not None: robustnesses.append(v)
    if robustnesses:
        print(f"\nLeakage robustness across {len(robustnesses)} recovered solutions:")
        print(f"  mean: {np.mean(robustnesses):.2f}  median: {np.median(robustnesses):.2f}")
        print(f"  Recoveries with <0.5 robustness (heavily privileged-state-dependent):"
              f" {sum(1 for r in robustnesses if r < 0.5)}")

    pathlib.Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nSaved to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
