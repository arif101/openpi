"""Day-1 diagnostic: perception-limited vs repertoire-limited failure regime.

Pulls grip-loss failures from the PhysVLA trace corpus, replays each
to its failure state, then tries a small library of canonical recovery
primitives. Reports per-failure recovery success.

Outcome:
  >= 3/5 recoverable by primitives → perception-limited
     → privileged-planner data engine is viable. Build it.
  <= 2/5 recoverable → repertoire-limited
     → the recovery behavior isn't reachable from these primitives.
     The planner's value would be in proposal distribution, not in
     privileged state. Rethink before investing weeks.

Primitives (each parameterized by 1-3 continuous values):
  retry_grasp(target_xyz)         — open gripper, re-approach, close
  lift_then_move(height, xyz)     — straight-up lift then lateral move
  push_and_regrasp(direction, m)  — nudge object, then retry_grasp
  open_then_close(delay)          — release-and-regrip in place

Each failure is replayed up to its last contact-loss point, then each
primitive is tried as the recovery action sequence. The privileged
planner part is: we evaluate success in the simulator (BDDL goal
predicate at end-of-rollout), not by what the deployed policy thinks.

Per the conversation that motivated this script: this is a falsifiable
gate. If primitives don't recover, the binding constraint is not
perception — it's repertoire — and the proposal distribution becomes
the central research problem rather than a detail.

Usage:
    PYTHONPATH=src:third_party/libero MUJOCO_GL=egl uv run python3 -u \\
        scripts/diagnose_recovery_regime.py \\
        --traces-dir data/contact_mpc/physvla_traces \\
        --n-failures 5
"""

from __future__ import annotations

import argparse
import glob
import pathlib
import sys
import time

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


def find_successful_traces(traces_dir: pathlib.Path, n: int,
                            task_filter: str | None = None,
                            pert_max: float = 10.0) -> list[pathlib.Path]:
    """Find PHYS_OK_*.npz traces where the task succeeded. Used by the
    controlled diagnostic: replay to T-60 of a SUCCESSFUL trace and see if
    primitives can complete the rest of the task from there.

    If primitives can complete a known-recoverable mid-trajectory state,
    they're functional. If they can't, they're inadequate regardless of
    replay drift.
    """
    import re
    candidates = []
    for p in sorted(traces_dir.glob("PHYS_OK_*.npz")):
        if task_filter and task_filter not in p.name:
            continue
        m = re.search(r"pert([\d.]+)cm", p.name)
        if m and float(m.group(1)) > pert_max:
            continue
        candidates.append(p)
        if len(candidates) >= n:
            break
    return candidates


def find_grip_loss_failures(traces_dir: pathlib.Path, n: int,
                              task_filter: str | None = None,
                              pert_max: float = 10.0) -> list[pathlib.Path]:
    """Find FAIL_*.npz traces where the EE got close to the object then lost it.

    Definition of grip-loss failure: min(EE-object distance) over trajectory
    < 8cm, AND the object never moved >5cm (so it was approached but not
    delivered).

    task_filter: substring that must appear in the filename (e.g. "task3").
    pert_max: only accept traces with perturbation cm <= this.
    """
    import re
    candidates = []
    for p in sorted(traces_dir.glob("PHYS_FAIL_*.npz")):
        if task_filter and task_filter not in p.name:
            continue
        m = re.search(r"pert([\d.]+)cm", p.name)
        if m and float(m.group(1)) > pert_max:
            continue
        d = np.load(p, allow_pickle=True)
        obj_pos = d["object_pos"]
        ee_pos = d["ee_pos"]
        if obj_pos.shape[1] == 0:
            continue
        # First tracked object (typically the one the task is about)
        ee_obj_dist = np.linalg.norm(ee_pos[:, None, :] - obj_pos, axis=-1)
        per_obj_min = ee_obj_dist.min(axis=0)   # closest EE got to each object
        per_obj_motion = np.linalg.norm(
            obj_pos.max(axis=0) - obj_pos.min(axis=0), axis=-1)
        # Find any object that the EE approached (<8cm) but that didn't move (<5cm)
        for i in range(obj_pos.shape[1]):
            if per_obj_min[i] < 0.08 and per_obj_motion[i] < 0.05:
                candidates.append((p, i, float(per_obj_min[i])))
                break
        if len(candidates) >= n:
            break
    return candidates[:n]


def replay_to_failure_point(env, trace_npz: pathlib.Path, restore_steps_before_end: int = 60):
    """Replay a trace up to `restore_steps_before_end` steps before its end,
    using the recorded action sequence. Returns the obs at that point."""
    d = np.load(trace_npz, allow_pickle=True)
    actions = d["action"]
    T = actions.shape[0]
    # Restore to ~60 steps before failure point. We pick failure_point = T-30
    # so as to be a bit before the final timeout. Replay all but the last
    # restore_steps_before_end steps.
    replay_until = max(0, T - restore_steps_before_end)
    # We don't know the env's exact init state; we re-execute actions from
    # whatever LIBERO gives us at reset. This is an approximation but for
    # the diagnostic it's enough — primitives operate on EE deltas relative
    # to the current pose, so they're robust to initial-state drift.
    env.reset()
    init_states = d.get("init_state", None)
    obs = None
    for i in range(replay_until):
        obs, _, done, _ = env.step(actions[i].tolist())
        if done:
            break
    return obs, replay_until


# -------------------- Primitive library --------------------


def _ee_delta_action(dx, dy, dz, droll=0.0, dpitch=0.0, dyaw=0.0, gripper=-1.0):
    return [float(dx), float(dy), float(dz), float(droll), float(dpitch), float(dyaw), float(gripper)]


def primitive_open_then_close(env, n_steps: int = 40):
    """Open the gripper for 20 steps, then close it for 20 steps. Tests if
    a release-and-regrip recovers a slipped grip."""
    actions = []
    for _ in range(n_steps // 2):
        actions.append(_ee_delta_action(0, 0, 0, gripper=1.0))   # open
    for _ in range(n_steps // 2):
        actions.append(_ee_delta_action(0, 0, 0, gripper=-1.0))  # close
    return _execute_actions(env, actions)


def primitive_lift_then_move(env, target_xyz, n_steps: int = 80):
    """Lift 5cm, then move toward target_xyz, gripper closed throughout."""
    obs = env.env._get_observations()
    cur_ee = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
    # Phase 1: lift 5cm (20 steps × 0.25cm)
    actions = []
    for _ in range(20):
        actions.append(_ee_delta_action(0, 0, 0.05, gripper=-1.0))
    # Phase 2: move horizontally toward target (60 steps)
    lifted_ee = cur_ee + np.array([0, 0, 0.05])
    delta = (np.asarray(target_xyz) - lifted_ee) / 60
    for _ in range(60):
        actions.append(_ee_delta_action(delta[0], delta[1], delta[2], gripper=-1.0))
    return _execute_actions(env, actions)


def primitive_retry_grasp(env, target_xyz, n_steps: int = 100):
    """Open gripper, approach target, close gripper."""
    obs = env.env._get_observations()
    cur_ee = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
    actions = []
    # Phase 1: open + lift slightly (10 steps)
    for _ in range(10):
        actions.append(_ee_delta_action(0, 0, 0.02, gripper=1.0))
    # Phase 2: move to target (60 steps)
    lifted_ee = cur_ee + np.array([0, 0, 0.02])
    delta = (np.asarray(target_xyz) - lifted_ee) / 60
    for _ in range(60):
        actions.append(_ee_delta_action(delta[0], delta[1], delta[2], gripper=1.0))
    # Phase 3: close gripper at target (10 steps)
    for _ in range(10):
        actions.append(_ee_delta_action(0, 0, 0, gripper=-1.0))
    # Phase 4: lift with object (20 steps)
    for _ in range(20):
        actions.append(_ee_delta_action(0, 0, 0.03, gripper=-1.0))
    return _execute_actions(env, actions)


def primitive_push_and_regrasp(env, target_xyz, push_direction, n_steps: int = 120):
    """Small nudge in push_direction, then retry_grasp at target_xyz."""
    obs = env.env._get_observations()
    cur_ee = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
    actions = []
    # Phase 1: nudge (20 steps)
    nudge = np.asarray(push_direction) * 0.05 / 20
    for _ in range(20):
        actions.append(_ee_delta_action(nudge[0], nudge[1], nudge[2], gripper=-1.0))
    # Phase 2-5: standard retry_grasp
    obs = env.env._get_observations()
    cur_ee = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
    for _ in range(10):
        actions.append(_ee_delta_action(0, 0, 0.02, gripper=1.0))
    lifted = cur_ee + np.array([0, 0, 0.02])
    delta = (np.asarray(target_xyz) - lifted) / 60
    for _ in range(60):
        actions.append(_ee_delta_action(delta[0], delta[1], delta[2], gripper=1.0))
    for _ in range(10):
        actions.append(_ee_delta_action(0, 0, 0, gripper=-1.0))
    for _ in range(20):
        actions.append(_ee_delta_action(0, 0, 0.03, gripper=-1.0))
    return _execute_actions(env, actions)


def _execute_actions(env, actions):
    """Step a list of actions, return final (obs, done, all_rewards)."""
    done = False
    obs = None
    for a in actions:
        if done:
            break
        try:
            obs, _, done, _ = env.step(a)
        except ValueError:
            break  # robosuite latched done
    return obs, bool(done)


# -------------------- Diagnostic loop --------------------


def diagnose_one(env, trace_path: pathlib.Path, obj_idx: int) -> dict:
    """For one failure trace, try the primitive library and report success."""
    d = np.load(trace_path, allow_pickle=True)
    task_desc = str(d["task_description"])
    # Approximate goal_xyz: the object's INTENDED location.
    # Heuristic: for grip-loss failures, the object should have been moved
    # somewhere — we use the last obj_pos as the failure-state position, and
    # add an offset in +y direction (basket / drawer typically). The task's
    # actual goal_xyz lives in reason_v3_targets.yaml but for diagnostic
    # purposes we use a simpler signal: object_pos at failure + 30cm in y.
    obj_xyz_at_failure = d["object_pos"][-1, obj_idx]

    # Read the actual goal from the yaml if available
    import yaml
    goal_yaml = pathlib.Path("/openpi/scripts/reason_v3_targets.yaml")
    goal_xyz = None
    if goal_yaml.exists():
        cfg = yaml.safe_load(goal_yaml.read_text())
        # task_idx from filename
        import re
        m = re.search(r"task(\d+)", trace_path.name)
        if m:
            tidx = int(m.group(1))
            entry = (cfg.get("libero_10", {}).get(tidx)
                     or cfg.get("libero_10", {}).get(str(tidx)))
            if entry:
                goal_xyz = np.array(entry["goal_xyz"], dtype=np.float64)
    if goal_xyz is None:
        goal_xyz = obj_xyz_at_failure + np.array([0, 0.3, 0])

    print(f"\n{'='*70}")
    print(f"Failure trace: {trace_path.name}")
    print(f"  task: {task_desc}")
    print(f"  object xyz at failure: {np.round(obj_xyz_at_failure, 3)}")
    print(f"  goal_xyz: {np.round(goal_xyz, 3)}")
    print(f"{'='*70}")

    primitives = [
        ("open_then_close", lambda: primitive_open_then_close(env, n_steps=40)),
        ("retry_grasp@obj",  lambda: primitive_retry_grasp(env, obj_xyz_at_failure)),
        ("retry_grasp@goal", lambda: primitive_retry_grasp(env, goal_xyz)),
        ("lift_then_move",   lambda: primitive_lift_then_move(env, goal_xyz)),
        ("push_and_regrasp", lambda: primitive_push_and_regrasp(env, obj_xyz_at_failure, [0, 0.1, 0])),
    ]

    results = {}
    for name, fn in primitives:
        print(f"\n  Trying {name}...", flush=True)
        # Re-replay to failure point for each primitive (state reset)
        obs, replay_steps = replay_to_failure_point(env, trace_path)
        # Try the primitive
        try:
            final_obs, done = fn()
        except Exception as e:
            print(f"    error: {type(e).__name__}: {e}")
            results[name] = False
            continue
        print(f"    after primitive: success={done}")
        results[name] = bool(done)

    any_success = any(results.values())
    print(f"\n  → recovered by primitives: {any_success}   ({sum(results.values())}/{len(results)} succeeded)")
    return {"trace": trace_path.name, "task": task_desc,
            "results": results, "any_success": any_success}


def diagnose_one_controlled(env, trace_path: pathlib.Path) -> dict:
    """Controlled: replay a SUCCESSFUL trace to T-60, then run primitives
    instead of the recorded actions. If the primitive can complete the
    task from a known-recoverable state, the primitive is functional.

    This rules out the 'replay drift' confound in the failure-diagnostic:
    if primitives work here, then the 0/5 on real failures is real
    repertoire inadequacy AND replay drift; if they don't work here, the
    primitives are simply inadequate.
    """
    d = np.load(trace_path, allow_pickle=True)
    task_desc = str(d["task_description"])
    # The original policy succeeded on this trace, so the state at T-60 is
    # known to be on a successful path. From there, can a primitive finish?
    obj_pos_at_freeze = d["object_pos"][-60, 0] if d["object_pos"].shape[0] > 60 \
        else d["object_pos"][0, 0]

    # Pull goal from yaml
    import yaml
    goal_yaml = pathlib.Path("/openpi/scripts/reason_v3_targets.yaml")
    goal_xyz = None
    if goal_yaml.exists():
        cfg = yaml.safe_load(goal_yaml.read_text())
        import re
        m = re.search(r"task(\d+)", trace_path.name)
        if m:
            tidx = int(m.group(1))
            entry = (cfg.get("libero_10", {}).get(tidx)
                     or cfg.get("libero_10", {}).get(str(tidx)))
            if entry:
                goal_xyz = np.array(entry["goal_xyz"], dtype=np.float64)
    if goal_xyz is None:
        goal_xyz = obj_pos_at_freeze + np.array([0, 0.3, 0])

    print(f"\n{'='*70}")
    print(f"CONTROLLED — successful trace, replay to T-60, run primitives")
    print(f"trace: {trace_path.name}")
    print(f"  task: {task_desc}")
    print(f"  object xyz at freeze: {np.round(obj_pos_at_freeze, 3)}")
    print(f"  goal_xyz: {np.round(goal_xyz, 3)}")
    print(f"{'='*70}")

    primitives = [
        ("retry_grasp@obj",  lambda: primitive_retry_grasp(env, obj_pos_at_freeze)),
        ("retry_grasp@goal", lambda: primitive_retry_grasp(env, goal_xyz)),
        ("lift_then_move",   lambda: primitive_lift_then_move(env, goal_xyz)),
        ("noop_continue",    lambda: _execute_actions(env, [LIBERO_DUMMY] * 200)),
    ]

    results = {}
    for name, fn in primitives:
        print(f"\n  Trying {name}...", flush=True)
        replay_to_failure_point(env, trace_path, restore_steps_before_end=60)
        try:
            _, done = fn()
        except Exception as e:
            print(f"    error: {type(e).__name__}: {e}")
            results[name] = False
            continue
        print(f"    after primitive: success={done}")
        results[name] = bool(done)

    any_success = any(results.values())
    print(f"\n  → completed by primitives: {any_success}   ({sum(results.values())}/{len(results)})")
    return {"trace": trace_path.name, "task": task_desc,
            "results": results, "any_success": any_success}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces-dir", required=True)
    ap.add_argument("--n-failures", type=int, default=5)
    ap.add_argument("--task-suite", default="libero_10")
    ap.add_argument("--task-filter", default=None,
                    help="Filename substring filter (e.g. 'task3') to focus "
                         "the diagnostic on a specific task.")
    ap.add_argument("--pert-max", type=float, default=10.0,
                    help="Max perturbation cm to include. Use 5.0 for the "
                         "publishable matrix regime; 10.0 for hard cases.")
    ap.add_argument("--mode", choices=["failure", "controlled"], default="failure",
                    help="failure: try primitives from grip-loss failure states. "
                         "controlled: try primitives from mid-trajectory of a "
                         "SUCCESSFUL trace — tests whether primitives are "
                         "functional at all, independent of replay drift.")
    args = ap.parse_args()

    traces_dir = pathlib.Path(args.traces_dir)

    if args.mode == "controlled":
        traces = find_successful_traces(
            traces_dir, args.n_failures,
            task_filter=args.task_filter, pert_max=args.pert_max,
        )
        if not traces:
            print("No successful traces found.")
            return 1
        print(f"Found {len(traces)} successful traces for controlled diagnostic.")
        for p in traces:
            print(f"  {p.name}")
        # Convert to (path, obj_idx, dist) shape so the env-build loop below works
        failures = [(p, 0, 0.0) for p in traces]
    else:
        failures = find_grip_loss_failures(
            traces_dir, args.n_failures,
            task_filter=args.task_filter, pert_max=args.pert_max,
        )
        print(f"Found {len(failures)} grip-loss failures to diagnose.\n")
        for p, obj_idx, dist in failures:
            print(f"  {p.name}  obj_idx={obj_idx}  closest_approach={dist*100:.1f}cm")
        if not failures:
            print("\nNo grip-loss failures found. Exiting.")
            return 1

    # Build an env once per task (env init is the slow part). Group failures
    # by task_idx, run them sequentially per task.
    import re
    bm = benchmark.get_benchmark_dict()[args.task_suite]()
    results = []
    for trace_path, obj_idx, _ in failures:
        m = re.search(r"task(\d+)", trace_path.name)
        if not m:
            continue
        tidx = int(m.group(1))
        task = bm.get_task(tidx)
        bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=128, camera_widths=128)
        env.seed(0)
        try:
            if args.mode == "controlled":
                r = diagnose_one_controlled(env, trace_path)
            else:
                r = diagnose_one(env, trace_path, obj_idx)
            results.append(r)
        finally:
            env.close()

    # Summary
    print(f"\n\n{'='*70}")
    print(f"DIAGNOSTIC SUMMARY ({len(results)} failures)")
    print(f"{'='*70}")
    n_recovered = sum(1 for r in results if r["any_success"])
    print(f"  Recovered by ≥1 primitive: {n_recovered}/{len(results)}")
    print()
    for r in results:
        flag = "✓" if r["any_success"] else "✗"
        prims = ",".join(k for k, v in r["results"].items() if v) or "none"
        print(f"  [{flag}] {r['trace'][:50]}  succeeded: {prims}")
    print()
    if args.mode == "controlled":
        # Different verdict logic for controlled mode.
        if n_recovered >= max(2, len(results) - 1):
            print("CONTROLLED VERDICT: PRIMITIVES FUNCTIONAL")
            print("  Primitives can complete the task from mid-trajectory of a")
            print("  successful trace. → The 0/5 result on real failures is")
            print("  likely due to replay-state-drift (we never reached the")
            print("  actual failure state). Need to add init_state save to the")
            print("  physics-trace logger and re-run failure diagnostic.")
        elif n_recovered == 0:
            print("CONTROLLED VERDICT: PRIMITIVES INADEQUATE")
            print("  Primitives can't complete the task even from a known-good")
            print("  mid-trajectory state. → The primitives themselves are too")
            print("  weak to be a recovery basis. Original REPERTOIRE-LIMITED")
            print("  verdict on failures is consistent with this. Two options:")
            print("  (a) expand the primitive library aggressively, (b) accept")
            print("  that scripted primitives aren't the right proposal mechanism")
            print("  and pivot to a learned proposal bootstrapped from successful")
            print("  trajectories.")
        else:
            print("CONTROLLED VERDICT: PARTIAL")
            print("  Some primitives work, others don't. Worth diagnosing why")
            print("  before committing to a planner architecture.")
        return 0

    if n_recovered >= 3:
        print("VERDICT: PERCEPTION-LIMITED regime")
        print("  Primitives can recover most failures. Privileged-state planning")
        print("  will work because the right action is reachable. Proceed to")
        print("  build the data engine with primitive-parameter search +")
        print("  noise-robust filter.")
    elif n_recovered <= 1:
        print("VERDICT: REPERTOIRE-LIMITED regime")
        print("  Primitives cannot recover. The right action isn't reachable")
        print("  from this primitive library. Privileged state buys nothing —")
        print("  the proposal distribution IS the problem. Rethink: either")
        print("  (a) expand the primitive library, (b) use a learned proposal")
        print("  bootstrapped from a different source, or (c) accept that")
        print("  these failures are NOT recoverable from observable state.")
    else:
        print("VERDICT: MARGINAL")
        print("  Primitives recover some failures, not most. Refine the")
        print("  primitive library and re-diagnose with a larger N before")
        print("  committing to the data-engine architecture.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
