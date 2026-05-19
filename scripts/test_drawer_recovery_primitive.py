"""Test the stuck-detect + joint-reset + push recovery primitive.

For each IN_TRUE_CLOSE_FALSE task-3 trace:
  1. State-jump env to the trace's inflection point (sustained +y cmd, no motion).
     Uses qpos[inflection] + qvel[inflection] from the trace — direct state jump,
     no action replay (replay drifts up to 40cm; state-jump does not).
  2. Apply the recovery primitive:
     phase A: write robot qpos[0:7] to the canonical push pose, settle 10 steps with no-op.
     phase B: command sustained +y EE-delta with gripper open for N_push steps.
  3. Evaluate BDDL atomics at the end.

This is a diagnostic test of the mechanism, not a deployable primitive yet.
Joint teleport bypasses OSC; if it succeeds where Pi0.5 couldn't, it confirms
the diagnosis (OSC kinematic limitation, not Pi0.5 generation failure).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import torch

_orig = torch.load
def _patched(*args, **kwargs):
    if "weights_only" not in kwargs: kwargs["weights_only"] = False
    return _orig(*args, **kwargs)
torch.load = _patched

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv


LIBERO_DUMMY = [0.0] * 6 + [-1.0]
PUSH_ACTION = [0.0, 0.7, 0.0, 0.0, 0.0, 0.0, -1.0]  # +y, gripper open


def find_inflection(action: np.ndarray, ee: np.ndarray, window: int = 30) -> int:
    """First step where mean(cmd_dy) > 0.4 over window and max |Δee_y| < 2mm."""
    T = action.shape[0]
    for t in range(window, T - window):
        win = slice(t, t + window)
        if np.mean(action[win, 1]) > 0.4 and np.max(np.abs(np.diff(ee[win, 1]))) < 0.002:
            return t
    return T - 100


def jump_to_state(env, qpos: np.ndarray, qvel: np.ndarray, t: float = 0.0) -> None:
    """Jump env to the given full sim state (qpos + qvel)."""
    flat = np.concatenate([[t], np.asarray(qpos).ravel(), np.asarray(qvel).ravel()])
    env.env.sim.set_state_from_flattened(flat)
    env.env.sim.forward()


def teleport_robot_to_pose(env, push_pose_q: np.ndarray) -> None:
    """Overwrite robot's first 7 qpos with the canonical push pose."""
    sim = env.env.sim
    q = sim.data.qpos.copy()
    qv = sim.data.qvel.copy()
    q[:7] = push_pose_q
    qv[:7] = 0.0
    t = sim.get_state().time
    sim.set_state_from_flattened(np.concatenate([[t], q, qv]))
    sim.forward()


def eval_atomics(env) -> dict[str, bool]:
    out = {}
    for state in env.env.parsed_problem["goal_state"]:
        key = "_".join(map(str, state))
        try:
            out[key] = bool(env.env._eval_predicate(state))
        except Exception as e:
            out[key] = f"ERROR: {e}"
    return out


def run_primitive(env, push_pose_q: np.ndarray, n_settle: int = 10, n_push: int = 60) -> dict:
    """Run phase A (teleport + settle) and phase B (push). Return atomics + diagnostics."""
    out = {}
    teleport_robot_to_pose(env, push_pose_q)
    # Phase A: settle
    for _ in range(n_settle):
        try: env.step(LIBERO_DUMMY)
        except Exception: pass
    out["post_settle_atomics"] = eval_atomics(env)
    out["post_settle_ee"] = list(map(float, env.env.sim.data.site_xpos[env.env.robots[0].eef_site_id])) \
        if hasattr(env.env.robots[0], "eef_site_id") else None
    # Phase B: push
    done_any = False
    for _ in range(n_push):
        try:
            _, _, done, _ = env.step(PUSH_ACTION)
            if done: done_any = True; break
        except Exception:
            break
    out["push_done_flag"] = bool(done_any)
    out["final_atomics"] = eval_atomics(env)
    # Final state diagnostics
    mdl = env.env.sim.model
    try:
        addr = int(mdl.jnt_qposadr[mdl.joint_name2id("white_cabinet_1_bottom_level")])
        out["final_drawer_qpos"] = float(env.env.sim.data.qpos[addr])
    except Exception:
        out["final_drawer_qpos"] = None
    try:
        bid = env.env.obj_body_id["akita_black_bowl_1"]
        out["final_bowl_pos"] = list(map(float, env.env.sim.data.body_xpos[bid]))
    except Exception:
        out["final_bowl_pos"] = None
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strata-json", required=True)
    ap.add_argument("--traces-dir", required=True)
    ap.add_argument("--push-pose", required=True, help="npz from find_drawer_push_pose.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--task-suite", default="libero_10")
    ap.add_argument("--task-idx", type=int, default=3)
    ap.add_argument("--n-push", type=int, default=60)
    args = ap.parse_args()

    push = np.load(args.push_pose)
    push_pose = push["push_pose_qpos"]
    print(f"Loaded push pose: {push_pose}")

    strata = json.loads(pathlib.Path(args.strata_json).read_text())
    itcf = [(n, r) for n, r in strata.items() if r.get("label") == "IN_TRUE_CLOSE_FALSE"]
    print(f"Testing primitive on {len(itcf)} IN_TRUE_CLOSE_FALSE traces.\n")

    bm = benchmark.get_benchmark_dict()[args.task_suite]()
    task = bm.get_task(args.task_idx)
    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file

    results = []
    traces_dir = pathlib.Path(args.traces_dir)
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=128, camera_widths=128)
    env.seed(0)
    try:
        for name, r in itcf:
            d = np.load(traces_dir / name, allow_pickle=True)
            inflection = find_inflection(d["action"], d["ee_pos"])
            try:
                env.reset()
                # State-jump to inflection
                jump_to_state(env, d["qpos"][inflection], d["qvel"][inflection],
                              t=float(d["t"][inflection]) if "t" in d.files else 0.0)
                # Apply primitive
                out = run_primitive(env, push_pose, n_push=args.n_push)
                out["trace"] = name
                out["inflection_step"] = inflection
                # Verdict
                atoms = out["final_atomics"]
                close_v = next((v for k, v in atoms.items() if k.lower().startswith("close")), None)
                in_v = next((v for k, v in atoms.items() if k.lower().startswith("in")), None)
                if in_v is True and close_v is True: verdict = "SUCCESS"
                elif in_v is True and close_v is False: verdict = "STILL_IN_TRUE_CLOSE_FALSE"
                elif in_v is False: verdict = "BOWL_FELL_OUT"
                else: verdict = "OTHER"
                out["verdict"] = verdict
            except Exception as e:
                out = {"trace": name, "error": f"{type(e).__name__}: {e}", "verdict": "EXCEPTION"}
            results.append(out)
            print(f"  {name[:60]}  →  {out.get('verdict','?')}  drawer_qpos={out.get('final_drawer_qpos')}",
                  flush=True)
    finally:
        env.close()

    counts: dict[str, int] = {}
    for r in results:
        counts[r.get("verdict", "?")] = counts.get(r.get("verdict", "?"), 0) + 1
    print(f"\nResults across {len(results)} traces:")
    for k, v in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {k:35s} {v:>3d}  ({v*100//max(len(results),1)}%)")

    pathlib.Path(args.out).write_text(json.dumps(results, indent=2, default=str))
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
