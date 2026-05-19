"""Re-stratify failure traces by ATOMIC BDDL predicate evaluation.

Replaces `scripts/stratify_failures.py`'s Euclidean heuristic
(`obj_to_goal_cm < 10` as proxy for `In(bowl, drawer)`) with direct
evaluation of each atomic in the BDDL goal expression. The Euclidean
heuristic mislabeled 4/5 PLACEMENT_OK_SUBPRED_FAIL traces — re-run is
to find out the true failure-mode distribution, not to validate the
stratifier label.

Predictions written before run live in:
  memory/project_in_predicate_failure_modes.md
Compare actual output to those predictions to attribute divergences to
metric vs world.

Per-trace output:
  task_idx
  bddl_goal:                     # list of (predicate, *args) tuples
  atomic_results:                # {atomic_str: bool} for each goal atomic
  extra_probes:                  # task-specific (e.g. wrong-drawer test for task 3)
  aabb_slack:                    # {site_name: (bowl_pos - site_center) / half_extent}
  label:                         # categorical label from atomic_results pattern

Labels for tasks of form `(And (In obj region) (Close region))`:
  IN_TRUE_CLOSE_TRUE             # both atomics True → trace shouldn't be a failure (replay bug)
  IN_TRUE_CLOSE_FALSE            # legitimate "drawer-close subpredicate failed"
  IN_FALSE_CLOSE_TRUE            # bowl misplaced but drawer closed
  IN_FALSE_CLOSE_FALSE           # combination failure (likely the dominant mode)
  WRONG_DRAWER                   # bowl in top or middle region (not the goal region)
  REPLAY_DIVERGED                # restoring + replaying didn't reproduce failure

Usage:
  PYTHONPATH=src:third_party/libero MUJOCO_GL=egl uv run python3 -u \\
      scripts/restratify_failures_bddl.py \\
      --traces-dir data/contact_mpc/recovery_source_traces \\
      --out data/contact_mpc/failure_strata_bddl.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
import traceback

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


def restore_to_end(env, trace_npz) -> bool:
    """Replay full action sequence from init_sim_state. Returns True on success."""
    try:
        env.reset()
        env.env.sim.set_state_from_flattened(trace_npz["init_sim_state"])
        env.env.sim.forward()
        for _ in range(10): env.step(LIBERO_DUMMY)
        for a in trace_npz["action"]:
            env.step(a.tolist())
        return True
    except Exception:
        return False


def goal_atomic_str(state) -> str:
    if len(state) == 3:
        return f"{state[0]}({state[1]}, {state[2]})"
    elif len(state) == 2:
        return f"{state[0]}({state[1]})"
    return "?"


def aabb_slack_for_site(env, site_name: str, point_pos: np.ndarray) -> dict:
    """Return slack = (point - site_center) / half_extent per axis.
    |slack| < 1 means inside AABB.  Diagnoses near-miss false-negatives."""
    try:
        sd = env.env.object_sites_dict.get(site_name)
        if sd is None:
            return {"error": "site_not_in_dict"}
        sim = env.env.sim
        site_pos = np.array(sim.data.get_site_xpos(site_name))
        site_mat = np.array(sim.data.get_site_xmat(site_name))
        total_size = np.abs(site_mat @ sd.size)
        half = total_size  # ub - lb = 2 * total_size, half_extent = total_size
        center = site_pos
        slack = (point_pos - center) / np.maximum(half, 1e-9)
        return {
            "site_center": center.tolist(),
            "half_extent": half.tolist(),
            "bowl_pos": point_pos.tolist(),
            "slack": slack.tolist(),
            "inside": bool(np.all(np.abs(slack) < 1.0)),
        }
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def restratify_one(env, d, tidx: int) -> dict:
    out = {"task_idx": tidx}
    parsed = env.env.parsed_problem
    goal = parsed["goal_state"]
    out["bddl_goal"] = [list(s) for s in goal]

    if not restore_to_end(env, d):
        out["label"] = "RESTORE_FAILED"
        return out

    # Evaluate each atomic
    atomic_results = {}
    for state in goal:
        key = goal_atomic_str(state)
        try:
            atomic_results[key] = bool(env.env._eval_predicate(state))
        except Exception as e:
            atomic_results[key] = f"ERROR: {type(e).__name__}: {e}"
    out["atomic_results"] = atomic_results

    # Task-3-specific probe: wrong-drawer test + AABB slack
    extra = {}
    if tidx == 3:
        bowl_state = env.env.object_states_dict.get("akita_black_bowl_1")
        if bowl_state is not None:
            bowl_pos = np.array(env.env.sim.data.body_xpos[
                env.env.obj_body_id["akita_black_bowl_1"]
            ])
            extra["bowl_pos"] = bowl_pos.tolist()
            for region in ("white_cabinet_1_top_region",
                           "white_cabinet_1_middle_region",
                           "white_cabinet_1_bottom_region"):
                try:
                    site_state = env.env.object_states_dict.get(region)
                    if site_state is not None:
                        in_val = bool(site_state.check_contain(bowl_state))
                        extra[f"In(bowl, {region})"] = in_val
                    extra[f"aabb_slack[{region}]"] = aabb_slack_for_site(env, region, bowl_pos)
                except Exception as e:
                    extra[f"In(bowl, {region})_error"] = f"{type(e).__name__}: {e}"
            # Drawer slider qpos
            mdl = env.env.sim.model
            for jname in ("white_cabinet_1_top_level",
                          "white_cabinet_1_middle_level",
                          "white_cabinet_1_bottom_level"):
                try:
                    jid = mdl.joint_name2id(jname)
                    addr = int(mdl.jnt_qposadr[jid])
                    extra[f"qpos[{jname}]"] = float(env.env.sim.data.qpos[addr])
                except Exception:
                    pass
    out["extra"] = extra

    # Label assignment
    label = label_for(tidx, atomic_results, extra)
    out["label"] = label
    return out


def label_for(tidx: int, atomic_results: dict, extra: dict) -> str:
    # Task 3 pattern: (And (Close ...region) (In ...region))
    # BDDL parser lowercases predicate names — match on lower.
    if tidx == 3:
        ar = {k.lower(): v for k, v in atomic_results.items()}
        close_v = next((v for k, v in ar.items() if k.startswith("close(")), None)
        in_v = next((v for k, v in ar.items() if k.startswith("in(")), None)
        if not (isinstance(in_v, bool) and isinstance(close_v, bool)):
            return "PARSE_ERR"
        wrong = any(
            extra.get(f"In(bowl, white_cabinet_1_{r}_region)") is True
            for r in ("top", "middle")
        )
        if in_v and close_v: return "IN_TRUE_CLOSE_TRUE"  # replay-diverged from corpus PHYS_FAIL label
        if in_v and not close_v: return "IN_TRUE_CLOSE_FALSE"  # legitimate drawer-close subpred fail
        if not in_v and close_v: return "WRONG_DRAWER" if wrong else "IN_FALSE_CLOSE_TRUE"
        if not in_v and not close_v: return "WRONG_DRAWER" if wrong else "IN_FALSE_CLOSE_FALSE"
    return "OTHER"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--traces-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--task-suite", default="libero_10")
    p.add_argument("--only-task", type=int, default=None,
                   help="Restrict to one task_idx (e.g. 3).")
    p.add_argument("--max-traces", type=int, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    bm = benchmark.get_benchmark_dict()[args.task_suite]()
    traces_dir = pathlib.Path(args.traces_dir)
    paths = sorted(traces_dir.glob("PHYS_FAIL_*.npz"))

    if args.only_task is not None:
        paths = [p for p in paths if f"task{args.only_task}_" in p.name]
    if args.max_traces is not None:
        paths = paths[:args.max_traces]
    print(f"Re-stratifying {len(paths)} failure traces (task filter={args.only_task})")

    out: dict[str, dict] = {}
    counts: dict[str, int] = {}
    # Group by task to reuse env where possible
    by_task: dict[int, list[pathlib.Path]] = {}
    for p in paths:
        m = re.search(r"task(\d+)", p.name)
        if not m: continue
        by_task.setdefault(int(m.group(1)), []).append(p)

    for tidx, task_paths in sorted(by_task.items()):
        task = bm.get_task(tidx)
        bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        print(f"\n== task {tidx} ({task.bddl_file}) — {len(task_paths)} traces ==")
        env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=128, camera_widths=128)
        env.seed(0)
        try:
            for tp in task_paths:
                try:
                    d = np.load(tp, allow_pickle=True)
                    info = restratify_one(env, d, tidx)
                    info["task_description"] = str(d.get("task_description", ""))
                    info["mode"] = str(d.get("mode", ""))
                except Exception as e:
                    info = {"task_idx": tidx, "label": "EXCEPTION",
                            "error": f"{type(e).__name__}: {e}",
                            "traceback": traceback.format_exc()}
                out[tp.name] = info
                counts[info["label"]] = counts.get(info["label"], 0) + 1
                print(f"  {tp.name[:70]}  →  {info['label']}", flush=True)
        finally:
            env.close()

    pathlib.Path(args.out).write_text(json.dumps(out, indent=2, default=str))
    print(f"\nLabel distribution across {len(out)} traces:")
    for k, v in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {k:30s} {v:>3d}  ({v*100//max(len(out),1)}%)")
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
