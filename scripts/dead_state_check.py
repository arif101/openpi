"""Dead-state check for PLACEMENT_OK_SUBPRED_FAIL traces.

The question this resolves:
  When real-P3 v2 returned 0/5 recovery on PLACEMENT_OK_SUBPRED_FAIL, was that
  evidence about RECOVERABILITY (the failure exists, just unreachable by
  inference-time search → policy improvement needed) or about the FAILURE SET
  (the failure states are dead, no action sequence could ever satisfy the
  goal → the failures shouldn't have been in the matrix in the first place)?

The two answers demand opposite next moves:
  Recoverable-but-unreachable → invest in RL / training-time methods.
  Dead-states → exclude these from EVERY metric, including retroactively
                Phase A's +6.7pp result. Failure set is contaminated.

Test design (cheap, no GPU):
  For each PLACEMENT_OK_SUBPRED_FAIL trace:
    1. Restore env to the same mid-trajectory state real-P3 v2 used
       (init_sim_state + 10 wait + replay 0..restore_step).
    2. STATE-TELEPORT ORACLE: find the drawer slider joint, set qpos to
       its closed position, sim.forward(), step env once with no-op.
       Did `done` fire? If yes → goal is physically satisfiable from
       here, just unreachable by our searches.
       If no → state is dead, BDDL goal cannot be satisfied from here
       at all.
    3. (Optional) Control-level oracle if teleport passed: hand-scripted
       EE move to drawer face + push forward. Tests whether the goal is
       reachable from the control space, not just from sim state.

The state-teleport check is the load-bearing one. It separates real
unrecoverability from dead-set contamination using one bit of evidence.

Usage:
  PYTHONPATH=src:third_party/libero MUJOCO_GL=egl uv run python3 -u \\
      scripts/dead_state_check.py \\
      --traces-dir data/contact_mpc/recovery_source_traces \\
      --strata-json data/contact_mpc/failure_strata.json \\
      --stratum PLACEMENT_OK_SUBPRED_FAIL \\
      --restore-frac 0.5
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
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


def find_drawer_joints(sim_model) -> list[tuple[int, str, float]]:
    """Find joints whose name suggests they control a drawer/door.

    Returns list of (jnt_idx, name, closed_qpos). closed_qpos is the qpos
    value that represents 'closed' — usually the joint's lower limit for
    a slide joint (drawer retracted) or 0 for a hinge.
    """
    out = []
    for j in range(sim_model.njnt):
        name = sim_model.joint_id2name(j) if hasattr(sim_model, "joint_id2name") else None
        if not name: continue
        nl = name.lower()
        if "drawer" in nl or "slide" in nl or ("cabinet" in nl and ("door" in nl or "level" in nl)):
            # closed qpos: lower limit if limited, else 0
            if sim_model.jnt_limited[j]:
                lo, hi = sim_model.jnt_range[j]
                # For a drawer-slide, "closed" is the smaller |q|. Pick whichever
                # endpoint is closer to 0.
                closed = lo if abs(lo) < abs(hi) else hi
            else:
                closed = 0.0
            out.append((j, name, float(closed)))
    return out


def restore_state(env, trace_npz, restore_step: int) -> bool:
    """Restore to failure mid-state. Returns True on success."""
    env.reset()
    env.env.sim.set_state_from_flattened(trace_npz["init_sim_state"])
    env.env.sim.forward()
    for _ in range(10): env.step(LIBERO_DUMMY)
    actions = trace_npz["action"]
    for i in range(min(restore_step, actions.shape[0])):
        try: env.step(actions[i].tolist())
        except Exception: return False
    return True


def teleport_close(env, drawer_jnts: list[tuple[int, str, float]]) -> tuple[bool, dict]:
    """Set every identified drawer joint to its closed qpos, settle briefly,
    step once with no-op action to trigger BDDL evaluation. Return (done, info).
    """
    sim = env.env.sim
    mdl = sim.model
    info = {"drawer_jnts": [], "qpos_before": {}, "qpos_after": {}}
    qpos = sim.data.qpos.copy()
    qvel = sim.data.qvel.copy()
    for (jnt, name, closed) in drawer_jnts:
        addr = int(mdl.jnt_qposadr[jnt])
        info["drawer_jnts"].append({"jnt": jnt, "name": name, "addr": addr,
                                     "closed": closed, "qpos_was": float(qpos[addr])})
        info["qpos_before"][name] = float(qpos[addr])
        qpos[addr] = closed
        info["qpos_after"][name] = float(qpos[addr])
        # Zero the joint's velocity too
        dof_addr = int(mdl.jnt_dofadr[jnt])
        qvel[dof_addr] = 0.0
    cur_time = sim.get_state().time
    sim.set_state_from_flattened(np.concatenate([[cur_time], qpos, qvel]))
    sim.forward()
    # Settle for a few steps with no-op to give object physics time to react
    done = False
    for _ in range(5):
        try:
            _, _, done, _ = env.step(LIBERO_DUMMY)
        except Exception:
            break
        if done: break
    info["teleport_done"] = bool(done)
    return bool(done), info


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--traces-dir", required=True)
    p.add_argument("--strata-json", required=True)
    p.add_argument("--stratum", required=True)
    p.add_argument("--n-failures", type=int, default=5)
    p.add_argument("--restore-frac", type=float, default=0.5,
                   help="Same as real-P3 v2's restore_frac for direct comparison.")
    p.add_argument("--task-suite", default="libero_10")
    p.add_argument("--out", default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.out is None:
        args.out = f"data/contact_mpc/dead_state_check_{args.stratum}.json"

    strata = json.loads(pathlib.Path(args.strata_json).read_text())
    target = [name for name, info in strata.items() if info["label"] == args.stratum]
    target = target[:args.n_failures]
    if not target:
        print(f"No traces in stratum {args.stratum}.")
        return 1
    print(f"Dead-state check on stratum {args.stratum} ({len(target)} traces)")

    bm = benchmark.get_benchmark_dict()[args.task_suite]()
    traces_dir = pathlib.Path(args.traces_dir)
    results = []

    for tn in target:
        src_path = traces_dir / tn
        d = np.load(src_path, allow_pickle=True)
        m = re.search(r"task(\d+)", tn); tidx = int(m.group(1))
        task = bm.get_task(tidx)
        bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=128, camera_widths=128)
        env.seed(0)
        try:
            restore_step = int(d["action"].shape[0] * args.restore_frac)
            ok = restore_state(env, d, restore_step)
            if not ok:
                results.append({"trace": tn, "error": "restore_failed"})
                continue
            drawer_jnts = find_drawer_joints(env.env.sim.model)
            if not drawer_jnts:
                results.append({"trace": tn, "error": "no_drawer_joint_found",
                                 "joint_names": [env.env.sim.model.joint_id2name(j) or "" for j in range(env.env.sim.model.njnt)]})
                continue
            done, info = teleport_close(env, drawer_jnts)
            results.append({"trace": tn, "task": tidx, "restore_step": restore_step,
                            "drawer_jnts_found": len(drawer_jnts),
                            "teleport_done": done, "info": info})
            print(f"  {tn[:60]}  joints={[(j,n,c) for j,n,c in drawer_jnts]}  done={done}", flush=True)
        finally:
            env.close()

    # Summary
    n_total = len(results)
    n_done = sum(1 for r in results if r.get("teleport_done"))
    n_err = sum(1 for r in results if "error" in r)
    print(f"\n=== DEAD-STATE CHECK VERDICT ===")
    print(f"Stratum: {args.stratum}  N={n_total}  errors={n_err}")
    print(f"  teleport-close fires done: {n_done}/{n_total} = {n_done*100//max(n_total,1)}%")
    print()
    if n_done == 0:
        print("VERDICT: DEAD-STATE SET")
        print("  No teleport-to-closed triggered the BDDL goal predicate.")
        print("  These failure states cannot satisfy the goal regardless of action.")
        print("  → real-P3 v2's 0/5 was NOT a finding about recoverability.")
        print("  → exclude these states from every metric, retroactively.")
        print("  → re-examine Phase A failure cases on the same task for similar")
        print("     contamination — the +6.7pp number is potentially inflated.")
    elif n_done == n_total:
        print("VERDICT: RECOVERABLE-IN-PRINCIPLE")
        print("  Teleport-to-closed satisfied the BDDL goal on every state.")
        print("  → real-P3 v2's 0/5 is a real finding: states are recoverable")
        print("     in principle but unreachable by inference-time search.")
        print("  → policy improvement (RL fine-tune) is the right next move.")
        print("  → Backward reachability (which uses recorded successful drawer-close")
        print("     subroutine) should still work; the issue is purely about")
        print("     forward search.")
    else:
        print("VERDICT: MIXED")
        print(f"  {n_done}/{n_total} states are recoverable-in-principle, the rest are dead.")
        print(f"  → Filter the failure set: drop the {n_total-n_done} dead states.")
        print(f"  → Real recovery rate is computed only over the recoverable subset.")

    pathlib.Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nSaved to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
