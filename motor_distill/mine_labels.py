"""Auto-label miner (design module: hindsight stage/firmness/done labels for the Responsibility Gate).

Mines per-timestep supervision for the learned gate from pi0.5 pick+place demos WITHOUT any manual
labels, images, or GPU — purely from the gripper-command edges + logged object/container positions.

Replaces the hardcoded `phase = lifted>0.03` (which is degenerate on real demos) with gripper-event
segmentation:  action[:,6]  −1(open)→+1(close=grasp)→+1(hold)→−1(open=release).

Per demo we emit:
  events : close_edge (pick grasp), open_edge (place release)  — the (noun, gripper-event) ladder timing
  phase  : per-step in {0 reach, 1 grasp_servo, 2 transport, 3 place_servo, 4 done}
  p_firm : per-step is-object-held (commanded-closed ∧ object lifted)  — gate's firmness target
  p_done : per-step sub-goal completion impulses (1 at pick-complete and place-complete frames)
  binding: tgt at close_edge (pick goal), cont at open_edge (place goal) — oracle binding from the demo

Gate labels need demo TIMING; emitter labels are generated synthetically elsewhere (gen_emitter_data.py).

Usage:  python motor_distill/mine_labels.py --data ~/openpi-box-backup/data --out data/labels
"""
from __future__ import annotations

import argparse, glob, json, os, pathlib
import numpy as np

# gripper command hysteresis (action[:,6]: +1 close, -1 open)
CLOSE_HI, OPEN_LO = 0.5, -0.5
PRE_SERVO_W = 8        # hindsight window (steps) of fine-servo before a gripper event
LIFT_HELD = 0.02       # m object rise to count as firmly held
DONE_W = 3             # impulse half-width around a completed sub-goal


def gripper_state(g6: np.ndarray) -> np.ndarray:
    """Hysteretic open(0)/closed(1) state from the gripper command channel."""
    st = np.zeros(len(g6), np.int8)
    cur = 0
    for t, v in enumerate(g6):
        if v > CLOSE_HI:
            cur = 1
        elif v < OPEN_LO:
            cur = 0
        st[t] = cur
    return st


def edges(st: np.ndarray):
    """First close-edge (0→1) and the first open-edge (1→0) AFTER it."""
    d = np.diff(st.astype(int))
    closes = np.where(d == 1)[0] + 1
    opens = np.where(d == -1)[0] + 1
    if len(closes) == 0:
        return None, None
    ce = int(closes[0])
    op_after = opens[opens > ce]
    oe = int(op_after[0]) if len(op_after) else None
    return ce, oe


def mine_one(d, is_place: bool):
    T = len(d["action"])
    g6 = d["action"][:, 6]
    st = gripper_state(g6)
    ce, oe = edges(st)

    ee = d["ee"]
    tgt = d.get("tgt", d.get("goal"))           # object position (place: tgt; reach: goal)
    cont = d["cont"] if is_place and "cont" in d else None
    obj_lift = (tgt[:, 2] - tgt[0, 2]) if tgt is not None else np.zeros(T)
    max_lift = float(obj_lift.max())

    phase = np.zeros(T, np.int8)                 # 0 reach
    if ce is not None:
        phase[max(0, ce - PRE_SERVO_W):ce] = 1   # 1 grasp_servo  (fine approach before close)
        phase[ce:] = 2                            # 2 transport    (holding, carrying)
    if is_place and oe is not None:
        phase[max(0, oe - PRE_SERVO_W):oe] = 3    # 3 place_servo  (fine approach before release)
        phase[oe:] = 4                            # 4 done
    elif ce is not None:
        # reach-only demo: "done" once grasped+lifted
        grasped_lifted = np.where((st == 1) & (obj_lift > LIFT_HELD))[0]
        if len(grasped_lifted):
            phase[grasped_lifted[0]:] = 4

    p_firm = ((st == 1) & (obj_lift > LIFT_HELD)).astype(np.float32)

    p_done = np.zeros(T, np.float32)
    pick_done = ce if (ce is not None and max_lift > LIFT_HELD) else None
    if pick_done is not None:
        # pick is "done" once lifted, not at the close instant — find first lifted frame after close
        lf = np.where(obj_lift > LIFT_HELD)[0]
        lf = lf[lf >= ce]
        if len(lf):
            pd = int(lf[0])
            p_done[max(0, pd - DONE_W):pd + DONE_W + 1] = 1.0
    if is_place and oe is not None and cont is not None:
        in_cont = np.linalg.norm(tgt[oe, :2] - cont[oe, :2]) < 0.12
        if in_cont:
            p_done[max(0, oe - DONE_W):oe + DONE_W + 1] = 1.0

    binding = {
        "pick_goal": (tgt[ce] if (ce is not None and tgt is not None) else None),
        "place_goal": (cont[oe] if (is_place and oe is not None and cont is not None) else None),
    }
    valid = ce is not None and (max_lift > LIFT_HELD) and (not is_place or oe is not None)
    return dict(phase=phase, p_firm=p_firm, p_done=p_done, close_edge=ce, open_edge=oe,
                max_lift=max_lift, valid=valid, binding=binding)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.expanduser("~/openpi-box-backup/data"))
    ap.add_argument("--out", default="data/labels")
    args = ap.parse_args()
    out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)

    summ = {"place": [], "reach": []}
    for kind, is_place in (("place", True), ("reach", False)):
        files = sorted(glob.glob(os.path.join(args.data, kind, "*.npz")))
        nvalid = 0
        for f in files:
            d = np.load(f)
            lab = mine_one(d, is_place)
            stem = pathlib.Path(f).stem
            np.savez(out / f"{kind}__{stem}.npz",
                     phase=lab["phase"], p_firm=lab["p_firm"], p_done=lab["p_done"])
            nvalid += int(lab["valid"])
            summ[kind].append(dict(stem=stem, close=lab["close_edge"], open=lab["open_edge"],
                                   max_lift_cm=round(lab["max_lift"] * 100, 1), valid=lab["valid"]))
        print(f"\n=== {kind.upper()}  ({nvalid}/{len(files)} valid gripper-event segmentations) ===")
        for s in summ[kind][:12]:
            ev = f"close@{s['close']}" + (f" open@{s['open']}" if s['open'] is not None else " open@—")
            print(f"  {s['stem'][:40]:42s} {ev:22s} lift={s['max_lift_cm']:5.1f}cm "
                  f"{'OK' if s['valid'] else 'INVALID'}")
        if len(summ[kind]) > 12:
            print(f"  ... (+{len(summ[kind]) - 12} more)")
    json.dump(summ, open(out / "summary.json", "w"), indent=2, default=str)
    print(f"\nlabels -> {out}  (summary.json written)")


if __name__ == "__main__":
    main()
