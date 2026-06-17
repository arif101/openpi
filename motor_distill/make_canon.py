"""Transform pi0.5 dual-goal distillation data -> SE(2) canonical frame, several MODES (ablation). The motor's position
leak is the in-plane BEARING of the target. We remove that DOF by construction (rotate world about gravity-z so a
reference goal points +x), apply identically to train data and at inference, and rotate predicted actions back. Modes
differ ONLY in which bearing defines the frame each step -- all are parameter-free symmetry choices, no state machine:

  episode      : ONE phi for the whole rollout = object's INITIAL bearing (fixed; image rotation goes stale in transport)
  perstep_obj  : phi each step = bearing to the object PICKUP spot (fixed world point -> well-defined even while carrying)
  gripkey      : phi each step = bearing to object while gripper OPEN, to container while CLOSED (keyed on the gripper
                 finger-separation SENSOR -- a coordinate choice from an observable, not a hand-coded subgoal predicate)
  none         : identity (no canon)

Run: .venv/bin/python motor_distill/make_canon.py --src data/dual_swap --out data/dual_swap_perstep --mode perstep_obj
"""
from __future__ import annotations
import argparse, glob, pathlib
import numpy as np
from canon import canon_angle, rot_image, rot_vec_xy, rot_chunk_xy

GRIP_OPEN_SEP = 0.05   # finger-separation (proprio[3]-proprio[4]) above this = OPEN; below = CLOSED (physical sensor gap)


def gripper_open(proprio) -> bool:
    return float(proprio[3] - proprio[4]) > GRIP_OPEN_SEP


def step_phi(mode, obj_rel, cont_rel, proprio):
    if mode == "perstep_obj":
        return -canon_angle(obj_rel)
    if mode == "gripkey":
        return -canon_angle(obj_rel if gripper_open(proprio) else cont_rel)
    return None   # episode / none handled by caller


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", default="data/dual_swap"); p.add_argument("--out", default="data/dual_swap_canon")
    p.add_argument("--mode", default="perstep_obj", choices=["episode", "perstep_obj", "gripkey", "none"])
    args = p.parse_args()
    out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)
    files = sorted(glob.glob(str(pathlib.Path(args.src) / "*.npz")))
    n = 0
    for f in files:
        d = np.load(f)
        if "wrist" not in d or "chunk" not in d or "obj_rel" not in d: continue
        W, O, Co, P, C = d["wrist"], d["obj_rel"], d["cont_rel"], d["proprio"], d["chunk"]
        ep_phi = -canon_angle(O[0]) if args.mode == "episode" else None
        wc, oc, cc2, pc, ch = [], [], [], [], []
        for i in range(len(W)):
            if args.mode == "none":
                phi = 0.0
            elif args.mode == "episode":
                phi = ep_phi
            else:
                phi = step_phi(args.mode, O[i], Co[i], P[i])
            wc.append(rot_image(W[i], phi) if phi else W[i])
            oc.append(rot_vec_xy(O[i], phi)); cc2.append(rot_vec_xy(Co[i], phi))
            p2 = np.asarray(P[i], np.float32).copy(); p2[:3] = rot_vec_xy(p2[:3], phi); pc.append(p2)
            ch.append(rot_chunk_xy(C[i], phi))
        np.savez_compressed(out / pathlib.Path(f).name,
                            wrist=np.asarray(wc, np.uint8), obj_rel=np.asarray(oc, np.float32),
                            cont_rel=np.asarray(cc2, np.float32), proprio=np.asarray(pc, np.float32),
                            chunk=np.asarray(ch, np.float32))
        n += 1
    print(f"=== canonicalized {n} dual-goal rollouts (mode={args.mode}) -> {out} ===", flush=True)
    print("MAKECANON_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
