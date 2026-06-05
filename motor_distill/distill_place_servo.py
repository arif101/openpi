"""P3 PLACE-SERVO primitive (MoE expert #3): descend-into-container + release.

The transport primitive carries the held object to ~1-3cm above the container but the hardcoded release
fires ~20cm too high -> object bounces out (RELEASED_MISSED). The place-servo expert is distilled from the
demos' OVER-CONTAINER segment (the descend + gripper-open motion pi0.5 actually executes), goal-relative to
the CONTAINER, and -- unlike transport -- it PREDICTS THE GRIPPER (learns WHEN to open from the descend
depth), so release is emergent from the learned motion rather than a hand-set threshold.

Place-servo steps = object horizontally over the container (xy dist < over_cm) AND after grasp (object was
lifted) -- this window contains the descend and the open. Input = (ee - container, quat, grip) -> 7-dim
action incl gripper.  Saves runs/place_servo_head.pkl.

Usage: python motor_distill/distill_place_servo.py --data ~/openpi-box-backup/data/place
"""
from __future__ import annotations

import argparse, glob, pathlib, pickle
import jax, jax.numpy as jnp, numpy as np, optax
from distill_reach import init_head, head_apply, _n


def select_place_servo(d, over_cm=0.12, lift_thr=0.02):
    """Steps where the (lifted) object is horizontally over the container -> descend+release window."""
    obj = d["tgt"]; cont = d["cont"]
    lift = obj[:, 2] - obj[0, 2]
    ever_lifted = lift > lift_thr
    over = np.linalg.norm(obj[:, :2] - cont[:, :2], axis=1) < over_cm
    # require it to have been grasped at least once before counting over-container steps
    if not ever_lifted.any():
        return np.zeros(len(obj), bool)
    first_lift = np.where(ever_lifted)[0][0]
    m = over.copy(); m[:first_lift] = False
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/place")
    ap.add_argument("--out", default="runs/place_servo_head.pkl")
    ap.add_argument("--steps", type=int, default=4000); ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--over-cm", type=float, default=0.12)
    args = ap.parse_args()

    files = sorted(glob.glob(str(pathlib.Path(args.data) / "*.npz")))
    EE, Q, G, CONT, A = [], [], [], [], []
    nsteps = 0
    for f in files:
        d = np.load(f)
        m = select_place_servo(d, over_cm=args.over_cm)
        if not m.any():
            continue
        EE.append(d["ee"][m]); Q.append(d["quat"][m]); G.append(d["grip"][m])
        CONT.append(d["cont"][m]); A.append(d["action"][m]); nsteps += int(m.sum())
    EE = np.concatenate(EE); Q = np.concatenate(Q); G = np.concatenate(G)
    CONT = np.concatenate(CONT); A = np.concatenate(A)
    ee_rel = (EE - CONT).astype(np.float32)
    open_frac = float((A[:, 6] < 0).mean())
    print(f"{len(files)} place demos -> {nsteps} place-servo (state,action) pairs; gripper-open frac={open_frac:.2f}", flush=True)
    print(f"  ee_rel z range: {ee_rel[:,2].min()*100:.0f}..{ee_rel[:,2].max()*100:.0f}cm (descends toward container)", flush=True)

    ee_rel, Q, G, A = map(jnp.asarray, (ee_rel, Q, G, A))
    params = init_head(jax.random.key(0))
    opt = optax.adam(args.lr); opt_state = opt.init(params)

    def loss_fn(params, er, q, g, a):
        return jnp.mean((head_apply(params, er, q, g) - a) ** 2)
    grad_fn = jax.jit(jax.value_and_grad(loss_fn))
    rng = np.random.default_rng(0); N = len(A)
    for step in range(args.steps):
        idx = rng.integers(0, N, args.batch)
        L, grads = grad_fn(params, ee_rel[idx], Q[idx], G[idx], A[idx])
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        if step % 500 == 0 or step == args.steps - 1:
            pred = np.asarray(head_apply(params, ee_rel[:2000], Q[:2000], G[:2000]))
            gt = np.asarray(A[:2000])
            dc = float(np.mean(np.sum(_n(pred[:, :3]) * _n(gt[:, :3]), -1)))
            grip_acc = float(np.mean((pred[:, 6] < 0) == (gt[:, 6] < 0)))   # open/close agreement
            print(f"step {step:5d}  mse={float(L):.4f}  dir_cos={dc:+.3f}  grip_acc={grip_acc:.3f}", flush=True)

    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as fh:
        pickle.dump(jax.tree.map(np.asarray, params), fh)
    print(f"SAVED {args.out}", flush=True)
    print("DISTILL_PLACE_SERVO_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
