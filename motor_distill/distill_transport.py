"""P1-TRANSPORT primitive (design module C / P1).

The R1 place diagnostic showed the REACH head DIVERGES when reused for transport (ee flies to ~-1.3m):
approach-grasp and carry-to-container are NOT the same primitive. So we distill a SEPARATE transport
attractor from the successful transport segments of the place demos — goal-relative to the CONTAINER,
gripper held closed, object lifted. Same attractor architecture as distill_reach (translation-equivariant,
object-agnostic), different data + goal.

Transport steps = gripper commanded-closed (action[:,6] > 0.5) AND object lifted (>2cm) AND before release.
Input = (ee - container, quat, grip) -> 7-dim action.  Saves runs/transport_head.pkl.

Usage: python motor_distill/distill_transport.py --data ~/openpi-box-backup/data/place
"""
from __future__ import annotations

import argparse, glob, pathlib, pickle
import jax, jax.numpy as jnp, numpy as np, optax
from distill_reach import init_head, head_apply, _n


def select_transport(d, lift_thr=0.02):
    """Boolean mask of held-and-carrying steps (after grasp, before release)."""
    g6 = d["action"][:, 6]
    closed = g6 > 0.5
    lift = d["tgt"][:, 2] - d["tgt"][0, 2]
    held = closed & (lift > lift_thr)
    if not held.any():
        return np.zeros(len(g6), bool)
    # restrict to the contiguous held span from first grasp to last hold (drop post-release)
    idx = np.where(held)[0]
    m = np.zeros(len(g6), bool); m[idx[0]:idx[-1] + 1] = True
    return m & closed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/place")
    ap.add_argument("--out", default="runs/transport_head.pkl")
    ap.add_argument("--steps", type=int, default=4000); ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    files = sorted(glob.glob(str(pathlib.Path(args.data) / "*.npz")))
    EE, Q, G, CONT, A = [], [], [], [], []
    nsteps = 0
    for f in files:
        d = np.load(f)
        m = select_transport(d)
        if not m.any():
            continue
        EE.append(d["ee"][m]); Q.append(d["quat"][m]); G.append(d["grip"][m])
        CONT.append(d["cont"][m]); A.append(d["action"][m]); nsteps += int(m.sum())
    EE = np.concatenate(EE); Q = np.concatenate(Q); G = np.concatenate(G)
    CONT = np.concatenate(CONT); A = np.concatenate(A)
    ee_rel = (EE - CONT).astype(np.float32)                  # goal-relative to CONTAINER
    print(f"{len(files)} place demos -> {nsteps} transport (state,action) pairs", flush=True)
    print(f"  ee_rel horiz range: {np.linalg.norm(ee_rel[:, :2], axis=1).min()*100:.0f}-"
          f"{np.linalg.norm(ee_rel[:, :2], axis=1).max()*100:.0f}cm", flush=True)

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
            print(f"step {step:5d}  mse={float(L):.4f}  dir_cos={dc:+.3f}  k={float(params['k_attract']):.2f}", flush=True)

    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as fh:
        pickle.dump(jax.tree.map(np.asarray, params), fh)
    print(f"SAVED {args.out}", flush=True)
    print("DISTILL_TRANSPORT_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
