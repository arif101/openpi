"""Unified POST-GRASP place head: one smooth motion = carry-to-container + descend + release, distilled from
the full post-grasp segment of the place demos (not the tiny over-container slice that gave place_servo low
dir_cos). Goal-relative to the CONTAINER; predicts the 7-dim action INCLUDING the gripper, so the release is
learned (when pi0.5 opens), replacing the hand-tuned transport-head + nadir-release scaffold that flings the
object.

Post-grasp steps = from first firm grasp (gripper closed AND object lifted) through release (open_edge)+tail.
Input = (ee - container, quat, grip) -> action.  Saves runs/place_full_head.pkl.

Run: python motor_distill/distill_place_full.py --data ~/openpi-box-backup/data/place
"""
from __future__ import annotations
import argparse, glob, pathlib, pickle
import jax, jax.numpy as jnp, numpy as np, optax
from distill_reach import init_head, head_apply, _n


def select_post_grasp(d, lift_thr=0.02, tail=8):
    g6 = d["action"][:, 6]; closed = g6 > 0.5
    lift = d["tgt"][:, 2] - d["tgt"][0, 2]
    held = closed & (lift > lift_thr)
    if not held.any():
        return np.zeros(len(g6), bool)
    start = np.where(held)[0][0]
    opens = np.where(np.diff(closed.astype(int)) == -1)[0] + 1
    oe = opens[opens > start]
    end = (int(oe[0]) + tail) if len(oe) else len(g6)
    m = np.zeros(len(g6), bool); m[start:min(end, len(g6))] = True
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/place")
    ap.add_argument("--out", default="runs/place_full_head.pkl")
    ap.add_argument("--steps", type=int, default=5000); ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()
    files = sorted(glob.glob(str(pathlib.Path(args.data) / "*.npz")))
    EE, Q, G, CONT, A = [], [], [], [], []; nsteps = 0
    for f in files:
        d = np.load(f); m = select_post_grasp(d)
        if not m.any(): continue
        EE.append(d["ee"][m]); Q.append(d["quat"][m]); G.append(d["grip"][m])
        CONT.append(d["cont"][m]); A.append(d["action"][m]); nsteps += int(m.sum())
    EE = np.concatenate(EE); Q = np.concatenate(Q); G = np.concatenate(G)
    CONT = np.concatenate(CONT); A = np.concatenate(A)
    ee_rel = (EE - CONT).astype(np.float32)
    print(f"{len(files)} demos -> {nsteps} post-grasp pairs; gripper-open frac={(A[:,6]<0).mean():.2f}; "
          f"ee_rel horiz {np.linalg.norm(ee_rel[:,:2],axis=1).min()*100:.0f}-{np.linalg.norm(ee_rel[:,:2],axis=1).max()*100:.0f}cm", flush=True)

    ee_rel, Q, G, A = map(jnp.asarray, (ee_rel, Q, G, A))
    params = init_head(jax.random.key(0)); opt = optax.adam(args.lr); opt_state = opt.init(params)
    def loss_fn(p, er, q, g, a): return jnp.mean((head_apply(p, er, q, g) - a) ** 2)
    grad_fn = jax.jit(jax.value_and_grad(loss_fn)); rng = np.random.default_rng(0); N = len(A)
    for step in range(args.steps):
        idx = rng.integers(0, N, args.batch)
        Lv, gr = grad_fn(params, ee_rel[idx], Q[idx], G[idx], A[idx])
        upd, opt_state = opt.update(gr, opt_state, params); params = optax.apply_updates(params, upd)
        if step % 500 == 0 or step == args.steps - 1:
            pred = np.asarray(head_apply(params, ee_rel[:2000], Q[:2000], G[:2000])); gt = np.asarray(A[:2000])
            dc = float(np.mean(np.sum(_n(pred[:,:3]) * _n(gt[:,:3]), -1)))
            ga = float(np.mean((pred[:,6] < 0) == (gt[:,6] < 0)))
            print(f"step {step:5d} mse={float(Lv):.4f} dir_cos={dc:+.3f} grip_acc={ga:.3f}", flush=True)
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(jax.tree.map(np.asarray, params), open(args.out, "wb"))
    print(f"SAVED {args.out}\nDISTILL_PLACE_FULL_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
