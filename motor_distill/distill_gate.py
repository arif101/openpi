"""LEARNED GATE (design module D, the keystone): decides the GRIPPER (close/open = grasp/release) from
CONVERGENCE signals, trained on the demos -- replacing hand-set release thresholds. The gate keys on how
close the EE is to each goal (|ee-object|, |ee-container|) + gripper state, which are closed-loop-robust:
it closes when converged on the object, opens when converged on the container. With a reaching motor
(attractor that actually arrives) the convergence happens, and the gate fires the release at the right time.

Features (all available at deploy -- NO privileged object tracking):
  [ |ee-obj_goal|_xy, (ee-obj_goal)_z, |ee-cont_goal|_xy, (ee-cont_goal)_z, grip0, grip1 ]   (6-dim)
Target: demo gripper command sign (close = action[6] > 0).  Saves runs/gate_head.pkl.

Run: python motor_distill/distill_gate.py --data ~/openpi-box-backup/data/place
"""
from __future__ import annotations
import argparse, glob, pathlib, pickle
import jax, jax.numpy as jnp, numpy as np, optax


def feats(ee, tgt, cont, grip):
    ro = ee - tgt; rc = ee - cont
    return np.concatenate([
        np.linalg.norm(ro[..., :2], axis=-1, keepdims=True), ro[..., 2:3],
        np.linalg.norm(rc[..., :2], axis=-1, keepdims=True), rc[..., 2:3],
        grip], axis=-1).astype(np.float32)                   # (...,6)


def init_gate(key, din=6, dh=64):
    k = jax.random.split(key, 3)
    s = lambda kk, a, b: jax.random.normal(kk, (a, b)) * (1.0 / np.sqrt(a))
    return {"w1": s(k[0], din, dh), "b1": jnp.zeros(dh),
            "w2": s(k[1], dh, dh), "b2": jnp.zeros(dh),
            "w3": s(k[2], dh, 1), "b3": jnp.zeros(1)}


def gate_apply(p, f):
    h = jax.nn.gelu(f @ p["w1"] + p["b1"])
    h = jax.nn.gelu(h @ p["w2"] + p["b2"])
    return (h @ p["w3"] + p["b3"])[..., 0]                    # logit P(close)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/place"); ap.add_argument("--out", default="runs/gate_head.pkl")
    ap.add_argument("--steps", type=int, default=4000); ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-3)
    args = ap.parse_args()
    files = sorted(glob.glob(str(pathlib.Path(args.data) / "*.npz")))
    F, Y = [], []
    for f in files:
        d = np.load(f)
        F.append(feats(d["ee"], d["tgt"], d["cont"], d["grip"]))
        Y.append((d["action"][:, 6] > 0).astype(np.float32))
    F = np.concatenate(F); Y = np.concatenate(Y)
    mu, sd = F.mean(0), F.std(0) + 1e-6
    Fn = (F - mu) / sd
    print(f"{len(files)} demos -> {len(F)} steps; close frac={Y.mean():.2f}", flush=True)
    Fn, Y = jnp.asarray(Fn), jnp.asarray(Y)
    params = init_gate(jax.random.key(0)); opt = optax.adam(args.lr); opt_state = opt.init(params)

    def loss_fn(p, f, y):
        return optax.sigmoid_binary_cross_entropy(gate_apply(p, f), y).mean()
    grad_fn = jax.jit(jax.value_and_grad(loss_fn)); rng = np.random.default_rng(0); N = len(F)
    for step in range(args.steps):
        idx = rng.integers(0, N, args.batch)
        Lv, gr = grad_fn(params, Fn[idx], Y[idx])
        upd, opt_state = opt.update(gr, opt_state, params); params = optax.apply_updates(params, upd)
        if step % 500 == 0 or step == args.steps - 1:
            pred = (np.asarray(gate_apply(params, Fn)) > 0) == (np.asarray(Y) > 0.5)
            print(f"step {step:5d} bce={float(Lv):.4f} gripper_acc={pred.mean():.3f}", flush=True)
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pickle.dump({"params": jax.tree.map(np.asarray, params), "mu": mu, "sd": sd}, open(args.out, "wb"))
    print(f"SAVED {args.out}\nDISTILL_GATE_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
