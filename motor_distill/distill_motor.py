"""Distill ONE phase/goal-conditioned attractor motor (not a per-action head) from pi0.5's full
pick+place demos. Input = [ee-active_goal, quat, grip, phase]; active_goal = target (phase 0) or
container (phase 1). The SAME head learns approach-grasp AND transport-release (incl. gripper timing).
Combines reach demos (phase-0 grasp data) + place demos (full pick+place). Saves runs/motor_head.pkl.

This is the scalable motor: one conditioned controller; tasks are plans over (goal, phase). No head-zoo.
"""
from __future__ import annotations

import argparse, glob, pathlib, pickle
import jax, jax.numpy as jnp, numpy as np, optax


def init_head(key, din=10, dh=256, dout=7):
    k1, k2, k3 = jax.random.split(key, 3)
    s = lambda k, a, b: jax.random.normal(k, (a, b)) * (1.0 / np.sqrt(a))
    return {"w1": s(k1, din, dh), "b1": jnp.zeros(dh), "w2": s(k2, dh, dh), "b2": jnp.zeros(dh),
            "w3": s(k3, dh, dout) * 0.1, "b3": jnp.zeros(dout), "k_attract": jnp.array(2.0)}


def head_apply(p, ee_rel, quat, grip, phase):
    """ee_rel = ee - active_goal. phase in {0,1} (broadcastable last dim 1). Returns 7-dim action."""
    x = jnp.concatenate([ee_rel, quat, grip, phase], axis=-1)
    h = jax.nn.gelu(x @ p["w1"] + p["b1"]); h = jax.nn.gelu(h @ p["w2"] + p["b2"])
    a = h @ p["w3"] + p["b3"]
    pad = jnp.zeros(a.shape[:-1] + (4,))
    return a + jnp.concatenate([p["k_attract"] * (-ee_rel), pad], axis=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--place", default="data/place"); ap.add_argument("--reach", default="data/reach")
    ap.add_argument("--out", default="runs/motor_head.pkl")
    ap.add_argument("--steps", type=int, default=6000); ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    ER, Q, G, PH, A = [], [], [], [], []
    # place demos: active goal switches by phase
    for f in sorted(glob.glob(str(pathlib.Path(args.place) / "*.npz"))):
        d = np.load(f)
        ph = d["phase"].astype(np.float32)
        active = np.where(ph[:, None] > 0.5, d["cont"], d["tgt"])    # container if place phase else target
        ER.append((d["ee"] - active).astype(np.float32)); Q.append(d["quat"]); G.append(d["grip"])
        PH.append(ph[:, None]); A.append(d["action"])
    # reach demos: all phase 0 (approach-grasp), goal = target
    for f in sorted(glob.glob(str(pathlib.Path(args.reach) / "*.npz"))):
        d = np.load(f)
        ER.append((d["ee"] - d["goal"]).astype(np.float32)); Q.append(d["quat"]); G.append(d["grip"])
        PH.append(np.zeros((len(d["action"]), 1), np.float32)); A.append(d["action"])
    ER = jnp.asarray(np.concatenate(ER)); Q = jnp.asarray(np.concatenate(Q)); G = jnp.asarray(np.concatenate(G))
    PH = jnp.asarray(np.concatenate(PH)); A = jnp.asarray(np.concatenate(A))
    N = len(A); print(f"{N} (state,action) pairs; place+reach demos", flush=True)

    params = init_head(jax.random.key(0)); opt = optax.adam(args.lr); st = opt.init(params)
    def loss_fn(p, er, q, g, ph, a):
        return jnp.mean((head_apply(p, er, q, g, ph) - a) ** 2)
    gf = jax.jit(jax.value_and_grad(loss_fn)); rng = np.random.default_rng(0)
    _n = lambda v: v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-8)
    for step in range(args.steps):
        idx = rng.integers(0, N, args.batch)
        L, grads = gf(params, ER[idx], Q[idx], G[idx], PH[idx], A[idx])
        up, st = opt.update(grads, st, params); params = optax.apply_updates(params, up)
        if step % 600 == 0 or step == args.steps - 1:
            pr = np.asarray(head_apply(params, ER[:2000], Q[:2000], G[:2000], PH[:2000]))
            dc = float(np.mean(np.sum(_n(pr[:, :3]) * _n(np.asarray(A[:2000])[:, :3]), -1)))
            print(f"step {step:5d}  mse={float(L):.4f}  dir_cos={dc:+.3f}  k={float(params['k_attract']):.2f}", flush=True)
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(jax.tree.map(np.asarray, params), open(args.out, "wb"))
    print(f"SAVED {args.out}"); print("DISTILL_MOTOR_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
