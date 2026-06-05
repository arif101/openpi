"""Stage 1: distill pi0.5's reaching motion into a GOAL-CONDITIONED ATTRACTOR head.

Head input is GOAL-RELATIVE (ee-goal, quat, grip) -> translation-equivariant: change the goal and
the output flows to the new goal BY CONSTRUCTION (redirect-by-goal). Output is the 7-dim env action.
An explicit attractor term k*(goal-ee) is ADDED on the position dims so the head pulls toward the goal
even off the training distribution (Exp1: plain goal-conditioned regression collapses in closed loop;
the attractor form survives). The MLP learns pi0.5's approach/grasp nuance on top.

The head NEVER sees object appearance or other objects -> it cannot memorize "which object"; it only
reaches the given goal. Trained by matching pi0.5's action on aligned reaching demos. Saves runs/reach_head.pkl.
"""
from __future__ import annotations

import argparse
import glob
import pathlib
import pickle

import jax
import jax.numpy as jnp
import numpy as np
import optax


def init_head(key, din=9, dh=256, dout=7):
    k1, k2, k3 = jax.random.split(key, 3)
    s = lambda k, a, b: jax.random.normal(k, (a, b)) * (1.0 / np.sqrt(a))
    return {"w1": s(k1, din, dh), "b1": jnp.zeros(dh),
            "w2": s(k2, dh, dh), "b2": jnp.zeros(dh),
            "w3": s(k3, dh, dout) * 0.1, "b3": jnp.zeros(dout),
            "k_attract": jnp.array(2.0)}                      # learned goal-attractor gain (meters->action)


def head_apply(p, ee_rel, quat, grip):
    """ee_rel = ee - goal (meters). Returns 7-dim action. Attractor pulls toward goal (-ee_rel)."""
    x = jnp.concatenate([ee_rel, quat, grip], axis=-1)
    h = jax.nn.gelu(x @ p["w1"] + p["b1"])
    h = jax.nn.gelu(h @ p["w2"] + p["b2"])
    a = h @ p["w3"] + p["b3"]
    attract_pos = p["k_attract"] * (-ee_rel)                  # pull toward goal on xyz dims
    pad = jnp.zeros(a.shape[:-1] + (4,))                      # no attractor on rot+gripper dims
    return a + jnp.concatenate([attract_pos, pad], axis=-1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/reach")
    p.add_argument("--out", default="runs/reach_head.pkl")
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    args = p.parse_args()

    files = sorted(glob.glob(str(pathlib.Path(args.data) / "*.npz")))
    EE, Q, G, GP, A = [], [], [], [], []
    for f in files:
        d = np.load(f)
        EE.append(d["ee"]); Q.append(d["quat"]); G.append(d["grip"]); GP.append(d["goal"]); A.append(d["action"])
    EE = np.concatenate(EE); Q = np.concatenate(Q); G = np.concatenate(G)
    GP = np.concatenate(GP); A = np.concatenate(A)
    ee_rel = (EE - GP).astype(np.float32)                    # goal-relative position
    print(f"{len(files)} demos, {len(A)} (state,action) pairs", flush=True)

    ee_rel, Q, G, A = map(jnp.asarray, (ee_rel, Q, G, A))
    params = init_head(jax.random.key(0))
    opt = optax.adam(args.lr); opt_state = opt.init(params)

    def loss_fn(params, er, q, g, a):
        pred = head_apply(params, er, q, g)
        return jnp.mean((pred - a) ** 2)

    grad_fn = jax.jit(jax.value_and_grad(loss_fn))
    rng = np.random.default_rng(0); N = len(A)
    for step in range(args.steps):
        idx = rng.integers(0, N, args.batch)
        L, grads = grad_fn(params, ee_rel[idx], Q[idx], G[idx], A[idx])
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        if step % 500 == 0 or step == args.steps - 1:
            # report directional reconstruction on a random batch (the keystone metric)
            pred = np.asarray(head_apply(params, ee_rel[:2000], Q[:2000], G[:2000]))
            gt = np.asarray(A[:2000])
            dc = float(np.mean(np.sum(_n(pred[:, :3]) * _n(gt[:, :3]), -1)))
            print(f"step {step:5d}  mse={float(L):.4f}  dir_cos={dc:+.3f}  k={float(params['k_attract']):.2f}", flush=True)

    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as fh:
        pickle.dump(jax.tree.map(np.asarray, params), fh)
    print(f"SAVED {args.out}", flush=True)
    print("DISTILL_REACH_EXIT=0", flush=True)


def _n(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-8)


if __name__ == "__main__":
    main()
