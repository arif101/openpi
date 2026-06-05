"""Learn box->3D planar localization (objects on a table). Tiny MLP: box(4 normalized) -> 3D pos.
Accurate because the detector already grounded WHICH object; this is just image->table geometry.
Optional --holdout objects for open-vocab generalization test. Saves runs/box_head.pkl.
"""
from __future__ import annotations

import argparse, glob, pathlib, pickle
import jax, jax.numpy as jnp, numpy as np, optax


def init(key, din=4, dh=128):
    k = jax.random.split(key, 3)
    s = lambda kk, a, b: jax.random.normal(kk, (a, b)) * (1.0 / np.sqrt(a))
    return {"w1": s(k[0], din, dh), "b1": jnp.zeros(dh), "w2": s(k[1], dh, dh), "b2": jnp.zeros(dh),
            "w3": s(k[2], dh, 3) * 0.1, "b3": jnp.zeros(3)}


def apply(p, box):
    h = jax.nn.gelu(box @ p["w1"] + p["b1"]); h = jax.nn.gelu(h @ p["w2"] + p["b2"])
    return h @ p["w3"] + p["b3"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/box"); ap.add_argument("--out", default="runs/box_head.pkl")
    ap.add_argument("--steps", type=int, default=3000); ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--holdout", default="")
    args = ap.parse_args()
    files = sorted(glob.glob(str(pathlib.Path(args.data) / "*.npz")))
    holdout = set(args.holdout.split(",")) if args.holdout else set()
    B, Y, tr, te = [], [], [], []
    for i, f in enumerate(files):
        d = np.load(f, allow_pickle=True); B.append(d["box"]); Y.append(d["pos"])
        (te if str(d["obj"]) in holdout else tr).append(i)
    B = jnp.asarray(np.stack(B)); Y = jnp.asarray(np.stack(Y)); tr = np.array(tr)
    din = int(B.shape[1])
    print(f"{len(files)} pairs, din={din}, train={len(tr)} holdout={len(te)}", flush=True)
    params = init(jax.random.key(0), din); opt = optax.adam(args.lr); st = opt.init(params)
    lf = lambda p, b, y: jnp.mean(jnp.sum((apply(p, b) - y) ** 2, -1))
    gf = jax.jit(jax.value_and_grad(lf)); rng = np.random.default_rng(0)
    for step in range(args.steps):
        idx = tr[rng.integers(0, len(tr), args.batch)]
        L, g = gf(params, B[idx], Y[idx]); up, st = opt.update(g, st, params); params = optax.apply_updates(params, up)
        if step % 500 == 0 or step == args.steps - 1:
            e = lambda ii: float(np.mean(np.linalg.norm(np.asarray(apply(params, B[np.array(ii)])) - np.asarray(Y[np.array(ii)]), -1))) * 100
            m = f"step {step:5d} loss={float(L):.4f} train_err={e(list(tr)):.1f}cm"
            if te: m += f" HOLDOUT_err={e(te):.1f}cm"
            print(m, flush=True)
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(jax.tree.map(np.asarray, params), open(args.out, "wb"))
    print(f"SAVED {args.out}"); print("DISTILL_BOX_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
