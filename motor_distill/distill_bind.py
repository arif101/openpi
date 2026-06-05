"""Stage 2: train the BINDING CHANNEL: language target phrase -> 3D goal, via open-vocab
cross-attention over frozen VLM image tokens.

q = pooled language tokens (the named target); attends over image tokens (k,v) -> attended feature
-> MLP -> 3D target position. Trained on counterfactual binding data (same image, varied instruction
-> varied target) so it MUST use language. Open-vocab generalization rides on the frozen VLM/SigLIP
tokens. Saves runs/bind_head.pkl.
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

D_VLM = 2048


def init_bind(key, d=256):
    k = jax.random.split(key, 6)
    s = lambda kk, a, b: jax.random.normal(kk, (a, b)) * (1.0 / np.sqrt(a))
    return {"Wq": s(k[0], D_VLM, d), "Wk": s(k[1], D_VLM, d), "Wv": s(k[2], D_VLM, d),
            "w1": s(k[3], d, d), "b1": jnp.zeros(d), "w2": s(k[4], d, 3) * 0.1, "b2": jnp.zeros(3),
            "scale": jnp.array(1.0 / np.sqrt(d))}


def bind_apply(p, prefix, n_img):
    """prefix [B, seq, 2048]; n_img static int. Returns predicted 3D target pos [B,3]."""
    img = prefix[:, :n_img]                                   # [B, n_img, 2048]
    lang = prefix[:, n_img:].mean(1)                          # [B, 2048] pooled instruction
    q = lang @ p["Wq"]                                        # [B, d]
    k = img @ p["Wk"]; v = img @ p["Wv"]                      # [B, n_img, d]
    attn = jax.nn.softmax(jnp.einsum("bd,bnd->bn", q, k) * p["scale"], axis=-1)   # [B, n_img]
    att = jnp.einsum("bn,bnd->bd", attn, v)                   # [B, d]
    h = jax.nn.gelu(att @ p["w1"] + p["b1"])
    return h @ p["w2"] + p["b2"]                              # [B, 3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/bind")
    ap.add_argument("--out", default="runs/bind_head.pkl")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--holdout", default="", help="comma objects to hold out (generalization test)")
    args = ap.parse_args()

    files = sorted(glob.glob(str(pathlib.Path(args.data) / "*.npz")))
    holdout = set(args.holdout.split(",")) if args.holdout else set()
    PF, NI, Y, tr_idx, te_idx = [], None, [], [], []
    for i, f in enumerate(files):
        d = np.load(f, allow_pickle=True)
        PF.append(d["prefix"].astype(np.float32)); NI = int(d["n_img"]); Y.append(d["target_pos"])
        (te_idx if str(d["obj"]) in holdout else tr_idx).append(i)
    PF = jnp.asarray(np.stack(PF)); Y = jnp.asarray(np.stack(Y))
    print(f"{len(files)} samples, n_img={NI}, train={len(tr_idx)} test(holdout)={len(te_idx)}", flush=True)
    tr = np.array(tr_idx)

    params = init_bind(jax.random.key(0))
    opt = optax.adam(args.lr); opt_state = opt.init(params)

    def loss_fn(params, pf, y):
        return jnp.mean(jnp.sum((bind_apply(params, pf, NI) - y) ** 2, -1))   # squared 3D error

    grad_fn = jax.jit(jax.value_and_grad(loss_fn))
    rng = np.random.default_rng(0)
    for step in range(args.steps):
        idx = tr[rng.integers(0, len(tr), args.batch)]
        L, grads = grad_fn(params, PF[idx], Y[idx])
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        if step % 400 == 0 or step == args.steps - 1:
            err = lambda ii: float(np.mean(np.linalg.norm(
                np.asarray(bind_apply(params, PF[np.array(ii)], NI)) - np.asarray(Y[np.array(ii)]), axis=-1))) * 100
            msg = f"step {step:5d}  loss={float(L):.4f}  train_err={err(tr_idx):.1f}cm"
            if te_idx:
                msg += f"  HOLDOUT_err={err(te_idx):.1f}cm"
            print(msg, flush=True)

    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as fh:
        pickle.dump({"params": jax.tree.map(np.asarray, params), "n_img": NI}, fh)
    print(f"SAVED {args.out}", flush=True)
    print("DISTILL_BIND_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
