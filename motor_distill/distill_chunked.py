"""CHUNKED unified motor: distill pi0.5's motor as an ACTION-CHUNK policy (predict a K-step sequence per
decision), NOT the single-step MLP we'd been using. Single-step distillation threw away pi0.5's chunking,
which is a primary source of its smoothness + robustness-to-compounding -> our carry->place drifted. Chunking
makes a 260-step task ~26-50 decisions instead of 260, killing most of the compounding error.

Object-agnostic (no image): input = goal-relative state to BOTH goals + grip = [ee-obj(3), ee-cont(3),
quat(4), grip(2)] = 12. Output = K x 7 action chunk. Targets are FREE: window each demo's logged action
sequence (state_t -> actions[t:t+K]). Saves runs/chunked_head.pkl.

Run: python motor_distill/distill_chunked.py --data ~/openpi-box-backup/data/place --K 10
"""
from __future__ import annotations
import argparse, glob, pathlib, pickle
import jax, jax.numpy as jnp, numpy as np, optax


DOUT = 7   # action dim (static, not a learned param)


def init_chunked(key, din=12, dh=256, K=10, dout=DOUT):
    k = jax.random.split(key, 3)
    s = lambda kk, a, b: jax.random.normal(kk, (a, b)) * (1.0 / np.sqrt(a))
    return {"w1": s(k[0], din, dh), "b1": jnp.zeros(dh),
            "w2": s(k[1], dh, dh), "b2": jnp.zeros(dh),
            "w3": s(k[2], dh, K * dout) * 0.1, "b3": jnp.zeros(K * dout)}


def head_apply_chunked(p, ee_rel_obj, ee_rel_cont, quat, grip):
    """Returns a (..., K, DOUT) action chunk from goal-relative state. K inferred from w3 shape."""
    x = jnp.concatenate([ee_rel_obj, ee_rel_cont, quat, grip], axis=-1)
    g = jax.nn.gelu(x @ p["w1"] + p["b1"])
    g = jax.nn.gelu(g @ p["w2"] + p["b2"])
    a = g @ p["w3"] + p["b3"]
    K = p["w3"].shape[1] // DOUT
    return a.reshape(a.shape[:-1] + (K, DOUT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/place")
    ap.add_argument("--out", default="runs/chunked_head.pkl")
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--steps", type=int, default=8000); ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()
    files = sorted(glob.glob(str(pathlib.Path(args.data) / "*.npz")))
    RO, RC, Q, G, AC = [], [], [], [], []
    for f in files:
        d = np.load(f); T = len(d["action"])
        ro = (d["ee"] - d["tgt"]).astype(np.float32); rc = (d["ee"] - d["cont"]).astype(np.float32)
        act = d["action"].astype(np.float32)
        for t in range(T):
            chunk = act[t:t + args.K]
            if len(chunk) < args.K:                          # pad tail by repeating last action
                chunk = np.concatenate([chunk, np.repeat(chunk[-1:], args.K - len(chunk), 0)])
            RO.append(ro[t]); RC.append(rc[t]); Q.append(d["quat"][t]); G.append(d["grip"][t]); AC.append(chunk)
    RO = np.stack(RO); RC = np.stack(RC); Q = np.stack(Q); G = np.stack(G); AC = np.stack(AC)  # AC [N,K,7]
    print(f"{len(files)} demos -> {len(RO)} (state, {args.K}-chunk) pairs", flush=True)
    RO, RC, Q, G, AC = map(jnp.asarray, (RO, RC, Q, G, AC))
    params = init_chunked(jax.random.key(0), K=args.K); opt = optax.adam(args.lr); opt_state = opt.init(params)

    def loss_fn(p, ro, rc, q, g, ac):
        return jnp.mean((head_apply_chunked(p, ro, rc, q, g) - ac) ** 2)
    grad_fn = jax.jit(jax.value_and_grad(loss_fn)); rng = np.random.default_rng(0); N = len(RO)
    for step in range(args.steps):
        idx = rng.integers(0, N, args.batch)
        Lv, gr = grad_fn(params, RO[idx], RC[idx], Q[idx], G[idx], AC[idx])
        upd, opt_state = opt.update(gr, opt_state, params); params = optax.apply_updates(params, upd)
        if step % 800 == 0 or step == args.steps - 1:
            pred = np.asarray(head_apply_chunked(params, RO[:2000], RC[:2000], Q[:2000], G[:2000]))
            gt = np.asarray(AC[:2000])
            d0 = pred[:, 0, :3]; g0 = gt[:, 0, :3]                 # first-action direction fidelity
            dc = float(np.mean(np.sum((d0/ (np.linalg.norm(d0,axis=-1,keepdims=True)+1e-8)) *
                                      (g0/ (np.linalg.norm(g0,axis=-1,keepdims=True)+1e-8)), -1)))
            ga = float(np.mean((pred[:, :, 6] < 0) == (gt[:, :, 6] < 0)))
            print(f"step {step:5d} mse={float(Lv):.4f} chunk_dir_cos(a0)={dc:+.3f} grip_acc={ga:.3f}", flush=True)
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(jax.tree.map(np.asarray, params), open(args.out, "wb"))
    print(f"SAVED {args.out}\nDISTILL_CHUNKED_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
