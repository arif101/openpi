"""Learned motor v2 (research recipe -- the non-hardcoded contribution). Fixes the chunked head's fidelity:
  - ACTION CHUNKING (K=10) -- already had it.
  - RELAY / single ACTIVE goal: condition on the object goal during approach+grasp, the container goal during
    carry+place (NOT both goals -> removes the 'drifts to container early' failure). Active goal selected by
    'held' (gripper closed & object lifted) -- a physical latch, not a hardcoded constant.
  - L1 loss on the 6-DoF pose chunk + separate BCE on the gripper bit (ACT: L1>L2 for precision; gripper TIMING
    needs BCE, not L2 regression -> fixes 'closes 8cm to the side').
Object-agnostic (goal-relative, no image) -> anti-memorization preserved. Saves runs/motor_v2_head.pkl.

Run: python motor_distill/distill_motor_v2.py --data ~/openpi-box-backup/data/place --K 10
"""
from __future__ import annotations
import argparse, glob, pathlib, pickle
import jax, jax.numpy as jnp, numpy as np, optax

DIN = 9   # (ee - active_goal)(3) + quat(4) + grip(2); +3 if obj-state (object in-hand pose, identity-free)
DOUT = 7


def init_motor(key, dh=256, K=10, din=DIN):
    k = jax.random.split(key, 3)
    s = lambda kk, a, b: jax.random.normal(kk, (a, b)) * (1.0 / np.sqrt(a))
    return {"w1": s(k[0], din, dh), "b1": jnp.zeros(dh),
            "w2": s(k[1], dh, dh), "b2": jnp.zeros(dh),
            "w3": s(k[2], dh, K * DOUT) * 0.1, "b3": jnp.zeros(K * DOUT)}


def motor_apply(p, ee_rel, quat, grip, obj_rel=None):
    """Returns (..., K, 7) chunk: [:6]=pose deltas, [6]=gripper logit. K inferred from w3.
    Input order: [ee_rel, (obj_rel if given), quat, grip]. obj_rel = object pos - ee (identity-free in-hand pose)."""
    parts = [ee_rel] + ([obj_rel] if obj_rel is not None else []) + [quat, grip]
    x = jnp.concatenate(parts, axis=-1)
    h = jax.nn.gelu(x @ p["w1"] + p["b1"])
    h = jax.nn.gelu(h @ p["w2"] + p["b2"])
    a = h @ p["w3"] + p["b3"]
    K = p["w3"].shape[1] // DOUT
    return a.reshape(a.shape[:-1] + (K, DOUT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", default=["data/place"]); ap.add_argument("--out", default="runs/motor_v2_head.pkl")
    ap.add_argument("--K", type=int, default=10); ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=256); ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--obj-filter", default="")   # comma object tokens (e.g. alphabet_soup,butter): keep only files whose name contains one -> held-out-object split
    ap.add_argument("--obj-state", action="store_true")   # +3 input dims: object in-hand pose (tgt-ee), identity-free -> place-precision observability
    args = ap.parse_args()
    files = sorted(f for d in args.data for f in glob.glob(str(pathlib.Path(d) / "*.npz")))
    if args.obj_filter:
        toks = [t.strip() for t in args.obj_filter.split(",") if t.strip()]
        files = [f for f in files if any(t in pathlib.Path(f).name for t in toks)]
        print(f"OBJ-FILTER {toks} -> {len(files)} files kept", flush=True)
    RR, Q, G, AC, OE = [], [], [], [], []
    for f in files:
        d = np.load(f); Tn = len(d["action"]); act = d["action"].astype(np.float32)
        lift = d["tgt"][:, 2] - d["tgt"][0, 2]
        held = (act[:, 6] > 0) & (lift > 0.02)                # carry phase
        active = np.where(held[:, None], d["cont"], d["tgt"]).astype(np.float32)   # RELAY: single active goal
        rr = (d["ee"] - active).astype(np.float32)
        oe = (d["tgt"] - d["ee"]).astype(np.float32)          # object in-hand pose (identity-free geometry)
        for t in range(Tn):
            chunk = act[t:t + args.K]
            if len(chunk) < args.K:
                chunk = np.concatenate([chunk, np.repeat(chunk[-1:], args.K - len(chunk), 0)])
            RR.append(rr[t]); Q.append(d["quat"][t]); G.append(d["grip"][t]); AC.append(chunk); OE.append(oe[t])
    RR = np.stack(RR); Q = np.stack(Q); G = np.stack(G); AC = np.stack(AC); OE = np.stack(OE)  # AC [N,K,7]
    print(f"{len(files)} demos -> {len(RR)} (state,{args.K}-chunk); carry-phase frac={(AC[:,:,6]>0).mean():.2f}; obj_state={args.obj_state}", flush=True)
    RR, Q, G, AC, OE = map(jnp.asarray, (RR, Q, G, AC, OE))
    din = DIN + (3 if args.obj_state else 0)
    params = init_motor(jax.random.key(0), K=args.K, din=din); opt = optax.adam(args.lr); opt_state = opt.init(params)

    def loss_fn(p, rr, q, g, ac, oe):
        pred = motor_apply(p, rr, q, g, oe if args.obj_state else None)   # [B,K,7]
        l1 = jnp.mean(jnp.abs(pred[..., :6] - ac[..., :6]))   # L1 on 6-DoF pose (ACT)
        bce = optax.sigmoid_binary_cross_entropy(pred[..., 6], (ac[..., 6] > 0).astype(jnp.float32)).mean()
        return l1 + 0.5 * bce
    grad_fn = jax.jit(jax.value_and_grad(loss_fn)); rng = np.random.default_rng(0); N = len(RR)
    for step in range(args.steps):
        idx = rng.integers(0, N, args.batch)
        Lv, gr = grad_fn(params, RR[idx], Q[idx], G[idx], AC[idx], OE[idx])
        upd, opt_state = opt.update(gr, opt_state, params); params = optax.apply_updates(params, upd)
        if step % 800 == 0 or step == args.steps - 1:
            pred = np.asarray(motor_apply(params, RR[:2000], Q[:2000], G[:2000], OE[:2000] if args.obj_state else None)); gt = np.asarray(AC[:2000])
            posl1 = float(np.abs(pred[:,0,:3] - gt[:,0,:3]).mean())
            ga = float(np.mean((pred[:, :, 6] < 0) == (gt[:, :, 6] < 0)))
            print(f"step {step:5d} loss={float(Lv):.4f} pos_a0_L1={posl1:.4f} grip_acc={ga:.3f}", flush=True)
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(jax.tree.map(np.asarray, params), open(args.out, "wb"))
    print(f"SAVED {args.out}\nDISTILL_MOTOR_V2_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
