"""UNIFIED goal-conditioned motor: ONE object-agnostic head that does the whole pick->place (reach, grasp,
carry, release) as a single smooth policy -- no discrete primitives, no explicit gate. Distilled from the
FULL pick+place demos.

Input = goal-relative state to BOTH goals + gripper (NO image, NO object identity -> cannot memorize):
  [ee-object_goal (3), ee-container_goal (3), quat (4), grip (2)]  = 12-dim
A learned soft "holding" weight h = sigmoid(MLP) (~0 pre-grasp, ~1 post-grasp, inferred from grip/state)
blends which goal the ATTRACTOR pursues:  active_rel = (1-h)*ee_rel_obj + h*ee_rel_cont. The phase transition
(reach->carry) and gripper timing are EMERGENT properties of one policy, not a switched gate.

Output = 7-dim action (incl gripper).  Trained on full place demos.  Saves runs/unified_head.pkl.
Run: python motor_distill/distill_unified.py --data ~/openpi-box-backup/data/place
"""
from __future__ import annotations
import argparse, glob, pathlib, pickle
import jax, jax.numpy as jnp, numpy as np, optax


K_ATTRACT = 4.0   # FIXED strong attractor gain (by construction) so the motor ARRIVES at its active goal


def init_unified(key, din=12, dh=256, dout=7):
    k = jax.random.split(key, 5)
    s = lambda kk, a, b: jax.random.normal(kk, (a, b)) * (1.0 / np.sqrt(a))
    return {"w1": s(k[0], din, dh), "b1": jnp.zeros(dh),
            "w2": s(k[1], dh, dh), "b2": jnp.zeros(dh),
            "w3": s(k[2], dh, dout) * 0.1, "b3": jnp.zeros(dout),
            "ph1": s(k[3], din, 64), "pb1": jnp.zeros(64),
            "ph2": s(k[4], 64, 1) * 0.1, "pb2": jnp.zeros(1)}


def head_apply_unified(p, ee_rel_obj, ee_rel_cont, quat, grip, k=K_ATTRACT):
    x = jnp.concatenate([ee_rel_obj, ee_rel_cont, quat, grip], axis=-1)
    hh = jax.nn.gelu(x @ p["ph1"] + p["pb1"])
    h = jax.nn.sigmoid(hh @ p["ph2"] + p["pb2"])              # holding weight in [0,1]
    g = jax.nn.gelu(x @ p["w1"] + p["b1"])
    g = jax.nn.gelu(g @ p["w2"] + p["b2"])
    a = g @ p["w3"] + p["b3"]
    active = (1.0 - h) * ee_rel_obj + h * ee_rel_cont          # soft-blend active goal (learned phase)
    attract = k * (-active)                                    # strong fixed attractor -> arrives at active goal
    pad = jnp.zeros(a.shape[:-1] + (4,))
    return a + jnp.concatenate([attract, pad], axis=-1)


def _n(v): return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/place")
    ap.add_argument("--out", default="runs/unified_head.pkl")
    ap.add_argument("--steps", type=int, default=6000); ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()
    files = sorted(glob.glob(str(pathlib.Path(args.data) / "*.npz")))
    RO, RC, Q, G, A = [], [], [], [], []
    for f in files:
        d = np.load(f)
        RO.append((d["ee"] - d["tgt"]).astype(np.float32))    # ee - object goal
        RC.append((d["ee"] - d["cont"]).astype(np.float32))   # ee - container goal
        Q.append(d["quat"]); G.append(d["grip"]); A.append(d["action"])
    RO = np.concatenate(RO); RC = np.concatenate(RC); Q = np.concatenate(Q)
    G = np.concatenate(G); A = np.concatenate(A)
    print(f"{len(files)} full demos -> {len(A)} (state,action) pairs", flush=True)
    RO, RC, Q, G, A = map(jnp.asarray, (RO, RC, Q, G, A))
    params = init_unified(jax.random.key(0)); opt = optax.adam(args.lr); opt_state = opt.init(params)

    def loss_fn(p, ro, rc, q, g, a):
        return jnp.mean((head_apply_unified(p, ro, rc, q, g) - a) ** 2)
    grad_fn = jax.jit(jax.value_and_grad(loss_fn)); rng = np.random.default_rng(0); N = len(A)
    for step in range(args.steps):
        idx = rng.integers(0, N, args.batch)
        Lv, gr = grad_fn(params, RO[idx], RC[idx], Q[idx], G[idx], A[idx])
        upd, opt_state = opt.update(gr, opt_state, params); params = optax.apply_updates(params, upd)
        if step % 500 == 0 or step == args.steps - 1:
            pred = np.asarray(head_apply_unified(params, RO[:2000], RC[:2000], Q[:2000], G[:2000]))
            gt = np.asarray(A[:2000])
            dc = float(np.mean(np.sum(_n(pred[:,:3]) * _n(gt[:,:3]), -1)))
            ga = float(np.mean((pred[:,6] < 0) == (gt[:,6] < 0)))
            print(f"step {step:5d} mse={float(Lv):.4f} dir_cos={dc:+.3f} grip_acc={ga:.3f}", flush=True)
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(jax.tree.map(np.asarray, params), open(args.out, "wb"))
    print(f"SAVED {args.out}\nDISTILL_UNIFIED_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
