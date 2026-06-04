"""Method A training loop (binder + language-adapter + vision-discriminator) on frozen pi0.5.

Demo-free DIRECTIONAL supervision: the bound policy's net EE displacement (unnormalized
dims 0:3 of the sampled action chunk, world frame) should point TOWARD the language-named
target and AWAY from the memorized distractor. Privileged target/distractor positions come
from the npz (never used at inference). pi0.5 stays frozen; only binder/adapter/disc train.

Adversarial vision-independence (the memorization-breaking certificate): a discriminator
regresses g from VISION-only pooled features; the adapter is pushed to make UNIT g
un-predictable from vision (bounded since ||g||=1). --penalty off disables the adapter's
independence term (disc still trains as a passive witness) — this is the KILL arm of the 2x2.

Outputs trained params to runs/cf_bind_<tag>.pkl for cf_harness eval.
"""
from __future__ import annotations

import argparse
import glob
import pickle
import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import optax

import cf_bind as B


def quat2axisangle(quat):  # robosuite convention (x,y,z,w); from examples/libero/main.py
    q = np.asarray(quat, np.float64)
    if q[3] > 1.0: q = q / np.linalg.norm(q)
    if q[3] < -1.0: q = q / np.linalg.norm(q)
    den = np.sqrt(max(1.0 - q[3] * q[3], 0.0))
    if den < 1e-8: return np.zeros(3)
    return (q[:3] * 2.0 * np.arccos(np.clip(q[3], -1, 1))) / den


def build_dataset(files, tf):
    """Run each npz through the real pi0.5 input transform; stack into a batched dict + supervision."""
    inps, ee, tpos, dpos = [], [], [], []
    for f in files:
        d = np.load(f, allow_pickle=True)
        state = np.concatenate([d["ee_pos"], quat2axisangle(d["ee_quat"]), d["gripper_qpos"]]).astype(np.float32)
        raw = {"observation/image": d["agentview"], "observation/wrist_image": d["wrist"],
               "observation/state": state, "prompt": str(d["instruction"])}
        inps.append(tf(raw))
        ee.append(d["ee_pos"].astype(np.float32))
        tpos.append(d["target_pos"].astype(np.float32))
        dpos.append(d["distractor_pos"].astype(np.float32))
    batched = {k: np.stack([x[k] for x in inps]) for k in inps[0] if not isinstance(inps[0][k], dict)}
    # nested image dicts
    for k in inps[0]:
        if isinstance(inps[0][k], dict):
            batched[k] = {kk: np.stack([x[k][kk] for x in inps]) for kk in inps[0][k]}
    sup = {"ee": np.stack(ee), "tpos": np.stack(tpos), "dpos": np.stack(dpos)}
    return batched, sup


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/cf_train")
    p.add_argument("--penalty", choices=["on", "off"], default="on")
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--dg", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--alpha", type=float, default=1.0, help="away-from-distractor weight in task loss")
    p.add_argument("--w-disc", type=float, default=1.0)
    p.add_argument("--w-indep", type=float, default=1.0)
    p.add_argument("--num-steps", type=int, default=10)
    p.add_argument("--out", default="runs")
    args = p.parse_args()

    from openpi.training import config as _config
    from openpi.policies import policy_config
    from openpi.models import model as _model
    from openpi.shared import download
    import json

    cfg = _config.get_config("pi05_libero")
    ckpt = download.maybe_download("gs://openpi-assets/checkpoints/pi05_libero/params")
    policy = policy_config.create_trained_policy(cfg, "gs://openpi-assets/checkpoints/pi05_libero")
    model = policy._model
    tf = policy._input_transform

    # action norm stats (quantile) for dims 0:3 -> physical net displacement
    ns = json.load(open(str(pathlib.Path(ckpt).parent /
        "assets/physical-intelligence/libero/norm_stats.json")))["norm_stats"]["actions"]
    q01 = np.array(ns["q01"][:3], np.float32); q99 = np.array(ns["q99"][:3], np.float32)
    a_scale = jnp.asarray((q99 - q01) / 2.0); a_bias = jnp.asarray((q99 + q01) / 2.0)

    files = sorted(glob.glob(str(pathlib.Path(args.data) / "*.npz")))
    print(f"{len(files)} training observations", flush=True)
    batched, sup = build_dataset(files, tf)
    N = len(files)

    # cache FROZEN prefix features once (adapter + disc read these; no grad to VLM)
    obs_all = jax.tree.map(jnp.asarray, _model.Observation.from_dict(batched))
    print("caching prefix features...", flush=True)
    prefix_cache = []
    for i in range(0, N, args.batch):
        ob = jax.tree.map(lambda x: x[i:i + args.batch], obs_all)
        pf, _ = model.extract_vlm_spatial_features(ob)
        prefix_cache.append(np.asarray(pf))
    prefix_cache = jnp.asarray(np.concatenate(prefix_cache, 0))   # [N, seq, 2048]
    L_txt = int(obs_all.tokenized_prompt.shape[1])
    n_img = int(prefix_cache.shape[1]) - L_txt                    # image tokens precede language tokens
    print(f"prefix_cache {prefix_cache.shape}  L_txt={L_txt}  n_img={n_img}", flush=True)

    key = jax.random.key(0)
    k1, k2, k3, key = jax.random.split(key, 4)
    params = {"binder": B.init_binder(k1, args.dg), "adapter": B.init_adapter(k2, args.dg),
              "disc": B.init_disc(k3, args.dg)}
    opt = optax.adam(args.lr); opt_state = opt.init(params)
    ee = jnp.asarray(sup["ee"]); tpos = jnp.asarray(sup["tpos"]); dpos = jnp.asarray(sup["dpos"])
    nrm = lambda v: v / (jnp.linalg.norm(v, axis=-1, keepdims=True) + 1e-8)
    w_indep = args.w_indep if args.penalty == "on" else 0.0

    def loss_fn(params, ob, pf, ee_b, tp_b, dp_b, noise):
        g = B.adapter_apply(params["adapter"], pf, n_img)
        g = nrm(g)                                                   # unit -> bounded adversary
        actions = model.sample_actions_bound_train(ob, g, B.binder_apply, params["binder"],
                                                   noise, num_steps=args.num_steps)
        S = actions[:, :, 0:3].sum(1)                                # net normalized displacement
        d = S * a_scale + args.num_steps * a_bias                    # -> physical (world frame)
        d_u = nrm(d)
        u_t = nrm(tp_b - ee_b); u_d = nrm(dp_b - ee_b)
        cos_t = jnp.sum(d_u * u_t, -1); cos_d = jnp.sum(d_u * u_d, -1)
        L_task = (-cos_t + args.alpha * jax.nn.relu(cos_d)).mean()
        disc_out = B.disc_apply(params["disc"], pf, n_img)          # disc pools vision internally
        L_disc = jnp.mean((disc_out - jax.lax.stop_gradient(g)) ** 2)        # train disc (witness)
        L_indep = jnp.mean((jax.lax.stop_gradient(disc_out) - g) ** 2)       # adapter strips vision info
        L = L_task + args.w_disc * L_disc - w_indep * L_indep
        return L, (L_task, L_disc, cos_t.mean(), cos_d.mean())

    grad_fn = jax.jit(jax.value_and_grad(loss_fn, has_aux=True))

    rng = np.random.default_rng(0)
    for step in range(args.steps):
        idx = rng.choice(N, size=min(args.batch, N), replace=False)
        ob = jax.tree.map(lambda x: x[idx], obs_all)
        pf = prefix_cache[idx]
        noise = jax.random.normal(jax.random.key(1000 + step), (len(idx), model.action_horizon, model.action_dim))
        (L, aux), grads = grad_fn(params, ob, pf, ee[idx], tpos[idx], dpos[idx], noise)
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        if step % 20 == 0 or step == args.steps - 1:
            Lt, Ld, ct, cd = (float(x) for x in aux)
            print(f"step {step:4d}  L={float(L):+.4f}  task={Lt:+.4f}  disc={Ld:.4f}  "
                  f"cos_tgt={ct:+.3f}  cos_distr={cd:+.3f}  gate={float(jnp.tanh(params['binder']['gate'])):+.3f}",
                  flush=True)

    out = pathlib.Path(args.out); out.mkdir(exist_ok=True)
    fn = out / f"cf_bind_{args.penalty}.pkl"
    with open(fn, "wb") as fh:
        pickle.dump(jax.tree.map(np.asarray, params), fh)
    print(f"SAVED {fn}", flush=True)
    print("CF_BIND_TRAIN_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
