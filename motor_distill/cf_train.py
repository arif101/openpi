"""Counterfactual-invariance LoRA trainer (brick #1 core). Fine-tunes a LoRA on FROZEN pi0.5 with the 3-arm loss:
  --loss bc            : Arm-2 CONTROL (CAST-style) = flow-matching BC on relabeled demos only.
  --loss bc_inv        : + INVARIANCE   (action velocity invariant to distractor-edited twin, same instr).
  --loss bc_inv_sens   : + SENSITIVITY  (action velocity DIVERGES between instr=M and instr=M2 = referent swap).
The HEADLINE ablation is bc (Arm-2) vs bc_inv_sens (Arm-3): does the OBJECTIVE beat the same data under plain BC?

Reuses openpi: pi05_libero_waypoint_lora model config (LoRA + freeze/trainable filters), CheckpointWeightLoader
to restore pi05 base weights (LoRA stays fresh & trainable), PaligemmaTokenizer, and model.compute_loss for BC.
The inv/sens terms compare the flow VELOCITY v=action_out_proj(suffix_out) at a shared (x_t,t) across paired obs.

!!! UNTESTED — debug on box with --selftest (loads model + 1 BC loss on a corpus batch) before full train.
Run: PYTHONPATH=third_party/libero:motor_distill .venv/bin/python motor_distill/cf_train.py --loss bc_inv_sens --steps 1500 --out runs/cf_lora_arm3
"""
from __future__ import annotations
import argparse, glob, json, pathlib, re
import numpy as np, jax, jax.numpy as jnp, optax
from flax import nnx
import einops


def quat2axisangle(q):
    q = np.asarray(q, np.float32);
    if q[3] > 1: q = q / np.linalg.norm(q)
    den = np.sqrt(max(1 - q[3] * q[3], 1e-12)); ang = 2 * np.arctan2(np.linalg.norm(q[:3]), q[3])
    return (q[:3] / den * ang).astype(np.float32) if den > 1e-6 else np.zeros(3, np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-name", default="pi05_libero_waypoint_lora")
    ap.add_argument("--checkpoint", default="gs://openpi-assets/checkpoints/pi05_libero")
    ap.add_argument("--corpus", default="data/cf_corpus")
    ap.add_argument("--loss", choices=["bc", "bc_inv", "bc_inv_sens"], default="bc_inv_sens")
    ap.add_argument("--obj-filter", default="")   # train-object tokens (held-out split)
    ap.add_argument("--steps", type=int, default=1500); ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--lam-inv", type=float, default=1.0)
    ap.add_argument("--lam-sens", type=float, default=0.5); ap.add_argument("--out", default="runs/cf_lora")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    from openpi.training import config as _config
    from openpi.training import weight_loaders
    from openpi.models import model as _model
    from openpi.models.tokenizer import PaligemmaTokenizer
    from openpi.shared import download

    cfg = _config.get_config(args.config_name)
    AD = cfg.model.action_dim; AH = cfg.model.action_horizon
    tok = PaligemmaTokenizer(max_len=cfg.model.max_token_len)
    # ---- norm stats (state+action) from the pi05_libero checkpoint assets ----
    apath = download.maybe_download(args.checkpoint + "/assets")
    ns = None
    for f in glob.glob(str(pathlib.Path(apath) / "**" / "norm_stats.json"), recursive=True):
        ns = json.load(open(f)); break
    def norm(x, key):
        if ns is None: return x
        s = ns["norm_stats"][key]; m = np.asarray(s["mean"], np.float32); sd = np.asarray(s["std"], np.float32) + 1e-6
        return (x - m[: x.shape[-1]]) / sd[: x.shape[-1]]

    # ---- model: LoRA config + restore pi05 base weights ----
    model = cfg.model.create(jax.random.key(0))
    loaded = weight_loaders.CheckpointWeightLoader(args.checkpoint + "/params").load(nnx.state(model).to_pure_dict())
    gdef, state = nnx.split(model); state.replace_by_pure_dict(loaded); model = nnx.merge(gdef, state)
    print(f"model loaded (LoRA={args.config_name}); action_dim={AD} horizon={AH}", flush=True)

    # ---- load corpus -> per-timestep samples ----
    files = sorted(glob.glob(str(pathlib.Path(args.corpus) / "*.npz")))
    if args.obj_filter:
        toks = [t.strip() for t in args.obj_filter.split(",") if t.strip()]
        files = [f for f in files if any(t in pathlib.Path(f).name for t in toks)]
    S = []
    for f in files:
        d = np.load(f, allow_pickle=True)
        instr = str(d["instr"]); instr2 = str(d["instr_alt"])
        ptok, pmask = tok.tokenize(instr); ptok2, pmask2 = tok.tokenize(instr2)
        T = len(d["action"])
        for t in range(T):
            st = d["state"][t]; ax = np.concatenate([st[:3], quat2axisangle(st[3:7]), st[7:9]]).astype(np.float32)
            S.append(dict(main=d["images_main"][t], wrist=d["images_wrist"][t], twin=d["twin_main"][t],
                          has_twin=int(d["has_twin"][t]), state=norm(np.pad(ax, (0, AD - len(ax))), "state"),
                          action=norm(np.pad(d["action"][t], ((0, 0)) if d["action"][t].ndim else (0, AD - 7)), "actions") if False else None,
                          act=d["action"][t], ptok=ptok, pmask=pmask, ptok2=ptok2, pmask2=pmask2))
    print(f"{len(files)} episodes -> {len(S)} timesteps", flush=True)
    if not S:
        print("NO DATA", flush=True); return

    def obs_of(batch, imgs_key="main", ptok_key="ptok", pmask_key="pmask"):
        img = jnp.asarray(np.stack([b[imgs_key] for b in batch]).astype(np.float32) / 127.5 - 1)
        wr = jnp.asarray(np.stack([b["wrist"] for b in batch]).astype(np.float32) / 127.5 - 1)
        B = len(batch)
        return _model.Observation(
            images={"base_0_rgb": img, "left_wrist_0_rgb": wr, "right_wrist_0_rgb": jnp.zeros_like(img)},
            image_masks={"base_0_rgb": jnp.ones(B, bool), "left_wrist_0_rgb": jnp.ones(B, bool),
                         "right_wrist_0_rgb": jnp.zeros(B, bool)},
            state=jnp.asarray(np.stack([b["state"] for b in batch])),
            tokenized_prompt=jnp.asarray(np.stack([b[ptok_key] for b in batch])),
            tokenized_prompt_mask=jnp.asarray(np.stack([b[pmask_key] for b in batch])))

    def actions_of(batch):
        a = np.stack([np.pad(b["act"], (0, AD - 7))[None].repeat(AH, 0) for b in batch]).astype(np.float32)
        return jnp.asarray(a)   # [B, AH, AD] (single-step action tiled over horizon as a simple BC target)

    def velocity(m, obs, x_t, t):
        from openpi.models.pi0 import make_attn_mask
        obs = _model.preprocess_observation(None, obs, train=False)
        pt, pm, par = m.embed_prefix(obs)
        st, sm, sar, ad = m.embed_suffix(obs, x_t, t, target_state=None)
        im = jnp.concatenate([pm, sm], 1); arm = jnp.concatenate([par, sar], 0)
        (_, so), _ = m.PaliGemma.llm([pt, st], mask=make_attn_mask(im, arm),
                                     positions=jnp.cumsum(im, 1) - 1, adarms_cond=[None, ad])
        return m.action_out_proj(so[:, -AH:])

    def loss_fn(m, rng, batch):
        obs = obs_of(batch); act = actions_of(batch)
        bc = m.compute_loss(rng, obs, act, train=True).mean()
        total = bc; aux = {"bc": bc}
        if args.loss in ("bc_inv", "bc_inv_sens"):
            B = act.shape[0]; noise = jax.random.normal(rng, act.shape); t = jnp.full((B,), 0.5)
            x_t = 0.5 * noise + 0.5 * act
            v_obs = velocity(m, obs, x_t, t)
            tw = obs_of(batch, imgs_key="twin")
            v_tw = velocity(m, tw, x_t, t)
            ht = jnp.asarray(np.array([b["has_twin"] for b in batch], np.float32))[:, None, None]
            inv = (jnp.sum(((v_obs - v_tw) ** 2) * ht) / (jnp.sum(ht) * AH * AD + 1e-6))
            total = total + args.lam_inv * inv; aux["inv"] = inv
            if args.loss == "bc_inv_sens":
                obs2 = obs_of(batch, ptok_key="ptok2", pmask_key="pmask2")
                v2 = velocity(m, obs2, x_t, t)
                sens = -jnp.mean((v_obs - v2) ** 2)         # push apart -> sensitive to instruction
                total = total + args.lam_sens * sens; aux["sens"] = sens
        return total, aux

    diff = nnx.DiffState(0, cfg.trainable_filter)
    gradfn = nnx.value_and_grad(loss_fn, argnums=diff, has_aux=True)
    tx = optax.adamw(args.lr); params = nnx.state(model).filter(cfg.trainable_filter); opt_state = tx.init(params)
    rng = jax.random.key(1); idx = np.random.default_rng(0)

    if args.selftest:
        b = [S[i] for i in idx.integers(0, len(S), 4)]
        (l, aux), _ = gradfn(model, rng, b); print("SELFTEST loss", float(l), {k: float(v) for k, v in aux.items()}, flush=True)
        print("CF_TRAIN_SELFTEST_OK", flush=True); return

    for step in range(args.steps):
        rng, sk = jax.random.split(rng)
        b = [S[i] for i in idx.integers(0, len(S), args.batch)]
        (l, aux), grads = gradfn(model, sk, b)
        gp = grads.filter(cfg.trainable_filter); upd, opt_state = tx.update(gp, opt_state, params)
        params = optax.apply_updates(params, upd); nnx.update(model, params)
        if step % 100 == 0 or step == args.steps - 1:
            print(f"step {step:5d} loss={float(l):.4f} " + " ".join(f"{k}={float(v):.4f}" for k, v in aux.items()), flush=True)
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    import pickle; pickle.dump(jax.tree.map(np.asarray, nnx.state(model).filter(cfg.trainable_filter).to_pure_dict()), open(args.out + ".pkl", "wb"))
    print(f"SAVED {args.out}.pkl\nCF_TRAIN_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
