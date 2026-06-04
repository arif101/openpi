"""Method A modules (pure-JAX, own params; pi0.5 stays frozen) + the gating
bit-identical check.

binder  : zero-init gated FiLM on the action-expert output (suffix_out, 1024-d).
          gate=0 at init -> tanh(0)=0 -> identity -> motor preserved by construction.
adapter : pooled LANGUAGE features (2048-d) -> target vector g (dg).
disc    : pooled VISION features (2048-d) -> predict g  (adversarial; grad-reversal
          makes g vision-independent = the memorization-breaking certificate).

Run as __main__ = the GATING kill-test: sample_actions_bound with a zero-init binder
must be bit-identical to stock sample_actions (motor untouched at init).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

N_IMG = 3 * 196          # 3 cameras x 14x14 patch tokens (vision prefix tokens)
D_VLM = 2048             # gemma_2b width (prefix/VLM)
D_ACT = 1024             # gemma_300m width (action expert / suffix_out)


# ----------------------------- modules (pure jax) -----------------------------
def init_mlp(key, din, dh, dout, zero_last=False):
    k1, k2 = jax.random.split(key)
    s = lambda a, b: jax.random.normal(k1, (a, b)) * (1.0 / np.sqrt(a))
    w2 = jnp.zeros((dh, dout)) if zero_last else jax.random.normal(k2, (dh, dout)) * (1.0 / np.sqrt(dh))
    return {"w1": jax.random.normal(k1, (din, dh)) * (1.0 / np.sqrt(din)), "b1": jnp.zeros(dh),
            "w2": w2, "b2": jnp.zeros(dout)}


def mlp(p, x):
    return jax.nn.silu(x @ p["w1"] + p["b1"]) @ p["w2"] + p["b2"]


def init_binder(key, dg, dh=256):
    ka, kb = jax.random.split(key)
    return {"hid": init_mlp(ka, dg, dh, dh),
            "scale": jnp.zeros((dh, D_ACT)), "shift": jnp.zeros((dh, D_ACT)),
            "gate": jnp.zeros(())}                                   # tanh(0)=0 -> identity at init


def binder_apply(p, suffix_out, g):
    gate = jnp.tanh(p["gate"])
    h = jax.nn.silu(mlp(p["hid"], g))                               # [B, dh]
    scale = h @ p["scale"]; shift = h @ p["shift"]                  # [B, D_ACT]
    out = suffix_out * (1.0 + gate * scale[:, None, :]) + gate * shift[:, None, :]
    return out.astype(suffix_out.dtype)                             # gate=0 -> exact identity (no bf16 upcast drift)


def init_adapter(key, dg, dh=512):
    return init_mlp(key, D_VLM, dh, dg)


def adapter_apply(p, prefix_out, n_img=N_IMG):
    lang = prefix_out[:, n_img:].mean(1)                            # pool language tokens
    return mlp(p, lang)


def init_disc(key, dg, dh=512):
    return init_mlp(key, D_VLM, dh, dg)


def disc_apply(p, prefix_out, n_img=N_IMG):
    vis = prefix_out[:, :n_img].mean(1)                            # pool vision tokens
    return mlp(p, vis)


# ----------------------------- gating bit-identical check -----------------------------
def _dummy_obs(model):
    from openpi.models import model as _model
    rng = np.random.default_rng(0)
    img = {k: rng.integers(0, 255, (1, 224, 224, 3), np.uint8) for k in
           ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")}
    data = {"image": img,
            "image_mask": {k: np.ones(1, bool) for k in img},
            "state": np.zeros((1, 32), np.float32),
            "tokenized_prompt": rng.integers(0, 1000, (1, 48), np.int32),
            "tokenized_prompt_mask": np.ones((1, 48), bool)}
    return jax.tree.map(jnp.asarray, _model.Observation.from_dict(data))


def main():
    from openpi.models import model as _model
    from openpi.models import pi0_config
    from openpi.shared import download
    ckpt = "gs://openpi-assets/checkpoints/pi05_libero/params"
    cfg = pi0_config.Pi0Config(pi05=True, action_horizon=10,
                               paligemma_variant="gemma_2b", action_expert_variant="gemma_300m")
    model = cfg.load(_model.restore_params(download.maybe_download(ckpt), dtype=jnp.bfloat16)); model.eval()
    obs = _dummy_obs(model)
    dg = 256
    binder_p = init_binder(jax.random.key(0), dg)
    g = jax.random.normal(jax.random.key(1), (1, dg))
    noise = jax.random.normal(jax.random.key(2), (1, 10, 32))
    a_stock = np.asarray(model.sample_actions(jax.random.key(3), obs, num_steps=10, noise=noise)[0])
    a_bound = np.asarray(model.sample_actions_bound(jax.random.key(3), obs, g, binder_apply, binder_p,
                                                    num_steps=10, noise=noise))
    diff = float(np.abs(a_stock - a_bound).max())
    print(f"max|stock - bound(zero-init)| = {diff:.2e}", flush=True)
    print("GATING: PASS (bit-identical, motor preserved)" if diff < 1e-4
          else "GATING: FAIL (zero-init not identity -> wiring bug)", flush=True)
    print("BIND_CHECK_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
