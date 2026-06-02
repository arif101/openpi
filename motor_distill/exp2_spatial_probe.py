"""Exp 2 (SPATIAL re-test): do Pi0.5's frozen VLM features encode object POSITION
when read SPATIALLY (per-token grid) instead of mean-pooled?

The pooled probe (exp2_probe.py) failed: 12.8cm in-dist, 24-31cm OOD. Diagnosis:
mean-pooling averages over all tokens -> destroys WHERE the object is. Pi0.5 itself
localizes via the per-token spatial grid (attention), not a pooled vector. This
re-test uses the un-pooled base-camera 14x14 patch grid with a spatial soft-argmax
keypoint head (the standard way to extract position from a feature map) -> object xyz.

Gate (unchanged): if pert5/pert10 error ~<=3cm -> spatial features localize OOD ->
world encoder viable. If still large -> perception is the genuine wall (the VLM, not
our handling). We also print a PREDICT-MEAN baseline + label spread per split so we
can tell "probe learned nothing (=mean)" apart from "probe localizes."

Train on pert0, eval on pert0-val + pert5 + pert10. Position error in cm.
"""
from __future__ import annotations

import argparse
import glob
import pathlib

import jax
import jax.numpy as jnp
import numpy as np

from openpi.contact_mpc.features.extractor import extract_spatial_features_from_dict
from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.shared import download
import rekey


GRID = 14                 # 224/16 patches per side
N_BASE = GRID * GRID       # 196 base-camera patch tokens (base camera = first image in prefix)
N_FRAMES = 6
BS = 8


def load_model(ckpt):
    cfg = pi0_config.Pi0Config(pi05=True, action_horizon=10,
                               paligemma_variant="gemma_2b", action_expert_variant="gemma_300m")
    params = _model.restore_params(download.maybe_download(ckpt), dtype=jnp.bfloat16)
    m = cfg.load(params); m.eval()
    return m


def resize224(imgs):
    imgs = imgs[:, ::-1, ::-1, :]
    out = jax.image.resize(imgs.astype(np.float32), (imgs.shape[0], 224, 224, 3), "bilinear")
    return np.asarray(jnp.clip(out, 0, 255).astype(jnp.uint8))


def extract_for_dir(model, files):
    bases, wrists, labels = [], [], []
    for f in files:
        d = np.load(f, allow_pickle=True)
        imgs, wrist, it = d["image"], d["wrist_image"], d["image_t"]
        if imgs.shape[0] == 0:
            continue
        k = min(N_FRAMES, imgs.shape[0])
        names = list(d["object_names"]); ti = rekey.select_target_object(d["object_pos"], names)
        it = np.clip(it[:k], 0, d["object_pos"].shape[0] - 1)
        bases.append(resize224(imgs[:k])); wrists.append(resize224(wrist[:k]))
        labels.append(d["object_pos"][it, ti])
    base = np.concatenate(bases); wr = np.concatenate(wrists); lab = np.concatenate(labels)
    grids = []
    for b in range(0, base.shape[0], BS):
        bb, ww = base[b:b+BS], wr[b:b+BS]
        pad = BS - bb.shape[0]
        if pad:
            bb = np.concatenate([bb, np.repeat(bb[-1:], pad, 0)])
            ww = np.concatenate([ww, np.repeat(ww[-1:], pad, 0)])
        data = {
            "image": {"base_0_rgb": bb, "left_wrist_0_rgb": ww, "right_wrist_0_rgb": np.zeros_like(bb)},
            "image_mask": {k_: np.ones(BS, bool) for k_ in ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")},
            "state": np.zeros((BS, 32), np.float32),
        }
        prefix_out, _ = extract_spatial_features_from_dict(model, data)   # [BS, seq, D]
        g = prefix_out[:, :N_BASE]                                        # base-camera 14x14 grid
        grids.append(g[: bb.shape[0] - pad] if pad else g)
        if (b // BS) % 10 == 0:
            print(f"    batch {b//BS}/{(base.shape[0]+BS-1)//BS}", flush=True)
    return np.concatenate(grids), lab                                    # [N,196,D], [N,3]


class KeypointHead:
    """Spatial soft-argmax keypoint readout: D-dim grid -> K heatmaps -> K (u,v) -> MLP -> xyz."""
    def __init__(self, D, K=16):
        import torch
        self.torch = torch
        self.detect = torch.nn.Linear(D, K)                              # 1x1 conv over grid
        self.mlp = torch.nn.Sequential(torch.nn.Linear(2 * K, 128), torch.nn.SiLU(),
                                       torch.nn.Linear(128, 64), torch.nn.SiLU(),
                                       torch.nn.Linear(64, 3))
        ys, xs = torch.meshgrid(torch.linspace(0, 1, GRID), torch.linspace(0, 1, GRID), indexing="ij")
        self.xs = xs.reshape(-1); self.ys = ys.reshape(-1)               # [196]
        self.params = list(self.detect.parameters()) + list(self.mlp.parameters())

    def forward(self, X):                                                # X [B,196,D]
        torch = self.torch
        heat = self.detect(X)                                            # [B,196,K]
        attn = torch.softmax(heat, dim=1)                                # spatial softmax per channel
        u = (attn * self.xs[None, :, None]).sum(1)                       # [B,K]
        v = (attn * self.ys[None, :, None]).sum(1)                       # [B,K]
        kp = torch.cat([u, v], dim=1)                                    # [B,2K]
        return self.mlp(kp)


def run_probe(Xtr, Ytr, Xte_dict, epochs=600, K=16):
    import torch
    head = KeypointHead(Xtr.shape[-1], K=K)
    Xtr = torch.tensor(Xtr); Ytr = torch.tensor(Ytr)
    mu, sd = Ytr.mean(0), Ytr.std(0) + 1e-6                              # normalize targets
    opt = torch.optim.Adam(head.params, 2e-3)
    for ep in range(epochs):
        opt.zero_grad()
        pred = head.forward(Xtr)
        loss = (((pred - Ytr) / sd) ** 2).mean()
        loss.backward(); opt.step()
        if ep % (epochs // 6) == 0 or ep == epochs - 1:
            print(f"    [head] ep {ep} loss {loss.item():.4f}", flush=True)
    out = {}
    ymean = Ytr.mean(0)                                                  # predict-mean baseline (from train)
    for name, (Xte, Yte) in Xte_dict.items():
        with torch.no_grad():
            pred = head.forward(torch.tensor(Xte)).numpy()
        err = np.linalg.norm(pred - Yte, axis=-1)
        base_err = np.linalg.norm(ymean.numpy() - Yte, axis=-1)         # error of always predicting train mean
        spread = np.linalg.norm(Yte - Yte.mean(0), axis=-1).mean()      # label spread within this split
        out[name] = (err.mean(), np.median(err), base_err.mean(), spread)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task-idx", type=int, default=3)
    p.add_argument("--task-suite", default="libero_10")
    p.add_argument("--ckpt", default="gs://openpi-assets/checkpoints/pi05_libero/params")
    p.add_argument("--data", default="data/keystone")
    p.add_argument("--K", type=int, default=16)
    args = p.parse_args()
    model = load_model(args.ckpt)
    pat = lambda pert: sorted(glob.glob(str(pathlib.Path(args.data) / f"pert{pert}" /
                              f"PHYS_OK_{args.task_suite}_task{args.task_idx}_*baseline*.npz")))
    print("extracting SPATIAL features ...", flush=True)
    X0, Y0 = extract_for_dir(model, pat(0))
    X5, Y5 = extract_for_dir(model, pat(5))
    X10, Y10 = extract_for_dir(model, pat(10))
    n = X0.shape[0]; ntr = int(n * 0.8)
    print(f"pert0 {n} grids {X0.shape}, pert5 {X5.shape[0]}, pert10 {X10.shape[0]}", flush=True)
    res = run_probe(X0[:ntr], Y0[:ntr],
                    {"pert0-val": (X0[ntr:], Y0[ntr:]), "pert5": (X5, Y5), "pert10": (X10, Y10)}, K=args.K)
    print("\n=== EXP 2 SPATIAL PROBE: keypoint readout -> object position (cm) ===", flush=True)
    print(f"{'split':10s}  {'probe-mean':>10s}  {'probe-med':>9s}  {'predict-mean':>12s}  {'label-spread':>12s}", flush=True)
    for k, (m, md, bm, sp) in res.items():
        print(f"{k:10s}  {m*100:9.1f}c  {md*100:8.1f}c  {bm*100:11.1f}c  {sp*100:11.1f}c", flush=True)
    print("\nRead: probe << predict-mean => features carry position. probe ~= predict-mean => learned nothing.", flush=True)
    print("Gate: pert5/pert10 probe-error ~<=3cm => spatial world encoder viable.", flush=True)
    print("SPATIAL_PROBE_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
