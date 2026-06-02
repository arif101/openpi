"""Exp 2 gate (cheapest decisive): do Pi0.5's FROZEN VLM features encode object
POSITION well enough to predict it, and does that mapping GENERALIZE to perturbed
positions (OOD)?

This is the make-or-break perception question for the world encoder. If a small
probe on VLM features predicts object position to ~cm in-dist AND on perturbed
scenes, the world encoder will work -> build the full NDP-with-predicted-pose
eval. If it can't (esp. OOD), perception is the bottleneck — an honest, important
finding that no NDP can fix.

Pipeline: keystone NPZ images -> Pi0.5 VLM features (extract_vlm_features) ->
probe (features -> target object position), supervised by privileged object_pos.
Train on pert0, eval on pert0-val + pert5 + pert10. Report position error (cm).
"""
from __future__ import annotations

import argparse
import glob
import pathlib

import jax.numpy as jnp
import numpy as np

from openpi.contact_mpc.features.extractor import extract_features_from_dict
from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.shared import download
from openpi.policies.libero_policy import _parse_image  # noqa (orientation handling reference)
import jax
import rekey


def load_model(ckpt):
    cfg = pi0_config.Pi0Config(pi05=True, action_horizon=10,
                               paligemma_variant="gemma_2b", action_expert_variant="gemma_300m")
    params = _model.restore_params(download.maybe_download(ckpt), dtype=jnp.bfloat16)
    m = cfg.load(params); m.eval()
    return m


def resize224(imgs):
    # imgs [K,H,W,3] uint8 (saved double-flipped) -> un-flip -> 224 float->uint8
    imgs = imgs[:, ::-1, ::-1, :]                              # recover Pi0.5's orientation
    out = jax.image.resize(imgs.astype(np.float32), (imgs.shape[0], 224, 224, 3), "bilinear")
    return np.asarray(jnp.clip(out, 0, 255).astype(jnp.uint8))


def extract_for_dir(model, files, bs=8):
    feats, pos = [], []
    for f in files:
        d = np.load(f, allow_pickle=True)
        imgs, wrist, it = d["image"], d["wrist_image"], d["image_t"]
        names = list(d["object_names"]); ti = rekey.select_target_object(d["object_pos"], names)
        if imgs.shape[0] == 0:
            continue
        base = resize224(imgs); wr = resize224(wrist)
        labels = d["object_pos"][it, ti]                      # object position at each image's timestep
        for b in range(0, base.shape[0], bs):
            data = {
                "image": {"base_0_rgb": base[b:b+bs],
                          "left_wrist_0_rgb": wr[b:b+bs],
                          "right_wrist_0_rgb": np.zeros_like(base[b:b+bs])},
                "image_mask": {k: np.ones(base[b:b+bs].shape[0], bool)
                               for k in ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")},
                "state": np.zeros((base[b:b+bs].shape[0], 32), np.float32),  # prefix features ignore state
            }
            feats.append(extract_features_from_dict(model, data))
            pos.append(labels[b:b+bs])
    return np.concatenate(feats), np.concatenate(pos)


def mlp_probe(Xtr, Ytr, Xte_dict, epochs=300):
    import torch
    Xtr = torch.tensor(Xtr); Ytr = torch.tensor(Ytr)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-5
    net = torch.nn.Sequential(torch.nn.Linear(Xtr.shape[1], 256), torch.nn.SiLU(),
                              torch.nn.Linear(256, 3))
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    for _ in range(epochs):
        opt.zero_grad(); loss = ((net((Xtr - mu) / sd) - Ytr) ** 2).mean(); loss.backward(); opt.step()
    out = {}
    for name, (Xte, Yte) in Xte_dict.items():
        with torch.no_grad():
            pred = net((torch.tensor(Xte) - mu) / sd).numpy()
        err = np.linalg.norm(pred - Yte, axis=-1)              # meters
        out[name] = (err.mean(), np.median(err))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task-idx", type=int, default=3)
    p.add_argument("--task-suite", default="libero_10")
    p.add_argument("--ckpt", default="gs://openpi-assets/checkpoints/pi05_libero/params")
    p.add_argument("--data", default="data/keystone")
    args = p.parse_args()
    model = load_model(args.ckpt)
    pat = lambda pert: sorted(glob.glob(str(pathlib.Path(args.data) / f"pert{pert}" /
                              f"PHYS_OK_{args.task_suite}_task{args.task_idx}_*baseline*.npz")))
    print("extracting features ...", flush=True)
    X0, Y0 = extract_for_dir(model, pat(0))
    X5, Y5 = extract_for_dir(model, pat(5))
    X10, Y10 = extract_for_dir(model, pat(10))
    n = X0.shape[0]; ntr = int(n * 0.8)
    print(f"pert0 {n} feats, pert5 {X5.shape[0]}, pert10 {X10.shape[0]}", flush=True)
    res = mlp_probe(X0[:ntr], Y0[:ntr],
                    {"pert0-val": (X0[ntr:], Y0[ntr:]), "pert5": (X5, Y5), "pert10": (X10, Y10)})
    print("\n=== VLM-feature -> object-position probe error (meters) ===", flush=True)
    for k, (m, md) in res.items():
        print(f"  {k:10s}: mean {m*100:.1f}cm  median {md*100:.1f}cm", flush=True)
    print("\nGate: if pert5/pert10 error is ~<=3cm, VLM features encode object position OOD -> "
          "world encoder viable. If large, perception is the bottleneck.", flush=True)


if __name__ == "__main__":
    main()
