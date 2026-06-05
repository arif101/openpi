"""B2 (Method A v2): train a keypoint head on pi0.5 LANGUAGE-CONDITIONED spatial features to localize the
NAMED object, and measure DISAMBIGUATION (does it point at the named object vs a distractor?) -- the kill-gate.

Head: per-token logit over the 14x14 grid -> softmax -> soft-argmax -> (row,col) in [0,1]. The language is
baked into the features (extracted WITH the instruction), so the same scene + different named target ->
different features -> different predicted pixel. The head just reads "which token is the named object".

Metrics (scene-split, held-out scenes):
  pixel err   : |pred - true named pixel| (grid units -> approx)
  DISAMBIG acc: predicted pixel nearest the NAMED object's pixel among all objects in the scene (vs distractors)
  baseline    : predict scene-centroid (no language) disambig acc = chance

Run: python motor_distill/train_bind_kp.py --data data/keystone/bind_feats_s7.npz
"""
from __future__ import annotations
import argparse, pathlib
import numpy as np
import torch

GRID = 14
DEV = "cuda" if torch.cuda.is_available() else "cpu"


class KPHead(torch.nn.Module):
    def __init__(self, D, hidden=256):
        super().__init__()
        self.mlp = torch.nn.Sequential(torch.nn.Linear(D, hidden), torch.nn.SiLU(),
                                       torch.nn.Linear(hidden, 1))
        gr, gc = np.meshgrid(np.arange(GRID), np.arange(GRID), indexing="ij")
        coords = np.stack([(gr.reshape(-1) + 0.5) / GRID, (gc.reshape(-1) + 0.5) / GRID], -1)  # [196,2] (row,col) in [0,1]
        self.register_buffer("coords", torch.tensor(coords, dtype=torch.float32))

    def forward(self, X):                       # X [B,196,D]
        logit = self.mlp(X).squeeze(-1)         # [B,196]
        w = torch.softmax(logit, dim=1)         # heatmap over grid
        return w @ self.coords                  # [B,2] soft-argmax pixel (row,col)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/keystone/bind_feats_s7.npz")
    ap.add_argument("--epochs", type=int, default=800); ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    z = np.load(args.data, allow_pickle=True)
    X, px, names, scene = z["X"].astype(np.float32), z["px"].astype(np.float32), z["names"], z["scene"]
    print(f"loaded {X.shape} feats, {len(set(scene.tolist()))} scenes, {len(set(names.tolist()))} object types", flush=True)

    uscenes = np.array(sorted(set(scene.tolist())))
    rng = np.random.default_rng(args.seed); rng.shuffle(uscenes)
    folds = np.array_split(uscenes, args.folds)

    all_pix, all_dis, all_base = [], [], []
    for fi, te_scenes in enumerate(folds):
        te = np.isin(scene, te_scenes); tr = ~te
        if tr.sum() == 0 or te.sum() == 0: continue
        head = KPHead(X.shape[-1]).to(DEV)
        Xtr = torch.tensor(X[tr], device=DEV); Ytr = torch.tensor(px[tr], device=DEV)
        Xte = torch.tensor(X[te], device=DEV)
        opt = torch.optim.Adam(head.parameters(), 2e-3)
        for ep in range(args.epochs):
            opt.zero_grad(); loss = ((head(Xtr) - Ytr) ** 2).mean(); loss.backward(); opt.step()
        with torch.no_grad():
            pred = head(Xte).cpu().numpy()
        te_idx = np.where(te)[0]
        pix_err = np.linalg.norm(pred - px[te], axis=1)
        # disambiguation: predicted pixel nearest the NAMED obj among all objs in that scene
        dis, base = [], []
        for k, gi in enumerate(te_idx):
            s = scene[gi]
            cand_idx = np.where(scene == s)[0]
            cand_px = px[cand_idx]                       # all object pixels in the scene
            d = np.linalg.norm(cand_px - pred[k], axis=1)
            chosen = cand_idx[d.argmin()]
            dis.append(int(chosen == gi))
            # baseline: pick nearest to the scene-centroid (no language signal) = chance-ish
            cen = cand_px.mean(0); db = np.linalg.norm(cand_px - cen, axis=1)
            base.append(int(cand_idx[db.argmin()] == gi))
        all_pix += pix_err.tolist(); all_dis += dis; all_base += base
        print(f"  fold {fi}: {te.sum()} held-out, pix_err {pix_err.mean():.3f}, disambig {np.mean(dis):.2f}", flush=True)

    all_pix, all_dis, all_base = map(np.array, (all_pix, all_dis, all_base))
    # approx grid-units -> cm is scene-dependent; report grid-normalized + disambiguation (the kill-gate)
    print(f"\n=== Method A v2 binder (language-conditioned, scene-split CV, N={len(all_dis)}) ===", flush=True)
    print(f"  pixel err (grid-norm [0,1]) : mean {all_pix.mean():.3f}  median {np.median(all_pix):.3f}", flush=True)
    print(f"  DISAMBIGUATION accuracy     : {all_dis.mean():.2f}   (named obj vs distractors)", flush=True)
    print(f"  centroid baseline (no lang) : {all_base.mean():.2f}", flush=True)
    print(f"  OWLv2 reference (tonight)   : ~0.40-0.50 disambiguation on this suite", flush=True)
    verdict = ("LANGUAGE SIGNAL RECOVERABLE -> Method A viable" if all_dis.mean() > 0.7
               else "weak -> out-weighting too strong, consider LoRA fine-tune")
    print(f"  VERDICT: {verdict}", flush=True)
    print("TRAIN_BIND_KP_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
