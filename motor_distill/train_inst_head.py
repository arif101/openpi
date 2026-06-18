"""SUPERVISED MULTI-INSTANCE depth-fused grounding head (deep-research 2026-06-17 fix for the spatial dark-bowl wall).
Differs from the single-softmax grounding head two ways the lit says matter:
  (1) MULTI-PEAK heatmap REGRESSION (sigmoid per-patch BCE to Gaussian bumps at ALL instances of a noun) instead of
      softmax-over-patches (which forces ONE peak) -> can light up BOTH identical bowls.
  (2) DEPTH fused in (per-image table-relative) -> texture-independent geometric separation of the 2nd instance that
      RGB correspondence loses in background noise (UOIS-Net/RoboRefer/IAM).
Proto still conditions WHICH noun (open-vocab). Deploy: head(feats,depth,proto) -> sigmoid -> top-K NMS peaks = instances.

Run: python motor_distill/train_inst_head.py --data data/ground_depth_spatial,data/ground_depth_obj --out data/inst_head.pt
"""
from __future__ import annotations
import argparse, glob, pathlib
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F


class InstHead(nn.Module):
    """feats(B,g,g,C) + depth(B,g,g) + proto(B,C) -> per-patch heatmap logits (B,g,g). Sigmoid -> multi-peak."""
    def __init__(self, c=1024, d=256):
        super().__init__()
        self.fp = nn.Linear(c, d); self.qp = nn.Linear(c, d)
        self.conv = nn.Sequential(nn.Conv2d(d + 2, 128, 3, 1, 1), nn.GroupNorm(8, 128), nn.ReLU(),
                                  nn.Conv2d(128, 128, 3, 1, 1), nn.GroupNorm(8, 128), nn.ReLU(),
                                  nn.Conv2d(128, 1, 1))

    def forward(self, feats, depth, proto):
        B, g, _, C = feats.shape
        f = self.fp(feats); q = self.qp(proto)
        cond = f * q[:, None, None, :]                              # FiLM by object query (which noun)
        corr = (f * q[:, None, None, :]).sum(-1, keepdim=True)      # correspondence channel
        dn = depth.unsqueeze(-1)                                    # texture-independent geometry
        x = torch.cat([cond, corr, dn], -1).permute(0, 3, 1, 2)     # B,d+2,g,g
        return self.conv(x).squeeze(1)                              # B,g,g logits


def _norm_depth(d):
    """Per-image table-relative depth: objects (closer, smaller raw depth) -> positive bump."""
    med = np.median(d)
    return np.clip((med - d) / 0.05, -2.0, 6.0).astype(np.float32)


def load(dirs):
    """Group objects by readable noun -> MULTI-instance targets (both bowls share 'akita black bowl')."""
    from bind_foveate import nm
    F_, D_, P_, Y_, G = [], [], [], [], None
    protos = {}
    for d in str(dirs).split(","):
        for fn in sorted(glob.glob(str(pathlib.Path(d) / "*.npz"))):
            z = np.load(fn, allow_pickle=True); g = int(z["g"]); G = g
            ft = z["feats"].astype(np.float32); dp = z["depth"].astype(np.float32)
            objs = z["objs"]; gts = z["gt"]; prs = z["protos"].astype(np.float32)
            byn = {}
            for j in range(len(objs)):
                byn.setdefault(nm(str(objs[j])), []).append(j)
            for noun, idxs in byn.items():
                pr = np.mean([prs[j] for j in idxs], 0); pr = pr / (np.linalg.norm(pr) + 1e-6)
                protos[noun] = pr.astype(np.float16)
                for ti in range(len(ft)):
                    pts = [(int(gts[j][ti][0]), int(gts[j][ti][1])) for j in idxs if gts[j][ti][0] >= 0]
                    if not pts: continue
                    F_.append(ft[ti]); D_.append(_norm_depth(dp[ti])); P_.append(pr); Y_.append(pts)
    return F_, D_, P_, Y_, G, protos


def heatmap_target(pts, g, sigma=1.0):
    """Gaussian bumps (max-combined) at every instance patch -> (g,g) in [0,1]."""
    t = np.zeros((g, g), np.float32)
    rr, cc = np.mgrid[0:g, 0:g]
    for (r, c) in pts:
        t = np.maximum(t, np.exp(-((rr - r) ** 2 + (cc - c) ** 2) / (2 * sigma ** 2)))
    return t


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True); p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=int, default=80); p.add_argument("--bs", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4); p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--sigma", type=float, default=1.0); p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed)
    Fl, Dl, Pl, Yl, g, protos = load(args.data)
    C = Fl[0].shape[-1]; n = len(Yl)
    Ft = torch.tensor(np.stack(Fl)); Dt = torch.tensor(np.stack(Dl)); Pt = torch.tensor(np.stack(Pl))
    Tt = torch.tensor(np.stack([heatmap_target(pts, g, args.sigma) for pts in Yl]))   # (n,g,g)
    ninst = np.array([len(pts) for pts in Yl])
    idx = rng.permutation(n); nval = max(1, int(n * args.val_frac)); vi, ti = idx[:nval], idx[nval:]
    print(f"{n} samples (g={g}, C={C}); train {len(ti)} / val {len(vi)}; {len(protos)} nouns; "
          f"multi-inst samples={int((ninst>=2).sum())}", flush=True)
    net = InstHead(C).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)

    def peaks(hm, k, nms=2):
        out = []; h = hm.copy()
        for _ in range(k):
            i = int(h.argmax()); r, c = i // g, i % g; out.append((r, c))
            h[max(0, r-nms):r+nms+1, max(0, c-nms):c+nms+1] = -1e9
        return out

    def evly():
        net.eval()
        with torch.no_grad():
            inst_hit = 0; inst_tot = 0; cov = 0; covn = 0
            for s in range(0, len(vi), 256):
                b = vi[s:s+256]
                lo = torch.sigmoid(net(Ft[b].to(dev), Dt[b].to(dev), Pt[b].to(dev))).cpu().numpy()
                for bi, si in enumerate(b):
                    pts = Yl[si]; k = len(pts); pk = peaks(lo[bi], k)
                    # coverage: fraction of GT instances within 1 patch of SOME predicted peak
                    for (gr, gc) in pts:
                        inst_tot += 1
                        if any(abs(pr-gr) <= 1 and abs(pc-gc) <= 1 for (pr, pc) in pk): inst_hit += 1
                    if k >= 2:  # the spatial case: did we recover BOTH instances?
                        covn += 1
                        if all(any(abs(pr-gr) <= 1 and abs(pc-gc) <= 1 for (pr, pc) in pk) for (gr, gc) in pts): cov += 1
            return inst_hit / max(inst_tot, 1), (cov / covn if covn else 0.0)
    best = 0.0
    Ti = Tt
    for ep in range(args.epochs):
        net.train(); perm = torch.randperm(len(ti))
        for s in range(0, len(ti), args.bs):
            b = ti[perm[s:s+args.bs].numpy()]
            lo = net(Ft[b].to(dev), Dt[b].to(dev), Pt[b].to(dev))
            loss = F.binary_cross_entropy_with_logits(lo, Ti[b].to(dev))
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        if ep % 5 == 0 or ep == args.epochs - 1:
            ihit, both = evly(); tag = ""
            if ihit > best:
                best = ihit
                torch.save({"state": net.state_dict(), "protos": protos, "g": g, "C": C, "res": 448}, args.out)
                tag = " *saved"
            print(f"ep{ep:3d} loss={loss.item():.4f} inst_recall@1patch={ihit:.3f} both-bowls={both:.3f}{tag}", flush=True)
    print(f"\n=== BEST inst_recall = {best:.3f} -> {args.out} ===", flush=True)
    print("INSTHEAD_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
