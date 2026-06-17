"""Train the LEARNED grounding head on LIBERO sim auto-labels (collect_ground_data.py output).
Frozen DINOv2 dense feats (g x g x C) conditioned on the target object's patch-space PROTO (open-vocab query)
-> softmax heatmap over the g x g patch grid; cross-entropy to the GT target patch. The proto makes it open-vocab
(swap the query for a different object); position diversity across inits = the generalization signal. Fair vs pi0.5
(itself LIBERO-fine-tuned). Deploy: head(dense_feats, proto) -> argmax patch -> pixel; NO body_pos.

Run: python motor_distill/train_ground_head.py --data data/ground_obj,data/ground_objobj --out data/ground_head.pt
"""
from __future__ import annotations
import argparse, glob, pathlib
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F


class GroundHead(nn.Module):
    """Proto-conditioned dense->heatmap. feats (B,g,g,C), proto (B,C) -> logits (B,g*g)."""
    def __init__(self, c=1024, d=256):
        super().__init__()
        self.fp = nn.Linear(c, d); self.qp = nn.Linear(c, d)
        self.conv = nn.Sequential(nn.Conv2d(d + 1, 128, 3, 1, 1), nn.GroupNorm(8, 128), nn.ReLU(),
                                  nn.Conv2d(128, 128, 3, 1, 1), nn.GroupNorm(8, 128), nn.ReLU(),
                                  nn.Conv2d(128, 1, 1))

    def forward(self, feats, proto):
        B, g, _, C = feats.shape
        f = self.fp(feats)                                   # B,g,g,d
        q = self.qp(proto)                                   # B,d
        cond = f * q[:, None, None, :]                       # FiLM-style gate by the object query
        sim = (f * q[:, None, None, :]).sum(-1, keepdim=True)  # B,g,g,1 correspondence channel
        x = torch.cat([cond, sim], -1).permute(0, 3, 1, 2)   # B,d+1,g,g
        return self.conv(x).flatten(1)                       # B,g*g


def load(dirs):
    F_, P_, Y_, G = [], [], [], None
    for d in str(dirs).split(","):
        for fn in sorted(glob.glob(str(pathlib.Path(d) / "*.npz"))):
            z = np.load(fn, allow_pickle=True); g = int(z["g"]); G = g
            ft = z["feats"].astype(np.float32); gt = z["gt"]; pr = z["proto"].astype(np.float32)
            for i in range(len(ft)):
                F_.append(ft[i]); P_.append(pr); Y_.append(int(gt[i][0]) * g + int(gt[i][1]))
    return np.stack(F_), np.stack(P_), np.array(Y_, np.int64), G


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True); p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=int, default=80); p.add_argument("--bs", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4); p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed)
    Fa, Pa, Ya, g = load(args.data)
    C = Fa.shape[-1]
    # also stash per-noun protos for eval (one proto per source npz -> keyed by noun)
    protos = {}
    for d in str(args.data).split(","):
        for fn in sorted(glob.glob(str(pathlib.Path(d) / "*.npz"))):
            z = np.load(fn, allow_pickle=True); protos[str(z["noun"])] = z["proto"].astype(np.float32)
    n = len(Ya); idx = rng.permutation(n); nval = max(1, int(n * args.val_frac))
    vi, ti = idx[:nval], idx[nval:]
    print(f"{n} samples (g={g}, C={C}); train {len(ti)} / val {len(vi)}; {len(protos)} nouns", flush=True)
    Ft = torch.tensor(Fa); Pt = torch.tensor(Pa); Yt = torch.tensor(Ya)
    net = GroundHead(C).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)

    def evly():
        net.eval()
        with torch.no_grad():
            corr = 0; d5 = 0
            for s in range(0, len(vi), 256):
                b = vi[s:s+256]
                lo = net(Ft[b].to(dev), Pt[b].to(dev)); pr = lo.argmax(1).cpu().numpy()
                gt = Ya[b]
                corr += int((pr == gt).sum())
                # within-1-patch (≈ the 18px tol at g=32/res256)
                pr_r, pr_c = pr // g, pr % g; gt_r, gt_c = gt // g, gt % g
                d5 += int((np.abs(pr_r - gt_r) <= 1) & (np.abs(pr_c - gt_c) <= 1)).sum()
            return corr / len(vi), d5 / len(vi)
    best = 0.0
    for ep in range(args.epochs):
        net.train(); perm = torch.randperm(len(ti))
        for s in range(0, len(ti), args.bs):
            b = ti[perm[s:s+args.bs].numpy()]
            lo = net(Ft[b].to(dev), Pt[b].to(dev))
            loss = F.cross_entropy(lo, Yt[b].to(dev))
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        if ep % 5 == 0 or ep == args.epochs - 1:
            acc, acc1 = evly(); tag = ""
            if acc1 > best:
                best = acc1
                torch.save({"state": net.state_dict(), "protos": protos, "g": g, "C": C, "res": 448}, args.out)
                tag = " *saved"
            print(f"ep{ep:3d} loss={loss.item():.3f} val_exact={acc:.3f} val_within1={acc1:.3f}{tag}", flush=True)
    print(f"\n=== BEST val within-1-patch = {best:.3f}  -> {args.out} ===", flush=True)
    print("GROUNDHEAD_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
