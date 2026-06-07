"""GOAL-CORRECTING motor (PI + broader-research recipe): condition on AGENTVIEW + target MASK + WRIST, drop the
brittle 3D point. The mask says WHICH object (general, from the binder); the policy grounds where it is from the
scene and servos with the wrist -> goal-CORRECTING by construction (no point to be wrong about). Trained on DART
recovery data (perturbed states + pi0.5 corrections). pi0.5 only the training supervisor; deployed motor is ours.

Run: .venv/bin/python motor_distill/train_maskmotor.py --data data/maskmotor_demos --out data/maskmotor.pt --epochs 60
"""
from __future__ import annotations
import argparse, glob, pathlib
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F


class MaskMotor(nn.Module):
    """agentview(RGB+mask, 4ch) + wrist(3ch) + proprio -> 10x7 action chunk. NO goal point."""
    def __init__(self, prop_dim=6, chunk=10, act=7):
        super().__init__()
        self.chunk, self.act = chunk, act
        c = lambda i, o, s: nn.Sequential(nn.Conv2d(i, o, 3, s, 1), nn.GroupNorm(min(8, o), o), nn.ReLU())
        self.ag = nn.Sequential(c(4, 32, 2), c(32, 64, 2), c(64, 128, 2), c(128, 128, 2), nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.wr = nn.Sequential(c(3, 32, 2), c(32, 64, 2), c(64, 128, 2), c(128, 128, 2), nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.pr = nn.Sequential(nn.Linear(prop_dim, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(384, 512), nn.ReLU(), nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, chunk * act))

    def forward(self, ag, wr, pr):
        f = torch.cat([self.ag(ag), self.wr(wr), self.pr(pr)], -1)
        return self.head(f).view(-1, self.chunk, self.act)


def load(dirs):
    files = []
    for d in str(dirs).split(","): files += sorted(glob.glob(str(pathlib.Path(d) / "*.npz")))
    rolls = []
    for f in files:
        d = np.load(f)
        if "agview" not in d or "tmask" not in d: continue
        rolls.append({k: d[k] for k in ("agview", "tmask", "wrist", "proprio", "held", "chunk")})
    return rolls


def stack(rolls):
    ag = np.concatenate([r["agview"] for r in rolls]).astype(np.float32) / 255.0           # [M,128,128,3]
    mk = np.concatenate([r["tmask"] for r in rolls]).astype(np.float32) / 255.0             # [M,128,128]
    agm = np.concatenate([np.transpose(ag, (0, 3, 1, 2)), mk[:, None]], 1)                  # [M,4,128,128]
    wr = np.transpose(np.concatenate([r["wrist"] for r in rolls]).astype(np.float32) / 255.0, (0, 3, 1, 2))
    pr = np.concatenate([np.concatenate([r["proprio"], r["held"][:, None]], 1) for r in rolls]).astype(np.float32)
    y = np.concatenate([r["chunk"] for r in rolls]).astype(np.float32)
    return agm, wr, pr, y


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/maskmotor_demos"); p.add_argument("--out", default="data/maskmotor.pt")
    p.add_argument("--epochs", type=int, default=60); p.add_argument("--bs", type=int, default=96)
    p.add_argument("--lr", type=float, default=3e-4); p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--grip-w", type=float, default=2.0); p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"; torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed)
    rolls = load(args.data); assert rolls, f"no mask npz in {args.data}"
    idx = rng.permutation(len(rolls)); nval = max(1, int(len(rolls) * args.val_frac))
    vr = [rolls[i] for i in idx[:nval]]; tr = [rolls[i] for i in idx[nval:]]
    Tag, Twr, Tpr, Ty = stack(tr); Vag, Vwr, Vpr, Vy = stack(vr)
    pm, ps = Tpr.mean(0), Tpr.std(0) + 1e-6; Tpr = (Tpr - pm) / ps; Vpr = (Vpr - pm) / ps
    print(f"{len(rolls)} rolls -> train {len(Ty)} / val {len(Vy)} samp", flush=True)
    tX = [torch.tensor(a) for a in (Tag, Twr, Tpr, Ty)]; vX = [torch.tensor(a) for a in (Vag, Vwr, Vpr, Vy)]
    net = MaskMotor(prop_dim=Tpr.shape[1]).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    n = len(Ty); w = torch.ones(7, device=dev); w[6] = args.grip_w; best = 1e9
    for ep in range(args.epochs):
        net.train(); perm = torch.randperm(n); tot = 0.0
        for s in range(0, n, args.bs):
            b = perm[s:s+args.bs]
            ag = tX[0][b].to(dev); wr = tX[1][b].to(dev); pr = tX[2][b].to(dev); y = tX[3][b].to(dev)
            loss = (F.smooth_l1_loss(net(ag, wr, pr), y, reduction="none") * w).mean()
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item() * len(b)
        sch.step()
        if ep % 5 == 0 or ep == args.epochs - 1:
            net.eval()
            with torch.no_grad():
                pr_ = [net(vX[0][s:s+256].to(dev), vX[1][s:s+256].to(dev), vX[2][s:s+256].to(dev)).cpu() for s in range(0, len(Vy), 256)]
                P = torch.cat(pr_); pos = (P[..., :3] - vX[3][..., :3]).abs().mean().item()
            tag = ""
            if pos < best:
                best = pos; torch.save({"state": net.state_dict(), "pm": pm, "ps": ps, "prop_dim": int(Tpr.shape[1])}, args.out); tag = " *saved"
            print(f"ep{ep:3d} train_l1={tot/n:.4f} val_pos={pos:.4f}{tag}", flush=True)
    print(f"\n=== MASK-MOTOR best val pos-L1 = {best:.4f} -> {args.out} ===", flush=True)
    print("MASKTRAIN_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
