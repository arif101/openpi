"""STAGE 2 (train) of the factored wrist-cam motor. Distill pi0.5's motor competence into a small
IDENTITY-AGNOSTIC, POSITION-AGNOSTIC head:
    input  = wrist_img(128x128, eye-in-hand) + goal_rel(3, relative target) + proprio(5, ee-orient+gripper) + held(1)
    output = 10x7 delta-EE action chunk (behavior-clone pi0.5's chunk)
NO base image, NO absolute position, NO object identity -> cannot memorize which/where; precision must come from the
wrist view + the relative goal. This directly tests the wrist-cam-precision thesis (validated by the ablation).

Split by ROLLOUT (no sample leakage). Torch only (run AFTER the jax collector exits -> no GPU contention).
Run: .venv/bin/python motor_distill/train_wristcam_motor.py --data data/motor_demos --out data/motor_head.pt --epochs 60
"""
from __future__ import annotations
import argparse, glob, pathlib
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F


class WristMotor(nn.Module):
    def __init__(self, prop_dim=6, chunk=10, act=7):
        super().__init__()
        self.chunk, self.act = chunk, act
        c = lambda i, o, s: nn.Sequential(nn.Conv2d(i, o, 3, s, 1), nn.GroupNorm(min(8, o), o), nn.ReLU())
        self.cnn = nn.Sequential(c(3, 32, 2), c(32, 64, 2), c(64, 128, 2), c(128, 128, 2),
                                 nn.AdaptiveAvgPool2d(1), nn.Flatten())          # -> 128
        self.prop = nn.Sequential(nn.Linear(prop_dim, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(256, 512), nn.ReLU(), nn.Linear(512, 512), nn.ReLU(),
                                  nn.Linear(512, chunk * act))

    def forward(self, img, vec):
        f = torch.cat([self.cnn(img), self.prop(vec)], -1)
        return self.head(f).view(-1, self.chunk, self.act)


def load(data_dir):
    files = []
    for d in str(data_dir).split(","):
        files += sorted(glob.glob(str(pathlib.Path(d) / "*.npz")))
    rolls = []
    for f in files:
        d = np.load(f)
        rolls.append({k: d[k] for k in ("wrist", "goal_rel", "proprio", "held", "chunk")})
    return rolls


def stack(rolls):
    img = np.concatenate([r["wrist"] for r in rolls]).astype(np.float32) / 255.0       # [M,128,128,3]
    img = np.transpose(img, (0, 3, 1, 2))
    vec = np.concatenate([np.concatenate([r["goal_rel"], r["proprio"], r["held"][:, None]], 1) for r in rolls]).astype(np.float32)
    y = np.concatenate([r["chunk"] for r in rolls]).astype(np.float32)                  # [M,10,7]
    return img, vec, y


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/motor_demos"); p.add_argument("--out", default="data/motor_head.pt")
    p.add_argument("--epochs", type=int, default=60); p.add_argument("--bs", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4); p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--grip-w", type=float, default=2.0); p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed)
    rolls = load(args.data)
    assert rolls, f"no npz in {args.data}"
    idx = rng.permutation(len(rolls)); nval = max(1, int(len(rolls) * args.val_frac))
    vr = [rolls[i] for i in idx[:nval]]; tr = [rolls[i] for i in idx[nval:]]
    Xi, Xv, Y = stack(tr); Vi, Vv, VY = stack(vr)
    print(f"{len(rolls)} rolls -> train {len(tr)} ({len(Y)} samp) / val {len(vr)} ({len(VY)} samp)", flush=True)
    # standardize the vec inputs (img is /255; targets stay raw action units)
    vm, vs = Xv.mean(0), Xv.std(0) + 1e-6
    Xv = (Xv - vm) / vs; Vv = (Vv - vm) / vs
    tX = [torch.tensor(a) for a in (Xi, Xv, Y)]; vX = [torch.tensor(a) for a in (Vi, Vv, VY)]
    net = WristMotor(prop_dim=Xv.shape[1]).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    n = len(Y); w = torch.ones(7, device=dev); w[6] = args.grip_w

    def run_eval():
        net.eval()
        with torch.no_grad():
            pr = []
            for s in range(0, len(VY), 512):
                pr.append(net(vX[0][s:s+512].to(dev), vX[1][s:s+512].to(dev)).cpu())
            P = torch.cat(pr); err = (P - vX[2]).abs()
            pos = err[..., :3].mean().item(); ori = err[..., 3:6].mean().item(); grp = err[..., 6].mean().item()
        return pos, ori, grp
    best = 1e9
    for ep in range(args.epochs):
        net.train(); perm = torch.randperm(n)
        tot = 0.0
        for s in range(0, n, args.bs):
            b = perm[s:s+args.bs]
            img = tX[0][b].to(dev); vec = tX[1][b].to(dev); y = tX[2][b].to(dev)
            pred = net(img, vec)
            loss = (F.smooth_l1_loss(pred, y, reduction="none") * w).mean()
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item() * len(b)
        sched.step()
        if ep % 5 == 0 or ep == args.epochs - 1:
            pos, ori, grp = run_eval(); score = pos + 0.5 * ori
            tag = ""
            if score < best:
                best = score
                torch.save({"state": net.state_dict(), "vm": vm, "vs": vs, "prop_dim": int(Xv.shape[1])}, args.out)
                tag = " *saved"
            print(f"ep{ep:3d} train_l1={tot/n:.4f}  val_pos={pos:.4f} ori={ori:.4f} grip={grp:.4f}{tag}", flush=True)
    print(f"\n=== BEST val pos-L1 (score)= {best:.4f}  saved -> {args.out} ===", flush=True)
    print("TRAIN_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
