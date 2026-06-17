"""Decisive motor-regression test: train the PLAIN proven architecture (WristMotorV0: cnn + prop([active_goal(3),
proprio(5), phase(1)]) + head) on OUR dual-goal data, with the active goal selected by the teacher gripper-phase. Eval
with eval_old_v0.py (privileged held + body goals + hide, NON-canon) -> if this hits ~0.60 like the old motor, my
fancy architecture (goal-embedding + phase-head + canon) was the regression, and the plain motor + a learned phase
classifier is the way back. Uses RAW (non-canon) body data to match the non-canon old motor.

Run: .venv/bin/python motor_distill/train_v0.py --data data/body_std,data/body_swap --out data/motor_head_v0my.pt
"""
from __future__ import annotations
import argparse, glob, pathlib
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from eval_old_v0 import WristMotorV0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/body_std,data/body_swap"); p.add_argument("--out", default="data/motor_head_v0my.pt")
    p.add_argument("--epochs", type=int, default=60); p.add_argument("--bs", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4); p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--grip-w", type=float, default=2.0); p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed)
    files = []
    for d in args.data.split(","): files += sorted(glob.glob(str(pathlib.Path(d) / "*.npz")))
    rolls = []
    for f in files:
        d = np.load(f)
        prop = d["proprio"]; sep = prop[:, 3] - prop[:, 4]; phase = (sep < 0.05).astype(np.float32)   # teacher gripper-phase
        active = np.where(phase[:, None] > 0.5, d["cont_rel"], d["obj_rel"]).astype(np.float32)        # active goal = cont if place else obj
        vec = np.concatenate([active, prop, phase[:, None]], 1).astype(np.float32)                     # [T,9] = old V0 layout
        rolls.append({"wrist": d["wrist"], "vec": vec, "chunk": d["chunk"]})
    assert rolls
    idx = rng.permutation(len(rolls)); nval = max(1, int(len(rolls) * args.val_frac))
    vr = [rolls[i] for i in idx[:nval]]; tr = [rolls[i] for i in idx[nval:]]

    def stack(rs):
        img = np.transpose(np.concatenate([r["wrist"] for r in rs]).astype(np.float32) / 255.0, (0, 3, 1, 2))
        vec = np.concatenate([r["vec"] for r in rs]).astype(np.float32)
        y = np.concatenate([r["chunk"] for r in rs]).astype(np.float32)
        return img, vec, y
    Xi, Xv, Y = stack(tr); Vi, Vv, VY = stack(vr)
    print(f"{len(rolls)} rolls -> train {len(Y)} / val {len(VY)} samp; place-frac={Xv[:,8].mean():.2f}", flush=True)
    vm, vs = Xv.mean(0), Xv.std(0) + 1e-6
    Xv = (Xv - vm) / vs; Vv = (Vv - vm) / vs
    tX = [torch.tensor(a) for a in (Xi, Xv, Y)]; vX = [torch.tensor(a) for a in (Vi, Vv, VY)]
    net = WristMotorV0(prop_dim=9).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    n = len(Y); w = torch.ones(7, device=dev); w[6] = args.grip_w; best = 1e9
    for ep in range(args.epochs):
        net.train(); perm = torch.randperm(n); tot = 0.0
        for s in range(0, n, args.bs):
            b = perm[s:s+args.bs]
            img = tX[0][b].to(dev); vec = tX[1][b].to(dev); y = tX[2][b].to(dev)
            loss = (F.smooth_l1_loss(net(img, vec), y, reduction="none") * w).mean()
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item() * len(b)
        sched.step()
        if ep % 5 == 0 or ep == args.epochs - 1:
            net.eval()
            with torch.no_grad():
                pr = torch.cat([net(vX[0][s:s+512].to(dev), vX[1][s:s+512].to(dev)).cpu() for s in range(0, len(VY), 512)])
                pos = (pr - vX[2])[..., :3].abs().mean().item()
            tag = ""
            if pos < best:
                best = pos; torch.save({"state": net.state_dict(), "vm": vm, "vs": vs, "prop_dim": 9}, args.out); tag = " *saved"
            print(f"ep{ep:3d} train={tot/n:.4f} val_pos={pos:.4f}{tag}", flush=True)
    print(f"\n=== V0-on-our-data BEST val pos-L1={best:.4f} -> {args.out} ===", flush=True)
    print("TRAINV0_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
