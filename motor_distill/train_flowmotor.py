"""PRECISION-PRESERVING DISTILLATION: replace the wrist motor's smooth_L1 REGRESSION head (which mode-AVERAGES
pi0.5's multimodal flow-matched actions -> blurred precision) with a FLOW-MATCHING (rectified-flow) action head that
distills the action DISTRIBUTION, not its mean. Same conditioning encoder (wrist CNN + goal_rel/proprio/held), same
pi0.5 data (data/motor_demos). Tests the hypothesis that part of our 'distillation caps' was the regression loss.
If standard AND swap rise vs the L1 head (87/50), mode-averaging was silently capping us (TinyVLA/SmolVLA-style).

Run: .venv/bin/python motor_distill/train_flowmotor.py --data data/motor_demos --out data/flowmotor.pt --epochs 120
"""
from __future__ import annotations
import argparse, pathlib
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from train_wristcam_motor import load, stack


class FlowMotor(nn.Module):
    """Conditioning = wrist CNN(128) + (goal_rel,proprio,held) MLP(128) -> 256. Rectified-flow velocity field
    v_theta(x_t, t, cond) over the flattened action chunk (chunk*act). Sampled by Euler integration noise->action."""
    def __init__(self, prop_dim=9, chunk=10, act=7, hid=512):
        super().__init__()
        self.chunk, self.act, self.dim = chunk, act, chunk * act
        c = lambda i, o, s: nn.Sequential(nn.Conv2d(i, o, 3, s, 1), nn.GroupNorm(min(8, o), o), nn.ReLU())
        self.cnn = nn.Sequential(c(3, 32, 2), c(32, 64, 2), c(64, 128, 2), c(128, 128, 2), nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.prop = nn.Sequential(nn.Linear(prop_dim, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU())
        self.tembed = nn.Sequential(nn.Linear(1, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU())
        self.vel = nn.Sequential(nn.Linear(self.dim + 64 + 256, hid), nn.SiLU(),
                                 nn.Linear(hid, hid), nn.SiLU(), nn.Linear(hid, hid), nn.SiLU(),
                                 nn.Linear(hid, self.dim))

    def cond(self, img, vec):
        return torch.cat([self.cnn(img), self.prop(vec)], -1)            # [B,256]

    def velocity(self, x, t, c):
        return self.vel(torch.cat([x, self.tembed(t.view(-1, 1)), c], -1))

    @torch.no_grad()
    def sample(self, img, vec, steps=10):
        c = self.cond(img, vec); B = c.shape[0]
        x = torch.randn(B, self.dim, device=c.device)
        for i in range(steps):
            t = torch.full((B,), i / steps, device=c.device)
            x = x + (1.0 / steps) * self.velocity(x, t, c)
        return x.view(B, self.chunk, self.act)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/motor_demos"); p.add_argument("--out", default="data/flowmotor.pt")
    p.add_argument("--epochs", type=int, default=120); p.add_argument("--bs", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4); p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--grip-w", type=float, default=2.0); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--infer-steps", type=int, default=10)
    args = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"; torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed)
    rolls = load(args.data); assert rolls, f"no npz in {args.data}"
    idx = rng.permutation(len(rolls)); nval = max(1, int(len(rolls) * args.val_frac))
    vr = [rolls[i] for i in idx[:nval]]; tr = [rolls[i] for i in idx[nval:]]
    Timg, Tvec, Ty = stack(tr); Vimg, Vvec, Vy = stack(vr)
    vm, vs = Tvec.mean(0), Tvec.std(0) + 1e-6; Tvec = (Tvec - vm) / vs; Vvec = (Vvec - vm) / vs
    Ty = Ty.reshape(len(Ty), -1); Vy = Vy.reshape(len(Vy), -1)                  # [M,70]
    cm, cs = Ty.mean(0), Ty.std(0) + 1e-6; Tyn = (Ty - cm) / cs; Vyn = (Vy - cm) / cs
    dim = Ty.shape[1]; w = np.ones(dim, np.float32); w[6::7] = args.grip_w       # weight gripper dims
    print(f"{len(rolls)} rolls -> train {len(Ty)} / val {len(Vy)} samp, dim={dim}", flush=True)
    T = [torch.tensor(a) for a in (Timg, Tvec, Tyn)]; V = [torch.tensor(a) for a in (Vimg, Vvec, Vyn)]
    wt = torch.tensor(w, device=dev); cmt = torch.tensor(cm, device=dev); cst = torch.tensor(cs, device=dev)
    net = FlowMotor(prop_dim=Tvec.shape[1]).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    n = len(Ty); best = 1e9
    for ep in range(args.epochs):
        net.train(); perm = torch.randperm(n); tot = 0.0
        for s in range(0, n, args.bs):
            b = perm[s:s + args.bs]
            img = T[0][b].to(dev); vec = T[1][b].to(dev); x1 = T[2][b].to(dev)
            x0 = torch.randn_like(x1); t = torch.rand(len(b), device=dev)
            xt = (1 - t)[:, None] * x0 + t[:, None] * x1; target = x1 - x0
            v = net.velocity(xt, t, net.cond(img, vec))
            loss = ((v - target) ** 2 * wt).mean()
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item() * len(b)
        sch.step()
        if ep % 5 == 0 or ep == args.epochs - 1:
            net.eval()
            with torch.no_grad():
                ps = []
                for s in range(0, len(Vy), 256):
                    ch = net.sample(V[0][s:s+256].to(dev), V[1][s:s+256].to(dev), steps=args.infer_steps)
                    ch = (ch.reshape(ch.shape[0], -1) * cst + cmt)            # un-standardize -> real action units
                    pos = (ch.reshape(-1, net.chunk, net.act)[..., :3] -
                           torch.tensor(Vy[s:s+256].reshape(-1, net.chunk, net.act)[..., :3], device=dev)).abs().mean().item()
                    ps.append(pos)
                pos = float(np.mean(ps))
            tag = ""
            if pos < best:
                best = pos; torch.save({"state": net.state_dict(), "vm": vm, "vs": vs, "cm": cm, "cs": cs,
                                        "prop_dim": int(Tvec.shape[1]), "infer_steps": args.infer_steps}, args.out); tag = " *saved"
            print(f"ep{ep:3d} flow_mse={tot/n:.4f} val_pos={pos:.4f}{tag}", flush=True)
    print(f"\n=== FLOW-MOTOR best val pos-L1 = {best:.4f} -> {args.out} ===", flush=True)
    print("FLOWTRAIN_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
