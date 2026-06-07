"""PHASE 2 v1: motor on a FROZEN DINOv2 BACKBONE (not a from-scratch CNN). The mask-motor failed (7% swap) because
its from-scratch agentview CNN could not GROUND -> it overfit positions. Fix: encode agentview + wrist with frozen
DINOv2 (validated: separates these objects 1.00, position-invariant web-pretrained features) and form a GROUNDED
target feature by mask-pooling the target object's DINOv2 patches. Head: [scene-CLS, target-pooled, wrist-CLS,
proprio] -> action chunk. Tests whether DINOv2 grounding gives the goal-CORRECTING/position-general property the
CNN lacked, BEFORE adding the flow-matching expert + FLARE. Re-distills pi0.5's competence (same maskmotor data).

Run: .venv/bin/python motor_distill/train_dinomotor.py --data data/maskmotor_demos --out data/dinomotor.pt --epochs 60
"""
from __future__ import annotations
import argparse, glob, pathlib
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

_D = {}


def dino():
    if "m" not in _D:
        from transformers import AutoModel, AutoImageProcessor
        _D["m"] = AutoModel.from_pretrained("facebook/dinov2-base").cuda().eval()
        _D["p"] = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    return _D["m"], _D["p"]


@torch.no_grad()
def encode(imgs, masks, bs=64):
    """imgs [N,128,128,3] uint8, masks [N,128,128] uint8 (or None). -> (cls[N,768], tgt[N,768])."""
    from PIL import Image
    m, p = dino(); cls_all, tgt_all = [], []
    for s in range(0, len(imgs), bs):
        ims = [Image.fromarray(imgs[i]) for i in range(s, min(s + bs, len(imgs)))]
        inp = p(images=ims, return_tensors="pt").to("cuda")
        out = m(**inp).last_hidden_state                       # [b, 1+256, 768]
        cls = out[:, 0]; patch = out[:, 1:].reshape(out.shape[0], 16, 16, 768)
        if masks is not None:
            mk = torch.tensor(masks[s:s+len(ims)].astype(np.float32) / 255.0, device="cuda")[:, None]
            m16 = F.interpolate(mk, size=(16, 16), mode="area")[:, 0] > 0.3          # [b,16,16]
            wsum = m16.sum((1, 2), keepdim=True).clamp_min(1.0)
            tgt = (patch * m16[..., None]).sum((1, 2)) / wsum[..., 0]
        else:
            tgt = patch.mean((1, 2))
        cls_all.append(cls.cpu()); tgt_all.append(tgt.cpu())
    return torch.cat(cls_all).numpy(), torch.cat(tgt_all).numpy()


class DinoMotor(nn.Module):
    """[scene-CLS(768), target-pooled(768), wrist-CLS(768), proprio] -> 10x7 chunk. Backbone is FROZEN DINOv2."""
    def __init__(self, prop_dim=6, chunk=10, act=7):
        super().__init__()
        self.chunk, self.act = chunk, act
        self.proj = nn.Sequential(nn.Linear(768 * 3, 512), nn.ReLU(), nn.Linear(512, 512), nn.ReLU())
        self.pr = nn.Sequential(nn.Linear(prop_dim, 128), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(640, 512), nn.ReLU(), nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, chunk * act))

    def forward(self, sc, tg, wr, pr):
        f = torch.cat([self.proj(torch.cat([sc, tg, wr], -1)), self.pr(pr)], -1)
        return self.head(f).view(-1, self.chunk, self.act)


def feats_for(rolls, cache):
    """Precompute & cache DINOv2 features per rollout file."""
    out = []
    for r, path in rolls:
        cf = pathlib.Path(cache) / (pathlib.Path(path).stem + ".dino.npz")
        if cf.exists():
            d = np.load(cf); out.append({k: d[k] for k in d.files}); continue
        sc, tg = encode(r["agview"], r["tmask"]); wc, _ = encode(r["wrist"], None)
        pr = np.concatenate([r["proprio"], r["held"][:, None]], 1).astype(np.float32)
        rec = {"sc": sc, "tg": tg, "wc": wc, "pr": pr, "chunk": r["chunk"].astype(np.float32)}
        np.savez_compressed(cf, **rec); out.append(rec)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/maskmotor_demos"); p.add_argument("--out", default="data/dinomotor.pt")
    p.add_argument("--cache", default="data/dino_cache"); p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--bs", type=int, default=256); p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--val-frac", type=float, default=0.15); p.add_argument("--grip-w", type=float, default=2.0); p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    pathlib.Path(args.cache).mkdir(parents=True, exist_ok=True)
    files = sorted(glob.glob(str(pathlib.Path(args.data) / "*.npz")))
    rolls = []
    for f in files:
        d = np.load(f)
        if "agview" in d and "tmask" in d: rolls.append(({k: d[k] for k in ("agview", "tmask", "wrist", "proprio", "held", "chunk")}, f))
    print(f"{len(rolls)} rollouts; precomputing DINOv2 features...", flush=True)
    recs = feats_for(rolls, args.cache)
    rng = np.random.default_rng(args.seed); idx = rng.permutation(len(recs)); nval = max(1, int(len(recs) * args.val_frac))
    def cat(rs, k): return np.concatenate([r[k] for r in rs])
    tr = [recs[i] for i in idx[nval:]]; vr = [recs[i] for i in idx[:nval]]
    Tsc, Ttg, Twc, Tpr, Ty = (cat(tr, "sc"), cat(tr, "tg"), cat(tr, "wc"), cat(tr, "pr"), cat(tr, "chunk"))
    Vsc, Vtg, Vwc, Vpr, Vy = (cat(vr, "sc"), cat(vr, "tg"), cat(vr, "wc"), cat(vr, "pr"), cat(vr, "chunk"))
    pm, ps = Tpr.mean(0), Tpr.std(0) + 1e-6; Tpr = (Tpr - pm) / ps; Vpr = (Vpr - pm) / ps
    dev = "cuda"; T = [torch.tensor(a) for a in (Tsc, Ttg, Twc, Tpr, Ty)]; V = [torch.tensor(a) for a in (Vsc, Vtg, Vwc, Vpr, Vy)]
    print(f"train {len(Ty)} / val {len(Vy)} samp", flush=True)
    net = DinoMotor(prop_dim=Tpr.shape[1]).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4); sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    n = len(Ty); w = torch.ones(7, device=dev); w[6] = args.grip_w; best = 1e9
    for ep in range(args.epochs):
        net.train(); perm = torch.randperm(n); tot = 0.0
        for s in range(0, n, args.bs):
            b = perm[s:s+args.bs]
            pred = net(T[0][b].to(dev), T[1][b].to(dev), T[2][b].to(dev), T[3][b].to(dev))
            loss = (F.smooth_l1_loss(pred, T[4][b].to(dev), reduction="none") * w).mean()
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item() * len(b)
        sch.step()
        if ep % 5 == 0 or ep == args.epochs - 1:
            net.eval()
            with torch.no_grad():
                P = torch.cat([net(V[0][s:s+512].to(dev), V[1][s:s+512].to(dev), V[2][s:s+512].to(dev), V[3][s:s+512].to(dev)).cpu() for s in range(0, len(Vy), 512)])
                pos = (P[..., :3] - V[4][..., :3]).abs().mean().item()
            tag = ""
            if pos < best: best = pos; torch.save({"state": net.state_dict(), "pm": pm, "ps": ps, "prop_dim": int(Tpr.shape[1])}, args.out); tag = " *saved"
            print(f"ep{ep:3d} train_l1={tot/n:.4f} val_pos={pos:.4f}{tag}", flush=True)
    print(f"\n=== DINO-MOTOR best val pos-L1 = {best:.4f} -> {args.out} ===", flush=True)
    print("DINOTRAIN_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
