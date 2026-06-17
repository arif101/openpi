"""STAGE 2 (train) of the factored wrist-cam motor. Distill pi0.5's motor competence into a small
IDENTITY-AGNOSTIC, POSITION-AGNOSTIC head:
    input  = wrist_img(128x128, eye-in-hand) + obj_rel(3) + cont_rel(3) + proprio(5, ee-orient+gripper)
    output = 10x7 delta-EE action chunk INCLUDING gripper (behavior-clone pi0.5's chunk)
NO base image, NO absolute position, NO object identity, NO held/phase flag -> cannot memorize which/where, and learns
grasp->transport->release + gripper timing from the wrist view itself (it sees the gripper holding the object). The
policy is given BOTH goals every step and learns the phase transition; there is NO hand-coded state machine.

Split by ROLLOUT (no sample leakage). Torch only (run AFTER the jax collector exits -> no GPU contention).
Run: .venv/bin/python motor_distill/train_wristcam_motor.py --data data/motor_demos --out data/motor_head.pt --epochs 60
"""
from __future__ import annotations
import argparse, glob, pathlib, re
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F


# ---- CONTAINER-CLASS CONDITIONING (general: derived from suite-regime + the goal instruction's container noun) ----
# Lets ONE motor host MULTIPLE place geometries (basket/plate/stove/cabinet/...) by SELECTING a mode from a learned
# embedding, instead of MODE-AVERAGING them (which collapsed object 0.70->0.30 when goal-fixture data was added).
CC_VOCAB = ["other", "basket", "plate", "bowl", "stove", "cabinet", "drawer", "rack", "caddy", "microwave"]
CC_IDX = {n: i for i, n in enumerate(CC_VOCAB)}


def cont_class_from_name(name: str) -> int:
    """Map a container noun/body-name (e.g. 'plate_1','akita_stove','wine_rack') -> class index. General token match."""
    s = (name or "").lower()
    for noun in ("microwave", "cabinet", "drawer", "stove", "rack", "caddy", "plate", "basket", "bowl"):
        if noun in s:
            return CC_IDX[noun]
    return 0


def cont_class_for_file(path) -> int:
    """Per-rollout container class. Object/coadapt/dag -> basket; spatial -> plate (place target is the plate, NOT the
    bowl's location named in the stem); goal -> parse the container noun from the instruction stem (put X on/in the Y)."""
    p = pathlib.Path(path); s = p.stem.lower(); d = p.parent.name.lower()
    if "goal" in d:
        return cont_class_from_name(s)            # goal stems name the place target
    if "spatial" in d:
        return CC_IDX["plate"]                    # spatial always places on the plate
    return CC_IDX["basket"]                       # object / coadapt / dag_place / bowls / dual


class WristMotor(nn.Module):
    """SINGLE-GOAL visual-servo motor + LEARNED PHASE SELECTOR (the proven 0.50 architecture, with the held lift-heuristic
    replaced by a learned classifier). vec = [obj_rel(3), cont_rel(3), proprio(pdim)]. A phase classifier predicts alpha
    ("am I in the place phase?") from wrist+proprio. The motor sees ONE HARD-SELECTED active goal (object while reaching,
    container while placing) -- NOT a blend -- so it stays a clean single-goal servo. At TRAIN the selection uses the
    teacher's gripper-phase label (motor always gets the correct goal); at DEPLOY the predicted alpha hard-selects. No
    hand-coded transition rule -- the switch is learned."""
    def __init__(self, prop_dim=11, chunk=10, act=7, n_cls=len(CC_VOCAB), cc_dim=16, conditioned=True):
        super().__init__()
        self.chunk, self.act, self.pdim, self.n_cls, self.cc_dim = chunk, act, prop_dim - 6, n_cls, cc_dim
        self.conditioned = conditioned
        c = lambda i, o, s: nn.Sequential(nn.Conv2d(i, o, 3, s, 1), nn.GroupNorm(min(8, o), o), nn.ReLU())
        self.cnn = nn.Sequential(c(3, 32, 2), c(32, 64, 2), c(64, 128, 2), c(128, 128, 2),
                                 nn.AdaptiveAvgPool2d(1), nn.Flatten())          # -> 128
        self.prop = nn.Sequential(nn.Linear(self.pdim, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU())
        self.goal = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 64))        # single active goal
        extra = 0
        if conditioned:
            self.cc_emb = nn.Embedding(n_cls, cc_dim)   # CONTAINER-CLASS conditioning -> selects place geometry (basket/plate/stove/...)
            extra = cc_dim
        self.phase = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1))    # from cnn+proprio (state), NOT goals
        self.head = nn.Sequential(nn.Linear(256 + 64 + 1 + extra, 512), nn.ReLU(), nn.Linear(512, 512), nn.ReLU(),
                                  nn.Linear(512, chunk * act))   # +1 phase bit (+cc_dim container-class embed if conditioned)

    def forward(self, img, vec, cc=None, phase_label=None, phase_override=None, return_phase=False):
        cnnf = self.cnn(img); pf = self.prop(vec[:, 6:])
        obj = vec[:, 0:3]; cont = vec[:, 3:6]
        a_logit = self.phase(torch.cat([cnnf, pf], -1)).squeeze(-1)
        if phase_label is not None: sel = phase_label                    # teacher-selected @train
        elif phase_override is not None: sel = phase_override            # latched phase @deploy
        else: sel = (torch.sigmoid(a_logit) > 0.5).float()              # raw predicted
        active = sel.unsqueeze(-1) * cont + (1 - sel.unsqueeze(-1)) * obj   # HARD single-goal selection (no blend)
        feats = [cnnf, pf, self.goal(active), sel.unsqueeze(-1)]
        if self.conditioned:
            if cc is None: cc = torch.zeros(img.shape[0], dtype=torch.long, device=img.device)   # -> 'other'
            feats.append(self.cc_emb(cc))
        chunk = self.head(torch.cat(feats, -1)).view(-1, self.chunk, self.act)
        if return_phase: return chunk, a_logit
        return chunk


def load(data_dir):
    files = []
    for d in str(data_dir).split(","):
        files += sorted(glob.glob(str(pathlib.Path(d) / "*.npz")))
    rolls = []
    for f in files:
        d = np.load(f)
        r = {k: d[k] for k in ("wrist", "obj_rel", "cont_rel", "proprio", "chunk")}
        # PER-ROLLOUT phase label = SUSTAINED gripper-close (closed for >=K consecutive steps), approximating the old
        # held's grip+lift confirmation -> switches LATER than raw gripper-close => robust to marginal/failed grasps.
        sep = r["proprio"][:, 3] - r["proprio"][:, 4]
        r["phase"] = (sep < 0.05).astype(np.float32)   # plain gripper-closed (place-frac ~0.42; best so far)
        r["cc"] = np.full(len(r["chunk"]), cont_class_for_file(f), np.int64)   # container-class conditioning label
        rolls.append(r)
    return rolls


def stack(rolls):
    img = np.concatenate([r["wrist"] for r in rolls]).astype(np.float32) / 255.0       # [M,128,128,3]
    img = np.transpose(img, (0, 3, 1, 2))
    vec = np.concatenate([np.concatenate([r["obj_rel"], r["cont_rel"], r["proprio"]], 1) for r in rolls]).astype(np.float32)
    y = np.concatenate([r["chunk"] for r in rolls]).astype(np.float32)                  # [M,10,7]
    ph = np.concatenate([r["phase"] for r in rolls]).astype(np.float32)                 # [M]
    cc = np.concatenate([r["cc"] for r in rolls]).astype(np.int64)                      # [M] container class
    return img, vec, y, ph, cc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/motor_demos"); p.add_argument("--out", default="data/motor_head.pt")
    p.add_argument("--epochs", type=int, default=60); p.add_argument("--bs", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4); p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--grip-w", type=float, default=2.0); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--goal-noise", type=float, default=0.0)   # meters of ISOTROPIC goal perturbation (breaks reach if large)
    p.add_argument("--gnz", type=float, default=0.0)          # meters of Z-ONLY goal noise -> teach wrist z-correction w/o breaking xy reach
    p.add_argument("--gn-close", type=float, default=0.0)     # meters of CLOSENESS-SCALED 3D goal noise -> wrist xy+z servo correction at grasp (goal-CORRECTING)
    p.add_argument("--phase-w", type=float, default=1.0)      # weight on the learned-phase BCE (alpha vs teacher gripper-closed)
    p.add_argument("--aug", type=int, default=0)              # multi-factor WRIST augmentation (validated LIBERO-PRO robustness lever)
    p.add_argument("--aug-str", type=float, default=1.0)      # aug strength (reduce if eye-in-hand precision degrades)
    args = p.parse_args()
    GRIP_SEP = 0.05   # finger-separation: raw vec col9-col10 < this => gripper CLOSED => place phase (train label only)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed)
    rolls = load(args.data)
    assert rolls, f"no npz in {args.data}"
    idx = rng.permutation(len(rolls)); nval = max(1, int(len(rolls) * args.val_frac))
    vr = [rolls[i] for i in idx[:nval]]; tr = [rolls[i] for i in idx[nval:]]
    Xi, Xv, Y, Ptr, Ctr = stack(tr); Vi, Vv, VY, Pval, Cval = stack(vr)   # phase=sustained-close, cc=container-class
    print(f"{len(rolls)} rolls -> train {len(tr)} ({len(Y)} samp) / val {len(vr)} ({len(VY)} samp)", flush=True)
    print(f"phase prior: train place-frac={Ptr.mean():.2f} val={Pval.mean():.2f}", flush=True)
    print("container-class hist (train): " + ", ".join(f"{CC_VOCAB[i]}={int((Ctr==i).sum())}" for i in range(len(CC_VOCAB)) if (Ctr==i).any()), flush=True)
    # standardize the vec inputs (img is /255; targets stay raw action units)
    vm, vs = Xv.mean(0), Xv.std(0) + 1e-6
    gn = (args.goal_noise / vs[:3]).astype(np.float32)   # raw meters -> standardized units, per goal dim
    gnz = float(args.gnz / vs[2]) if args.gnz > 0 else 0.0   # z-only noise in standardized units (vec idx 2 = goal_rel z)
    import torch as _t; vmt3 = _t.tensor(vm[:3]); vst3 = _t.tensor(vs[:3])   # for closeness-scaled goal noise
    Xv = (Xv - vm) / vs; Vv = (Vv - vm) / vs
    tX = [torch.tensor(a) for a in (Xi, Xv, Y, Ptr, Ctr)]; vX = [torch.tensor(a) for a in (Vi, Vv, VY, Pval, Cval)]
    net = WristMotor(prop_dim=Xv.shape[1]).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    n = len(Y); w = torch.ones(7, device=dev); w[6] = args.grip_w

    def run_eval():
        net.eval()
        with torch.no_grad():
            pr = []
            for s in range(0, len(VY), 512):
                pr.append(net(vX[0][s:s+512].to(dev), vX[1][s:s+512].to(dev), cc=vX[4][s:s+512].to(dev)).cpu())
            P = torch.cat(pr); err = (P - vX[2]).abs()
            pos = err[..., :3].mean().item(); ori = err[..., 3:6].mean().item(); grp = err[..., 6].mean().item()
        return pos, ori, grp
    best = 1e9
    for ep in range(args.epochs):
        net.train(); perm = torch.randperm(n)
        tot = 0.0
        for s in range(0, n, args.bs):
            b = perm[s:s+args.bs]
            img = tX[0][b].to(dev); vec = tX[1][b].to(dev); y = tX[2][b].to(dev); ph = tX[3][b].to(dev); ccb = tX[4][b].to(dev)
            if args.aug:   # multi-factor wrist aug: brightness/contrast/color + small translation (preserve eye-in-hand signal)
                Bn = img.shape[0]; sgn = args.aug_str
                bri = 1.0 + (torch.rand(Bn, 1, 1, 1, device=dev) - 0.5) * 0.4 * sgn
                con = 1.0 + (torch.rand(Bn, 1, 1, 1, device=dev) - 0.5) * 0.4 * sgn
                col = 1.0 + (torch.rand(Bn, 3, 1, 1, device=dev) - 0.5) * 0.2 * sgn
                img = (((img - 0.5) * con + 0.5) * bri * col).clamp(0, 1)
                pad = int(round(6 * sgn))
                if pad > 0:
                    img = F.pad(img, (pad, pad, pad, pad), mode="replicate")
                    ox = int(torch.randint(0, 2 * pad + 1, (1,)).item()); oy = int(torch.randint(0, 2 * pad + 1, (1,)).item())
                    img = img[:, :, oy:oy + 128, ox:ox + 128]
            if args.goal_noise > 0:
                vec = vec.clone(); vec[:, :3] += torch.randn_like(vec[:, :3]) * torch.tensor(gn, device=dev)
            if gnz > 0:
                vec = vec.clone(); vec[:, 2] += torch.randn_like(vec[:, 2]) * gnz   # z-only: keep xy reach clean
            if args.gn_close > 0:
                # closeness-scaled FULL-3D goal noise: full noise when AT the object (use wrist to servo),
                # zero when far (>10cm) so the coarse reach stays clean. Teaches goal-CORRECTION at the grasp.
                vm3 = vmt3.to(dev); vs3 = vst3.to(dev)
                raw_xy = (vec[:, :3] * vs3 + vm3)[:, :2]; dist = raw_xy.norm(dim=1)
                close = (1.0 - (dist / 0.10).clamp(0, 1))                            # 1 at object -> 0 at >=10cm
                noise_m = torch.randn(len(b), 3, device=dev) * args.gn_close * close[:, None]
                vec = vec.clone(); vec[:, :3] += noise_m / vs3
            pred, a_logit = net(img, vec, cc=ccb, phase_label=ph, return_phase=True)   # teacher-selected active goal at train
            loss = (F.smooth_l1_loss(pred, y, reduction="none") * w).mean() \
                   + args.phase_w * F.binary_cross_entropy_with_logits(a_logit, ph)
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item() * len(b)
        sched.step()
        if ep % 5 == 0 or ep == args.epochs - 1:
            pos, ori, grp = run_eval(); score = pos + 0.5 * ori
            tag = ""
            if score < best:
                best = score
                torch.save({"state": net.state_dict(), "vm": vm, "vs": vs, "prop_dim": int(Xv.shape[1]), "n_cls": int(net.n_cls)}, args.out)
                tag = " *saved"
            print(f"ep{ep:3d} train_l1={tot/n:.4f}  val_pos={pos:.4f} ori={ori:.4f} grip={grp:.4f}{tag}", flush=True)
    print(f"\n=== BEST val pos-L1 (score)= {best:.4f}  saved -> {args.out} ===", flush=True)
    print("TRAIN_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
