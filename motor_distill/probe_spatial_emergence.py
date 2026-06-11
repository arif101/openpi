"""
Decisive de-risking experiment for PILLAR 3 (action-grounded spatial intelligence,
RGB-only, NO depth, NO VLM) of the predictive-sensorimotor VLA.

QUESTION (open Q1 from wf_010ac610-dbe): does predicting future EGOCENTRIC wrist-cam
RGB produce latents that linearly decode object position AND stay relocation-invariant
(the property that would collapse into our prior learned-WM failure mode if absent)?

DESIGN
  Encoder E: conv -> spatial-softmax bottleneck (Finn&Levine 2015) -> "where" feature.
  Self-supervision (RGB-only, no depth/VLM):
    - recon: decode feat_t -> downsampled grayscale img_t   (anchors bottleneck, anti-collapse)
    - predict: MLP(feat_t, action_chunk_t) -> feat_{t+1}     (the "predict egocentric perception" property)
  Then FREEZE E, linear-probe feat -> obj_rel (object pos relative to EE).

ARMS (isolate what creates spatial grounding):
  random : untrained encoder (conv architecture only)            <- baseline
  recon  : recon loss only (plain spatial autoencoder)           <- does prediction add anything?
  predict: recon + predictive dynamics loss                      <- headline (our pillar)

GO/CUT for pillar 3:
  GO  if predict-probe error << random, AND swap_err ~= std_err (relocation-invariant), AND abs err small.
  CUT if predict ~= random (no emergent structure) OR swap_err >> std_err (memorizes, not invariant).
"""
import argparse, glob, numpy as np, torch, torch.nn as nn, torch.nn.functional as F

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def load_set(dirname, root="data"):
    """Return list of episodes: dict(wrist[T,128,128,3] uint8, obj_rel[T,3], chunk[T,10,7])."""
    eps = []
    for f in sorted(glob.glob(f"{root}/{dirname}/*.npz")):
        d = np.load(f, allow_pickle=True)
        if "wrist" not in d.files or "obj_rel" not in d.files:
            continue
        eps.append(dict(wrist=d["wrist"], obj_rel=d["obj_rel"].astype(np.float32),
                        chunk=d["chunk"].astype(np.float32)))
    return eps


def make_pairs(eps):
    """Frame-level samples: (img_t, action_t flat, img_{t+1}, obj_rel_t, frac_t).
    frac_t = within-episode time fraction (0=start, 1=end); the wrist only sees the
    object in the LATE approach phase, so we stratify probe error by frac."""
    I, A, In, Y, Fr = [], [], [], [], []
    for e in eps:
        T = e["wrist"].shape[0]
        for t in range(T - 1):
            I.append(e["wrist"][t]); In.append(e["wrist"][t + 1])
            A.append(e["chunk"][t].reshape(-1)); Y.append(e["obj_rel"][t])
            Fr.append(t / max(T - 2, 1))
    I = np.stack(I).astype(np.float32) / 255.0
    In = np.stack(In).astype(np.float32) / 255.0
    A = np.stack(A).astype(np.float32); Y = np.stack(Y).astype(np.float32)
    Fr = np.array(Fr, dtype=np.float32)
    I = np.transpose(I, (0, 3, 1, 2)); In = np.transpose(In, (0, 3, 1, 2))
    return (torch.tensor(I), torch.tensor(A), torch.tensor(In), torch.tensor(Y), torch.tensor(Fr))


class SpatialSoftmax(nn.Module):
    def __init__(self, h, w):
        super().__init__()
        ys, xs = torch.meshgrid(torch.linspace(-1, 1, h), torch.linspace(-1, 1, w), indexing="ij")
        self.register_buffer("xs", xs.reshape(-1)); self.register_buffer("ys", ys.reshape(-1))

    def forward(self, feat):  # B,C,H,W
        B, C, H, W = feat.shape
        a = torch.softmax(feat.reshape(B, C, -1), dim=-1)
        ex = (a * self.xs).sum(-1); ey = (a * self.ys).sum(-1)
        return torch.cat([ex, ey], dim=-1)  # B, 2C


class Encoder(nn.Module):
    def __init__(self, ch=32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 7, 2, 3), nn.ReLU(),     # 64
            nn.Conv2d(32, 64, 5, 2, 2), nn.ReLU(),    # 32
            nn.Conv2d(64, ch, 5, 1, 2), nn.ReLU(),    # 32
        )
        self.ss = SpatialSoftmax(32, 32)
        self.dim = 2 * ch

    def forward(self, x):
        return self.ss(self.conv(x))  # B, 2ch


class Recon(nn.Module):
    """Decode bottleneck -> 32x32 grayscale (Finn&Levine anchor)."""
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 512), nn.ReLU(), nn.Linear(512, 32 * 32))

    def forward(self, z):
        return self.net(z).reshape(-1, 1, 32, 32)


class Dyn(nn.Module):
    """Predict feat_{t+1} from (feat_t, action)."""
    def __init__(self, d, a):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d + a, 256), nn.ReLU(),
                                 nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, d))

    def forward(self, z, a):
        return self.net(torch.cat([z, a], -1))


def train_encoder(arm, I, A, In, adim, epochs, bs, seed):
    torch.manual_seed(seed)
    enc = Encoder().to(DEV)
    if arm == "random":
        enc.eval()
        return enc
    rec = Recon(enc.dim).to(DEV)
    dyn = Dyn(enc.dim, adim).to(DEV)
    params = list(enc.parameters()) + list(rec.parameters())
    if arm == "predict":
        params += list(dyn.parameters())
    opt = torch.optim.Adam(params, 1e-3)
    N = I.shape[0]
    tgt_gray = (0.299 * In[:, 0] + 0.587 * In[:, 1] + 0.114 * In[:, 2]).unsqueeze(1)
    tgt_gray = F.avg_pool2d(tgt_gray, 4)  # 32x32 next-frame gray
    for ep in range(epochs):
        perm = torch.randperm(N)
        tot = 0.0
        for i in range(0, N, bs):
            idx = perm[i:i + bs]
            xb = I[idx].to(DEV); xn = In[idx].to(DEV); ab = A[idx].to(DEV); gb = tgt_gray[idx].to(DEV)
            z = enc(xb)
            # recon predicts the NEXT frame's appearance from current feat+action via decode of predicted feat
            if arm == "predict":
                zpred = dyn(z, ab)
                with torch.no_grad():
                    znext = enc(xn)
                loss_pred = F.mse_loss(zpred, znext)
                loss_rec = F.mse_loss(rec(zpred), gb)        # decode predicted feat -> next gray
                loss = loss_rec + loss_pred
            else:  # recon-only autoencoder: decode current feat -> current gray
                gcur = F.avg_pool2d((0.299 * xb[:, 0] + 0.587 * xb[:, 1] + 0.114 * xb[:, 2]).unsqueeze(1), 4)
                loss = F.mse_loss(rec(z), gcur)
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item() * len(idx)
        if ep % 5 == 0 or ep == epochs - 1:
            print(f"   [{arm}] ep{ep} loss {tot / N:.5f}", flush=True)
    enc.eval()
    return enc


def feats(enc, I, bs=512):
    out = []
    with torch.no_grad():
        for i in range(0, I.shape[0], bs):
            out.append(enc(I[i:i + bs].to(DEV)).cpu())
    return torch.cat(out)


def probe(enc, tr, te_std, te_swap, epochs=300):
    """Train linear probe feat->obj_rel on std-train; eval on std-heldout + swap."""
    Ztr = feats(enc, tr[0]); Ytr = tr[3]
    mu, sd = Ztr.mean(0), Ztr.std(0) + 1e-6
    nrm = lambda z: (z - mu) / sd
    lin = nn.Linear(Ztr.shape[1], 3)
    opt = torch.optim.Adam(lin.parameters(), 1e-2, weight_decay=1e-4)
    Ztn = nrm(Ztr)
    for ep in range(epochs):
        opt.zero_grad(); loss = F.mse_loss(lin(Ztn), Ytr); loss.backward(); opt.step()

    def err(te):
        Z = nrm(feats(enc, te[0]))
        with torch.no_grad():
            pred = lin(Z)
        e = (pred - te[3]).norm(dim=1)          # per-frame euclidean error (m)
        late = te[4] >= 0.6                      # last 40% = object in wrist view
        em = lambda m: (100.0 * e[m].mean().item(), 100.0 * e[m].median().item()) if m.sum() > 0 else (float("nan"),) * 2
        return dict(all=em(torch.ones_like(e, dtype=torch.bool)), late=em(late))

    return dict(std=err(te_std), swap=err(te_swap))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--std", default="coadapt_std")
    ap.add_argument("--swap", default="coadapt_swap")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--probe-epochs", type=int, default=400)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print(f"DEV={DEV}  std={args.std} swap={args.swap}", flush=True)
    std_eps = load_set(args.std); swap_eps = load_set(args.swap)
    rng = np.random.default_rng(args.seed); rng.shuffle(std_eps)
    n_tr = int(0.8 * len(std_eps))
    tr_eps, hd_eps = std_eps[:n_tr], std_eps[n_tr:]
    print(f"episodes: std_train={len(tr_eps)} std_heldout={len(hd_eps)} swap={len(swap_eps)}", flush=True)

    tr = make_pairs(tr_eps); te_std = make_pairs(hd_eps); te_swap = make_pairs(swap_eps)
    adim = tr[1].shape[1]
    print(f"frames: train={tr[0].shape[0]} std_heldout={te_std[0].shape[0]} swap={te_swap[0].shape[0]} adim={adim}", flush=True)
    base = (te_swap[3] - tr[3].mean(0)).norm(dim=1)
    print(f"NAIVE mean-predictor swap err: mean {100*base.mean():.2f}cm median {100*base.median():.2f}cm", flush=True)

    res = {}
    for arm in ["random", "recon", "predict"]:
        print(f"\n=== ARM: {arm} ===", flush=True)
        enc = train_encoder(arm, tr[0], tr[1], tr[2], adim, args.epochs, args.bs, args.seed)
        res[arm] = probe(enc, tr, te_std, te_swap, args.probe_epochs)
        r = res[arm]
        print(f"   obj_rel err ALL  std={r['std']['all'][0]:.2f} swap={r['swap']['all'][0]:.2f}cm | "
              f"LATE(obj in view) std={r['std']['late'][0]:.2f} swap={r['swap']['late'][0]:.2f}cm", flush=True)

    for phase in ["all", "late"]:
        print(f"\n========== VERDICT TABLE [{phase} frames] (obj_rel decode err, cm) ==========", flush=True)
        print(f"{'arm':<9} {'std_mean':>9} {'std_med':>8} {'swap_mean':>10} {'swap_med':>9} {'swap/std':>9}", flush=True)
        for arm in ["random", "recon", "predict"]:
            r = res[arm]; sm = r['std'][phase][0]; wm = r['swap'][phase][0]
            print(f"{arm:<9} {sm:>9.2f} {r['std'][phase][1]:>8.2f} {wm:>10.2f} {r['swap'][phase][1]:>9.2f} {wm/max(sm,1e-6):>9.2f}", flush=True)
        rnd, rec, pred = (res[a]['swap'][phase][0] for a in ["random", "recon", "predict"])
        inv = res['predict']['swap'][phase][0] / max(res['predict']['std'][phase][0], 1e-6)
        print(f"  predict vs random (swap): {rnd:.2f} -> {pred:.2f}cm ({100*(rnd-pred)/max(rnd,1e-6):.0f}% lower) | "
              f"vs recon: {rec:.2f} -> {pred:.2f} | reloc-invariance ratio {inv:.2f}", flush=True)
    print("\nGO if predict<<random (prediction creates structure) AND swap/std~1 (relocation-invariant) "
          "AND late-phase abs err small (~2-4cm); CUT/rethink otherwise.", flush=True)


if __name__ == "__main__":
    main()
