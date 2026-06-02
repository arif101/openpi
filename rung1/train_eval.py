"""Rung 1 diagnostic: does a STRUCTURED (object-centric) world model generalize to novel
arrangements / object counts / sizes where a MONOLITHIC one fails? = structure > coverage
for the world model. Foundation for 'simulate a candidate approach on a novel object'.

Both models train ONLY on K=3 objects (default radii range). Then we measure next-state
prediction error on:
  - K=3 novel arrangements (held-out rollouts)   [interpolation — both should do ok]
  - K=2 / K=4 / K=5 (UNSEEN object counts)        [compositional — structure should win]
  - K=3 with held-out LARGE radii (unseen sizes)  [novel-object proxy]
Report contact-MSE (transitions where an object actually moved) — the dynamics that matter.
"""
from __future__ import annotations

import numpy as np
import torch

from physics2d import collect, collect_rollouts
from world_models import MonolithicWM, InteractionWM

DEV = "cpu"
SMALL_R = lambda rng: rng.uniform(0.05, 0.09, 3)   # train sizes
BIG_R = lambda rng: rng.uniform(0.11, 0.14, 3)     # held-out (unseen) sizes


@torch.no_grad()
def rollout_mse(model, roll, horizon=12):
    """Multi-step simulation: feed the model's own prediction back H steps; compare to truth.
    This is what a world model is FOR (simulate a candidate forward). Errors compound, so a
    memorizing (monolithic) model degrades far worse OOD than a structural one."""
    pusher = torch.tensor(roll["pusher"])      # (R,T,2)
    action = torch.tensor(roll["action"])      # (R,T,2)
    objects = torch.tensor(roll["objects"])    # (R,T,n,4)
    radii = torch.tensor(roll["radii"])        # (R,T,n)
    R, T, n, _ = objects.shape
    H = min(horizon, T)
    obj = objects[:, 0].clone()                # start state
    err = 0.0
    for t in range(H):
        delta = model(pusher[:, t], action[:, t], obj, radii[:, 0])
        obj = obj + delta
        truth = objects[:, t + 1 if t + 1 < T else t]            # (R,n,4)
        err += ((obj[..., :2] - truth[..., :2]) ** 2).mean().item()
    return err / H


def to_t(d):
    return {k: torch.tensor(v, device=DEV) for k, v in d.items()}


def batchify(d, bs, shuffle=True):
    n = len(d["pusher"])
    idx = np.random.permutation(n) if shuffle else np.arange(n)
    for i in range(0, n - bs + 1, bs):
        j = idx[i : i + bs]
        yield {k: v[j] for k, v in d.items()}


def train(model, data, epochs=40, bs=256, lr=1e-3):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    d = to_t(data)
    for ep in range(epochs):
        for b in batchify(d, bs):
            target = b["next_objects"] - b["objects"]            # delta
            pred = model(b["pusher"], b["action"], b["objects"], b["radii"])
            loss = ((pred - target) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    return model


@torch.no_grad()
def contact_mse(model, data):
    d = to_t(data)
    pred = model(d["pusher"], d["action"], d["objects"], d["radii"])
    target = d["next_objects"] - d["objects"]
    err = ((pred - target) ** 2).mean(-1)                        # (T,n) per object
    motion = (d["next_objects"][..., :2] - d["objects"][..., :2]).norm(dim=-1)  # (T,n)
    mask = motion > 0.01
    overall = err.mean().item()
    contact = err[mask].mean().item() if mask.any() else float("nan")
    return overall, contact


TINY_R = lambda rng: rng.uniform(0.03, 0.045, 3)   # held-out SMALL (lighter, faster, harder)


def run_once(seed):
    torch.manual_seed(seed); np.random.seed(seed)
    train_data = collect(3, 400, 24, seed=1 + seed, radii_sampler=SMALL_R)
    mono = train(MonolithicWM(n_max=8), train_data)
    inter = train(InteractionWM(), train_data)
    rolls = {
        "K3": collect_rollouts(3, 60, 16, seed=199 + seed, radii_sampler=SMALL_R),
        "K8": collect_rollouts(8, 60, 16, seed=203 + seed),
        "K3_tiny": collect_rollouts(3, 60, 16, seed=211 + seed, radii_sampler=TINY_R),
    }
    out = {}
    for name, rd in rolls.items():
        out[name] = (rollout_mse(mono, rd), rollout_mse(inter, rd))
    return out


def multiseed(n_seeds=3):
    print(f"=== multi-seed robustness ({n_seeds} seeds) — 12-step rollout sim ===")
    agg = {}
    for s in range(n_seeds):
        r = run_once(s)
        for k, v in r.items():
            agg.setdefault(k, []).append(v)
    print(f"{'eval':10s} {'Mono mean±std':>20s} {'Inter mean±std':>20s} {'gap(OOD/in-dist)':>22s}")
    base_m = np.mean([v[0] for v in agg['K3']]); base_i = np.mean([v[1] for v in agg['K3']])
    for k, vs in agg.items():
        m = np.array([v[0] for v in vs]); i = np.array([v[1] for v in vs])
        gm = m.mean() / base_m; gi = i.mean() / base_i
        print(f"{k:10s} {m.mean():>9.4f}±{m.std():.4f}    {i.mean():>9.4f}±{i.std():.4f}    "
              f"mono {gm:.2f}x / inter {gi:.2f}x")
    print("\nGate: across seeds, Interaction's degradation ratio on unseen counts (K8) and "
          "harder sizes (K3_tiny) stays near 1x while Monolithic blows up => structure > coverage.")


def main():
    torch.manual_seed(0); np.random.seed(0)
    print("collecting data...")
    train_data = collect(3, 400, 24, seed=1, radii_sampler=SMALL_R)
    evals = {
        "K3_novel  (interp)": collect(3, 80, 24, seed=99, radii_sampler=SMALL_R),
        "K2  (unseen count)": collect(2, 80, 24, seed=7),
        "K4  (unseen count)": collect(4, 80, 24, seed=8),
        "K5  (unseen count)": collect(5, 80, 24, seed=9),
        "K3_BIG (unseen size)": collect(3, 80, 24, seed=11, radii_sampler=BIG_R),
    }
    print(f"train transitions: {len(train_data['pusher'])}\n")

    print("training MonolithicWM (coverage)...")
    mono = train(MonolithicWM(n_max=8), train_data)   # capacity 8 so it can RUN on K<=8 (still capped + trained only on K3)
    print("training InteractionWM (structure)...")
    inter = train(InteractionWM(), train_data)

    # ---- single-step contact-MSE ----
    print(f"\n--- single-step contact-MSE ---")
    print(f"{'eval set':22s} {'Monolithic':>12s} {'Interaction':>12s}  winner")
    for name, ed in evals.items():
        _, mc = contact_mse(mono, ed)
        _, ic = contact_mse(inter, ed)
        print(f"{name:22s} {mc:>12.5f} {ic:>12.5f}  {'STRUCT' if ic < mc else 'mono'}")

    # ---- multi-step rollout simulation (the real WM use case) + generalization gap ----
    rolls = {
        "K3 (in-dist)": collect_rollouts(3, 60, 16, seed=199, radii_sampler=SMALL_R),
        "K4 (unseen N)": collect_rollouts(4, 60, 16, seed=201),
        "K6 (unseen N)": collect_rollouts(6, 60, 16, seed=202),
        "K8 (unseen N)": collect_rollouts(8, 60, 16, seed=203),
    }
    print(f"\n--- 12-step rollout simulation MSE (errors compound) ---")
    print(f"{'eval set':16s} {'Monolithic':>12s} {'Interaction':>12s}  winner")
    base_m = base_i = None
    for name, rd in rolls.items():
        mc = rollout_mse(mono, rd)
        ic = rollout_mse(inter, rd)
        if base_m is None:
            base_m, base_i = mc, ic
        print(f"{name:16s} {mc:>12.5f} {ic:>12.5f}  {'STRUCT' if ic < mc else 'mono'}")
    # generalization gap (degradation ratio vs in-distribution)
    mc8 = rollout_mse(mono, rolls["K8 (unseen N)"]); ic8 = rollout_mse(inter, rolls["K8 (unseen N)"])
    print(f"\nGeneralization gap K3->K8 (OOD/in-dist):  Monolithic {mc8/base_m:.2f}x   "
          f"Interaction {ic8/base_i:.2f}x")
    print("Gate: Interaction degrades far LESS on unseen object counts under multi-step "
          "simulation => structure > coverage for the world model.")


if __name__ == "__main__":
    main()
