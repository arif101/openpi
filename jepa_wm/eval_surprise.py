"""M2: does the JEPA's surprise signal spike at physically-impossible events?

We roll out the pusher and, at a known step, inject an anomaly (teleport / levitate
/ freeze). Each transition (img_t, a_t, img_t1) gets a surprise score from the frozen
JEPA. The metacognition claim is that surprise is high at injected anomalies but NOT
at legal-but-hard *contact* transitions (ball genuinely pushed). We therefore report:

  - mean surprise per category: free-space / legal-contact / anomaly
  - AUROC(anomaly vs ALL normal)            -- can it detect impossible events?
  - AUROC(anomaly vs legal-contact only)    -- the HARD test: not just "big motion"
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from jepa_wm.data import stack_state, to_float
from jepa_wm.model import JEPA
from jepa_wm.sim.env import PusherEnv


def auroc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    # rank-based Mann-Whitney U / (n_pos*n_neg)
    allv = np.concatenate([pos, neg])
    order = allv.argsort()
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(allv) + 1)
    r_pos = ranks[: len(pos)].sum()
    u = r_pos - len(pos) * (len(pos) + 1) / 2
    return u / (len(pos) * len(neg))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="jepa_wm/data/jepa.pt")
    p.add_argument("--rollouts", type=int, default=200)
    p.add_argument("--horizon", type=int, default=16)
    p.add_argument("--contact-thresh", type=float, default=0.01, help="ball motion (m) that counts as legal contact")
    args = p.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    stack = ck["args"].get("stack", 1)
    model = JEPA(dim=ck["dim"], in_ch=3 * stack).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"loaded {args.ckpt}  (stack={stack})")

    def make_stack(frame_list):
        fs = list(frame_list[-stack:])
        while len(fs) < stack:
            fs = [fs[0]] + fs
        return np.transpose(np.concatenate(fs, axis=-1), (2, 0, 1))

    env = PusherEnv(seed=2024)
    rng = np.random.default_rng(7)
    anomalies = ["teleport", "ghost_push", "pass_through"]
    inject_fn = {
        "teleport": lambda a: env.inject_teleport(),
        "ghost_push": lambda a: env.step_ghost(a),
        "pass_through": lambda a: env.step_passthrough(a),
    }

    # collect transitions with labels: 0=free, 1=legal-contact, 2=anomaly(+type)
    rows = []  # (state_t, a_t, state_t1, label)
    for r in range(args.rollouts):
        obs = env.reset()
        frames = [obs["image"]]
        inj_step = rng.integers(4, args.horizon - 2)
        inj_type = anomalies[r % len(anomalies)]
        for t in range(args.horizon):
            a = (obs["ball_pos"] + rng.normal(0, 0.05, size=2)).clip(-0.3, 0.3).astype(np.float32)
            ball_t = obs["ball_pos"]
            state_t = make_stack(frames)
            if t == inj_step:
                obs = inject_fn[inj_type](a)
                label = 2
            else:
                obs = env.step(a)
                motion = np.linalg.norm(obs["ball_pos"] - ball_t)
                label = 1 if motion > args.contact_thresh else 0
            frames.append(obs["image"])
            state_t1 = make_stack(frames)
            rows.append((state_t, a, state_t1, label))

    acts = torch.from_numpy(np.stack([x[1] for x in rows])).to(device)
    states_t = [x[0] for x in rows]
    states_t1 = [x[2] for x in rows]
    labels = np.array([x[3] for x in rows])

    def batch_states(arr):
        return torch.from_numpy(np.stack(arr)).float().div_(255.0).to(device)

    surp = []
    bs = 512
    with torch.no_grad():
        for i in range(0, len(rows), bs):
            s = model.surprise(batch_states(states_t[i : i + bs]), acts[i : i + bs], batch_states(states_t1[i : i + bs]))
            surp.append(s.cpu().numpy())
    surp = np.concatenate(surp)

    free = surp[labels == 0]
    contact = surp[labels == 1]
    anom = surp[labels == 2]
    print("=== M2: surprise by category ===")
    print(f"  free-space     n={len(free):4d}  surprise mean {free.mean():.4f}  p90 {np.percentile(free,90):.4f}")
    print(f"  legal-contact  n={len(contact):4d}  surprise mean {contact.mean():.4f}  p90 {np.percentile(contact,90):.4f}")
    print(f"  ANOMALY        n={len(anom):4d}  surprise mean {anom.mean():.4f}  p90 {np.percentile(anom,90):.4f}")
    a_all = auroc(anom, np.concatenate([free, contact]))
    a_hard = auroc(anom, contact)
    print(f"\n  AUROC(anomaly vs all normal)   = {a_all:.3f}")
    print(f"  AUROC(anomaly vs legal-contact)= {a_hard:.3f}   <- HARD test (not just 'big motion')")
    # per-anomaly-type breakdown
    print("\n  per-type AUROC vs legal-contact:")
    types = anomalies
    type_scores = {t: [] for t in types}
    anom_idx = np.where(labels == 2)[0]
    for k, idx in enumerate(anom_idx):
        type_scores[types[k % 3]].append(surp[idx])
    for t in types:
        print(f"    {t:9s}: AUROC={auroc(type_scores[t], contact):.3f}  mean_surprise={np.mean(type_scores[t]):.4f}")

    gate = a_hard > 0.8
    print(f"\nM2 {'PASS' if gate else 'FAIL'}  (gate: AUROC(anomaly vs legal-contact) > 0.80)")


if __name__ == "__main__":
    main()
