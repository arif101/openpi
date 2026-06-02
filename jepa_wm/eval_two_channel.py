"""M3: world-surprise vs self-residual on two surgically-clean anomalies.

  ball_jump   (OBJECT anomaly): pusher steps normally, then ball jumps.
  pusher_stuck(EFFECTOR anomaly): pusher commanded but doesn't move; ball normal.

Two channels:
  world-surprise  = JEPA latent prediction error (action-conditioned).
  self-residual   = ‖predicted pusher delta - realized pusher delta‖ (self-model).

Prints the 2x2 AUROC matrix (channel x anomaly, each vs normal transitions).
Clean complementarity = each channel fires on its OWN anomaly and not the other's.
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from jepa_wm.data import stack_state
from jepa_wm.model import JEPA
from jepa_wm.self_model import SelfModel
from jepa_wm.sim.env import PusherEnv


def auroc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    allv = np.concatenate([pos, neg])
    order = allv.argsort()
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(allv) + 1)
    u = ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2
    return u / (len(pos) * len(neg))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--jepa", default="jepa_wm/data/jepa_s2.pt")
    p.add_argument("--self", dest="self_ckpt", default="jepa_wm/data/self_model.pt")
    p.add_argument("--rollouts", type=int, default=240)
    p.add_argument("--horizon", type=int, default=16)
    args = p.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    ck = torch.load(args.jepa, map_location=device, weights_only=False)
    stack = ck["args"].get("stack", 1)
    jepa = JEPA(dim=ck["dim"], in_ch=3 * stack).to(device); jepa.load_state_dict(ck["model"]); jepa.eval()
    sm = SelfModel(); sm.load_state_dict(torch.load(args.self_ckpt, map_location="cpu")["model"]); sm.eval()
    print(f"jepa stack={stack}; self-model loaded")

    def mk_stack(frames):
        fs = list(frames[-stack:])
        while len(fs) < stack:
            fs = [fs[0]] + fs
        return np.transpose(np.concatenate(fs, axis=-1), (2, 0, 1))

    env = PusherEnv(seed=2024)
    rng = np.random.default_rng(7)
    # label: 0 free, 1 contact, 2 ball_jump(OBJECT), 3 pusher_stuck(EFFECTOR)
    rows = []  # (state_t, a, state_t1, proprio_t, realized_pdelta, label)
    for r in range(args.rollouts):
        obs = env.reset()
        frames = [obs["image"]]
        inj_step = rng.integers(4, args.horizon - 2)
        eff = r % 2 == 0  # alternate which anomaly
        for t in range(args.horizon):
            a = (obs["ball_pos"] + rng.normal(0, 0.05, 2)).clip(-0.3, 0.3).astype(np.float32)
            proprio_t, ball_t = obs["proprio"].copy(), obs["ball_pos"].copy()
            state_t = mk_stack(frames)
            if t == inj_step:
                if eff:
                    obs = env.step_pusher_stuck(a); label = 3
                else:
                    env.step(a); obs = env.displace_ball(); label = 2
            else:
                obs = env.step(a)
                label = 1 if np.linalg.norm(obs["ball_pos"] - ball_t) > 0.01 else 0
            realized_pdelta = obs["proprio"][:2] - proprio_t[:2]
            frames.append(obs["image"])
            rows.append((state_t, a, mk_stack(frames), proprio_t, realized_pdelta, label))

    labels = np.array([x[5] for x in rows])
    acts = np.stack([x[1] for x in rows])
    proprio = np.stack([x[3] for x in rows])
    pdelta = np.stack([x[4] for x in rows])

    # world-surprise (batched on device)
    st = np.stack([x[0] for x in rows]); st1 = np.stack([x[2] for x in rows])
    ws = []
    with torch.no_grad():
        for i in range(0, len(rows), 512):
            f = lambda a: torch.from_numpy(a[i:i+512]).float().div(255).to(device)
            ws.append(jepa.surprise(f(st), torch.from_numpy(acts[i:i+512]).to(device), f(st1)).cpu().numpy())
    world = np.concatenate(ws)

    # self-residual
    feat = torch.tensor(np.concatenate([proprio, acts], -1), dtype=torch.float32)
    with torch.no_grad():
        pred_delta = sm(feat).numpy()
    selfr = np.linalg.norm(pred_delta - pdelta, axis=-1)

    normal = (labels == 0) | (labels == 1)
    obj = labels == 2
    eff = labels == 3
    print(f"\nn: free={int((labels==0).sum())} contact={int((labels==1).sum())} "
          f"ball_jump={int(obj.sum())} pusher_stuck={int(eff.sum())}\n")

    def line(name, sig):
        a_obj = auroc(sig[obj], sig[normal])
        a_eff = auroc(sig[eff], sig[normal])
        print(f"  {name:16s}  ball_jump(OBJECT)={a_obj:.3f}   pusher_stuck(EFFECTOR)={a_eff:.3f}")
        return a_obj, a_eff

    print("AUROC (anomaly vs normal):")
    wo, we = line("world-surprise", world)
    so, se = line("self-residual", selfr)
    print(f"\n  mean surprise -- normal: world {world[normal].mean():.4f}  self {selfr[normal].mean():.5f}")
    print(f"  ball_jump   : world {world[obj].mean():.4f}  self {selfr[obj].mean():.5f}")
    print(f"  pusher_stuck: world {world[eff].mean():.4f}  self {selfr[eff].mean():.5f}")

    # complementarity verdict
    world_specialises = wo > 0.8 and we < wo - 0.15
    self_specialises = se > 0.8 and so < se - 0.15
    print("\n=== M3 verdict ===")
    print(f"  world-surprise specialises on OBJECT anomaly: {world_specialises}")
    print(f"  self-residual specialises on EFFECTOR anomaly: {self_specialises}")
    if world_specialises and self_specialises:
        print("  COMPLEMENTARY ✅  — channels are orthogonal; both needed.")
    elif wo > 0.8 and we > 0.8:
        print("  world-surprise catches BOTH — action-conditioned WM may suffice; self-channel redundant.")
    else:
        print("  mixed — inspect means above.")


if __name__ == "__main__":
    main()
