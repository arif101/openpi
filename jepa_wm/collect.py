"""Collect random-policy rollouts from PusherEnv into an npz dataset.

Policy mix designed to produce contact events (the predictable/unpredictable
split JEPA exploits): half the rollouts chase the ball (guarantees pushing),
half use random target walks (free-space + occasional contact).
"""
from __future__ import annotations

import argparse
import pathlib
import time

import numpy as np

from jepa_wm.sim.env import PusherEnv


def collect(n_rollouts: int, horizon: int, seed: int, out: str):
    env = PusherEnv(seed=seed)
    rng = np.random.default_rng(seed + 1)
    H, W = env.img_size, env.img_size

    images = np.zeros((n_rollouts, horizon + 1, H, W, 3), dtype=np.uint8)
    actions = np.zeros((n_rollouts, horizon, 2), dtype=np.float32)
    proprio = np.zeros((n_rollouts, horizon + 1, 4), dtype=np.float32)
    ball = np.zeros((n_rollouts, horizon + 1, 2), dtype=np.float32)

    t0 = time.time()
    for r in range(n_rollouts):
        obs = env.reset()
        images[r, 0] = obs["image"]
        proprio[r, 0] = obs["proprio"]
        ball[r, 0] = obs["ball_pos"]
        chase = r % 2 == 0
        target = rng.uniform(-0.25, 0.25, size=2)
        for t in range(horizon):
            if chase:
                a = obs["ball_pos"] + rng.normal(0, 0.04, size=2)
            else:
                if t % 6 == 0:
                    target = rng.uniform(-0.25, 0.25, size=2)
                a = target + rng.normal(0, 0.03, size=2)
            a = np.clip(a, -0.3, 0.3).astype(np.float32)
            obs = env.step(a)
            actions[r, t] = a
            images[r, t + 1] = obs["image"]
            proprio[r, t + 1] = obs["proprio"]
            ball[r, t + 1] = obs["ball_pos"]
        if (r + 1) % 100 == 0:
            print(f"  {r + 1}/{n_rollouts} rollouts  ({(r + 1) / (time.time() - t0):.0f} roll/s)")

    out_path = pathlib.Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, images=images, actions=actions, proprio=proprio, ball=ball)
    dt = time.time() - t0
    mb = out_path.stat().st_size / 1e6
    print(f"saved {out_path}  ({n_rollouts}x{horizon}) in {dt:.1f}s  [{mb:.0f} MB]")
    # quick stats: ball displacement = how much contact dynamics we captured
    disp = np.linalg.norm(np.diff(ball, axis=1), axis=-1).sum(axis=1)
    print(f"  ball path length: mean {disp.mean():.3f} m, frac with motion>0.05m: {(disp > 0.05).mean():.2f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--horizon", type=int, default=24)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="jepa_wm/data/pusher_train.npz")
    a = p.parse_args()
    collect(a.n, a.horizon, a.seed, a.out)
