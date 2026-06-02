"""Our VLA's System-1 action head: a small goal-conditioned policy.

state = [px, py, vx, vy, bx, by, gx, gy]  (pusher, ball, goal)
action = pusher target (2-d)

Trained by BC on a get-behind-then-push expert, in OPEN space only. At eval we test
out-of-distribution starts (ball against a wall) where the open-space policy stalls — the
failure the metacognitive loop is meant to recover. This mirrors our real pipeline: train
in-distribution, fail OOD, recover via retract-and-reapproach.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

BEHIND = 0.055  # ball radius + pusher radius


def expert_action(px, py, vx, vy, bx, by, gx, gy):
    p = np.array([px, py]); b = np.array([bx, by]); g = np.array([gx, gy])
    pd = g - b
    n = np.linalg.norm(pd) + 1e-8
    push_dir = pd / n
    behind = b - push_dir * BEHIND
    if np.linalg.norm(behind - p) > 0.03:
        target = behind            # reposition behind the ball
    else:
        target = b + push_dir * 0.10  # push through ball toward goal
    return np.clip(target, -0.3, 0.3).astype(np.float32)


class PusherPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(8, 128), nn.GELU(), nn.Linear(128, 128), nn.GELU(), nn.Linear(128, 2)
        )

    def forward(self, s):
        return self.net(s)

    @torch.no_grad()
    def act(self, state8):
        s = torch.tensor(state8, dtype=torch.float32)[None]
        return self.net(s)[0].numpy()


def generate_demos(env, n, rng, open_only=True):
    X, Y = [], []
    for _ in range(n):
        obs = env.reset()
        # open-space ball + goal
        bx, by = rng.uniform(-0.15, 0.15, 2)
        env.data.qpos[env._ball_qadr + 0] = bx
        env.data.qpos[env._ball_qadr + 1] = by
        import mujoco
        mujoco.mj_forward(env.model, env.data)
        obs = env._obs()
        goal = rng.uniform(-0.15, 0.15, 2).astype(np.float32)
        for t in range(40):
            px, py, vx, vy = obs["proprio"]
            bx, by = obs["ball_pos"]
            if np.linalg.norm(np.array([bx, by]) - goal) < 0.04:
                break
            a = expert_action(px, py, vx, vy, bx, by, goal[0], goal[1])
            X.append([px, py, vx, vy, bx, by, goal[0], goal[1]])
            Y.append(a)
            obs = env.step(a)
    return np.array(X, np.float32), np.array(Y, np.float32)


def train_policy(out="jepa_wm/data/policy.pt", n_demos=400, seed=0):
    from jepa_wm.sim.env import PusherEnv
    env = PusherEnv(seed=seed)
    rng = np.random.default_rng(seed + 5)
    X, Y = generate_demos(env, n_demos, rng)
    Xt, Yt = torch.tensor(X), torch.tensor(Y)
    model = PusherPolicy()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-5)
    n = len(Xt)
    for ep in range(150):
        perm = torch.randperm(n)
        for i in range(0, n, 512):
            idx = perm[i : i + 512]
            loss = (model(Xt[idx]) - Yt[idx]).pow(2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        fit = (model(Xt) - Yt).pow(2).mean().item()
    print(f"policy BC fit MSE {fit:.5f} on {n} (state,action) pairs")
    torch.save({"model": model.state_dict()}, out)
    return model


if __name__ == "__main__":
    train_policy()
