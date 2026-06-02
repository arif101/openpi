"""Wire the frozen JEPA vision encoder into the action head.

Tests whether perception composes: does an action head that sees ONLY the JEPA
latent z(image) + goal imitate the expert as well as one that sees privileged
ground-truth state? If held-out action MSE is comparable, the frozen vision encoder
is a sufficient perceptual front-end for the policy — i.e. the 'V' is real, not a
ground-truth-state cheat.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from jepa_wm.data import stack_state
from jepa_wm.model import JEPA
from jepa_wm.policy import expert_action
from jepa_wm.sim.env import PusherEnv


class Head(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 128), nn.GELU(), nn.Linear(128, 128), nn.GELU(), nn.Linear(128, 2))

    def forward(self, x):
        return self.net(x)


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    ck = torch.load("jepa_wm/data/jepa_s2.pt", map_location=device, weights_only=False)
    stack = ck["args"]["stack"]
    jepa = JEPA(dim=ck["dim"], in_ch=3 * stack).to(device); jepa.load_state_dict(ck["model"]); jepa.eval()

    env = PusherEnv(seed=1)
    rng = np.random.default_rng(3)
    Z, S, Y = [], [], []  # latent, privileged-state, action
    import mujoco
    for ep in range(300):
        env.reset()
        bx, by = rng.uniform(-0.13, 0.13, 2)
        env.data.qpos[env._ball_qadr + 0] = bx; env.data.qpos[env._ball_qadr + 1] = by
        mujoco.mj_forward(env.model, env.data); obs = env._obs()
        goal = rng.uniform(-0.13, 0.13, 2).astype(np.float32)
        frames = [obs["image"]]
        for t in range(30):
            px, py, vx, vy = obs["proprio"]; bx, by = obs["ball_pos"]
            a = expert_action(px, py, vx, vy, bx, by, goal[0], goal[1])
            st = stack_state(np.array(frames), len(frames) - 1, stack)
            with torch.no_grad():
                z = jepa.online(torch.from_numpy(st[None]).float().div(255).to(device)).cpu().numpy()[0]
            Z.append(np.concatenate([z, goal]))
            S.append([px, py, vx, vy, bx, by, goal[0], goal[1]])
            Y.append(a)
            obs = env.step(a); frames.append(obs["image"])

    Z = torch.tensor(np.array(Z, np.float32)); S = torch.tensor(np.array(S, np.float32)); Y = torch.tensor(np.array(Y, np.float32))
    n = len(Y); ntr = int(0.85 * n); idx = torch.randperm(n)
    tr, te = idx[:ntr], idx[ntr:]

    def train(X):
        head = Head(X.shape[1]); opt = torch.optim.AdamW(head.parameters(), lr=2e-3, weight_decay=1e-5)
        for ep in range(150):
            p = tr[torch.randperm(len(tr))]
            for i in range(0, len(p), 512):
                b = p[i : i + 512]
                loss = (head(X[b]) - Y[b]).pow(2).mean()
                opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            return (head(X[te]) - Y[te]).pow(2).mean().item()

    print(f"demos: {n} (state,action) pairs; latent dim {Z.shape[1]-2}+2(goal)")
    mse_state = train(S)
    mse_vision = train(Z)
    print(f"  held-out action MSE  | privileged state : {mse_state:.5f}")
    print(f"  held-out action MSE  | JEPA vision latent: {mse_vision:.5f}")
    ratio = mse_vision / mse_state
    print(f"  vision/state ratio: {ratio:.2f}  -> {'composes ✅ (vision ~ state)' if ratio < 2.0 else 'vision worse — encoder insufficient'}")


if __name__ == "__main__":
    main()
