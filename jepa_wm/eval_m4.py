"""M4: does ACTING on the surprise signal recover failures a bare policy cannot?

Setup: BC policy trained in open space. OOD eval = ball spawned against the right
wall, goal to the left → the open-space policy stalls trying to get 'behind' the ball
through the wall (effector stuck against ball+wall). The self-residual fires; the
metacognitive loop responds with retract-and-reapproach.

Compares, on identical OOD episodes:
  - bare policy
  - metacognitive policy (self-residual stall detector -> retract tool-call)
Reports task success (ball within tol of goal).
"""
from __future__ import annotations

import argparse

import mujoco
import numpy as np
import torch

from jepa_wm.policy import PusherPolicy
from jepa_wm.self_model import SelfModel
from jepa_wm.sim.env import PusherEnv

TOL = 0.05


def set_scene(env, bx, by, gx, gy):
    env.data.qpos[env._ball_qadr + 0] = bx
    env.data.qpos[env._ball_qadr + 1] = by
    env.data.qpos[env._ball_qadr + 2] = 0.03
    env.data.qpos[0:2] = [0.0, 0.0]  # pusher starts center
    mujoco.mj_forward(env.model, env.data)
    return env._obs()


def run_episode(env, policy, sm, obs, goal, horizon, meta, device="cpu"):
    """Returns (success, fired) — fired = whether recovery triggered."""
    stall = 0
    cooldown = 0
    retract_left = 0
    retract_target = None
    fired = False
    for t in range(horizon):
        px, py, vx, vy = obs["proprio"]
        bx, by = obs["ball_pos"]
        if np.linalg.norm(np.array([bx, by]) - goal) < TOL:
            return True, fired
        state = np.array([px, py, vx, vy, bx, by, goal[0], goal[1]], np.float32)

        if meta and retract_left > 0:
            a = retract_target
            retract_left -= 1
        else:
            a = policy.act(state)
            if meta and cooldown == 0:
                # self-residual: predicted pusher delta vs realized (computed post-step)
                feat = torch.tensor(np.concatenate([obs["proprio"], a]), dtype=torch.float32)
                with torch.no_grad():
                    pred_delta = sm(feat).numpy()

        prev_p = np.array([px, py])
        obs = env.step(a)
        new_p = obs["proprio"][:2]
        realized = new_p - prev_p

        if meta and retract_left == 0 and cooldown == 0:
            resid = np.linalg.norm(pred_delta - realized)
            if resid > 0.004:  # commanded motion, little realized motion
                stall += 1
            else:
                stall = 0
            if stall >= 3:  # sustained stall -> retract-and-reapproach
                fired = True
                away = prev_p - np.array([bx, by])
                away = away / (np.linalg.norm(away) + 1e-8)
                perp = np.array([-away[1], away[0]]) * 0.08  # lateral, change approach angle
                retract_target = np.clip(prev_p + away * 0.18 + perp, -0.3, 0.3).astype(np.float32)
                retract_left = 8
                stall = 0
                cooldown = 14
        if cooldown > 0:
            cooldown -= 1
    bx, by = obs["ball_pos"]
    return np.linalg.norm(np.array([bx, by]) - goal) < TOL, fired


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=60)
    p.add_argument("--horizon", type=int, default=90)
    args = p.parse_args()

    env = PusherEnv(seed=2024)
    policy = PusherPolicy(); policy.load_state_dict(torch.load("jepa_wm/data/policy.pt", map_location="cpu")["model"]); policy.eval()
    sm = SelfModel(); sm.load_state_dict(torch.load("jepa_wm/data/self_model.pt", map_location="cpu")["model"]); sm.eval()
    rng = np.random.default_rng(11)

    # sanity: open-space success
    ok = 0
    for _ in range(args.episodes):
        env.reset()
        bx, by = rng.uniform(-0.13, 0.13, 2)
        gx, gy = rng.uniform(-0.13, 0.13, 2)
        obs = set_scene(env, bx, by, gx, gy)
        s, _ = run_episode(env, policy, sm, obs, np.array([gx, gy], np.float32), args.horizon, meta=False)
        ok += s
    print(f"sanity (open-space, bare): {ok}/{args.episodes} = {ok/args.episodes:.0%}")

    # OOD: ball against right wall, goal left
    scenes = []
    for _ in range(args.episodes):
        bx = rng.uniform(0.24, 0.28); by = rng.uniform(-0.10, 0.10)
        gx = rng.uniform(-0.20, -0.10); gy = rng.uniform(-0.10, 0.10)
        scenes.append((bx, by, gx, gy))

    bare_ok = meta_ok = fired_n = 0
    for (bx, by, gx, gy) in scenes:
        env.reset(); obs = set_scene(env, bx, by, gx, gy)
        s_b, _ = run_episode(env, policy, sm, obs, np.array([gx, gy], np.float32), args.horizon, meta=False)
        env.reset(); obs = set_scene(env, bx, by, gx, gy)
        s_m, fired = run_episode(env, policy, sm, obs, np.array([gx, gy], np.float32), args.horizon, meta=True)
        bare_ok += s_b; meta_ok += s_m; fired_n += fired
    n = args.episodes
    print(f"\nOOD (ball vs wall, goal left), {n} matched episodes:")
    print(f"  bare policy:          {bare_ok}/{n} = {bare_ok/n:.0%}")
    print(f"  metacognitive policy: {meta_ok}/{n} = {meta_ok/n:.0%}   (recovery fired in {fired_n})")
    print(f"  net: {(meta_ok-bare_ok)/n:+.0%}")


if __name__ == "__main__":
    main()
