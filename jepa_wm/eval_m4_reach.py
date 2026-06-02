"""M4': does ACTING on the self-residual recover an effector-stuck a reactive policy can't?

Task: reactive reach policy (action = goal) drives the pusher straight to a goal on the
far side of an immovable peg. Bare policy jams on the peg and never arrives. The
metacognitive loop detects the jam (self-residual: commanded motion, none realized) and
runs retract-and-sidestep, then resumes — going around the peg.

Sanity: no-obstacle reach (peg cleared by starting/goal off-axis) ~ 100%.
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from jepa_wm.self_model import SelfModel
from jepa_wm.sim.reach_env import ReachObstacleEnv

TOL = 0.04
R = 0.22


def step_toward(pos, target, mag=0.06):
    """Bounded-velocity command: a position target a short step toward `target`.
    Keeps per-step motion in the self-model's training distribution."""
    d = np.asarray(target) - pos
    n = np.linalg.norm(d) + 1e-8
    return np.clip(pos + d / n * min(mag, n), -0.3, 0.3).astype(np.float32)


def run(env, sm, start, goal, horizon, meta):
    obs = env.reset(start)
    side = 1
    stall = 0
    detour_target = None
    detour_timer = 0
    fired = False
    for t in range(horizon):
        p = obs["proprio"][:2]
        if np.linalg.norm(p - goal) < TOL:
            return True, fired
        if meta and detour_target is not None:
            # drive to the detour waypoint until reached or timeout, THEN resume goal
            a = step_toward(p, detour_target)
            detour_timer -= 1
            if np.linalg.norm(p - detour_target) < 0.04 or detour_timer <= 0:
                detour_target = None
        else:
            a = step_toward(p, goal)  # reactive: bounded step toward goal
            if meta:
                feat = torch.tensor(np.concatenate([obs["proprio"], a]), dtype=torch.float32)
                with torch.no_grad():
                    pred_delta = sm(feat).numpy()
        prev = p.copy()
        obs = env.step(a)
        if meta and detour_target is None:
            realized = obs["proprio"][:2] - prev
            resid = np.linalg.norm(pred_delta - realized)
            stall = stall + 1 if resid > 0.004 else 0
            if stall >= 3:  # jam detected -> set a go-around waypoint offset from the jam point
                fired = True
                gdir = goal - prev; gdir = gdir / (np.linalg.norm(gdir) + 1e-8)
                perp = np.array([-gdir[1], gdir[0]]) * side
                detour_target = np.clip(prev + perp * 0.17, -0.3, 0.3).astype(np.float32)
                detour_timer = 20
                side *= -1  # alternate sides across triggers
                stall = 0
    return np.linalg.norm(obs["proprio"][:2] - goal) < TOL, fired


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=48)
    p.add_argument("--horizon", type=int, default=90)
    args = p.parse_args()

    env = ReachObstacleEnv(seed=0)
    sm = SelfModel(); sm.load_state_dict(torch.load("jepa_wm/data/self_model.pt", map_location="cpu")["model"]); sm.eval()
    rng = np.random.default_rng(7)

    # sanity: off-axis reach that doesn't cross the peg (start->goal offset so the line misses center)
    ok = 0
    for _ in range(args.episodes):
        th = rng.uniform(0, 2 * np.pi)
        start = R * np.array([np.cos(th), np.sin(th)])
        goal = start + np.array([0.10, 0.0])  # short reach, no peg crossing
        goal = np.clip(goal, -0.28, 0.28)
        s, _ = run(env, sm, start, goal, args.horizon, meta=False)
        ok += s
    print(f"sanity (short off-peg reach, bare): {ok}/{args.episodes} = {ok/args.episodes:.0%}")

    # obstacle: start and goal on opposite sides -> straight path crosses the peg
    scenes = []
    for _ in range(args.episodes):
        th = rng.uniform(0, 2 * np.pi)
        start = R * np.array([np.cos(th), np.sin(th)])
        goal = -start
        scenes.append((start, goal))

    bare = meta = fired_n = 0
    for start, goal in scenes:
        sb, _ = run(env, sm, start, goal, args.horizon, meta=False)
        sm_ok, fired = run(env, sm, start, goal, args.horizon, meta=True)
        bare += sb; meta += sm_ok; fired_n += fired
    n = args.episodes
    print(f"\nobstacle (path crosses peg), {n} matched episodes:")
    print(f"  bare reactive policy: {bare}/{n} = {bare/n:.0%}")
    print(f"  metacognitive policy: {meta}/{n} = {meta/n:.0%}   (recovery fired in {fired_n})")
    print(f"  net: {(meta-bare)/n:+.0%}")


if __name__ == "__main__":
    main()
