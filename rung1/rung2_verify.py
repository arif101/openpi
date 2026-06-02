"""Rung 2 — reasoning via world-model VERIFICATION (non-exploitable), enabled by the structured WM.

Task: push a TARGET disk to a GOAL location. In the NOVEL condition a BLOCKER disk sits between
target and goal, so a reactive "push straight at the goal" policy rams the blocker and fails.

Agents:
  Reactive        : every step, push the target toward the goal. No model, no reasoning.
  WM-Verify(model): MPC. Enumerate K candidate push DIRECTIONS (a small grounded set, NOT free
                    optimization -> non-exploitable). For each, simulate the object dynamics H steps
                    with `model` (pusher kinematics known analytically), score by predicted
                    target->goal distance, pick the best, execute one step, replan.

Claim: WM-Verify with the STRUCTURED world model reasons about the blocker (it generalizes to the
novel arrangement, Rung 1) and routes the target AROUND it; the same agent with a MONOLITHIC WM
mis-simulates the novel arrangement and fails; the reactive baseline gets blocked.
"""
from __future__ import annotations

import numpy as np
import torch

from physics2d import World, BOUND, PUSH_R, STEP, collect
from world_models import MonolithicWM, InteractionWM
from train_eval import train, SMALL_R

DEV = "cpu"
TOL = 0.10
K_DIRS = 12
HORIZON = 10
REPLAN = 3


def pusher_step(pusher, action):
    """Analytic pusher kinematics (agent knows its own body)."""
    d = action - pusher
    dist = np.linalg.norm(d) + 1e-9
    return pusher + d / dist * min(STEP, dist)


def behind(target_pos, direction, obj_r):
    """Pusher target that pushes `target` along `direction`: sit on the opposite side."""
    return np.clip(target_pos - direction * (PUSH_R + obj_r + 0.04), -BOUND, BOUND)


def push_along_action(pusher, tpos, obj_r, d):
    """Reliable 2-phase 'push object along direction d' feedback primitive (stateless):
    if not yet behind the object -> move to the behind-point; once behind -> drive THROUGH it."""
    rel = pusher - tpos
    behind_amt = rel @ (-d)                                    # >0 means pusher is on the -d side
    lateral = np.linalg.norm(rel - (rel @ (-d)) * (-d))
    if behind_amt > 0 and lateral < obj_r + PUSH_R:
        return np.clip(tpos + d * 0.6, -BOUND, BOUND)          # push through toward d
    return behind(tpos, d, obj_r)                              # get into position behind


@torch.no_grad()
def simulate_dir(model, pusher, objects, radii, tgt_idx, d, horizon):
    """Simulate pushing the target along d for `horizon` steps using the WM for object dynamics
    and the analytic push-along primitive for the pusher. Returns predicted target end position."""
    p = pusher.copy()
    obj = torch.tensor(objects[None], dtype=torch.float32)
    rad = torch.tensor(radii[None], dtype=torch.float32)
    for _ in range(horizon):
        tpos = obj[0, tgt_idx, :2].numpy()
        a = push_along_action(p, tpos, radii[tgt_idx], d)
        p = pusher_step(p, a)
        pt = torch.tensor(p[None], dtype=torch.float32)
        at = torch.tensor(a[None], dtype=torch.float32)
        obj = obj + model(pt, at, obj, rad)
    return obj[0, tgt_idx, :2].numpy()


def wm_verify_action(model, state, tgt_idx, goal):
    """Verify K candidate push directions by simulation; act on the best (non-exploitable: a small
    grounded candidate set, no optimization-through-model)."""
    pusher, objects, radii = state["pusher"], state["objects"], state["radii"]
    tpos = objects[tgt_idx, :2]
    best_d, best_score = None, 1e9
    for k in range(K_DIRS):
        th = 2 * np.pi * k / K_DIRS
        d = np.array([np.cos(th), np.sin(th)])
        end = simulate_dir(model, pusher, objects, radii, tgt_idx, d, HORIZON)
        score = np.linalg.norm(end - goal)
        if score < best_score:
            best_score, best_d = score, d
    return push_along_action(pusher, tpos, radii[tgt_idx], best_d)


def reactive_action(state, tgt_idx, goal):
    tpos = state["objects"][tgt_idx, :2]
    d = goal - tpos
    d = d / (np.linalg.norm(d) + 1e-9)
    return push_along_action(state["pusher"], tpos, state["radii"][tgt_idx], d)


import copy


def true_progress(world, tgt_idx, goal, d, H):
    """Oracle: roll the TRUE env H steps pushing along d; return goal-distance reduction (progress)."""
    w = copy.deepcopy(world)
    d0 = np.linalg.norm(w.pos[tgt_idx] - goal)
    for _ in range(H):
        s = w.state()
        a = push_along_action(s["pusher"], s["objects"][tgt_idx, :2], w.r[tgt_idx], d)
        w.step(a)
    return d0 - np.linalg.norm(w.pos[tgt_idx] - goal)


def candidate_dirs():
    return [np.array([np.cos(2 * np.pi * k / K_DIRS), np.sin(2 * np.pi * k / K_DIRS)]) for k in range(K_DIRS)]


def chosen_dir(kind, world, model, tgt_idx, goal):
    s = world.state()
    pusher, objects, radii = s["pusher"], s["objects"], s["radii"]
    tpos = objects[tgt_idx, :2]
    if kind == "react":
        d = goal - tpos
        return d / (np.linalg.norm(d) + 1e-9)
    best_d, best = None, 1e9
    for d in candidate_dirs():
        end = simulate_dir(model, pusher, objects, radii, tgt_idx, d, HORIZON)
        sc = np.linalg.norm(end - goal)
        if sc < best:
            best, best_d = sc, d
    return best_d


def make_scene(rng, n_blockers, block_between):
    """Target + blockers + goal. If block_between, place a blocker on the target->goal line."""
    n = 1 + n_blockers
    radii = rng.uniform(0.05, 0.09, n)
    w = World(n, radii=radii, seed=int(rng.integers(1 << 30)))
    # target at left, goal at right
    w.pos[0] = np.array([-0.5, rng.uniform(-0.2, 0.2)])
    goal = np.array([0.5, w.pos[0, 1] + rng.uniform(-0.1, 0.1)])
    for b in range(n_blockers):
        if block_between and b == 0:
            w.pos[1 + b] = (w.pos[0] + goal) / 2 + rng.normal(0, 0.03, 2)  # ON the path
        else:
            w.pos[1 + b] = rng.uniform(-0.3, 0.3, 2)
    w.vel[:] = 0
    return w, 0, goal


def main():
    torch.manual_seed(0); np.random.seed(0)
    print("training world models (struct + mono) on K=3...")
    td = collect(3, 400, 24, seed=1, radii_sampler=SMALL_R)
    mono = train(MonolithicWM(n_max=8), td)
    inter = train(InteractionWM(), td)

    conds = {
        "no blocker (easy)":       dict(n_blockers=1, block_between=False),
        "BLOCKER on path (novel)": dict(n_blockers=1, block_between=True),
        "2 blockers (novel)":      dict(n_blockers=2, block_between=True),
    }
    N, H = 40, 20
    print(f"\nDecision-quality: REGRET = oracle_best_progress - chosen_dir_true_progress (lower=better)")
    print(f"{'condition':24s} {'Reactive':>10s} {'Verify-Mono':>12s} {'Verify-Struct':>14s} {'oracle':>9s}")
    for cname, cfg in conds.items():
        rng = np.random.default_rng(123)
        reg = {"react": [], "mono": [], "struct": []}
        oracles = []
        for _ in range(N):
            seed = int(rng.integers(1 << 30))
            w, ti, goal = make_scene(np.random.default_rng(seed), **cfg)
            # oracle: best true progress over candidate dirs
            tp = {tuple(d): true_progress(w, ti, goal, d, H) for d in candidate_dirs()}
            oracle = max(tp.values()); oracles.append(oracle)
            for kind, model in [("react", None), ("mono", mono), ("struct", inter)]:
                d = chosen_dir(kind, w, model, ti, goal)
                # snap chosen dir to nearest candidate to read its true progress
                near = min(candidate_dirs(), key=lambda c: np.linalg.norm(c - d))
                reg[kind].append(oracle - tp[tuple(near)])
        print(f"{cname:24s} {np.mean(reg['react']):>10.3f} {np.mean(reg['mono']):>12.3f} "
              f"{np.mean(reg['struct']):>14.3f} {np.mean(oracles):>9.3f}")
    print("\nGate: on NOVEL blocker arrangements, Verify-Struct regret << Reactive and < Verify-Mono "
          "=> structured WM enables verification-reasoning that picks the right approach where a "
          "reactive policy (always straight) and a mis-generalizing monolithic WM fail. "
          "Non-exploitable: 12 grounded candidates, no optimization-through-model.")


if __name__ == "__main__":
    main()
