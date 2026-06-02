"""Rung 3 — surprise-gated online correction: metacognition makes an imperfect world model usable.

Rung 2 boundary: the structured WM errs on hard multi-step blocker plans (22%/2% vs oracle 100/95),
because its residual prediction error compounds. Rung 3 fixes this WITHOUT retraining: as the agent
acts, it compares the WM's prediction to reality. When they diverge (epsilon high = SURPRISE), it
writes the true outcome to memory and plans with the corrected model. Surprise-gated memory turns
"the WM is wrong here" into "I now know the truth here" — online, grounded, non-exploitable.

Ablations:
  WM-only        : plan with the structured WM, no correction (= Rung 2).
  +surprise-mem  : on epsilon-high steps, write true (state,action)->outcome to memory; plan with it.
  oracle (TrueWM): perfect model upper bound.
"""
from __future__ import annotations

import numpy as np
import torch
from collections import deque

from gridworld_rung2 import (Grid, DELTAS, collect, MonoWM, InterWM, train, make_task, _key)


class Corrected:
    """WM + a memory of surprise-triggered corrections. predict() uses memory if present, else WM."""
    def __init__(self, wm, use_memory=True):
        self.wm = wm
        self.mem = {}
        self.use_memory = use_memory
        self.n_writes = 0

    @torch.no_grad()
    def wm_delta(self, agent, objects, action):
        return (self.wm(torch.tensor(agent[None], dtype=torch.float32),
                        torch.tensor(objects[None], dtype=torch.float32),
                        torch.tensor([action])).round().numpy()[0])

    def delta(self, agent, objects, action):
        if self.use_memory:
            k = _key(agent, objects) + (action,)
            if k in self.mem:
                return self.mem[k]
        return self.wm_delta(agent, objects, action)

    def observe(self, agent, objects, action, true_next_objects):
        """Surprise gate: if WM prediction != reality, store the truth (epsilon-triggered write)."""
        if not self.use_memory:
            return
        pred = self.wm_delta(agent, objects, action)
        true_delta = true_next_objects - objects
        if not np.array_equal(pred, true_delta):          # epsilon high -> surprise
            self.mem[_key(agent, objects) + (action,)] = true_delta
            self.n_writes += 1


def sim_step(model, agent, objects, action, S=6):
    delta = DELTAS[action]
    na = agent + delta
    od = model.delta(agent, objects, action)
    new_obj = objects + od
    if np.any(na < 0) or np.any(na >= S):
        return agent, objects
    hit = [k for k, o in enumerate(objects) if np.array_equal(o, na)]
    if hit:
        moved = not np.array_equal(new_obj[hit[0]], objects[hit[0]])
        return (na, new_obj) if moved else (agent, objects)
    return na, objects


def bfs_first(model, agent, objects, target, goal, max_nodes=4000):
    if np.array_equal(objects[target], goal):
        return None
    visited = {_key(agent, objects)}
    q = deque([(agent, objects, None)])
    nodes = 0
    while q and nodes < max_nodes:
        ag, ob, first = q.popleft()
        for a in range(4):
            nag, nob = sim_step(model, ag, ob, a)
            k = _key(nag, nob)
            if k in visited:
                continue
            visited.add(k); nodes += 1
            f = a if first is None else first
            if np.array_equal(nob[target], goal):
                return f
            q.append((nag, nob, f))
    return np.random.randint(4)


def run(g, model, max_steps=40):
    for _ in range(max_steps):
        if np.array_equal(g.obj[g.target], g.goal):
            return True
        a = bfs_first(model, g.agent, g.obj, g.target, g.goal)
        ag0, ob0 = g.agent.copy(), g.obj.copy()
        g.step(a)
        model.observe(ag0, ob0, a, g.obj.copy())          # surprise-gated memory write
    return np.array_equal(g.obj[g.target], g.goal)


class TrueWM:
    def delta(self, agent, objects, action):
        gg = Grid(6, agent.astype(int), objects.astype(int).copy())
        gg.step(action)
        return gg.obj - objects
    def observe(self, *a):
        pass


def main():
    torch.manual_seed(0); np.random.seed(0)
    td = collect(6, 3, 600, 20, 2)
    inter = train(InterWM(), td)

    conds = {
        "no blocker":          dict(n_blockers=1, block_between=False),
        "BLOCKER on path":     dict(n_blockers=1, block_between=True),
        "blocker + 2 clutter": dict(n_blockers=3, block_between=True),
    }
    N = 40
    print(f"{'condition':22s} {'WM-only':>9s} {'+surprise-mem':>14s} {'oracle':>8s} {'(mem writes/ep)':>16s}")
    for cname, cfg in conds.items():
        rng = np.random.default_rng(7)
        sc = {"wm": 0, "mem": 0, "true": 0}
        writes = []
        for _ in range(N):
            seed = int(rng.integers(1 << 30))
            g = make_task(np.random.default_rng(seed), **cfg); sc["wm"] += run(g, Corrected(inter, use_memory=False))
            g = make_task(np.random.default_rng(seed), **cfg)
            cm = Corrected(inter, use_memory=True); sc["mem"] += run(g, cm); writes.append(cm.n_writes)
            g = make_task(np.random.default_rng(seed), **cfg); sc["true"] += run(g, TrueWM())
        print(f"{cname:22s} {sc['wm']/N:>9.0%} {sc['mem']/N:>14.0%} {sc['true']/N:>8.0%} {np.mean(writes):>16.1f}")
    print("\nGate: +surprise-mem recovers hard-case planning toward the oracle, vs WM-only "
          "(= Rung 2). Surprise-gated memory makes the imperfect WM usable online, non-exploitably "
          "(write the truth where the model was wrong; no retraining, no optimization-through-model).")


if __name__ == "__main__":
    main()
