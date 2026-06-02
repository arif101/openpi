"""Rung 4 — surprise-gated STRUCTURAL memory: corrections that GENERALIZE across episodes.

Rung 3 fixed the WM's errors online, but keyed corrections on the EXACT global (state,action) — so a
correction learned in one episode does NOT transfer to a new arrangement. Rung 4 keys the correction
on the LOCAL, translation/permutation-invariant interaction config around the pushed object (relative
offsets of nearby objects + wall-proximity in the push direction). A correction learned once ("pushing
into an adjacent object -> blocked") then applies to ANY future arrangement with the same local config.

Test of GENERALIZATION (the point): build memory on TRAIN episodes, FREEZE it, evaluate on HELD-OUT
episodes with NO further writes.
  WM-only           : structured WM, no memory (= Rung 2)            -> fails on hard configs
  exact-key (Rung3) : memory keyed on global state, frozen          -> no transfer (keys never match)
  struct-key (Rung4): memory keyed on local interaction config, frozen -> TRANSFERS to new arrangements

This is surprise-gated STRUCTURAL retrieval (the unoccupied wedge) on the world model — and it merges
the WM program with the hippocampus/structural-key (Exp 2a) thread.
"""
from __future__ import annotations

import numpy as np
import torch
from collections import deque

from gridworld_rung2 import (Grid, DELTAS, collect, InterWM, train, make_task, _key)

S = 6
WIN = 2  # Chebyshev window for the local structural key


def struct_key(agent, objects, action):
    """Local, translation+permutation-invariant key for the PUSHED object's interaction config.
    Returns None if this action doesn't push an object (WM gets free moves right)."""
    d = DELTAS[action]
    na = agent + d
    hit = [k for k, o in enumerate(objects) if np.array_equal(o, na)]
    if not hit:
        return None
    k = hit[0]
    p = objects[k]
    rel = []
    for j, o in enumerate(objects):
        if j == k:
            continue
        off = o - p
        if np.max(np.abs(off)) <= WIN:
            rel.append((int(off[0]), int(off[1])))         # relative offset (translation-invariant)
    # wall distance in the push direction (clipped) — captures "pushed against a wall"
    dest = p + d
    wall = 0 if (np.any(dest < 0) or np.any(dest >= S)) else 1
    return (int(action), frozenset(rel), wall)


class StructMem:
    def __init__(self, wm, mode="struct"):
        self.wm = wm
        self.mode = mode                 # "wm" | "exact" | "struct"
        self.mem = {}
        self.frozen = False
        self.n = 0

    @torch.no_grad()
    def wm_delta(self, agent, objects, action):
        return (self.wm(torch.tensor(agent[None], dtype=torch.float32),
                        torch.tensor(objects[None], dtype=torch.float32),
                        torch.tensor([action])).round().numpy()[0])

    def _key(self, agent, objects, action):
        if self.mode == "exact":
            return _key(agent, objects) + (action,)
        return struct_key(agent, objects, action)            # struct

    def delta(self, agent, objects, action):
        od = self.wm_delta(agent, objects, action)
        if self.mode != "wm":
            kk = self._key(agent, objects, action)
            if kk is not None and kk in self.mem:
                # override the PUSHED object's delta with the remembered truth
                d = DELTAS[action]; na = agent + d
                hit = [i for i, o in enumerate(objects) if np.array_equal(o, na)]
                if hit:
                    od = od.copy(); od[hit[0]] = self.mem[kk]
        return od

    def observe(self, agent, objects, action, true_next_objects):
        if self.mode == "wm" or self.frozen:
            return
        pred = self.wm_delta(agent, objects, action)
        true_delta = true_next_objects - objects
        d = DELTAS[action]; na = agent + d
        hit = [i for i, o in enumerate(objects) if np.array_equal(o, na)]
        if hit and not np.array_equal(pred[hit[0]], true_delta[hit[0]]):   # surprise on the pushed object
            kk = self._key(agent, objects, action)
            if kk is not None:
                self.mem[kk] = true_delta[hit[0]]
                self.n += 1


def sim_step(model, agent, objects, action):
    delta = DELTAS[action]; na = agent + delta
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
    visited = {_key(agent, objects)}; q = deque([(agent, objects, None)]); nodes = 0
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


def run(g, model, learn, max_steps=40):
    model.frozen = not learn
    for _ in range(max_steps):
        if np.array_equal(g.obj[g.target], g.goal):
            return True
        a = bfs_first(model, g.agent, g.obj, g.target, g.goal)
        ag0, ob0 = g.agent.copy(), g.obj.copy()
        g.step(a)
        if learn:
            model.observe(ag0, ob0, a, g.obj.copy())
    return np.array_equal(g.obj[g.target], g.goal)


def main():
    torch.manual_seed(0); np.random.seed(0)
    inter = train(InterWM(), collect(6, 3, 600, 20, 2))

    cfg = dict(n_blockers=3, block_between=True)        # the hardest condition (Rung 2/3: 0% / 70%)
    n_train, n_test = 60, 60

    print("building memory on TRAIN episodes, then FROZEN eval on HELD-OUT episodes (hardest config)\n")
    print(f"{'memory mode':16s} {'held-out success':>18s} {'mem entries':>12s}")
    for mode in ["wm", "exact", "struct"]:
        m = StructMem(inter, mode=mode)
        # train phase: accumulate memory (learn=True)
        rng = np.random.default_rng(1000)
        for _ in range(n_train):
            g = make_task(np.random.default_rng(int(rng.integers(1 << 30))), **cfg)
            run(g, m, learn=True)
        entries = len(m.mem)
        # test phase: FROZEN memory, held-out episodes (different seeds), no writes
        rng = np.random.default_rng(9999)
        succ = 0
        for _ in range(n_test):
            g = make_task(np.random.default_rng(int(rng.integers(1 << 30))), **cfg)
            succ += run(g, m, learn=False)
        print(f"{mode:16s} {succ / n_test:>18.0%} {entries:>12d}")
    print("\nGate: struct-key memory TRANSFERS to held-out arrangements (>> wm-only and >> exact-key, "
          "which can't transfer because global keys never recur). Surprise-gated STRUCTURAL memory = "
          "corrections that generalize. Merges WM program + hippocampus/structural-key thread.")


if __name__ == "__main__":
    main()
