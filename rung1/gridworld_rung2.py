"""Rung 2 (clean substrate) — verification-reasoning on a DETERMINISTIC Sokoban-style grid.

No contact noise: a push either succeeds (destination free) or is blocked (occupied/wall) — clean
yes/no. This isolates the Rung-2 claim: a STRUCTURED world model generalizes its predictions to
novel arrangements (Rung 1), so an agent that VERIFIES candidate plans by simulating them with the
structured WM solves novel blocker-configs where (a) a reactive policy deadlocks and (b) the same
agent using a MONOLITHIC WM mis-predicts and fails.

Dynamics: agent moves in 4 directions; moving into object k pushes it one cell along the move IFF
the destination cell is in-bounds and unoccupied by another object (else nothing moves).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

DELTAS = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])  # up,down,left,right


class Grid:
    def __init__(self, size, agent, objects, goal=None, target=0):
        self.S = size
        self.agent = np.array(agent)
        self.obj = np.array(objects)            # (n,2)
        self.goal = None if goal is None else np.array(goal)
        self.target = target

    def occupied(self, cell, exclude=-1):
        for k, o in enumerate(self.obj):
            if k != exclude and np.array_equal(o, cell):
                return True
        return False

    def step(self, action):
        d = DELTAS[action]
        na = self.agent + d
        if np.any(na < 0) or np.any(na >= self.S):
            return                               # wall: agent blocked
        hit = [k for k, o in enumerate(self.obj) if np.array_equal(o, na)]
        if hit:
            k = hit[0]
            dest = self.obj[k] + d
            if np.all(dest >= 0) and np.all(dest < self.S) and not self.occupied(dest, exclude=k):
                self.obj[k] = dest               # push
                self.agent = na
            # else: blocked, nothing moves
        else:
            self.agent = na

    def state(self):
        return self.agent.copy(), self.obj.copy()


def random_grid(size, n_obj, rng):
    cells = rng.permutation(size * size)[: n_obj + 1]
    pts = np.array([[c // size, c % size] for c in cells])
    return Grid(size, pts[0], pts[1:])


def collect(size, n_obj, n_roll, horizon, seed):
    rng = np.random.default_rng(seed)
    A, O, ACT, ON = [], [], [], []
    for _ in range(n_roll):
        g = random_grid(size, n_obj, rng)
        for t in range(horizon):
            a = rng.integers(4)
            ag, ob = g.state()
            g.step(a)
            _, ob2 = g.state()
            A.append(ag); O.append(ob); ACT.append(a); ON.append(ob2)
    return {"agent": np.array(A, np.float32), "objects": np.array(O, np.float32),
            "action": np.array(ACT, np.int64), "next_objects": np.array(ON, np.float32)}


def mlp(sizes):
    L = []
    for i in range(len(sizes) - 1):
        L += [nn.Linear(sizes[i], sizes[i + 1])]
        if i < len(sizes) - 2:
            L += [nn.SiLU()]
    return nn.Sequential(*L)


def onehot(a, device):
    return torch.eye(4, device=device)[a]


class MonoWM(nn.Module):
    def __init__(self, n_max=8, hidden=256):
        super().__init__()
        self.n_max = n_max
        self.net = mlp([2 + 4 + n_max * 2, hidden, hidden, hidden, n_max * 2])

    def forward(self, agent, objects, action):
        B, n, _ = objects.shape
        obj = objects
        if n < self.n_max:
            obj = torch.cat([obj, torch.zeros(B, self.n_max - n, 2, device=obj.device)], 1)
        x = torch.cat([agent, onehot(action, agent.device), obj.reshape(B, -1)], -1)
        return self.net(x).reshape(B, self.n_max, 2)[:, :n]


class InterWM(nn.Module):
    """Shared per-object rule + pairwise (destination-occupancy) messages on relative cells."""
    def __init__(self, hidden=128, emb=64):
        super().__init__()
        self.enc = mlp([2 + 4, emb, emb])                  # obj pos + action
        self.agent_msg = mlp([emb + 2, hidden, emb])       # + rel(agent->obj)
        self.pair_msg = mlp([emb + 2, hidden, emb])        # + rel(other->obj)
        self.dec = mlp([emb + emb + emb, hidden, 2])

    def forward(self, agent, objects, action):
        B, n, _ = objects.shape
        act = onehot(action, agent.device).unsqueeze(1).expand(B, n, 4)
        h = self.enc(torch.cat([objects, act], -1))
        rel_a = objects - agent.unsqueeze(1)
        m_a = self.agent_msg(torch.cat([h, rel_a], -1))
        rel_ij = objects.unsqueeze(2) - objects.unsqueeze(1)   # (B,n,n,2)
        hj = h.unsqueeze(1).expand(B, n, n, h.shape[-1])
        pair = self.pair_msg(torch.cat([hj, rel_ij], -1))
        eye = torch.eye(n, device=h.device).view(1, n, n, 1)
        m_p = (pair * (1 - eye)).sum(2)
        return self.dec(torch.cat([h, m_a, m_p], -1))


def train(model, data, epochs=60, bs=256, lr=1e-3, dev="cpu"):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    d = {k: torch.tensor(v, device=dev) for k, v in data.items()}
    nT = len(d["agent"])
    for ep in range(epochs):
        idx = torch.randperm(nT)
        for i in range(0, nT - bs + 1, bs):
            j = idx[i : i + bs]
            tgt = d["next_objects"][j] - d["objects"][j]
            pred = model(d["agent"][j], d["objects"][j], d["action"][j])
            loss = ((pred - tgt) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    return model


@torch.no_grad()
def exact_acc(model, data, dev="cpu", push_only=False):
    """Exact next-cell accuracy. push_only: restrict to PUSH EVENTS (agent moves into an object) —
    the discriminating dynamics (whether the push succeeds depends on destination occupancy)."""
    d = {k: torch.tensor(v, device=dev) for k, v in data.items()}
    pred = (model(d["agent"], d["objects"], d["action"]) + d["objects"]).round()
    correct = (pred == d["next_objects"]).all(-1)            # (T,n) per object
    if not push_only:
        return correct.float().mean().item()
    # push event for object k: agent + delta == objects[k]
    agent = data["agent"]; objs = data["objects"]; act = data["action"]
    deltas = DELTAS[act]                                      # (T,2)
    na = agent + deltas                                      # (T,2)
    pushed = (np.abs(na[:, None, :] - objs).sum(-1) == 0)    # (T,n) which object the agent steps into
    mask = torch.tensor(pushed, device=dev)
    if mask.sum() == 0:
        return float("nan")
    return correct[mask].float().mean().item()


def main():
    torch.manual_seed(0); np.random.seed(0)
    S = 6
    print("collecting grid data (train K=3 only)...")
    train_data = collect(S, 3, 600, 20, 2)
    print(f"train transitions: {len(train_data['agent'])}")

    mono = train(MonoWM(n_max=8), train_data)
    inter = train(InterWM(), train_data)

    print(f"\n--- exact accuracy on PUSH EVENTS (the discriminating dynamics) ---")
    print(f"{'objects':10s} {'MonoWM':>10s} {'InterWM':>10s}")
    for k in [2, 3, 4, 5, 6, 8]:
        ed = collect(S, k, 300, 16, 50 + k)
        print(f"K={k:<8d} {exact_acc(mono, ed, push_only=True):>10.3f} "
              f"{exact_acc(inter, ed, push_only=True):>10.3f}")
    print("\nGate: on push events, InterWM stays ~exact on UNSEEN counts (K>=4) while MonoWM drops "
          "=> structured WM generalizes the (occupancy-dependent) push/block rule by construction.")


if __name__ == "__main__":
    main()


# ---------------- Rung 2: verification-planning ----------------
import itertools


@torch.no_grad()
def wm_sim_step(model, agent, objects, action, dev="cpu"):
    """One step using the WM for object dynamics + derived agent kinematics (deterministic grid)."""
    S_ = 6
    delta = DELTAS[action]
    na = agent + delta
    od = model(torch.tensor(agent[None], dtype=torch.float32),
               torch.tensor(objects[None], dtype=torch.float32),
               torch.tensor([action])).round().numpy()[0]
    new_obj = objects + od
    if np.any(na < 0) or np.any(na >= S_):
        return agent, objects
    hit = [k for k, o in enumerate(objects) if np.array_equal(o, na)]
    if hit:
        moved = not np.array_equal(new_obj[hit[0]], objects[hit[0]])
        return (na, new_obj) if moved else (agent, objects)
    return na, objects


def manhattan(a, b):
    return int(np.abs(np.array(a) - np.array(b)).sum())


def plan_action(model, agent, objects, target, goal, L=3):
    best_first, best = 0, 1e9
    for seq in itertools.product(range(4), repeat=L):
        ag, ob = agent.copy(), objects.copy()
        for a in seq:
            ag, ob = wm_sim_step(model, ag, ob, a)
        sc = manhattan(ob[target], goal) + 0.1 * manhattan(ag, ob[target])
        if sc < best:
            best, best_first = sc, seq[0]
    return best_first


def reactive_plan(agent, objects, target, goal):
    """Greedy: move to push target toward goal (no model). Picks the action reducing target->goal."""
    t = objects[target]
    d = np.sign(goal - t)
    # choose axis with larger gap; push from behind
    if abs(goal[0] - t[0]) >= abs(goal[1] - t[1]) and d[0] != 0:
        want = 1 if d[0] > 0 else 0           # down/up
    elif d[1] != 0:
        want = 3 if d[1] > 0 else 2           # right/left
    else:
        want = 0
    behind = t - DELTAS[want]
    if np.array_equal(agent, behind):
        return want                            # behind target -> push
    # else navigate toward behind cell (greedy)
    dd = np.sign(behind - agent)
    if dd[0] != 0:
        return 1 if dd[0] > 0 else 0
    return 3 if dd[1] > 0 else 2


def make_task(rng, n_blockers, block_between):
    S_ = 6
    # target left-ish, goal right-ish, agent somewhere
    t = np.array([rng.integers(1, 5), 1])
    goal = np.array([t[0], 4])
    objs = [t]
    if block_between:
        objs.append(np.array([t[0], 3]))       # blocker on the straight path
    for _ in range(n_blockers - (1 if block_between else 0)):
        while True:
            c = np.array([rng.integers(0, S_), rng.integers(0, S_)])
            if not any(np.array_equal(c, o) for o in objs) and not np.array_equal(c, goal):
                objs.append(c); break
    while True:
        ag = np.array([rng.integers(0, S_), rng.integers(0, S_)])
        if not any(np.array_equal(ag, o) for o in objs):
            break
    return Grid(S_, ag, np.array(objs), goal=goal, target=0)


def run_plan(g, policy, max_steps=40):
    for _ in range(max_steps):
        if np.array_equal(g.obj[g.target], g.goal):
            return True
        a = policy(g.agent, g.obj, g.target, g.goal)
        g.step(a)
    return np.array_equal(g.obj[g.target], g.goal)


def planning_eval(mono, inter, n=40):
    conds = {
        "no blocker":          dict(n_blockers=1, block_between=False),
        "BLOCKER on path":     dict(n_blockers=1, block_between=True),
        "blocker + 2 clutter": dict(n_blockers=3, block_between=True),
    }
    print(f"\n--- verification-planning success (deterministic grid) ---")
    print(f"{'condition':22s} {'Reactive':>10s} {'Plan-Mono':>10s} {'Plan-Inter':>11s}")
    for cname, cfg in conds.items():
        rng = np.random.default_rng(7)
        sc = {"react": 0, "mono": 0, "inter": 0}
        for _ in range(n):
            seed = int(rng.integers(1 << 30))
            for tag in ("react", "mono", "inter"):
                g = make_task(np.random.default_rng(seed), **cfg)
                if tag == "react":
                    pol = reactive_plan
                elif tag == "mono":
                    pol = lambda a, o, t, gl: plan_action(mono, a, o, t, gl)
                else:
                    pol = lambda a, o, t, gl: plan_action(inter, a, o, t, gl)
                sc[tag] += run_plan(g, pol)
        print(f"{cname:22s} {sc['react']/n:>10.0%} {sc['mono']/n:>10.0%} {sc['inter']/n:>11.0%}")
    print("\nGate: on NOVEL blocker arrangements, Plan-Inter > Reactive and > Plan-Mono => the "
          "structured WM enables verification-planning that routes around blockers (reactive "
          "deadlocks; monolithic mis-predicts). Non-exploitable: short-horizon enumeration, no grad opt.")


# ---------------- competent planner: BFS over the WM's dynamics ----------------
from collections import deque


def _key(agent, objects):
    return (int(agent[0]), int(agent[1])) + tuple(map(lambda o: (int(o[0]), int(o[1])), objects))


def bfs_plan(model, agent, objects, target, goal, max_nodes=4000):
    """BFS to push target to goal using `model` as the transition function. Returns first action."""
    start = (agent.copy(), objects.copy())
    if np.array_equal(objects[target], goal):
        return None
    visited = {_key(agent, objects)}
    q = deque([(agent, objects, None)])
    nodes = 0
    while q and nodes < max_nodes:
        ag, ob, first = q.popleft()
        for a in range(4):
            nag, nob = wm_sim_step(model, ag, ob, a)
            k = _key(nag, nob)
            if k in visited:
                continue
            visited.add(k); nodes += 1
            f = a if first is None else first
            if np.array_equal(nob[target], goal):
                return f
            q.append((nag, nob, f))
    return np.random.randint(4)   # no plan found -> arbitrary


def planning_eval_bfs(mono, inter, n=40):
    conds = {
        "no blocker":          dict(n_blockers=1, block_between=False),
        "BLOCKER on path":     dict(n_blockers=1, block_between=True),
        "blocker + 2 clutter": dict(n_blockers=3, block_between=True),
    }
    print(f"\n--- BFS verification-planning (plan with WM, execute in TRUE env, replan) ---")
    print(f"{'condition':22s} {'Plan-Mono':>10s} {'Plan-Inter':>11s} {'(oracle TrueWM)':>16s}")

    class TrueWM:
        def __call__(self, ag, ob, act):
            # ground-truth object delta (for oracle upper bound)
            import torch as T
            a = int(act[0]); ob0 = ob[0].numpy()
            g = Grid(6, ag[0].numpy().astype(int), ob0.astype(int).copy())
            g.step(a)
            return T.tensor((g.obj - ob0)[None], dtype=T.float32)
    true_wm = TrueWM()

    for cname, cfg in conds.items():
        rng = np.random.default_rng(7)
        sc = {"mono": 0, "inter": 0, "true": 0}
        for _ in range(n):
            seed = int(rng.integers(1 << 30))
            for tag, model in [("mono", mono), ("inter", inter), ("true", true_wm)]:
                g = make_task(np.random.default_rng(seed), **cfg)
                pol = lambda a, o, t, gl, m=model: bfs_plan(m, a, o, t, gl)
                sc[tag] += run_plan(g, pol, max_steps=40)
        print(f"{cname:22s} {sc['mono']/n:>10.0%} {sc['inter']/n:>11.0%} {sc['true']/n:>16.0%}")
    print("\nGate: Plan-Inter approaches the oracle (TrueWM) and >> Plan-Mono on NOVEL blocker "
          "arrangements => the structured WM's accurate generalization enables planning that works.")
