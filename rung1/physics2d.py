"""Lightweight 2D multi-object pusher physics — clean substrate for testing whether a
world model generalizes by STRUCTURE (per-object transferable dynamics) vs COVERAGE
(memorized whole-scene configs).

No rendering, no GPU. State is low-dim so we isolate the structural question from perception.
A pusher (position-controlled) shoves free disks around; disks collide with pusher, each other,
and walls, with friction. The dynamics are the SAME local rule for every disk — exactly the
structure an object-centric world model should capture and a monolithic one should miss.
"""
from __future__ import annotations

import numpy as np

BOUND = 1.0
PUSH_R = 0.08
STEP = 0.05          # pusher max move per step
FRICTION = 0.85
RESTITUTION = 0.4


class World:
    def __init__(self, n_objects: int, radii=None, seed=0):
        self.n = n_objects
        self.rng = np.random.default_rng(seed)
        self.r = np.asarray(radii, float) if radii is not None else self.rng.uniform(0.05, 0.10, n_objects)
        self.mass = (self.r / 0.07) ** 2          # mass ∝ size² → size genuinely affects dynamics
        self.reset()

    def reset(self):
        self.pusher = self.rng.uniform(-0.8, 0.8, 2)
        # place objects without heavy overlap
        pos = []
        for _ in range(self.n):
            for _try in range(50):
                p = self.rng.uniform(-0.7, 0.7, 2)
                if all(np.linalg.norm(p - q) > self.r[i] + self.r[len(pos)] + 0.02 for i, q in enumerate(pos)):
                    break
            pos.append(p)
        self.pos = np.array(pos) if self.n else np.zeros((0, 2))
        self.vel = np.zeros((self.n, 2))
        return self.state()

    def state(self):
        # per-object [x, y, vx, vy]; plus pusher pos and radii returned separately
        return {
            "pusher": self.pusher.copy(),
            "objects": np.concatenate([self.pos, self.vel], axis=1),  # (n,4)
            "radii": self.r.copy(),
        }

    def step(self, action):
        action = np.clip(action, -BOUND, BOUND)
        # move pusher toward target (bounded)
        d = action - self.pusher
        dist = np.linalg.norm(d) + 1e-9
        self.pusher = self.pusher + d / dist * min(STEP, dist)

        # pusher -> object contact: push radially out + impart velocity
        for i in range(self.n):
            off = self.pos[i] - self.pusher
            dd = np.linalg.norm(off) + 1e-9
            overlap = (PUSH_R + self.r[i]) - dd
            if overlap > 0:
                n = off / dd
                self.pos[i] += n * overlap
                self.vel[i] += n * overlap * 3.0 / self.mass[i]   # heavier objects move less

        # object-object contact: separate + simple velocity exchange
        for i in range(self.n):
            for j in range(i + 1, self.n):
                off = self.pos[i] - self.pos[j]
                dd = np.linalg.norm(off) + 1e-9
                overlap = (self.r[i] + self.r[j]) - dd
                if overlap > 0:
                    n = off / dd
                    mi, mj = self.mass[i], self.mass[j]
                    self.pos[i] += n * overlap * mj / (mi + mj)
                    self.pos[j] -= n * overlap * mi / (mi + mj)
                    vrel = (self.vel[i] - self.vel[j]) @ n
                    if vrel < 0:
                        imp = -(1 + RESTITUTION) * vrel / (1 / mi + 1 / mj)  # momentum-conserving
                        self.vel[i] += imp / mi * n
                        self.vel[j] -= imp / mj * n

        # integrate + friction + walls
        self.pos += self.vel
        self.vel *= FRICTION
        for i in range(self.n):
            for k in range(2):
                lim = BOUND - self.r[i]
                if self.pos[i, k] > lim:
                    self.pos[i, k] = lim; self.vel[i, k] *= -RESTITUTION
                if self.pos[i, k] < -lim:
                    self.pos[i, k] = -lim; self.vel[i, k] *= -RESTITUTION
        return self.state()


def collect(n_objects, n_rollouts, horizon, seed=0, radii_sampler=None):
    """Return transition arrays. Pusher policy: random walk toward sampled targets (causes contacts)."""
    rng = np.random.default_rng(seed)
    P, A, O, R, Onext = [], [], [], [], []
    for r in range(n_rollouts):
        radii = radii_sampler(rng) if radii_sampler else None
        w = World(n_objects, radii=radii, seed=int(rng.integers(1 << 30)))
        s = w.state()
        target = rng.uniform(-0.8, 0.8, 2)
        for t in range(horizon):
            if t % 5 == 0:
                # aim near a random object to force interactions
                if n_objects and rng.random() < 0.7:
                    target = s["objects"][rng.integers(n_objects), :2] + rng.normal(0, 0.05, 2)
                else:
                    target = rng.uniform(-0.8, 0.8, 2)
            a = np.clip(target, -BOUND, BOUND)
            ns = w.step(a)
            P.append(s["pusher"]); A.append(a)
            O.append(s["objects"]); R.append(s["radii"]); Onext.append(ns["objects"])
            s = ns
    return {
        "pusher": np.array(P, np.float32),       # (T,2)
        "action": np.array(A, np.float32),       # (T,2)
        "objects": np.array(O, np.float32),      # (T,n,4)
        "radii": np.array(R, np.float32),        # (T,n)
        "next_objects": np.array(Onext, np.float32),  # (T,n,4)
    }


def collect_rollouts(n_objects, n_rollouts, horizon, seed=0, radii_sampler=None):
    """Rollout-structured (R, T, ...) data for multi-step world-model simulation eval."""
    rng = np.random.default_rng(seed)
    P, A, O, R, Onext = [], [], [], [], []
    for r in range(n_rollouts):
        radii = radii_sampler(rng) if radii_sampler else None
        w = World(n_objects, radii=radii, seed=int(rng.integers(1 << 30)))
        s = w.state()
        target = rng.uniform(-0.8, 0.8, 2)
        rp, ra, ro, rr, rn = [], [], [], [], []
        for t in range(horizon):
            if t % 5 == 0:
                if n_objects and rng.random() < 0.8:
                    target = s["objects"][rng.integers(n_objects), :2] + rng.normal(0, 0.04, 2)
                else:
                    target = rng.uniform(-0.8, 0.8, 2)
            a = np.clip(target, -BOUND, BOUND)
            ns = w.step(a)
            rp.append(s["pusher"]); ra.append(a); ro.append(s["objects"]); rr.append(s["radii"]); rn.append(ns["objects"])
            s = ns
        P.append(rp); A.append(ra); O.append(ro); R.append(rr); Onext.append(rn)
    return {
        "pusher": np.array(P, np.float32), "action": np.array(A, np.float32),
        "objects": np.array(O, np.float32), "radii": np.array(R, np.float32),
        "next_objects": np.array(Onext, np.float32),
    }


if __name__ == "__main__":
    d = collect(n_objects=3, n_rollouts=20, horizon=24, seed=0)
    disp = np.linalg.norm(d["next_objects"][..., :2] - d["objects"][..., :2], axis=-1)
    print("transitions:", d["objects"].shape, "| mean per-object motion:", disp.mean().round(4),
          "| frac moving>0.01:", (disp > 0.01).mean().round(2))
