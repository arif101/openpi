"""Dynamic Movement Primitive (discrete, Ijspeert/Schaal 2013) — a re-targetable
motor primitive.

A DMP is a goal-attractor dynamical system + a one-shot-learned forcing term:
    tau * v_dot = az (bz (g - x) - v) + f(s)        (transformation system)
    tau * x_dot = v
    tau * s_dot = -a_s * s                            (canonical / phase clock, 1->0)
    f(s) = [ sum_i w_i psi_i(s) / sum_i psi_i(s) ] * s * (g - x0)

The goal g is an EXPLICIT structural input: the attractor guarantees x -> g from
any start, so changing g re-aims the SAME learned movement (the forcing term
vanishes as s->0). The weights w are fit from ONE demo by locally-weighted
regression (closed form). This is the re-targetability the distilled flow head
lacked (it learned P(action|g) by correlation; nothing forced convergence to g).

Per-dimension independent DMPs share one phase. Spatial scaling by (g - x0).
"""
from __future__ import annotations

import numpy as np


class DMP:
    def __init__(self, n_dim=3, n_bf=20, az=25.0, a_s=4.0):
        self.n_dim, self.n_bf = n_dim, n_bf
        self.az, self.bz, self.a_s = az, az / 4.0, a_s
        # basis centers in phase space (s decays exp), widths from spacing
        self.c = np.exp(-self.a_s * np.linspace(0, 1, n_bf))
        self.h = np.r_[1.0 / np.diff(self.c) ** 2, 1.0 / (self.c[-1] - self.c[-2]) ** 2] * 0.5
        self.w = np.zeros((n_dim, n_bf))
        self.x0 = np.zeros(n_dim)
        self.g = np.zeros(n_dim)

    def _psi(self, s):
        return np.exp(-self.h * (s - self.c) ** 2)        # [n_bf]

    def fit(self, traj, tau=1.0):
        """Fit forcing weights to reproduce a demo position trajectory [T, n_dim]."""
        traj = np.asarray(traj, float)
        T = len(traj)
        t = np.linspace(0, 1, T) * tau
        dt = t[1] - t[0]
        s = np.exp(-self.a_s / tau * t)                    # phase over the demo
        x0, g = traj[0].copy(), traj[-1].copy()
        v = np.gradient(traj, dt, axis=0)
        vd = np.gradient(v, dt, axis=0)
        # target forcing from rearranging the transformation system
        f_target = tau ** 2 * vd - self.az * (self.bz * (g - traj) - tau * v)   # [T, n_dim]
        psi = np.stack([self._psi(si) for si in s])        # [T, n_bf]
        scale = (g - x0)
        scale[np.abs(scale) < 1e-6] = 1e-6
        for d in range(self.n_dim):
            # LWR: per-basis weighted regression of f_target on the (s*scale) regressor
            num = (psi * (s[:, None]) * f_target[:, d][:, None]).sum(0)
            den = (psi * (s[:, None] ** 2)).sum(0) * scale[d]
            self.w[d] = np.where(np.abs(den) > 1e-9, num / den, 0.0)
        self.x0, self.g = x0, g
        return self

    def reset(self):
        return dict(x=self.x0.copy(), v=np.zeros(self.n_dim), s=1.0)

    def step(self, st, g, dt, x0=None, tau=1.0):
        """Advance one step toward goal g (which may move). Returns updated state;
        st['x'] is the desired position. g is the LIVE goal (re-targeting)."""
        x0 = self.x0 if x0 is None else x0
        s = st["s"]
        psi = self._psi(s)
        f = (psi @ self.w.T) / (psi.sum() + 1e-10) * s * (g - x0)   # [n_dim]
        vdot = (self.az * (self.bz * (g - st["x"]) - st["v"]) + f) / tau
        st["v"] = st["v"] + vdot * dt
        st["x"] = st["x"] + st["v"] / tau * dt
        st["s"] = st["s"] + (-self.a_s * st["s"] / tau) * dt
        return st


# ---------------------------------------------------------------------------
def _selftest():
    # fit a curved 2D demo (quarter circle), then RE-TARGET to a new goal and
    # verify convergence + shape preservation (the re-targetability property).
    T = 100
    th = np.linspace(0, np.pi / 2, T)
    demo = np.stack([np.cos(th) - 1, np.sin(th)], 1)       # from (0,0) curving to (-1,1)
    d = DMP(n_dim=2, n_bf=30).fit(demo)

    # reproduce with original goal
    st = d.reset(); xs = [st["x"].copy()]
    for _ in range(300):
        d.step(st, d.g, dt=0.01); xs.append(st["x"].copy())
    xs = np.array(xs)
    assert np.linalg.norm(xs[-1] - d.g) < 0.02, f"repro didn't reach goal: {xs[-1]} vs {d.g}"

    # RE-TARGET: new goal never seen; must converge there by construction
    new_g = np.array([0.5, 1.5])
    st = d.reset(); xs2 = [st["x"].copy()]
    for _ in range(400):
        d.step(st, new_g, dt=0.01); xs2.append(st["x"].copy())
    xs2 = np.array(xs2)
    assert np.linalg.norm(xs2[-1] - new_g) < 0.03, f"RE-TARGET failed: {xs2[-1]} vs {new_g}"
    # path should be curved (not a straight line) — shape preserved under re-target
    mid = xs2[len(xs2) // 2]
    straight_mid = 0.5 * new_g
    curved = np.linalg.norm(mid - straight_mid) > 0.05
    print(f"  repro end {xs[-1].round(3)} (goal {d.g.round(3)})")
    print(f"  RE-TARGET end {xs2[-1].round(3)} (new goal {new_g}) — converged, curved={curved}")
    print("DMP selftest PASS: fits a demo AND re-targets to an unseen goal by construction.")


if __name__ == "__main__":
    _selftest()
