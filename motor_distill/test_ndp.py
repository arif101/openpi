"""Unit tests for the NDP. The decisive one (test_ndp_retargets_OOD_plain_does_not)
is a CPU mini-version of committed-program Stage 1: train NDP and PlainHead on the
SAME in-range goals, then test on OUT-OF-RANGE goals. The attractor head should
re-target by construction; the plain head should memorize the training range.

Run: python motor_distill/test_ndp.py
"""
from __future__ import annotations

import numpy as np
import torch

from ndp import DMPLayer, NDP, PlainHead

torch.manual_seed(0)


def test_attractor_converges_to_goal():
    # zero forcing -> pure spring -> converges to g from any x0, for ANY g.
    L = DMPLayer(n_dim=2, n_bf=10, T=60)
    B = 8
    w = torch.zeros(B, 2, 10)
    g = torch.randn(B, 2) * 3
    x0 = torch.randn(B, 2) * 3
    traj = L(w, g, x0)
    err = (traj[:, -1] - g).norm(dim=-1)
    assert err.max() < 0.05, f"attractor didn't converge: max err {err.max():.3f}"


def test_gradients_flow():
    net = NDP(cond_dim=4, n_dim=2, n_bf=12, T=24)
    cond = torch.randn(5, 4); g = torch.randn(5, 2); x0 = torch.randn(5, 2)
    out = net(cond, g, x0)
    out.sum().backward()
    grads = [p.grad.abs().sum().item() for p in net.enc.parameters() if p.grad is not None]
    assert len(grads) > 0 and max(grads) > 0, "no gradient reached the forcing net"


def _make_task(goals, seed=0):
    """Synthetic: trajectory from x0=origin to goal with a FIXED curved style
    (a quarter-circle bulge). cond encodes the goal direction (the 'perception')."""
    g = torch.manual_seed(seed)
    n = len(goals)
    x0 = torch.zeros(n, 2)
    # curved demo: straight line to goal + a perpendicular sinusoidal bulge
    T = 24
    ts = torch.linspace(0, 1, T)
    trajs = []
    for gg in goals:
        line = ts[:, None] * gg[None, :]                         # [T,2]
        perp = torch.tensor([-gg[1], gg[0]])
        perp = perp / (perp.norm() + 1e-6)
        bulge = 0.25 * torch.sin(np.pi * ts)[:, None] * perp[None, :]
        trajs.append(line + bulge)
    trajs = torch.stack(trajs)                                   # [n,T,2]
    cond = goals / (goals.norm(dim=-1, keepdim=True) + 1e-6)     # direction only (NOT magnitude)
    return cond, goals, x0, trajs


def _train(model, cond, g, x0, target, steps=1500, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        loss = ((model(cond, g, x0) - target) ** 2).mean()
        loss.backward(); opt.step()
    return loss.item()


def test_ndp_retargets_OOD_plain_does_not():
    """THE decisive unit test. Train on goals at radius ~1; test at radius ~2.5
    (out of range). NDP must still land near the OOD goal (attractor); PlainHead
    should undershoot/miss (memorized training radius)."""
    # training goals: radius ~1, varied angle
    ang = torch.linspace(0, 2 * np.pi, 40)
    train_g = torch.stack([torch.cos(ang), torch.sin(ang)], 1) * (1.0 + 0.1 * torch.randn(40, 1))
    cond, g, x0, tgt = _make_task(train_g, seed=1)

    ndp = NDP(cond_dim=2, n_dim=2, n_bf=16, T=24)
    plain = PlainHead(cond_dim=2, n_dim=2, T=24)
    ln = _train(ndp, cond, g, x0, tgt)
    lp = _train(plain, cond, g, x0, tgt)
    assert ln < 0.05 and lp < 0.05, f"a head didn't fit in-range: ndp {ln:.3f} plain {lp:.3f}"

    # OOD goals: radius ~2.5 (NEVER seen), same angles
    ang2 = torch.linspace(0.1, 2 * np.pi - 0.1, 20)
    ood_g = torch.stack([torch.cos(ang2), torch.sin(ang2)], 1) * 2.5
    cond2 = ood_g / ood_g.norm(dim=-1, keepdim=True)
    x02 = torch.zeros(20, 2)
    with torch.no_grad():
        ndp_end = ndp(cond2, ood_g, x02)[:, -1]
        plain_end = plain(cond2, ood_g, x02)[:, -1]
    ndp_err = (ndp_end - ood_g).norm(dim=-1).mean().item()
    plain_err = (plain_end - ood_g).norm(dim=-1).mean().item()
    print(f"  in-range fit: ndp {ln:.4f} plain {lp:.4f}")
    print(f"  OOD goal (r=2.5) endpoint error:  NDP {ndp_err:.3f}  vs  Plain {plain_err:.3f}")
    # FINDING (honest): the NDP re-targets OOD by construction (attractor guarantees it).
    assert ndp_err < 0.15, f"NDP failed to re-target OOD: {ndp_err:.3f}"
    # FINDING (the sharpener): on a SMOOTH task, a PLAIN head GIVEN the goal as an explicit
    # input ALSO extrapolates to OOD goals — the attractor structure is NOT necessary here.
    # => The lever for curing the lookup table is likely PHYSICAL-STATE CONDITIONING (give the
    # head a live goal/object input at all), NOT the attractor per se. The attractor's value, if
    # any, must show up in the NONLINEAR / closed-loop / imperfect-forcing regime (the real grasp
    # test on GPU), not on this smooth toy. Stage 1 must therefore be a 3-arm comparison:
    #   appearance-conditioned (pi0.5-like, no goal) vs plain physical-goal-conditioned vs NDP.
    assert plain_err < 0.20, f"unexpected: plain did NOT extrapolate ({plain_err:.3f}) — re-examine"
    print(f"  => BOTH re-target given the goal (NDP {ndp_err:.3f}, Plain {plain_err:.3f}). "
          f"On smooth tasks the attractor is NOT the lever; physical-goal CONDITIONING is. "
          f"Attractor's value (if any) is in the nonlinear/closed-loop regime -> GPU Stage 1, 3 arms.")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t(); print(f"PASS {t.__name__}")
    print(f"\nAll {len(tests)} NDP tests passed.")
