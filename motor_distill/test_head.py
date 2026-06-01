"""Tests for the distilled motor head. Run: python motor_distill/test_head.py

The load-bearing test is EQUIVARIANCE BY CONSTRUCTION: under a global yaw of the
scene, the equivariant head's conditioning is invariant and its output chunk
rotates by the same yaw — without any training. Plus a tiny train-convergence
smoke so we know the flow objective actually fits.
"""
from __future__ import annotations

import torch

import head as H


def _sample(B=4, Hh=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    return dict(
        g_pos=torch.randn(B, 3, generator=g),
        g_quat=torch.nn.functional.normalize(torch.randn(B, 4, generator=g), dim=-1),
        proprio=torch.randn(B, 9, generator=g),
        chunk=torch.randn(B, Hh, 7, generator=g),
        obj_pos=torch.randn(B, 3, generator=g),
        obj_quat=torch.nn.functional.normalize(torch.randn(B, 4, generator=g), dim=-1),
    )


def _rotate_scene(s, theta):
    """Apply a global yaw of theta about the world z-axis to a sample."""
    th = torch.full((s["g_pos"].shape[0],), theta)
    yq = H.yaw_quat(th)
    out = dict(s)
    out["obj_pos"] = H.rotz(s["obj_pos"], th)
    out["obj_quat"] = H.quat_mul(yq, s["obj_quat"])
    ee_pos, ee_quat, grip = s["proprio"][:, :3], s["proprio"][:, 3:7], s["proprio"][:, 7:9]
    out["proprio"] = torch.cat([H.rotz(ee_pos, th), H.quat_mul(yq, ee_quat), grip], dim=-1)
    # g is object-relative -> unchanged under global yaw
    return out, th


def test_canonical_roundtrip():
    s = _sample()
    psi = H.quat_yaw(s["obj_quat"])
    c = H.chunk_to_canonical(s["chunk"], psi)
    back = H.chunk_from_canonical(c, psi)
    assert torch.allclose(back, s["chunk"], atol=1e-5)


def test_cond_invariant_under_yaw_equivariant():
    s = _sample()
    cond0, _ = H.cond_vector(s["g_pos"], s["g_quat"], s["proprio"], s["obj_pos"], s["obj_quat"], True)
    for theta in (0.4, 1.7, -2.3):
        sr, _ = _rotate_scene(s, theta)
        cond, _ = H.cond_vector(sr["g_pos"], sr["g_quat"], sr["proprio"], sr["obj_pos"], sr["obj_quat"], True)
        assert torch.allclose(cond, cond0, atol=1e-5), f"equivariant cond moved at {theta}"


def test_cond_NOT_invariant_plain():
    """Sanity that the ablation differs: plain conditioning DOES change with yaw."""
    s = _sample()
    cond0, _ = H.cond_vector(s["g_pos"], s["g_quat"], s["proprio"], s["obj_pos"], s["obj_quat"], False)
    sr, _ = _rotate_scene(s, 1.0)
    cond, _ = H.cond_vector(sr["g_pos"], sr["g_quat"], sr["proprio"], sr["obj_pos"], sr["obj_quat"], False)
    assert not torch.allclose(cond, cond0, atol=1e-3)


def test_output_equivariance_by_construction():
    """Untrained equivariant head: rotating the scene by theta rotates the
    sampled world chunk's pos/rot deltas by theta (gripper unchanged)."""
    torch.manual_seed(0)
    net = H.MotorHead(); net.equivariant = True
    s = _sample(B=4)
    torch.manual_seed(123); out0 = net.sample(s["g_pos"], s["g_quat"], s["proprio"], s["obj_pos"], s["obj_quat"], steps=8)
    theta = 0.9
    sr, th = _rotate_scene(s, theta)
    torch.manual_seed(123); out1 = net.sample(sr["g_pos"], sr["g_quat"], sr["proprio"], sr["obj_pos"], sr["obj_quat"], steps=8)
    # out1 pos/rot deltas should equal out0 rotated by +theta; gripper equal
    p = th[:, None].expand(-1, out0.shape[1])
    exp_pos = H.rotz(out0[..., :3], p)
    exp_rot = H.rotz(out0[..., 3:6], p)
    assert torch.allclose(out1[..., :3], exp_pos, atol=1e-4), "pos delta not equivariant"
    assert torch.allclose(out1[..., 3:6], exp_rot, atol=1e-4), "rot delta not equivariant"
    assert torch.allclose(out1[..., 6], out0[..., 6], atol=1e-4), "gripper should be yaw-invariant"


def test_train_converges():
    """Synthetic: canonical chunk is a fixed linear map of canonical cond.
    Flow loss should drop substantially in a few hundred steps."""
    torch.manual_seed(0)
    net = H.MotorHead(hidden=128); net.equivariant = False
    B = 256
    W = torch.randn(16, 16 * 7) * 0.3
    s = _sample(B=B, seed=7)
    # build a learnable target: canonical chunk = (cond @ W) broadcast over H
    cond, psi = H.cond_vector(s["g_pos"], s["g_quat"], s["proprio"], s["obj_pos"], s["obj_quat"], False)
    base = (cond @ W).reshape(B, 16, 7)
    s["chunk"] = H.chunk_from_canonical(base, psi)        # world chunk (plain psi=0 -> identity)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    losses = []
    for it in range(400):
        opt.zero_grad()
        l = net.loss(s["g_pos"], s["g_quat"], s["proprio"], s["chunk"], s["obj_pos"], s["obj_quat"])
        l.backward(); opt.step()
        losses.append(l.item())
    assert losses[-1] < 0.5 * losses[0], f"flow loss did not drop: {losses[0]:.3f}->{losses[-1]:.3f}"
    print(f"  train smoke: loss {losses[0]:.3f} -> {losses[-1]:.3f}")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"\nAll {len(tests)} head tests passed.")
