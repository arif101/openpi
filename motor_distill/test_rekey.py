"""Unit tests for hindsight re-keying — pins the SE(3) math and the convention
gotcha before any GPU collection. Run: python motor_distill/test_rekey.py
"""
from __future__ import annotations

import tempfile

import numpy as np

import rekey as rk


def _approx(a, b, tol=1e-9):
    return np.allclose(np.asarray(a), np.asarray(b), atol=tol)


def test_quat_conventions():
    # scalar-last identity [x,y,z,w]=[0,0,0,1] -> scalar-first [1,0,0,0]
    assert _approx(rk.quat_xyzw_to_wxyz([0, 0, 0, 1]), [1, 0, 0, 0])
    # 90deg about z, scalar-last -> scalar-first
    th = np.pi / 2
    xyzw = [0, 0, np.sin(th / 2), np.cos(th / 2)]
    assert _approx(rk.quat_xyzw_to_wxyz(xyzw), [np.cos(th / 2), 0, 0, np.sin(th / 2)])


def test_rotate_and_yaw():
    q = rk.yaw_quat(np.pi / 2)                      # +90deg about z
    assert _approx(rk.quat_rotate(q, [1, 0, 0]), [0, 1, 0])   # x -> y
    assert abs(rk.quat_yaw(q) - np.pi / 2) < 1e-9


def test_ee_in_object_frame_basic():
    # object at origin, no rotation; EE 0.1 in +x. rel pos should be [0.1,0,0].
    rp, rq = rk.ee_in_object_frame([0.1, 0, 0], [1, 0, 0, 0], [0, 0, 0], [1, 0, 0, 0])
    assert _approx(rp, [0.1, 0, 0])
    assert _approx(rq, [1, 0, 0, 0])

    # object rotated +90deg about z, EE 0.1 in WORLD +x.
    # In the object frame (rotated +90), world +x maps to object -y.
    qobj = rk.yaw_quat(np.pi / 2)
    rp, _ = rk.ee_in_object_frame([0.1, 0, 0], [1, 0, 0, 0], [0, 0, 0], qobj)
    assert _approx(rp, [0, -0.1, 0])


def test_global_yaw_invariance():
    """THE equivariance property: g is invariant to a global rotation about
    gravity applied to BOTH the EE and the object. This is what makes the
    goal-relative target generalize over scene orientation by construction."""
    rng = np.random.default_rng(0)
    ee_pos = rng.normal(size=3)
    ee_q = rk.quat_normalize(rng.normal(size=4))
    obj_pos = rng.normal(size=3)
    obj_q = rk.quat_normalize(rng.normal(size=4))
    rp0, rq0 = rk.ee_in_object_frame(ee_pos, ee_q, obj_pos, obj_q)

    for theta in (0.3, 1.1, -2.0, np.pi):
        R = rk.yaw_quat(theta)
        ee_pos2 = rk.quat_rotate(R, ee_pos)
        obj_pos2 = rk.quat_rotate(R, obj_pos)
        ee_q2 = rk.quat_mul(R, ee_q)
        obj_q2 = rk.quat_mul(R, obj_q)
        rp, rq = rk.ee_in_object_frame(ee_pos2, ee_q2, obj_pos2, obj_q2)
        assert _approx(rp, rp0, 1e-9), f"pos not yaw-invariant at {theta}"
        # quats equal up to sign
        assert _approx(rq, rq0, 1e-9) or _approx(rq, -rq0, 1e-9), f"quat not yaw-invariant at {theta}"


def test_full_translation_rotation_invariance():
    """Stronger: g invariant to any global SE(3) transform of both frames."""
    rng = np.random.default_rng(1)
    ee_pos, ee_q = rng.normal(size=3), rk.quat_normalize(rng.normal(size=4))
    obj_pos, obj_q = rng.normal(size=3), rk.quat_normalize(rng.normal(size=4))
    rp0, rq0 = rk.ee_in_object_frame(ee_pos, ee_q, obj_pos, obj_q)
    R = rk.quat_normalize(rng.normal(size=4))       # arbitrary rotation
    tr = rng.normal(size=3)                          # arbitrary translation
    f = lambda p: rk.quat_rotate(R, p) + tr
    rp, rq = rk.ee_in_object_frame(f(ee_pos), rk.quat_mul(R, ee_q), f(obj_pos), rk.quat_mul(R, obj_q))
    assert _approx(rp, rp0, 1e-8)
    assert _approx(rq, rq0, 1e-8) or _approx(rq, -rq0, 1e-8)


def _make_synthetic_npz(path, T=40, n_obj=2, A=7):
    """Mimic the run_reason_v3_mppi.py --log-physics-traces NPZ schema."""
    rng = np.random.default_rng(2)
    ee_pos = np.cumsum(rng.normal(scale=0.01, size=(T, 3)), axis=0)
    ee_quat = np.tile([0, 0, 0, 1.0], (T, 1))        # scalar-last identity
    # object 0 stationary (fixed pose + tiny jitter); object 1 is the
    # "manipulated" one, displaced 0.3m in +x -> largest first-vs-last delta.
    obj_pos = np.zeros((T, n_obj, 3))
    obj_pos[:, 0, :] = np.array([0.5, -0.2, 0.1]) + rng.normal(scale=1e-4, size=(T, 3))
    obj_pos[:, 1, :] = np.array([-0.3, 0.4, 0.05]) + np.linspace(0, 1, T)[:, None] * np.array([0.3, 0.0, 0.0])
    obj_quat = np.tile([1.0, 0, 0, 0], (T, n_obj, 1))  # scalar-first identity
    np.savez_compressed(
        path,
        ee_pos=ee_pos, ee_quat=ee_quat,
        gripper_qpos=rng.normal(size=(T, 2)),
        action=rng.normal(size=(T, A)),
        object_pos=obj_pos, object_quat=obj_quat,
        object_names=np.array(["akita_bowl", "wooden_cabinet"], dtype=object),
        success=True, task_description="put the bowl in the drawer",
    )


def test_build_pairs_end_to_end():
    with tempfile.TemporaryDirectory() as tdir:
        p = f"{tdir}/PHYS_OK_synthetic.npz"
        _make_synthetic_npz(p, T=40, n_obj=2)
        out = rk.build_pairs(p, rk.RekeyConfig(horizon=16))
        N = 40 - 16
        assert out["g_pos"].shape == (N, 3)
        assert out["g_quat"].shape == (N, 4)
        assert out["proprio"].shape == (N, 9)
        assert out["chunk"].shape == (N, 16, 7)
        assert out["obj_pos"].shape == (N, 3)
        assert out["obj_quat"].shape == (N, 4)
        # most-displaced object is index 1 (the 0.3m-moving one)
        assert out["target_index"] == 1
        # explicit name selection works
        out2 = rk.build_pairs(p, rk.RekeyConfig(horizon=16, target_name="bowl"))
        assert out2["target_index"] == 0
        print(f"  build_pairs: N={N}, target='{out['target_name']}' (auto), "
              f"g_pos[0]={out['g_pos'][0].round(3)}")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"\nAll {len(tests)} tests passed.")
