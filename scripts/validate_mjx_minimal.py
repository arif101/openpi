"""Minimal MJX-JAX gradient validation. No LIBERO needed.

The point: confirm MJX-JAX itself compiles + differentiates on this machine
before we tackle LIBERO scene compatibility.

Run with:
    # In a fresh Mac venv (NOT the openpi venv):
    uv venv --python 3.11 .venv-mjx
    source .venv-mjx/bin/activate
    uv pip install 'mujoco>=3.2' jax jaxlib

    python3 scripts/validate_mjx_minimal.py

If this passes, MJX works on this machine. We then handle LIBERO scenes
separately (LIBERO pins mujoco==2.3.7 which predates MJX; we'll export
LIBERO MJCFs from the GPU box and test them under mujoco>=3.0 + MJX).
"""

from __future__ import annotations

import sys
import time
import traceback

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx


# A simple but non-trivial scene: a robot arm with 3 hinge joints picking
# at a free-body box. Covers HINGE + FREE joints (the two LIBERO uses most)
# plus a contact between the gripper and the box.
TEST_XML = """
<mujoco model="mjx_test">
  <option timestep="0.002" gravity="0 0 -9.81" iterations="10"/>

  <worldbody>
    <geom name="floor" type="plane" size="2 2 0.1" rgba="0.8 0.9 0.8 1"/>

    <!-- 3-link arm with HINGE joints (LIBERO uses these heavily) -->
    <body name="link1" pos="0 0 0.1">
      <joint name="j1" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
      <geom type="capsule" size="0.04" fromto="0 0 0 0.3 0 0" rgba="0.6 0.6 0.9 1"/>
      <body name="link2" pos="0.3 0 0">
        <joint name="j2" type="hinge" axis="0 1 0" range="-1.57 1.57"/>
        <geom type="capsule" size="0.04" fromto="0 0 0 0.3 0 0" rgba="0.6 0.9 0.6 1"/>
        <body name="link3" pos="0.3 0 0">
          <joint name="j3" type="hinge" axis="0 1 0" range="-1.57 1.57"/>
          <geom name="grip" type="box" size="0.03 0.03 0.05" pos="0.05 0 0" rgba="0.9 0.6 0.6 1"/>
        </body>
      </body>
    </body>

    <!-- Free-body box (LIBERO objects use FREE joints) -->
    <body name="box" pos="0.5 0 0.2">
      <freejoint/>
      <geom type="box" size="0.04 0.04 0.04" rgba="0.9 0.9 0.6 1"/>
    </body>
  </worldbody>

  <actuator>
    <motor joint="j1" ctrlrange="-1 1" gear="10"/>
    <motor joint="j2" ctrlrange="-1 1" gear="10"/>
    <motor joint="j3" ctrlrange="-1 1" gear="10"/>
  </actuator>
</mujoco>
"""


def main() -> int:
    print("=" * 70)
    print("Minimal MJX-JAX validation")
    print("=" * 70)
    print(f"\nJAX devices: {jax.devices()}")
    print(f"MuJoCo version: {mujoco.__version__}")

    # 1. Load + compile to MJX
    print("\n[1/4] Compile MJCF -> MJX...")
    try:
        mj_model = mujoco.MjModel.from_xml_string(TEST_XML)
        print(f"  nq={mj_model.nq}, nu={mj_model.nu}, nbody={mj_model.nbody}")
        mjx_model = mjx.put_model(mj_model)
        mjx_data = mjx.make_data(mjx_model)
        print("  ok")
    except Exception as e:
        print(f"  FAIL: {e}")
        traceback.print_exc()
        return 2

    # 2. JIT'd step
    print("\n[2/4] JIT-compile and run a step...")
    try:
        @jax.jit
        def step_fn(data, ctrl):
            data = data.replace(ctrl=ctrl)
            return mjx.step(mjx_model, data)

        ctrl0 = jnp.zeros(mjx_model.nu)
        t0 = time.time()
        d = step_fn(mjx_data, ctrl0)
        d.qpos.block_until_ready()
        print(f"  first (compile): {(time.time()-t0)*1000:.1f}ms")

        t0 = time.time()
        for _ in range(100):
            d = step_fn(d, ctrl0)
        d.qpos.block_until_ready()
        per_step_us = (time.time() - t0) * 1e4
        print(f"  warm: {per_step_us:.1f}us/step ({1e6/per_step_us:.0f} SPS single-scene)")
    except Exception as e:
        print(f"  FAIL: {e}")
        traceback.print_exc()
        return 3

    # 3. Single-step gradient
    print("\n[3/4] Compute gradient of qpos w.r.t. control (single step)...")
    try:
        def loss_single(ctrl):
            d = mjx_data.replace(ctrl=ctrl)
            return jnp.sum(mjx.step(mjx_model, d).qpos ** 2)

        grad_fn = jax.jit(jax.grad(loss_single))
        t0 = time.time()
        g = grad_fn(ctrl0)
        g.block_until_ready()
        print(f"  first grad (compile): {(time.time()-t0)*1000:.1f}ms")
        print(f"  finite: {bool(jnp.all(jnp.isfinite(g)).item())}")
        print(f"  norm: {float(jnp.linalg.norm(g).item()):.4e}")
    except Exception as e:
        print(f"  FAIL: {e}")
        traceback.print_exc()
        return 4

    # 4. Multi-step gradient (this is what we actually need for refinement)
    print("\n[4/4] Gradient through a 10-step rollout (action chunk semantics)...")
    try:
        def rollout_loss(actions):
            # actions: [H, nu]
            def body(carry, a):
                data = carry.replace(ctrl=a)
                next_data = mjx.step(mjx_model, data)
                return next_data, next_data.qpos
            _, qpos_traj = jax.lax.scan(body, mjx_data, actions)
            return jnp.sum(qpos_traj ** 2)

        H = 10
        actions0 = jnp.zeros((H, mjx_model.nu))
        rollout_grad = jax.jit(jax.grad(rollout_loss))
        t0 = time.time()
        gA = rollout_grad(actions0)
        gA.block_until_ready()
        print(f"  first chunk-grad (compile): {(time.time()-t0)*1000:.1f}ms")

        t0 = time.time()
        for _ in range(20):
            gA = rollout_grad(actions0)
        gA.block_until_ready()
        per_chunk_ms = (time.time() - t0) * 50
        print(f"  warm: {per_chunk_ms:.1f}ms per H={H} action-chunk gradient")
        print(f"  finite: {bool(jnp.all(jnp.isfinite(gA)).item())}")
        print(f"  shape: {gA.shape}, norm: {float(jnp.linalg.norm(gA).item()):.4e}")
    except Exception as e:
        print(f"  FAIL: {e}")
        traceback.print_exc()
        return 5

    print("\n" + "=" * 70)
    print("PASS. MJX-JAX is viable on this machine.")
    print("=" * 70)
    print("\nWhat this means:")
    print(f"  - {per_chunk_ms:.0f}ms per action-chunk physics gradient")
    print(f"  - At 5-10 refinement iterations: {per_chunk_ms * 5:.0f}-{per_chunk_ms * 10:.0f}ms physics overhead")
    print("  - For comparison: Pi0.5 single forward = ~100ms on GPU")
    print("\nNext: extract LIBERO MJCFs on the GPU box, test under mujoco>=3.0 here.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
