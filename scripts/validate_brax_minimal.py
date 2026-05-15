"""Minimal Brax validation — replaces validate_mjx_minimal.py.

MJX-JAX's constraint solver uses jax.lax.while_loop which doesn't support
reverse-mode autodiff. Brax was built for differentiable RL and uses
fixed-iteration solvers that are reverse-mode friendly. This script
validates Brax can do what we need for REASON-VLA.

Run in the same .venv-mjx after:
    pip install brax

    python3 scripts/validate_brax_minimal.py
"""

from __future__ import annotations

import sys
import time
import traceback

import jax
import jax.numpy as jnp


# Same scene as the MJX test: 3-DOF arm + free-body box.
TEST_XML = """
<mujoco model="brax_test">
  <option timestep="0.005"/>
  <worldbody>
    <geom name="floor" type="plane" size="2 2 0.1" rgba="0.8 0.9 0.8 1"/>
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
    print("Minimal Brax validation")
    print("=" * 70)
    print(f"\nJAX devices: {jax.devices()}")

    try:
        import brax
        from brax.io import mjcf
        from brax import envs
        from brax.generalized import pipeline as gen_pipeline
        from brax.positional import pipeline as pos_pipeline
        print(f"Brax version: {brax.__version__ if hasattr(brax, '__version__') else 'unknown'}")
    except Exception as e:
        print(f"  Brax import FAIL: {e}")
        traceback.print_exc()
        return 1

    # 1. Load scene
    print("\n[1/4] Load MJCF into Brax...")
    try:
        sys_spec = mjcf.loads(TEST_XML)
        print(f"  ok: nq={sys_spec.q_size()}, nu={sys_spec.act_size()}")
    except Exception as e:
        print(f"  FAIL: {e}")
        traceback.print_exc()
        return 2

    # 2. JIT step. Brax has multiple pipelines; "generalized" matches MuJoCo's
    # generalized-coordinate dynamics.
    print("\n[2/4] JIT step via generalized pipeline...")
    try:
        state = gen_pipeline.init(sys_spec, sys_spec.init_q, jnp.zeros(sys_spec.qd_size()))

        @jax.jit
        def step_fn(state, ctrl):
            return gen_pipeline.step(sys_spec, state, ctrl)

        ctrl0 = jnp.zeros(sys_spec.act_size())
        t0 = time.time()
        state = step_fn(state, ctrl0)
        state.q.block_until_ready()
        print(f"  first (compile): {(time.time()-t0)*1000:.1f}ms")

        t0 = time.time()
        for _ in range(100):
            state = step_fn(state, ctrl0)
        state.q.block_until_ready()
        per_step_us = (time.time() - t0) * 1e4
        print(f"  warm: {per_step_us:.1f}us/step ({1e6/per_step_us:.0f} SPS single-scene)")
    except Exception as e:
        print(f"  FAIL: {e}")
        traceback.print_exc()
        return 3

    # 3. Single-step gradient — the actual gating test
    print("\n[3/4] Reverse-mode gradient of qpos w.r.t. control...")
    try:
        init_state = gen_pipeline.init(sys_spec, sys_spec.init_q, jnp.zeros(sys_spec.qd_size()))

        def loss_single(ctrl):
            s = gen_pipeline.step(sys_spec, init_state, ctrl)
            return jnp.sum(s.q ** 2)

        grad_fn = jax.jit(jax.grad(loss_single))
        t0 = time.time()
        g = grad_fn(ctrl0)
        g.block_until_ready()
        print(f"  first grad (compile): {(time.time()-t0)*1000:.1f}ms")
        print(f"  finite: {bool(jnp.all(jnp.isfinite(g)).item())}")
        print(f"  norm: {float(jnp.linalg.norm(g).item()):.4e}")

        t0 = time.time()
        for _ in range(100):
            g = grad_fn(ctrl0)
        g.block_until_ready()
        per_grad_us = (time.time() - t0) * 1e4
        print(f"  warm: {per_grad_us:.1f}us/grad ({1e6/per_grad_us:.0f} grad/s)")
    except Exception as e:
        print(f"  FAIL: {e}")
        traceback.print_exc()
        return 4

    # 4. Multi-step chunk gradient — what we need for REASON-VLA
    print("\n[4/4] Gradient through H=10 step action chunk...")
    try:
        H = 10

        def chunk_loss(actions):
            def body(s, a):
                s2 = gen_pipeline.step(sys_spec, s, a)
                return s2, s2.q
            _, qs = jax.lax.scan(body, init_state, actions)
            return jnp.sum(qs ** 2)

        chunk_grad = jax.jit(jax.grad(chunk_loss))
        actions0 = jnp.zeros((H, sys_spec.act_size()))
        t0 = time.time()
        gA = chunk_grad(actions0)
        gA.block_until_ready()
        print(f"  first chunk-grad (compile): {(time.time()-t0)*1000:.1f}ms")

        t0 = time.time()
        for _ in range(20):
            gA = chunk_grad(actions0)
        gA.block_until_ready()
        per_chunk_ms = (time.time() - t0) * 50
        print(f"  warm: {per_chunk_ms:.1f}ms per H={H} chunk gradient")
        print(f"  finite: {bool(jnp.all(jnp.isfinite(gA)).item())}")
        print(f"  shape: {gA.shape}, norm: {float(jnp.linalg.norm(gA).item()):.4e}")
    except Exception as e:
        print(f"  FAIL: {e}")
        traceback.print_exc()
        return 5

    print("\n" + "=" * 70)
    print("PASS. Brax differentiable physics is viable.")
    print("=" * 70)
    print(f"\nPer-action-chunk gradient: {per_chunk_ms:.0f}ms")
    print(f"At 5-10 refinement iterations: {per_chunk_ms*5:.0f}-{per_chunk_ms*10:.0f}ms physics overhead")
    print("\nNext: try Brax with LIBERO MJCF (export from GPU box, load via brax.io.mjcf)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
