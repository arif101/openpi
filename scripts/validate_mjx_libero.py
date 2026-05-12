"""Validate that MJX-JAX compiles + differentiates LIBERO scenes.

The single most important pre-flight check for REASON-VLA. If LIBERO's MJCF
doesn't compile in MJX-JAX or gradients don't flow, the differentiable-physics
plank of REASON-VLA's architecture is dead for our benchmark — we'd need to
pivot to Robosuite/Panda or hand-port scenes.

This test compiles a single LIBERO scene to MJX, runs a step, and takes a
gradient of qpos w.r.t. control input. If it returns finite gradients,
we're unblocked. If it fails, the error message tells us specifically what
MJX feature LIBERO needs that's missing.

Usage:
    PYTHONPATH=src:third_party/libero uv run python3 -u scripts/validate_mjx_libero.py
"""

from __future__ import annotations

import pathlib
import sys
import time
import traceback

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from libero.libero import benchmark, get_libero_path


def _pick_task_bddl(task_suite_name: str = "libero_10", task_idx: int = 0) -> pathlib.Path:
    bm = benchmark.get_benchmark_dict()[task_suite_name]()
    task = bm.get_task(task_idx)
    return pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file


def _bddl_to_mjcf(bddl_path: pathlib.Path) -> str:
    """LIBERO ships with a converter from BDDL → MuJoCo XML. Use the same path
    that OffScreenRenderEnv uses internally so we test the actual scene."""
    from libero.libero.envs import OffScreenRenderEnv
    env = OffScreenRenderEnv(bddl_file_name=str(bddl_path),
                              camera_heights=256, camera_widths=256)
    xml = env.env.sim.model.get_xml()
    env.close()
    return xml


def main() -> int:
    print("=" * 70, flush=True)
    print("MJX-LIBERO Compatibility Test", flush=True)
    print("=" * 70, flush=True)

    print(f"\nJAX devices: {jax.devices()}", flush=True)
    print(f"MuJoCo version: {mujoco.__version__}", flush=True)

    # --- 1. Load a LIBERO scene and convert to MJCF -------------------------
    print("\n[1/4] Loading LIBERO scene 0 from libero_10...", flush=True)
    try:
        bddl = _pick_task_bddl("libero_10", 0)
        xml = _bddl_to_mjcf(bddl)
        print(f"  ✓ Loaded BDDL: {bddl.name}", flush=True)
        print(f"  ✓ MJCF length: {len(xml)} chars", flush=True)
    except Exception as e:
        print(f"  ✗ Failed to load LIBERO scene: {e}", flush=True)
        traceback.print_exc()
        return 1

    # --- 2. Compile to MJX --------------------------------------------------
    print("\n[2/4] Compiling MJCF → MJX model...", flush=True)
    try:
        mj_model = mujoco.MjModel.from_xml_string(xml)
        print(f"  ✓ MuJoCo model: nq={mj_model.nq}, nu={mj_model.nu}, nbody={mj_model.nbody}",
              flush=True)
        mjx_model = mjx.put_model(mj_model)
        mjx_data = mjx.make_data(mjx_model)
        print(f"  ✓ MJX model + data on device", flush=True)
    except Exception as e:
        print(f"  ✗ MJX put_model failed: {e}", flush=True)
        traceback.print_exc()
        print("\n  This likely means LIBERO uses an MJX-unsupported feature:", flush=True)
        print("    - non-FREE/BALL/SLIDE/HINGE joint type", flush=True)
        print("    - unsupported integrator", flush=True)
        print("    - too-large mesh (>200 verts)", flush=True)
        print("    - unsupported constraint type", flush=True)
        print("  Action: either patch the MJCF or pivot benchmark.", flush=True)
        return 2

    # --- 3. Step the MJX model ---------------------------------------------
    print("\n[3/4] JIT-compiling + running one MJX step...", flush=True)
    try:
        @jax.jit
        def step_fn(data, ctrl):
            data = data.replace(ctrl=ctrl)
            return mjx.step(mjx_model, data)

        zero_ctrl = jnp.zeros(mjx_model.nu)
        t0 = time.time()
        new_data = step_fn(mjx_data, zero_ctrl)
        new_data.qpos.block_until_ready()  # force JIT compile + execution
        t_jit = time.time() - t0
        print(f"  ✓ First step (JIT compile): {t_jit*1000:.1f}ms", flush=True)

        # Warm step
        t0 = time.time()
        for _ in range(100):
            new_data = step_fn(new_data, zero_ctrl)
        new_data.qpos.block_until_ready()
        t_warm = (time.time() - t0) / 100
        print(f"  ✓ Warm step: {t_warm*1e6:.1f}μs/step ({1/t_warm:.0f} SPS single-scene)",
              flush=True)
    except Exception as e:
        print(f"  ✗ MJX step failed: {e}", flush=True)
        traceback.print_exc()
        return 3

    # --- 4. Take a gradient -- THE actual gating test ----------------------
    print("\n[4/4] Computing gradient of post-step qpos w.r.t. control...", flush=True)
    try:
        def loss_fn(ctrl):
            data = mjx_data.replace(ctrl=ctrl)
            new_data = mjx.step(mjx_model, data)
            # Scalar: sum of qpos magnitudes after one step.
            return jnp.sum(new_data.qpos ** 2)

        grad_fn = jax.jit(jax.grad(loss_fn))
        t0 = time.time()
        grad = grad_fn(zero_ctrl)
        grad.block_until_ready()
        t_jit_grad = time.time() - t0
        print(f"  ✓ First gradient (JIT compile): {t_jit_grad*1000:.1f}ms", flush=True)

        t0 = time.time()
        for _ in range(100):
            grad = grad_fn(zero_ctrl)
        grad.block_until_ready()
        t_warm_grad = (time.time() - t0) / 100
        print(f"  ✓ Warm gradient: {t_warm_grad*1e6:.1f}μs/grad", flush=True)

        finite = bool(jnp.all(jnp.isfinite(grad)).item())
        magnitude = float(jnp.linalg.norm(grad).item())
        print(f"  ✓ Gradient is finite: {finite}", flush=True)
        print(f"  ✓ Gradient L2 norm: {magnitude:.6e}", flush=True)
        if not finite:
            print("  ⚠ Gradient has NaN/Inf — physics is differentiable but unstable here.",
                  flush=True)
            return 4
    except Exception as e:
        print(f"  ✗ MJX gradient failed: {e}", flush=True)
        traceback.print_exc()
        return 4

    # --- Done ----------------------------------------------------------------
    print("\n" + "=" * 70, flush=True)
    print("✓ ALL CHECKS PASSED. MJX-LIBERO is viable for REASON-VLA.", flush=True)
    print("=" * 70, flush=True)
    print(f"\nSingle-scene MJX speed: {1/t_warm:.0f} SPS step, {1/t_warm_grad:.0f} SPS grad",
          flush=True)
    print("\nFor REASON-VLA Phase 2:", flush=True)
    print(f"  - Each action chunk is ~10 sim steps", flush=True)
    print(f"  - Per-chunk gradient: ~{10 * t_warm_grad * 1000:.1f}ms", flush=True)
    print(f"  - At 5-10 refinement iterations: ~{50 * t_warm_grad * 1000:.0f}-{100 * t_warm_grad * 1000:.0f}ms physics overhead",
          flush=True)
    print("\nNext step: implement REASON-VLA Phase 1 (iterative refinement)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
