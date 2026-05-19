"""Build a reachability field for the LIBERO Franka + OSC_POSE controller.

The field encodes: given a joint configuration q ∈ R^7 and a commanded EE-delta
action a ∈ R^7 (xyz + axis-angle + gripper), how much will the EE actually
move when env.step(a) is executed under the deployment controller?

This is the embodiment-feasibility signal. A configuration where commanded
+y motion realizes <2mm of EE-y displacement is "infeasible for +y push" —
the failure mode we measured on 11/14 IN_TRUE_CLOSE_FALSE traces.

Stages:
  1. Data generation: sample (q, a) pairs, query env+OSC, log realized EE deltas.
  2. Train a small MLP: f(q, a) → realized_delta_xyz (R^3).
  3. Validate on held-out: R² per axis. Acceptance: mean R² > 0.5.

Object isolation: during each query the bowl and wine_bottle are moved to
(5, 5, 0.5) — well outside the robot's workspace — so contact dynamics
don't confound the controller-reachability signal we're measuring.

Two-phase script: `--stage gen` writes the dataset; `--stage train` reads
it and trains. Saves model + metrics to the output dir.

Usage:
  # Phase 1: generate ~10k samples (CPU only, ~20 min)
  PYTHONPATH=src:third_party/libero MUJOCO_GL=egl uv run python3 -u \\
      scripts/build_reachability_field.py --stage gen \\
      --n-samples 10000 --steps-per-query 5 \\
      --out-dir data/contact_mpc/reachability_field

  # Phase 2: train + validate (CPU/GPU, fast)
  uv run python3 -u scripts/build_reachability_field.py --stage train \\
      --out-dir data/contact_mpc/reachability_field
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch
import torch.nn as nn

_orig = torch.load
def _patched(*args, **kwargs):
    if "weights_only" not in kwargs: kwargs["weights_only"] = False
    return _orig(*args, **kwargs)
torch.load = _patched


# ----- Data generation -----

def isolate_objects(env) -> None:
    """Move free-joint objects far away so they don't interact with the robot."""
    sim = env.env.sim
    mdl = sim.model
    q = sim.data.qpos.copy()
    # For each free joint (joint type 0 == mjJNT_FREE in MuJoCo)
    for j in range(mdl.njnt):
        if mdl.jnt_type[j] == 0:  # free joint
            addr = int(mdl.jnt_qposadr[j])
            q[addr:addr + 3] = [5.0, 5.0, 0.5]   # position far away
            q[addr + 3:addr + 7] = [1.0, 0.0, 0.0, 0.0]  # identity quat
    sim.data.qpos[:] = q
    sim.data.qvel[:] = 0.0
    sim.forward()


def sample_joint_config(rng: np.random.Generator, mdl, traj_configs: list[np.ndarray] | None = None) -> np.ndarray:
    """50/50 mix of uniform-in-limits and trajectory-derived configs."""
    if traj_configs and rng.random() < 0.5:
        return traj_configs[int(rng.integers(0, len(traj_configs)))].copy()
    # Uniform within Franka joint limits (read from model)
    limits = np.array([mdl.jnt_range[j] for j in range(7)])  # (7, 2)
    return rng.uniform(limits[:, 0], limits[:, 1])


def sample_action(rng: np.random.Generator) -> np.ndarray:
    """Mix uniform-in-[-1,1] and Gaussian-near-0."""
    if rng.random() < 0.5:
        a = rng.uniform(-1.0, 1.0, size=7)
    else:
        a = rng.normal(0.0, 0.5, size=7).clip(-1.0, 1.0)
    # Last element is gripper command; restrict to {-1, +1} half the time
    if rng.random() < 0.5:
        a[6] = -1.0 if rng.random() < 0.5 else 1.0
    return a.astype(np.float32)


def load_trajectory_configs(traces_dir: pathlib.Path | None) -> list[np.ndarray]:
    """Pull qpos[0:7] from existing traces to bias sampling toward
    Pi0.5-visited configurations."""
    if traces_dir is None or not traces_dir.exists():
        return []
    out = []
    for p in sorted(traces_dir.glob("PHYS_*.npz"))[:200]:
        try:
            d = np.load(p, allow_pickle=True)
            qp = d["qpos"]
            T = qp.shape[0]
            # Sample every 50 steps
            for t in range(0, T, 50):
                out.append(qp[t, :7].copy())
        except Exception:
            pass
    print(f"  loaded {len(out)} trajectory configs from {traces_dir}")
    return out


def stage_gen(args) -> int:
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    bm = benchmark.get_benchmark_dict()["libero_10"]()
    task = bm.get_task(3)
    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=64, camera_widths=64)
    env.seed(0)
    env.reset()
    mdl = env.env.sim.model
    eef_site_id = env.env.robots[0].eef_site_id

    traj_cfgs = load_trajectory_configs(pathlib.Path(args.traces_dir)) if args.traces_dir else []
    rng = np.random.default_rng(0)
    K = args.steps_per_query
    N = args.n_samples

    configs = np.zeros((N, 7), dtype=np.float32)
    actions = np.zeros((N, 7), dtype=np.float32)
    realized_xyz = np.zeros((N, 3), dtype=np.float32)
    joint_motion = np.zeros((N,), dtype=np.float32)

    print(f"Generating {N} (config, action) samples, K={K} OSC steps each...")
    t0 = time.time()
    for i in range(N):
        q = sample_joint_config(rng, mdl, traj_cfgs)
        a = sample_action(rng)
        # Reset env state
        env.reset()
        isolate_objects(env)
        sim = env.env.sim
        # Overwrite robot qpos
        sim.data.qpos[:7] = q
        sim.data.qvel[:7] = 0.0
        sim.forward()
        ee0 = np.array(sim.data.site_xpos[eef_site_id])
        q0 = sim.data.qpos[:7].copy()
        # Run K OSC steps with the commanded action
        try:
            for _ in range(K):
                env.step(a.tolist())
        except Exception:
            pass
        ee1 = np.array(sim.data.site_xpos[eef_site_id])
        q1 = sim.data.qpos[:7].copy()
        configs[i] = q.astype(np.float32)
        actions[i] = a
        realized_xyz[i] = (ee1 - ee0).astype(np.float32)
        joint_motion[i] = float(np.linalg.norm(q1 - q0))
        if (i + 1) % 200 == 0 or i == N - 1:
            elapsed = time.time() - t0
            rate = (i + 1) / max(elapsed, 1e-9)
            print(f"  {i + 1}/{N}  ({rate:.1f}/s, ETA {(N - i - 1)/max(rate,1e-9):.0f}s)", flush=True)
    env.close()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "dataset.npz",
             configs=configs, actions=actions,
             realized_xyz=realized_xyz, joint_motion=joint_motion,
             steps_per_query=K)
    print(f"\nWrote {out_dir/'dataset.npz'}  ({N} samples)")
    # Quick summary
    norms = np.linalg.norm(realized_xyz, axis=1)
    print(f"  realized EE displacement: mean={norms.mean():.4f}m  median={np.median(norms):.4f}m  max={norms.max():.4f}m")
    print(f"  joint motion magnitude:   mean={joint_motion.mean():.4f}  max={joint_motion.max():.4f}")
    return 0


# ----- Training -----

class ReachabilityField(nn.Module):
    def __init__(self, hidden: int = 128):
        super().__init__()
        # Input: 7 joint angles + 7 action dims = 14
        self.net = nn.Sequential(
            nn.Linear(14, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 3),  # realized EE delta xyz
        )

    def forward(self, q, a):
        return self.net(torch.cat([q, a], dim=-1))


def stage_train(args) -> int:
    out_dir = pathlib.Path(args.out_dir)
    d = np.load(out_dir / "dataset.npz")
    configs = d["configs"]; actions = d["actions"]; y = d["realized_xyz"]
    N = configs.shape[0]
    rng = np.random.default_rng(1)
    perm = rng.permutation(N)
    n_val = N // 10
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    print(f"Train: {len(train_idx)}   Val: {len(val_idx)}")

    Xc = torch.tensor(configs); Xa = torch.tensor(actions); Y = torch.tensor(y)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    model = ReachabilityField().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    Xc_train = Xc[train_idx].to(device); Xa_train = Xa[train_idx].to(device); Y_train = Y[train_idx].to(device)
    Xc_val = Xc[val_idx].to(device); Xa_val = Xa[val_idx].to(device); Y_val = Y[val_idx].to(device)

    n_epochs = args.epochs
    batch = 1024
    best_val = float("inf")
    metrics = []
    for epoch in range(n_epochs):
        idxs = torch.randperm(len(train_idx), device=device)
        model.train()
        total = 0.0
        for s in range(0, len(idxs), batch):
            ix = idxs[s:s+batch]
            yp = model(Xc_train[ix], Xa_train[ix])
            loss = nn.functional.mse_loss(yp, Y_train[ix])
            opt.zero_grad(); loss.backward(); opt.step()
            total += float(loss) * ix.numel()
        train_loss = total / len(idxs)
        model.eval()
        with torch.no_grad():
            yp = model(Xc_val, Xa_val)
            val_loss = float(nn.functional.mse_loss(yp, Y_val))
            ss_res = ((yp - Y_val) ** 2).sum(dim=0)
            ss_tot = ((Y_val - Y_val.mean(dim=0)) ** 2).sum(dim=0)
            r2_per_axis = (1 - ss_res / ss_tot.clamp(min=1e-9)).cpu().numpy()
        metrics.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                        "r2_x": float(r2_per_axis[0]), "r2_y": float(r2_per_axis[1]),
                        "r2_z": float(r2_per_axis[2]), "r2_mean": float(r2_per_axis.mean())})
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), out_dir / "reachability_field.pt")
        if epoch % 5 == 0 or epoch == n_epochs - 1:
            print(f"  ep{epoch:>3}  train={train_loss:.6f}  val={val_loss:.6f}  R²=({r2_per_axis[0]:+.3f},{r2_per_axis[1]:+.3f},{r2_per_axis[2]:+.3f})  mean R²={r2_per_axis.mean():+.3f}", flush=True)

    (out_dir / "training_metrics.json").write_text(json.dumps(metrics, indent=2))
    final = metrics[-1]
    print()
    print("=== Reachability field training complete ===")
    print(f"  Final R² per axis:  x={final['r2_x']:+.3f}  y={final['r2_y']:+.3f}  z={final['r2_z']:+.3f}")
    print(f"  Mean R²:            {final['r2_mean']:+.3f}")
    print(f"  Acceptance gate (mean R² > 0.5): {'PASS' if final['r2_mean'] > 0.5 else 'FAIL'}")
    print(f"\nSaved: {out_dir/'reachability_field.pt'} (best on val)")
    print(f"       {out_dir/'training_metrics.json'}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["gen", "train"], required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--n-samples", type=int, default=10000)
    p.add_argument("--steps-per-query", type=int, default=5)
    p.add_argument("--traces-dir", default="data/contact_mpc/recovery_source_traces")
    p.add_argument("--epochs", type=int, default=50)
    args = p.parse_args()
    return stage_gen(args) if args.stage == "gen" else stage_train(args)


if __name__ == "__main__":
    sys.exit(main())
