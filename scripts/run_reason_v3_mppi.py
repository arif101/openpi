"""REASON-VLA v3 — MPPI refinement integration with LIBERO.

Each decision point:
  1. Pi0.5 emits initial action chunk a_0 (in EE-delta space, 7 dims).
  2. MPPI: pick which body to track (object-mode tasks: the candidate object
     currently farthest from its goal_xyz; ee-mode tasks: the end-effector).
     Save env state, sample K perturbations, step each through env, score by
     (tracked-body-to-goal, joint_limit, robot_collision, anchor), restore.
  3. Take softmin-weighted average → refined chunk a*.
  4. Execute a*.

Uses LIBERO's env directly for forward sim — guarantees identical
action-space semantics to the eval execution path. State save/restore
via robosuite's sim.get_state() / set_state_from_flattened().

Object-tracking cost handles "put X in Y" tasks without explicit stage
detection: once the robot grasps X, X's pos tracks the EE, so driving
X → Y encodes both pick and place. For compound goals ("put both X1 and
X2 in Y"), MPPI picks the candidate currently farthest from Y at each
decision step — sequencing emerges from the cost shape.

Targets per task live in scripts/reason_v3_targets.yaml (schema documented
inline). Generate / update via scripts/inspect_libero_scene.py.

Kill criterion: if MPPI-refined success < Pi0.5 baseline on the gated
task, MPPI mechanism doesn't help and we tune (K, sigma, lambda, weights).

Usage:
    # Default (physics scorer) — forward-sim K candidates in MuJoCo
    PYTHONPATH=src:third_party/libero uv run python3 -u \\
        scripts/run_reason_v3_mppi.py \\
        --task-suite libero_10 --task-idx 3 \\
        --num-trials 5 --perturbation-cm 5.0 --seed 7 \\
        --mppi-K 16 --mppi-sigma 0.1 --mppi-lambda 0.1 \\
        --scorer-type physics

    # Head-to-head — same loop, learned Q(h, a) scorer
    PYTHONPATH=src:third_party/libero uv run python3 -u \\
        scripts/run_reason_v3_mppi.py \\
        --task-suite libero_10 --task-idx 3 \\
        --num-trials 5 --perturbation-cm 5.0 --seed 7 \\
        --mppi-K 16 --mppi-sigma 0.1 --mppi-lambda 0.1 \\
        --scorer-type learned-wm \\
        --q-function data/contact_mpc/q_function_libero90/q_function.pt
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import pathlib
import time

import numpy as np
import torch
import yaml

_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import image_tools

from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config
from openpi.contact_mpc.value_function.architecture import ActionConditionalValueFunction


def load_q_function(ckpt_path: str, cfg_path: str | None, device) -> ActionConditionalValueFunction:
    """Load a trained ActionConditionalValueFunction Q(h, a)."""
    if cfg_path is None:
        p = pathlib.Path(ckpt_path)
        cfg_path = str(p.parent / (p.stem + "_config.pt"))
    cfg = torch.load(cfg_path, weights_only=False)
    if cfg.get("type") and cfg["type"] != "ActionConditionalValueFunction":
        raise ValueError(
            f"Expected ActionConditionalValueFunction at {cfg_path}, got {cfg.get('type')!r}"
        )
    q = ActionConditionalValueFunction(
        hidden_state_dim=cfg["hidden_state_dim"],
        action_chunk_horizon=cfg["action_chunk_horizon"],
        action_dim=cfg["action_dim"],
        action_emb_dim=cfg.get("action_emb_dim", 128),
        hidden_dim=cfg.get("hidden_dim", 256),
        dropout=0.0,
    )
    q.load_state_dict(torch.load(ckpt_path, weights_only=True))
    return q.to(device).eval()


def score_chunks_with_q(
    q_fn: ActionConditionalValueFunction,
    hidden_state: np.ndarray,    # [hidden_state_dim] from Pi0.5 vlm_features
    chunks: np.ndarray,          # [K, H, action_dim]
    device,
) -> np.ndarray:
    """Batch-score K action chunks with Q(h, a). Returns [K] scores.

    Pads / truncates each chunk to the Q-function's expected horizon. Higher Q
    = better candidate; we negate to produce a "cost" so MPPI softmin logic
    treats higher-Q candidates as preferred (the physics path produces costs
    where lower is better).
    """
    Hq = q_fn.action_chunk_horizon
    Aq = q_fn.action_dim
    K, H, A = chunks.shape
    if A != Aq:
        raise ValueError(f"action_dim mismatch: chunks have {A}, Q wants {Aq}")
    if H >= Hq:
        chunks_fit = chunks[:, :Hq]
    else:
        pad = np.zeros((K, Hq - H, A), dtype=chunks.dtype)
        chunks_fit = np.concatenate([chunks, pad], axis=1)
    h = torch.tensor(hidden_state, dtype=torch.float32, device=device).unsqueeze(0).expand(K, -1)
    a = torch.tensor(chunks_fit, dtype=torch.float32, device=device)
    with torch.no_grad():
        q_vals = q_fn(h, a).squeeze(-1).cpu().numpy()
    # Return cost = -Q so MPPI's softmin (prefer low cost) prefers high Q.
    return -q_vals.astype(np.float64)


LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256
RESIZE_SIZE = 224

MAX_STEPS = {
    "libero_spatial": 220, "libero_object": 280, "libero_goal": 300,
    "libero_10": 520, "libero_90": 400,
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config-name", default="pi05_libero")
    p.add_argument("--checkpoint", default="gs://openpi-assets/checkpoints/pi05_libero")
    p.add_argument("--task-suite", default="libero_10")
    p.add_argument("--task-idx", type=int, default=0, help="Single task to evaluate")
    p.add_argument("--num-trials", type=int, default=10)
    p.add_argument("--num-steps-wait", type=int, default=10)
    p.add_argument("--replan-steps", type=int, default=5)
    p.add_argument("--perturbation-cm", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--episode-timeout", type=int, default=300)
    p.add_argument("--output-dir", default="data/contact_mpc/reason_v3_mppi")
    p.add_argument("--targets-yaml", default="scripts/reason_v3_targets.yaml")
    # MPPI knobs
    p.add_argument("--mppi-K", type=int, default=16, help="Number of MPPI samples")
    p.add_argument("--mppi-sigma", type=float, default=0.1, help="Sample noise std")
    p.add_argument("--mppi-lambda", type=float, default=1.0, help="Softmin temperature")
    p.add_argument("--mppi-iterations", type=int, default=1)
    p.add_argument("--mppi-trigger", choices=["always", "contact"], default="always")
    p.add_argument("--w-target", type=float, default=10.0,
                   help="Weight for tracked-body → goal distance term")
    p.add_argument("--w-approach", type=float, default=5.0,
                   help="Weight for EE → tracked-body distance. Provides MPPI a "
                        "gradient signal before the robot grasps the object — "
                        "when the bowl is on the table 100ms of physics can't "
                        "move it, but the EE moves several cm so ||EE - bowl|| "
                        "discriminates candidates.")
    p.add_argument("--w-anchor", type=float, default=0.05,
                   help="Weight for ||action - prior||² anchor term")
    p.add_argument("--w-grip", type=float, default=20.0,
                   help="Weight for grip-stability term. Penalizes positive "
                        "growth of ||EE - tracked_obj|| ONLY when the previous "
                        "step's offset was within --grip-radius (i.e., EE was "
                        "in / near grasp range). This prevents the term from "
                        "firing during approach-phase jitter (which broke the "
                        "first grip matrix iteration: pooled -13.3pp on the "
                        "task 3 cells). Set 0 to disable.")
    p.add_argument("--grip-radius", type=float, default=0.05,
                   help="EE-object distance (m) below which the grip-stability "
                        "term activates. Default 0.05m = 5cm, the rough size of "
                        "LIBERO objects + gripper closing range. Above this, the "
                        "EE is in 'approach' phase and natural trajectory jitter "
                        "shouldn't be penalized.")
    p.add_argument("--w-collision", type=float, default=100.0,
                   help="Weight for robot-involved penetration depth (in meters)")
    p.add_argument("--w-joint-limit", type=float, default=10.0,
                   help="Weight for joint-limit violations")
    p.add_argument("--verbose-mppi", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # Head-to-head ablation: physics-grounded scorer (default) vs learned-WM scorer.
    # The paper's core differentiator vs SITCOM / VLAPS / VLA-Reasoner is that
    # physics scores generalize where learned ones drift. To prove that we need
    # the same MPPI loop run with both scorers, on the same seeds.
    p.add_argument("--scorer-type", choices=["physics", "learned-wm"], default="physics",
                   help="physics: forward-sim each candidate in MuJoCo, score by "
                        "collision+joint_limit+target+approach. learned-wm: score "
                        "each candidate via a trained Q(h, action_chunk) — no sim.")
    p.add_argument("--q-function",
                   help="Path to a trained ActionConditionalValueFunction .pt for "
                        "--scorer-type=learned-wm.")
    p.add_argument("--q-function-config", default=None,
                   help="Optional explicit config path; defaults to <ckpt>_config.pt.")
    # Phase B distillation data engine: capture (obs, prior, refined) per MPPI
    # call so a downstream LoRA fine-tune can train Pi0.5 to anticipate the
    # physics-grounded correction. Each saved file is one successful episode.
    p.add_argument("--log-refined-rollouts", default=None,
                   help="Directory to write per-episode .npz files containing "
                        "(obs, prior_action, refined_action) tuples per MPPI "
                        "decision. Only successful episodes are saved (no point "
                        "distilling failures). Set to a path under data/ to "
                        "build the Phase B fine-tune dataset.")
    p.add_argument("--log-failure-traces", default=None,
                   help="Directory to write per-step traces for FAILED episodes "
                        "only. Captures EE position, tracked-object positions, "
                        "gripper command, plus a camera frame every ~50 steps. "
                        "Lets us distinguish 'wrong intent' (EE drifts to phantom "
                        "location) from 'right intent / bad control' (EE near "
                        "object but misses). Drives Phase B vs guided-exploration "
                        "choice as the capability fix.")
    # PhysVLA: per-step physics-rich trace logger. Captures (image, wrist,
    # state, action, body poses, contact wrenches, gripper qpos, contact
    # summary) at every env.step for every episode (success AND failure).
    # This is the foundation dataset for training the latent dynamics model
    # + physics-supervised aux heads. Distinct from --log-refined-rollouts
    # (per-decision MPPI input/output) and --log-failure-traces (failed
    # episodes only, downsampled).
    p.add_argument("--log-physics-traces", default=None,
                   help="Directory to write per-step physics-rich .npz files "
                        "for ALL episodes. Captures the supervision data for "
                        "the PhysVLA latent dynamics model + aux heads "
                        "(contact wrenches, object poses, slip velocities, "
                        "contact normals).")
    p.add_argument("--physics-trace-image-stride", type=int, default=5,
                   help="Save images every N steps to keep dataset size "
                        "manageable. Physics-state arrays are still saved "
                        "per-step regardless. Default 5 → ~30MB/episode "
                        "at 500 steps with 224x224 images.")
    # Perturbation validity. Audit of the first 160 traces showed ~50% of
    # 5cm episodes had pathological initial conditions (objects launched by
    # contact resolver or spawned mid-air). Reject-sample until a clean
    # perturbation passes the settling test.
    p.add_argument("--validate-perturbations", action="store_true", default=True,
                   help="After perturbing free-body xyz, settle the env for "
                        "--validation-settle-steps and check that no movable "
                        "body moved >--validation-max-launch m. Resample on "
                        "failure. Default ON.")
    p.add_argument("--no-validate-perturbations", dest="validate_perturbations",
                   action="store_false",
                   help="Disable perturbation validity check (old behaviour).")
    p.add_argument("--validation-settle-steps", type=int, default=20,
                   help="Zero-action settling steps before the end-state "
                        "check. 20 is enough for any non-pathological initial "
                        "condition to come to rest in LIBERO.")
    p.add_argument("--validation-max-final-speed", type=float, default=0.02,
                   help="After settling, all movable bodies must have linear "
                        "speed below this (m/s). Means the object came to rest "
                        "instead of continuing to bounce / fall. Default 2cm/s.")
    p.add_argument("--validation-min-table-z", type=float, default=0.40,
                   help="No movable body's final z may be below this. Catches "
                        "objects that fell through the floor.")
    p.add_argument("--validation-max-drift", type=float, default=10.0,
                   help="DISABLED by default (very large value). After settling, "
                        "body position can be far from where we wrote it as long "
                        "as it's stable (speed) and above the table (z). The "
                        "resolver moving an object to a different stable resting "
                        "spot is fine — the policy / dynamics model see the "
                        "actual sim state. Set this low (e.g. 0.15) to enforce "
                        "intent-matching.")
    p.add_argument("--validation-max-retries", type=int, default=40,
                   help="Max perturbation resamples before giving up and "
                        "using the last sample anyway. Multi-object scenes "
                        "(task 0/1/7 have 8 free bodies) need more retries "
                        "since P(any pair penetrates) grows with object count.")
    p.add_argument("--mppi-early-exit-calls", type=int, default=6,
                   help="Number of consecutive MPPI calls with no signal "
                        "(spread<eps AND |refined-nominal|<eps) before giving "
                        "up MPPI for the rest of this episode and executing "
                        "the Pi0.5 prior directly. Saves ~150s/trial on the "
                        "unrescuable stuck-policy failure mode at high "
                        "perturbations. Set to 0 to disable.")
    p.add_argument("--mppi-early-exit-spread", type=float, default=0.08,
                   help="Spread threshold for early-exit no-signal detection.")
    p.add_argument("--mppi-early-exit-delta", type=float, default=0.01,
                   help="|refined-nominal| threshold for early-exit detection.")
    # Trust-region gate: only commit MPPI when it claims a meaningful win.
    # Without this, MPPI averages noise from K candidates around an already-
    # correct prior and can derail trials that baseline would have solved.
    # The 3-seed matrix at N=10 showed MPPI's lift is monotonic in baseline
    # weakness — +30pp when baseline=40%, but -20pp when baseline=90%. The
    # gate suppresses the regression at high baselines without losing the
    # rescue at low baselines.
    p.add_argument("--mppi-trust-threshold", type=float, default=0.0,
                   help="Only use MPPI's refined chunk if "
                        "best_sample_cost < nominal_cost * (1 - threshold). "
                        "0.0 disables the gate (current behaviour). Try 0.03 "
                        "to suppress noise-level refinements. Higher = "
                        "more conservative (use prior more often).")
    return p.parse_args()


# -------------------- LIBERO helpers --------------------


def _quat2axisangle(quat):
    quat = quat.copy()
    if quat[3] > 1.0: quat[3] = 1.0
    elif quat[3] < -1.0: quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0): return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def _candidate_perturbed_state(init_state, free_joint_addrs, perturbation_m, rng):
    """Sample one perturbed init state without applying it to the env."""
    state = init_state.clone() if hasattr(init_state, "clone") else init_state.copy()
    for addr in free_joint_addrs:
        state[addr:addr + 3] += rng.normal(0, perturbation_m, size=3)
    return state


def _state_is_settled(env, free_body_ids, settle_steps: int,
                       max_final_speed: float, min_table_z: float,
                       max_drift_from_intended: float,
                       intended_positions: dict | None = None) -> tuple[bool, str]:
    """After perturbation, settle and check that the FINAL state is valid.

    We can't use per-step motion as the gate (the resolver's first-step
    correction is ~7cm even on unperturbed states). Instead check the
    end-state:
      1. Final velocity of every movable body is below `max_final_speed`
         (the object has settled, isn't bouncing around)
      2. No movable body's final z is below `min_table_z` (didn't fall
         through the floor)
      3. If we know what position we INTENDED for each body (the perturbed
         qpos write), the settled position is within `max_drift_from_intended`
         of that intent (resolver didn't have to shove it far)
    """
    sim = env.env.sim
    if not free_body_ids:
        return True, "no movable bodies to check"
    for _ in range(settle_steps):
        env.step(LIBERO_DUMMY_ACTION)

    # End-state velocity: free joint qvel for each body's free joint
    # gives 6-DoF (linear 3 + angular 3) twist. Use linear component magnitude.
    mdl = sim.model
    d = sim.data
    for bid in free_body_ids:
        # Find the qvel slice for this body's free joint.
        for j in range(mdl.njnt):
            if mdl.jnt_type[j] == 0 and int(mdl.jnt_bodyid[j]) == bid:
                qvel_addr = int(mdl.jnt_dofadr[j])  # free joint is 6 DOFs
                v_lin = d.qvel[qvel_addr:qvel_addr+3]
                speed = float(np.linalg.norm(v_lin))
                if speed > max_final_speed:
                    return False, f"body {bid} settled at speed {speed*100:.1f}cm/s (>{max_final_speed*100:.0f}cm/s)"
                break

        z = float(d.body_xpos[bid][2])
        if z < min_table_z:
            return False, f"body {bid} below table (z={z:.3f} < {min_table_z:.3f})"

        if intended_positions is not None and bid in intended_positions:
            drift = float(np.linalg.norm(d.body_xpos[bid] - intended_positions[bid]))
            if drift > max_drift_from_intended:
                return False, f"body {bid} drifted {drift*100:.1f}cm from intended ({max_drift_from_intended*100:.0f}cm cap)"

    return True, "ok"


def perturb_object_positions(env, init_state, perturbation_m, rng,
                              validate: bool = True,
                              settle_steps: int = 20,
                              max_final_speed: float = 0.02,
                              min_table_z: float = 0.40,
                              max_drift_from_intended: float = 10.0,
                              max_retries: int = 40):
    """Sample a perturbed init state that survives a settling check.

    Iterates: sample → set_init_state → settle for `settle_steps` zero-action
    steps → check that no movable body moved >`max_launch_per_step` per step.
    If invalid, resample. After `max_retries`, fall back to the unperturbed
    init_state (rare, only happens at very large σ).
    """
    sim = env.env.sim
    model = sim.model
    free_joint_addrs = [
        int(model.jnt_qposadr[j]) for j in range(model.njnt) if model.jnt_type[j] == 0
    ]
    # Resolve free-joint body ids by walking joint→body map.
    free_body_ids: list[int] = []
    for j in range(model.njnt):
        if model.jnt_type[j] == 0:
            free_body_ids.append(int(model.jnt_bodyid[j]))

    if not validate or perturbation_m <= 0.0:
        # Legacy path: single sample, no validation, no env-step side effects.
        state = _candidate_perturbed_state(init_state, free_joint_addrs, perturbation_m, rng)
        return state

    last_state = None
    last_reason = ""
    for attempt in range(max_retries):
        candidate = _candidate_perturbed_state(init_state, free_joint_addrs, perturbation_m, rng)
        # Read the candidate's intended xyz for each free body so the validity
        # check can verify the resolver didn't move it far during settling.
        intended = {}
        for j in range(model.njnt):
            if model.jnt_type[j] == 0:
                addr = int(model.jnt_qposadr[j])
                bid = int(model.jnt_bodyid[j])
                intended[bid] = np.array(candidate[addr:addr+3], dtype=np.float64)
        env.set_init_state(candidate)
        ok, reason = _state_is_settled(
            env, free_body_ids, settle_steps,
            max_final_speed=max_final_speed,
            min_table_z=min_table_z,
            max_drift_from_intended=max_drift_from_intended,
            intended_positions=intended,
        )
        if ok:
            # Restore the env to the candidate so the caller can use it as the
            # "starting state" — we re-apply set_init_state to undo settling.
            env.set_init_state(candidate)
            return candidate
        last_state = candidate
        last_reason = reason
        # else: loop; the env will be re-set with a fresh candidate next attempt.

    print(f"  [perturb] no valid sample after {max_retries} retries "
          f"(perturbation σ={perturbation_m*100:.1f}cm; last reason: {last_reason}). "
          f"Using last sample anyway.", flush=True)
    # Re-apply the last attempt's state so the caller's env is in a consistent
    # state. We accept whatever it is — bias toward returning *some* perturbation
    # over collapsing to the unperturbed init state.
    if last_state is None:
        last_state = _candidate_perturbed_state(init_state, free_joint_addrs, perturbation_m, rng)
    env.set_init_state(last_state)
    return last_state


def build_obs_element(obs, task_description):
    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, RESIZE_SIZE, RESIZE_SIZE))
    wrist = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist, RESIZE_SIZE, RESIZE_SIZE))
    state = np.concatenate((
        obs["robot0_eef_pos"], _quat2axisangle(obs["robot0_eef_quat"]),
        obs["robot0_gripper_qpos"],
    ))
    return {
        "observation/image": img, "observation/wrist_image": wrist,
        "observation/state": state, "prompt": str(task_description),
    }


def detect_contact_in_chunk(action_chunk: np.ndarray) -> bool:
    gripper_cmds = action_chunk[:, -1] if action_chunk.ndim == 2 else action_chunk[-1:]
    for i in range(1, len(gripper_cmds)):
        if (gripper_cmds[i] > 0) != (gripper_cmds[i - 1] > 0):
            return True
    return False


# -------------------- MPPI cost using LIBERO env directly --------------------


def _robot_geom_mask(model) -> np.ndarray:
    """Identify robot-owned geoms (for filtering contacts)."""
    robot_prefixes = ("robot0_", "gripper0_", "panda")
    is_robot_body = np.zeros(model.nbody, dtype=bool)
    for b in range(model.nbody):
        name = model.body_id2name(b) if hasattr(model, "body_id2name") else ""
        if any((name or "").startswith(p) for p in robot_prefixes):
            is_robot_body[b] = True
    for b in range(model.nbody):
        parent = b
        while parent > 0:
            if is_robot_body[parent]:
                is_robot_body[b] = True
                break
            parent = model.body_parentid[parent]
    is_robot_geom = np.zeros(model.ngeom, dtype=bool)
    for g in range(model.ngeom):
        is_robot_geom[g] = is_robot_body[model.geom_bodyid[g]]
    return is_robot_geom


def compute_step_cost(sim, is_robot_geom, w_collision, w_joint_limit):
    """Compute per-step physics cost terms from current sim state."""
    # Robot-involved penetration depth
    pen = 0.0
    for i in range(sim.data.ncon):
        c = sim.data.contact[i]
        depth = max(0.0, -float(c.dist))
        if depth == 0:
            continue
        if is_robot_geom[c.geom1] or is_robot_geom[c.geom2]:
            pen += depth

    # Joint-limit violations (limited joints only)
    jl_viol = 0.0
    model = sim.model
    for j in range(model.njnt):
        if not model.jnt_limited[j]:
            continue
        jt = model.jnt_type[j]
        # Hinge / slide joints have a single qpos value at jnt_qposadr[j]
        if jt in (2, 3):  # mjJNT_SLIDE=2, mjJNT_HINGE=3
            q = sim.data.qpos[model.jnt_qposadr[j]]
            lo, hi = model.jnt_range[j]
            margin = min(q - lo, hi - q)
            if margin < 0:
                jl_viol += -margin

    return w_collision * pen + w_joint_limit * jl_viol


def _resolve_target_body_id(sim, candidate_body_ids: list[int], goal_xyz: np.ndarray) -> int:
    """Pick whichever candidate body is currently farthest from ``goal_xyz``.

    This is how staging emerges without explicit stage logic. For "put both X
    and Y in basket": as soon as X lands in the basket, its distance to
    goal_xyz drops near 0, and Y (still on the table) becomes the new tracked
    body — automatically retargeting the cost.
    """
    pos = sim.data.body_xpos
    best_id = candidate_body_ids[0]
    best_d = -1.0
    for bid in candidate_body_ids:
        d = float(np.linalg.norm(pos[bid] - goal_xyz))
        if d > best_d:
            best_d = d
            best_id = bid
    return best_id


def mppi_refine(
    env,
    prior_action: np.ndarray,    # [H, 7] from Pi0.5
    goal_xyz: np.ndarray,
    tracked_body_id: int,        # which body's xyz the target-distance cost measures
    K: int,
    sigma: float,
    lam: float,
    num_iterations: int,
    w_target: float,
    w_approach: float,
    w_anchor: float,
    w_collision: float,
    w_joint_limit: float,
    w_grip: float,
    grip_radius: float,
    is_robot_geom: np.ndarray,
    replan_steps: int,
    rng: np.random.Generator,
    scorer_type: str = "physics",
    q_fn=None,
    hidden_state: np.ndarray = None,
    device=None,
):
    """Run MPPI refinement on top of Pi0.5's action chunk using LIBERO env.

    Cost decomposition (in priority of providing gradient signal):
        - target_distance: ||tracked_body - goal_xyz|| (final step). The
          end-state objective.
        - approach: ||EE - tracked_body|| (final step). When the object isn't
          moving yet (no grasp), the target term is constant across candidates;
          the approach term gives discriminative signal toward the object.
        - collision_penetration / joint_limit: per-step robot-only physics
          violations.
        - anchor: ||candidate - prior||² over the noised window only.

    Robosuite's done-flag is latched: a candidate that satisfies the BDDL goal
    sets ``env.env.done = True``, and a subsequent ``env.step`` raises. We
    clear the flag after every state restore and also break out of a
    candidate's rollout on done so the partial cost is kept.

    Returns:
        refined: [H, 7] same shape as prior_action
        diag: dict with mppi diagnostics
    """
    H, action_dim = prior_action.shape
    sim = env.env.sim
    saved_state = sim.get_state().flatten()
    # Robosuite increments env.env.timestep on every step. Candidate
    # evaluations call env.step → timestep grows by K*replan_steps per MPPI
    # call. After enough MPPI calls it exceeds env.env.horizon and `done`
    # latches True — set_state_from_flattened restores physics but NOT the
    # step counter, so the crash persists. Save+restore these scalars
    # alongside the sim state.
    saved_timestep = getattr(env.env, "timestep", None)
    saved_cur_time = getattr(env.env, "cur_time", None)
    saved_done = getattr(env.env, "done", False)
    # Only the first `noised_horizon` actions are actually executed during a
    # rollout (we replan every replan_steps), so noising beyond that wastes
    # exploration and adds garbage to the anchor term.
    noised_horizon = min(replan_steps, H)

    def _restore_env_scalars():
        if saved_timestep is not None:
            env.env.timestep = saved_timestep
        if saved_cur_time is not None:
            env.env.cur_time = saved_cur_time
        env.env.done = saved_done

    def evaluate(candidate: np.ndarray) -> float:
        """Step env forward over the noised window; compute scalar cost; restore."""
        sim.set_state_from_flattened(saved_state)
        sim.forward()
        _restore_env_scalars()

        step_cost_sum = 0.0
        last_ee_pos = None
        # Grip-stability: track ||EE - tracked_obj|| per step. If the offset
        # grows (object slipping away from EE), we penalize. The penalty
        # rewards candidates that *keep* the object near the EE — the dominant
        # failure mode (71% of trace verdicts) is "reached object, then lost it."
        ee_obj_offsets: list[float] = []
        for action in candidate[:noised_horizon]:
            try:
                obs_local, _, d_flag, _ = env.step(action.tolist())
            except ValueError:
                break  # defensive: robosuite still raised despite the reset
            last_ee_pos = np.asarray(obs_local["robot0_eef_pos"], dtype=np.float64)
            ee_obj_offsets.append(float(
                np.linalg.norm(last_ee_pos - sim.data.body_xpos[tracked_body_id])
            ))
            step_cost_sum += compute_step_cost(
                sim, is_robot_geom, w_collision, w_joint_limit,
            )
            if d_flag:
                break  # candidate completed the task; partial cost is fine

        final_tracked = sim.data.body_xpos[tracked_body_id]
        target_cost = float(np.linalg.norm(final_tracked - goal_xyz))
        if last_ee_pos is not None:
            approach_cost = float(np.linalg.norm(last_ee_pos - final_tracked))
        else:
            approach_cost = 0.0
        # Grip-instability: penalize positive growth of ||EE - obj|| but
        # ONLY when the previous step's offset was within `grip_radius` —
        # i.e., the EE was actually in grasp range, so a growth means
        # "object slipping out of grasp," not "EE jittering during approach."
        # The first matrix iteration without this gating gave −13.3pp pooled
        # because approach-phase jitter dominated the cost.
        grip_cost = 0.0
        if len(ee_obj_offsets) >= 2:
            offsets = np.array(ee_obj_offsets)
            deltas = np.diff(offsets)
            in_grasp = (offsets[:-1] < grip_radius).astype(np.float64)
            grip_cost = float(np.sum(np.clip(deltas, 0.0, None) * in_grasp))
        anchor_cost = float(np.sum(
            (candidate[:noised_horizon] - prior_action[:noised_horizon]) ** 2
        ))
        return (step_cost_sum
                + w_target * target_cost
                + w_approach * approach_cost
                + w_grip * grip_cost
                + w_anchor * anchor_cost)

    def score_batch_learned(chunks: np.ndarray) -> np.ndarray:
        """Learned-WM scorer path: no env simulation, just Q(h, candidate)."""
        # Anchor term still matters — without it Q can pick arbitrarily far actions.
        learned_cost = score_chunks_with_q(q_fn, hidden_state, chunks, device)
        anchor = np.sum(
            (chunks[:, :noised_horizon] - prior_action[None, :noised_horizon]) ** 2,
            axis=(1, 2),
        )
        return learned_cost + w_anchor * anchor

    nominal = prior_action.copy()
    sample_costs = np.zeros(K)
    if scorer_type == "physics":
        nominal_cost = evaluate(nominal)
    else:  # learned-wm
        nominal_cost = float(score_batch_learned(nominal[None, ...])[0])
    final_weights = None

    t0 = time.time()
    for it in range(num_iterations):
        # Noise only the first noised_horizon positions (rest is unchanged prior).
        noise = np.zeros((K, H, action_dim), dtype=prior_action.dtype)
        noise[:, :noised_horizon] = (
            rng.standard_normal((K, noised_horizon, action_dim)) * sigma
        ).astype(prior_action.dtype)
        candidates = np.clip(nominal[None, ...] + noise, -1.0, 1.0)
        if scorer_type == "physics":
            for k in range(K):
                sample_costs[k] = evaluate(candidates[k])
        else:  # learned-wm: one batched forward pass
            sample_costs[:] = score_batch_learned(candidates)

        costs_shifted = sample_costs - sample_costs.min()
        weights = np.exp(-costs_shifted / lam)
        weights = weights / weights.sum()
        final_weights = weights

        weighted_perturbation = np.einsum(
            "k,khd->hd", weights, candidates - nominal[None, ...],
        )
        nominal = np.clip(nominal + weighted_perturbation, -1.0, 1.0).astype(prior_action.dtype)

    if scorer_type == "physics":
        refined_cost = evaluate(nominal)
    else:
        refined_cost = float(score_batch_learned(nominal[None, ...])[0])

    # Restore env to pre-MPPI state so the outer execution proceeds normally.
    sim.set_state_from_flattened(saved_state)
    sim.forward()
    _restore_env_scalars()

    diag = {
        "K": K,
        "iterations": num_iterations,
        "nominal_cost": float(nominal_cost),
        "refined_cost": float(refined_cost),
        "best_sample_cost": float(sample_costs.min()),
        "worst_sample_cost": float(sample_costs.max()),
        "cost_spread": float(sample_costs.max() - sample_costs.min()),
        "weight_entropy": float(-np.sum(final_weights * np.log(final_weights + 1e-12))),
        "wall_seconds": time.time() - t0,
    }
    return nominal, diag


# -------------------- Episode + suite eval --------------------


def run_episode(
    *, policy, env, init_state, task_description, goal_xyz, tracked_body_ids,
    is_robot_geom, perturbation_m, perturb_rng, max_steps, args, mode, mppi_rng,
    q_fn=None, device=None,
):
    env.reset()
    init_state_np = init_state.clone() if hasattr(init_state, "clone") else init_state.copy()
    if perturbation_m > 0:
        init_state_np = perturb_object_positions(
            env, init_state_np, perturbation_m, perturb_rng,
            validate=args.validate_perturbations,
            settle_steps=args.validation_settle_steps,
            max_final_speed=args.validation_max_final_speed,
            min_table_z=args.validation_min_table_z,
            max_drift_from_intended=args.validation_max_drift,
            max_retries=args.validation_max_retries,
        )
    obs = env.set_init_state(init_state_np)
    # Capture the actual MjSim state right after init. This is the only way
    # to make perturbed episodes reproducible — perturbation sampling is
    # deterministic given the rng, but env.set_init_state's effect on the
    # raw MjData state depends on robosuite internals we don't control. The
    # validity-check perturb_object_positions also calls sim.forward() and
    # extra steps during settling; capturing the post-settle state is what
    # we need for any future state-restoration diagnostic (and the planner's
    # save/restore loop).
    try:
        episode_init_sim_state = env.env.sim.get_state().flatten().copy()
    except Exception:
        episode_init_sim_state = None

    plan = collections.deque()
    done = False
    t = 0
    episode_start = time.time()
    last_action_chunk = None
    refinements_run = 0
    total_decisions = 0
    cost_improvements: list[tuple[float, float]] = []
    # Phase B logging buffer: appended per MPPI call. Each entry is (obs at the
    # decision point, the Pi0.5 prior chunk, the refined chunk produced by
    # MPPI). The Pi0.5 LoRA target = refined chunk, given the same obs.
    refined_log: list[dict] = []
    # Failure trace buffer: per-step EE + object positions + gripper command.
    # Camera frames at periodic snapshots only (full per-step frames would
    # produce ~500MB per failed episode). Used to diagnose stuck-policy mode.
    trace_step: list[dict] = []
    trace_frames: list[dict] = []
    FRAME_EVERY = 50
    # PhysVLA: full per-step physics-rich trace. Captures the supervision
    # signals for the latent dynamics model + aux heads. Saved for every
    # episode (success or failure). Images are stored every Nth step to
    # keep dataset size manageable; physics scalars saved per step.
    phys_step: list[dict] = []
    phys_images: list[dict] = []
    # Build the list of all body ids whose pose we'll track. These are the
    # non-robot, non-mount, non-table bodies (the actual movable scene
    # objects). Resolved lazily on first env.step because env.env.sim is
    # rebuilt at reset.
    phys_object_body_ids: list[tuple[int, str]] | None = None
    phys_ee_body_id: int | None = None
    # Early-exit state: track consecutive no-signal MPPI calls within this
    # episode. If we hit the threshold we stop trying MPPI for the rest of
    # the episode (executes pure Pi0.5 prior instead). Resets per-episode.
    no_signal_streak = 0
    mppi_disabled_this_episode = False
    early_exit_threshold = max(0, int(args.mppi_early_exit_calls))

    while t < max_steps + args.num_steps_wait:
        if time.time() - episode_start > args.episode_timeout:
            break

        if t < args.num_steps_wait:
            obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
            t += 1
            continue

        if not plan:
            total_decisions += 1
            obs_element = build_obs_element(obs, task_description)
            result = policy.infer(dict(obs_element))
            initial_action = np.asarray(result["actions"], dtype=np.float32)
            # vlm_features = Pi0.5's hidden state for Q(h, a) scoring.
            # Physics path doesn't need it, but pull it conditionally so the
            # cost of extraction is paid only when needed.
            hidden_state = None
            if args.scorer_type == "learned-wm":
                if "vlm_features" not in result:
                    raise KeyError(
                        "Policy did not return 'vlm_features' — required for "
                        "scorer-type=learned-wm. Make sure the openpi build "
                        "exposes vlm hidden states."
                    )
                hidden_state = np.asarray(result["vlm_features"], dtype=np.float32)

            use_mppi = False
            if mode == "mppi" and not mppi_disabled_this_episode:
                if args.mppi_trigger == "always":
                    use_mppi = True
                elif (args.mppi_trigger == "contact" and last_action_chunk is not None
                        and detect_contact_in_chunk(last_action_chunk)):
                    use_mppi = True

            if use_mppi:
                # Pick the tracked body for this decision: among candidates, the
                # one currently farthest from goal_xyz still needs work most.
                tracked_id = _resolve_target_body_id(
                    env.env.sim, tracked_body_ids, goal_xyz,
                )
                refined, diag = mppi_refine(
                    env, initial_action, goal_xyz, tracked_body_id=tracked_id,
                    K=args.mppi_K, sigma=args.mppi_sigma, lam=args.mppi_lambda,
                    num_iterations=args.mppi_iterations,
                    w_target=args.w_target, w_approach=args.w_approach,
                    w_anchor=args.w_anchor, w_grip=args.w_grip,
                    grip_radius=args.grip_radius,
                    w_collision=args.w_collision, w_joint_limit=args.w_joint_limit,
                    is_robot_geom=is_robot_geom, replan_steps=args.replan_steps,
                    rng=mppi_rng,
                    scorer_type=args.scorer_type, q_fn=q_fn,
                    hidden_state=hidden_state, device=device,
                )
                # Trust-region gate: only commit MPPI's refined chunk if its
                # best candidate beat the nominal by a meaningful margin. The
                # N=10 matrix showed MPPI's softmin-average of K candidates
                # introduces noise that can derail trials Pi0.5 was solving
                # — the gate keeps us on the prior unless MPPI's best sample
                # claims a real win.
                trust = args.mppi_trust_threshold
                claimed_win = diag["nominal_cost"] - diag["best_sample_cost"]
                trust_floor = trust * abs(diag["nominal_cost"])
                gate_passed = (trust <= 0.0) or (claimed_win >= trust_floor)
                action_chunk = refined if gate_passed else initial_action
                refinements_run += 1
                cost_improvements.append((diag["nominal_cost"], diag["refined_cost"]))
                if args.log_refined_rollouts:
                    refined_log.append({
                        "t": t,
                        "image": obs_element["observation/image"].copy(),
                        "wrist_image": obs_element["observation/wrist_image"].copy(),
                        "state": obs_element["observation/state"].copy(),
                        "prompt": obs_element["prompt"],
                        "prior_action": initial_action.copy(),
                        "refined_action": refined.copy(),
                        "nominal_cost": float(diag["nominal_cost"]),
                        "refined_cost": float(diag["refined_cost"]),
                    })
                if args.verbose_mppi:
                    import math as _m
                    log_K = _m.log(args.mppi_K)
                    # Entropy gap from uniform: log(K) - entropy. 0 = uniform
                    # weights (MPPI noop), large = peaked weights (MPPI active).
                    ent_gap = log_K - diag["weight_entropy"]
                    gate_tag = "" if gate_passed else " GATE-REJECT→prior"
                    print(
                        f"    [MPPI t={t}] nominal={diag['nominal_cost']:.3f} "
                        f"refined={diag['refined_cost']:.3f} "
                        f"best={diag['best_sample_cost']:.3f} "
                        f"spread={diag['cost_spread']:.4f} "
                        f"ent_gap_from_uniform={ent_gap:.4f} "
                        f"(refined-nominal={diag['refined_cost']-diag['nominal_cost']:+.3f}, "
                        f"best-nominal={diag['best_sample_cost']-diag['nominal_cost']:+.3f}) "
                        f"wall={diag['wall_seconds']*1000:.0f}ms{gate_tag}",
                        flush=True,
                    )

                # Early-exit accounting: if this call shows no useful signal
                # (cost barely varies across candidates AND refined is barely
                # different from nominal), bump the no-signal streak. After N
                # consecutive no-signal calls, give up MPPI for the rest of
                # this episode — Pi0.5 is stuck out-of-workspace and MPPI
                # cannot help.
                if early_exit_threshold > 0:
                    abs_delta = abs(diag["refined_cost"] - diag["nominal_cost"])
                    if (diag["cost_spread"] < args.mppi_early_exit_spread
                            and abs_delta < args.mppi_early_exit_delta):
                        no_signal_streak += 1
                    else:
                        no_signal_streak = 0
                    if no_signal_streak >= early_exit_threshold:
                        mppi_disabled_this_episode = True
                        print(f"    [MPPI t={t}] early-exit: {no_signal_streak} "
                              f"consecutive no-signal calls. Falling back to "
                              f"Pi0.5 prior for the rest of the episode.",
                              flush=True)
            else:
                action_chunk = initial_action

            last_action_chunk = action_chunk
            plan.extend(action_chunk[:args.replan_steps])

        action = plan.popleft()
        obs, _, done, _ = env.step(action.tolist())

        # PhysVLA physics trace: per-step contact wrenches, body poses, etc.
        # Saved for every episode (success or failure). This is the dataset
        # the latent dynamics model + aux heads train on.
        if args.log_physics_traces:
            sim_now = env.env.sim
            mdl = sim_now.model
            d = sim_now.data
            if phys_object_body_ids is None:
                # First step: resolve which bodies to log. Anything not the
                # robot, the mount, the floor, or the table is a "scene object."
                skip_prefixes = ("robot0_", "gripper0_", "panda", "link",
                                 "world", "mount0_", "floor", "table")
                phys_object_body_ids = []
                for b in range(1, mdl.nbody):
                    name = mdl.body_id2name(b) or ""
                    if not any(name.startswith(p) for p in skip_prefixes):
                        phys_object_body_ids.append((b, name))
                # EE body: try common robosuite Franka names
                for nm in ("robot0_right_hand", "gripper0_eef", "robot0_link7"):
                    try:
                        phys_ee_body_id = int(mdl.body_name2id(nm))
                        break
                    except Exception:
                        continue
                if phys_ee_body_id is None:
                    phys_ee_body_id = mdl.nbody - 1

            ee_wrench = d.cfrc_ext[phys_ee_body_id].copy()  # [6] force+torque
            obj_states = []
            for bid, _ in phys_object_body_ids:
                obj_states.append({
                    "pos": d.body_xpos[bid].copy(),
                    "quat": d.body_xquat[bid].copy(),
                    "wrench": d.cfrc_ext[bid].copy(),
                })

            # Per-contact summary — small structured list (active contacts only)
            contacts = []
            for ci in range(int(d.ncon)):
                c = d.contact[ci]
                depth = max(0.0, -float(c.dist))
                if depth < 1e-6:
                    continue
                contacts.append({
                    "geom1": int(c.geom1),
                    "geom2": int(c.geom2),
                    "pos": np.array(c.pos, dtype=np.float64).copy(),
                    "depth": depth,
                })

            phys_step.append({
                "t": t,
                "ee_pos": np.asarray(obs["robot0_eef_pos"], dtype=np.float64).copy(),
                "ee_quat": np.asarray(obs["robot0_eef_quat"], dtype=np.float64).copy(),
                "ee_wrench": ee_wrench,
                "gripper_qpos": np.asarray(obs["robot0_gripper_qpos"], dtype=np.float64).copy(),
                "action": np.asarray(action, dtype=np.float64).copy(),
                "qpos": d.qpos.copy(),
                "qvel": d.qvel.copy(),
                "objects": obj_states,
                "contacts": contacts,
            })
            if t % args.physics_trace_image_stride == 0:
                phys_images.append({
                    "t": t,
                    "image": np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]).copy(),
                    "wrist_image": np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1]).copy(),
                })

        # Failure trace: cheap per-step data + occasional camera snapshots.
        # We always collect during the episode; only persist if it fails.
        if args.log_failure_traces:
            sim_now = env.env.sim
            object_xyz = {
                f"body_{bid}_{sim_now.model.body_id2name(bid) or bid}":
                    sim_now.data.body_xpos[bid].copy().tolist()
                for bid in tracked_body_ids
            }
            trace_step.append({
                "t": t,
                "ee_pos": [float(x) for x in obs["robot0_eef_pos"]],
                "gripper_cmd": float(action[-1]) if hasattr(action, "__len__") else None,
                "ee_action_delta": [float(x) for x in action[:6]],
                "objects": object_xyz,
            })
            if t % FRAME_EVERY == 0:
                trace_frames.append({
                    "t": t,
                    "agentview": np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]).copy(),
                })

        if done:
            break
        t += 1

    # Phase B: only save successful trajectories — failed episodes don't tell us
    # what the policy *should* have done, just that it didn't. Distilling
    # failures would teach Pi0.5 to imitate bad rollouts.
    saved_path = None
    if args.log_refined_rollouts and bool(done) and refined_log:
        log_dir = pathlib.Path(args.log_refined_rollouts)
        log_dir.mkdir(parents=True, exist_ok=True)
        stem = (f"{args.task_suite}_task{args.task_idx}_"
                f"seed{args.seed}_pert{args.perturbation_cm}cm_"
                f"ts{int(time.time()*1000)}")
        saved_path = log_dir / f"{stem}.npz"
        np.savez_compressed(
            saved_path,
            t=np.array([e["t"] for e in refined_log], dtype=np.int32),
            image=np.stack([e["image"] for e in refined_log]),
            wrist_image=np.stack([e["wrist_image"] for e in refined_log]),
            state=np.stack([e["state"] for e in refined_log]),
            prompt=np.array([e["prompt"] for e in refined_log], dtype=object),
            prior_action=np.stack([e["prior_action"] for e in refined_log]),
            refined_action=np.stack([e["refined_action"] for e in refined_log]),
            nominal_cost=np.array([e["nominal_cost"] for e in refined_log]),
            refined_cost=np.array([e["refined_cost"] for e in refined_log]),
            task_description=str(task_description),
            steps=t,
        )

    # PhysVLA physics trace: dump for every episode (success and failure).
    # This is the foundation dataset for the latent dynamics model.
    phys_path = None
    if args.log_physics_traces and phys_step:
        phys_dir = pathlib.Path(args.log_physics_traces)
        phys_dir.mkdir(parents=True, exist_ok=True)
        outcome = "OK" if bool(done) else "FAIL"
        stem = (f"PHYS_{outcome}_{args.task_suite}_task{args.task_idx}_"
                f"seed{args.seed}_pert{args.perturbation_cm}cm_"
                f"{mode}_ts{int(time.time()*1000)}")
        phys_path = phys_dir / f"{stem}.npz"
        # Pack per-step physics arrays
        ts = np.array([s["t"] for s in phys_step], dtype=np.int32)
        ee_pos = np.stack([s["ee_pos"] for s in phys_step])
        ee_quat = np.stack([s["ee_quat"] for s in phys_step])
        ee_wrench = np.stack([s["ee_wrench"] for s in phys_step])
        gripper_qpos = np.stack([s["gripper_qpos"] for s in phys_step])
        actions = np.stack([s["action"] for s in phys_step])
        qpos = np.stack([s["qpos"] for s in phys_step])
        qvel = np.stack([s["qvel"] for s in phys_step])
        # Per-object trajectories, keyed by body id+name
        obj_names = [n for _, n in (phys_object_body_ids or [])]
        obj_pos = np.stack([
            np.stack([s["objects"][i]["pos"] for i in range(len(obj_names))])
            for s in phys_step
        ]) if obj_names else np.zeros((len(phys_step), 0, 3))
        obj_quat = np.stack([
            np.stack([s["objects"][i]["quat"] for i in range(len(obj_names))])
            for s in phys_step
        ]) if obj_names else np.zeros((len(phys_step), 0, 4))
        obj_wrench = np.stack([
            np.stack([s["objects"][i]["wrench"] for i in range(len(obj_names))])
            for s in phys_step
        ]) if obj_names else np.zeros((len(phys_step), 0, 6))
        # Contacts: variable-length per step → save as object array of dicts
        contact_records = np.array([s["contacts"] for s in phys_step], dtype=object)
        # Images at sparse stride
        img_ts = np.array([f["t"] for f in phys_images], dtype=np.int32) \
            if phys_images else np.zeros(0, dtype=np.int32)
        imgs = (np.stack([f["image"] for f in phys_images])
                if phys_images else np.zeros((0, 0, 0, 3), dtype=np.uint8))
        wrist_imgs = (np.stack([f["wrist_image"] for f in phys_images])
                      if phys_images else np.zeros((0, 0, 0, 3), dtype=np.uint8))
        np.savez_compressed(
            phys_path,
            t=ts,
            ee_pos=ee_pos, ee_quat=ee_quat, ee_wrench=ee_wrench,
            gripper_qpos=gripper_qpos, action=actions,
            qpos=qpos, qvel=qvel,
            object_names=np.array(obj_names, dtype=object),
            object_pos=obj_pos, object_quat=obj_quat, object_wrench=obj_wrench,
            contacts=contact_records,
            image_t=img_ts, image=imgs, wrist_image=wrist_imgs,
            task_description=str(task_description),
            prompt=str(task_description),
            success=bool(done),
            steps_total=t, mode=mode,
            ee_body_id=int(phys_ee_body_id) if phys_ee_body_id is not None else -1,
            image_stride=int(args.physics_trace_image_stride),
            # Episode initial MjSim state — captured right after set_init_state,
            # post validity-check settling. Enables deterministic replay via
            # sim.set_state_from_flattened(init_sim_state). Required for any
            # state-restoration diagnostic or planner that needs to revisit
            # the same physical state across runs.
            init_sim_state=(episode_init_sim_state
                            if episode_init_sim_state is not None
                            else np.zeros(0, dtype=np.float64)),
            init_state_libero=np.asarray(init_state_np, dtype=np.float64),
            perturbation_cm=float(args.perturbation_cm),
        )

    # Failure trace: dump only if episode failed. Successful trajectories are
    # the Phase B distillation target; failed ones get traced for diagnosis.
    trace_path = None
    if args.log_failure_traces and not bool(done) and trace_step:
        trace_dir = pathlib.Path(args.log_failure_traces)
        trace_dir.mkdir(parents=True, exist_ok=True)
        stem = (f"FAIL_{args.task_suite}_task{args.task_idx}_"
                f"seed{args.seed}_pert{args.perturbation_cm}cm_"
                f"{mode}_ts{int(time.time()*1000)}")
        trace_path = trace_dir / f"{stem}.npz"
        # Flatten per-step records into arrays for cheap loading
        ts = np.array([s["t"] for s in trace_step], dtype=np.int32)
        ee_pos = np.array([s["ee_pos"] for s in trace_step], dtype=np.float64)
        gripper = np.array([s["gripper_cmd"] for s in trace_step], dtype=np.float64)
        ee_delta = np.array([s["ee_action_delta"] for s in trace_step], dtype=np.float64)
        # Object xyz keyed by body name
        obj_keys = list(trace_step[0]["objects"].keys()) if trace_step else []
        obj_xyz = {
            k: np.array([s["objects"][k] for s in trace_step], dtype=np.float64)
            for k in obj_keys
        }
        frame_ts = np.array([f["t"] for f in trace_frames], dtype=np.int32) \
            if trace_frames else np.zeros(0, dtype=np.int32)
        frame_images = (np.stack([f["agentview"] for f in trace_frames])
                        if trace_frames else np.zeros((0, 0, 0, 3), dtype=np.uint8))
        np.savez_compressed(
            trace_path,
            t=ts, ee_pos=ee_pos, gripper_cmd=gripper, ee_action_delta=ee_delta,
            frame_t=frame_ts, frame_images=frame_images,
            mode=mode, task_description=str(task_description),
            steps_total=t, refinements_run=refinements_run,
            tracked_body_names=np.array(obj_keys, dtype=object),
            **{f"obj_{k}_xyz": v for k, v in obj_xyz.items()},
        )

    return {
        "success": bool(done),
        "steps": t,
        "refinements_run": refinements_run,
        "total_decisions": total_decisions,
        "wall_seconds": time.time() - episode_start,
        "cost_improvements": cost_improvements,
        "rollout_log_path": str(saved_path) if saved_path is not None else None,
        "failure_trace_path": str(trace_path) if trace_path is not None else None,
        "physics_trace_path": str(phys_path) if phys_path is not None else None,
        "mppi_early_exit": bool(mppi_disabled_this_episode),
    }


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.task_suite not in MAX_STEPS:
        raise ValueError(f"Unknown task suite: {args.task_suite}")

    # Load MPPI cost targets from YAML
    targets_path = pathlib.Path(args.targets_yaml)
    if not targets_path.exists():
        raise FileNotFoundError(f"Targets YAML not found: {targets_path}")
    targets = yaml.safe_load(targets_path.read_text())
    suite_targets = (targets or {}).get(args.task_suite, {}) or {}
    entry = suite_targets.get(args.task_idx) or suite_targets.get(str(args.task_idx))
    if entry is None:
        raise KeyError(
            f"No target entry for {args.task_suite}/task_{args.task_idx} in {targets_path}."
        )
    if "goal_xyz" not in entry:
        raise KeyError(
            f"Entry {args.task_suite}/{args.task_idx} missing 'goal_xyz'. "
            f"Schema: mode (object|ee), track_bodies (list, for mode=object), goal_xyz [x,y,z]."
        )
    mode = entry.get("mode", "object")
    goal_xyz = np.array(entry["goal_xyz"], dtype=np.float64)
    print(f"Target for {args.task_suite}/task_{args.task_idx}: mode={mode} goal_xyz={goal_xyz} "
          f"({entry.get('description', 'no description')})", flush=True)

    # Load Pi0.5
    print("Loading Pi0.5 policy...", flush=True)
    from openpi.shared import download
    download.maybe_download(args.checkpoint + "/assets")
    download.maybe_download(args.checkpoint + "/params")
    train_cfg = _config.get_config(args.config_name)
    policy = _policy_config.create_trained_policy(train_cfg, args.checkpoint)
    print("Policy loaded.", flush=True)

    # Load learned Q(h, a) if running the learned-WM head-to-head ablation.
    q_fn = None
    if args.scorer_type == "learned-wm":
        if not args.q_function:
            raise ValueError("--scorer-type=learned-wm requires --q-function <ckpt>.pt")
        device = torch.device(args.device)
        print(f"Loading Q(h, a) from {args.q_function} (device={device})...", flush=True)
        q_fn = load_q_function(args.q_function, args.q_function_config, device)
        print(f"Q loaded: hidden_state_dim={q_fn.hidden_state_dim} "
              f"horizon={q_fn.action_chunk_horizon} action_dim={q_fn.action_dim}", flush=True)
    device = torch.device(args.device) if args.scorer_type == "learned-wm" else torch.device("cpu")

    bm = benchmark.get_benchmark_dict()[args.task_suite]()
    task = bm.get_task(args.task_idx)
    init_states = bm.get_task_init_states(args.task_idx)
    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl),
        camera_heights=LIBERO_ENV_RESOLUTION, camera_widths=LIBERO_ENV_RESOLUTION,
    )
    env.seed(args.seed)
    sim_model = env.env.sim.model
    is_robot_geom = _robot_geom_mask(sim_model)
    print(f"Robot geoms identified: {is_robot_geom.sum()} / {sim_model.ngeom}",
          flush=True)

    # Resolve tracked-body names → MuJoCo body ids
    def _body_id(name: str) -> int:
        if hasattr(sim_model, "body_name2id"):
            try:
                return int(sim_model.body_name2id(name))
            except Exception:
                pass
        # Fall back: linear scan
        for b in range(sim_model.nbody):
            if (sim_model.body_id2name(b) or "") == name:
                return b
        raise KeyError(f"Body '{name}' not found in MuJoCo model for {args.task_suite}/{args.task_idx}")

    if mode == "ee":
        # The robosuite obs key "robot0_eef_pos" isn't a body name; the wrist
        # body is typically "robot0_right_hand". Try a few aliases.
        ee_aliases = ("robot0_right_hand", "gripper0_eef", "robot0_link7", "robot0_eef", "eef")
        resolved = None
        for nm in ee_aliases:
            try:
                resolved = (nm, _body_id(nm))
                break
            except KeyError:
                continue
        if resolved is None:
            raise KeyError(f"No EE body found among {ee_aliases}")
        tracked_body_ids = [resolved[1]]
        print(f"Tracking EE body '{resolved[0]}' (id={resolved[1]})", flush=True)
    elif mode == "object":
        track_names = entry.get("track_bodies") or []
        if not track_names:
            raise KeyError(
                f"mode=object for {args.task_suite}/{args.task_idx} but no 'track_bodies' list."
            )
        tracked_body_ids = [_body_id(n) for n in track_names]
        print(f"Tracking object bodies {list(zip(track_names, tracked_body_ids))} "
              f"(MPPI picks farthest from goal at each decision)", flush=True)
    else:
        raise ValueError(f"Unknown mode '{mode}'; expected 'object' or 'ee'")

    max_steps = MAX_STEPS[args.task_suite]
    perturbation_m = args.perturbation_cm / 100.0
    trials = min(args.num_trials, len(init_states))
    mppi_rng = np.random.default_rng(args.seed + 99)

    print(f"\n{'='*70}", flush=True)
    print(f"Task: {task.language}", flush=True)
    print(f"{'='*70}", flush=True)

    results = {}
    for eval_mode in ("baseline", "mppi"):
        perturb_rng = np.random.default_rng(args.seed + 1000)
        successes = 0
        per_trial = []
        print(f"\n--- {eval_mode} (K={args.mppi_K}, σ={args.mppi_sigma}, λ={args.mppi_lambda}) ---",
              flush=True)
        for ti in range(trials):
            r = run_episode(
                policy=policy, env=env, init_state=init_states[ti],
                task_description=task.language,
                goal_xyz=goal_xyz, tracked_body_ids=tracked_body_ids,
                is_robot_geom=is_robot_geom, perturbation_m=perturbation_m,
                perturb_rng=perturb_rng, max_steps=max_steps, args=args,
                mode=eval_mode, mppi_rng=mppi_rng,
                q_fn=q_fn, device=device,
            )
            successes += int(r["success"])
            per_trial.append(r)
            tag = "OK" if r['success'] else "FAIL"
            extra = " early-exit" if r.get("mppi_early_exit") else ""
            print(f"  trial {ti+1}/{trials}: {tag} "
                  f"({r['steps']} steps, {r['refinements_run']} refinements, "
                  f"{r['wall_seconds']:.1f}s{extra})",
                  flush=True)
        results[eval_mode] = {
            "successes": successes,
            "trials": trials,
            "success_rate": successes / trials,
            "per_trial": per_trial,
        }
        print(f"  → {eval_mode}: {successes}/{trials} = {successes/trials*100:.1f}%", flush=True)

    env.close()

    base = results["baseline"]["success_rate"]
    mp = results["mppi"]["success_rate"]
    delta_pp = (mp - base) * 100
    print(f"\n{'='*70}", flush=True)
    print(f"baseline:   {base*100:5.1f}% ({results['baseline']['successes']}/{trials})", flush=True)
    print(f"mppi:       {mp*100:5.1f}% ({results['mppi']['successes']}/{trials})", flush=True)
    print(f"delta:      {delta_pp:+.1f} pp", flush=True)
    print(f"{'='*70}", flush=True)
    if delta_pp >= 3:
        print("✓ PASS: MPPI mechanism validates (≥+3pp).", flush=True)
    elif delta_pp >= -2:
        print("≈ FLAT: tune K / σ / λ / weights, OR target may be wrong.", flush=True)
    else:
        print("✗ FAIL: MPPI actively hurting — diagnose cost function.", flush=True)

    payload = {
        "args": vars(args),
        "task_description": task.language,
        "goal_xyz": goal_xyz.tolist(),
        "mode": mode,
        "tracked_body_ids": tracked_body_ids,
        "tracked_body_names": entry.get("track_bodies") if mode == "object" else ["robot0_eef"],
        "results": {
            m: {k: v for k, v in r.items() if k != "per_trial"}
            for m, r in results.items()
        },
        "delta_pp": delta_pp,
    }
    out = (output_dir
           / f"v3mppi_{args.task_suite}_task{args.task_idx}_"
             f"{args.perturbation_cm}cm_seed{args.seed}_K{args.mppi_K}_"
             f"scorer-{args.scorer_type}.json")
    out.write_text(json.dumps(payload, indent=2, default=str))
    print(f"\nSaved to {out}", flush=True)


if __name__ == "__main__":
    main()
