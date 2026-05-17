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
    return p.parse_args()


# -------------------- LIBERO helpers --------------------


def _quat2axisangle(quat):
    quat = quat.copy()
    if quat[3] > 1.0: quat[3] = 1.0
    elif quat[3] < -1.0: quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0): return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def perturb_object_positions(env, init_state, perturbation_m, rng):
    state = init_state.clone() if hasattr(init_state, "clone") else init_state.copy()
    sim = env.env.sim
    model = sim.model
    for joint_idx in range(model.njnt):
        if model.jnt_type[joint_idx] == 0:
            pos_start = model.jnt_qposadr[joint_idx]
            state[pos_start:pos_start + 3] += rng.normal(0, perturbation_m, size=3)
    return state


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
        for action in candidate[:noised_horizon]:
            try:
                obs_local, _, d_flag, _ = env.step(action.tolist())
            except ValueError:
                break  # defensive: robosuite still raised despite the reset
            last_ee_pos = np.asarray(obs_local["robot0_eef_pos"], dtype=np.float64)
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
        anchor_cost = float(np.sum(
            (candidate[:noised_horizon] - prior_action[:noised_horizon]) ** 2
        ))
        return (step_cost_sum
                + w_target * target_cost
                + w_approach * approach_cost
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
        init_state_np = perturb_object_positions(env, init_state_np, perturbation_m, perturb_rng)
    obs = env.set_init_state(init_state_np)

    plan = collections.deque()
    done = False
    t = 0
    episode_start = time.time()
    last_action_chunk = None
    refinements_run = 0
    total_decisions = 0
    cost_improvements: list[tuple[float, float]] = []

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
            if mode == "mppi":
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
                    w_anchor=args.w_anchor,
                    w_collision=args.w_collision, w_joint_limit=args.w_joint_limit,
                    is_robot_geom=is_robot_geom, replan_steps=args.replan_steps,
                    rng=mppi_rng,
                    scorer_type=args.scorer_type, q_fn=q_fn,
                    hidden_state=hidden_state, device=device,
                )
                action_chunk = refined
                refinements_run += 1
                cost_improvements.append((diag["nominal_cost"], diag["refined_cost"]))
                if args.verbose_mppi:
                    import math as _m
                    log_K = _m.log(args.mppi_K)
                    # Entropy gap from uniform: log(K) - entropy. 0 = uniform
                    # weights (MPPI noop), large = peaked weights (MPPI active).
                    ent_gap = log_K - diag["weight_entropy"]
                    print(
                        f"    [MPPI t={t}] nominal={diag['nominal_cost']:.3f} "
                        f"refined={diag['refined_cost']:.3f} "
                        f"best={diag['best_sample_cost']:.3f} "
                        f"spread={diag['cost_spread']:.4f} "
                        f"ent_gap_from_uniform={ent_gap:.4f} "
                        f"(refined-nominal={diag['refined_cost']-diag['nominal_cost']:+.3f}, "
                        f"best-nominal={diag['best_sample_cost']-diag['nominal_cost']:+.3f}) "
                        f"wall={diag['wall_seconds']*1000:.0f}ms",
                        flush=True,
                    )
            else:
                action_chunk = initial_action

            last_action_chunk = action_chunk
            plan.extend(action_chunk[:args.replan_steps])

        action = plan.popleft()
        obs, _, done, _ = env.step(action.tolist())
        if done:
            break
        t += 1

    return {
        "success": bool(done),
        "steps": t,
        "refinements_run": refinements_run,
        "total_decisions": total_decisions,
        "wall_seconds": time.time() - episode_start,
        "cost_improvements": cost_improvements,
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
            print(f"  trial {ti+1}/{trials}: {'OK' if r['success'] else 'FAIL'} "
                  f"({r['steps']} steps, {r['refinements_run']} refinements, "
                  f"{r['wall_seconds']:.1f}s)",
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
