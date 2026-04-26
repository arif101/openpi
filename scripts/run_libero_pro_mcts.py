"""LIBERO-PRO evaluation: base Pi0.5 vs. Pi0.5 + MCTS over latent world model.

This is Experiment A of the world-model research program: test whether
inference-time tree search over our existing Phase 4 world model + value
function lifts Pi0.5's LIBERO-PRO performance above the 46% perturbed
baseline.

Comparison:
  - Baseline: vanilla Pi0.5 action-chunk sampling (K=1 per decision).
  - MCTS: at each contact-triggered decision, run the MCTSPlanner with
    policy prior from Pi0.5 action samples, transitions from our WM,
    and leaf evaluation from our value function.

The LIBERO-PRO 5cm object-position perturbation is applied identically
to both runs (so any delta is attributable to the planner, not to
perturbation stochasticity).

Usage:
    PYTHONPATH=/workspace/openpi/src:/workspace/openpi/third_party/libero \\
    MUJOCO_GL=egl \\
    /workspace/openpi/.venv/bin/python -u scripts/run_libero_pro_mcts.py \\
        --world-model data/contact_mpc/mpc_results/world_model.pt \\
        --value-function data/contact_mpc/value_function/value_function.pt \\
        --task-suite libero_10 --num-trials 5 --perturbation-cm 5.0 \\
        --mcts-width-k 4 --mcts-max-depth 1 --mcts-num-simulations 32 \\
        --output-dir data/contact_mpc/libero_pro_mcts
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

from openpi.contact_mpc.planner.mcts import MCTSConfig
from openpi.contact_mpc.planner.mcts import MCTSPlanner
from openpi.contact_mpc.value_function.architecture import (
    ActionConditionalValueFunction,
    PairwiseValueFunction,
)
from openpi.contact_mpc.world_model.architecture import LatentWorldModel
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256
RESIZE_SIZE = 224

MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config-name", default="pi05_libero")
    p.add_argument("--checkpoint", default="gs://openpi-assets/checkpoints/pi05_libero")
    p.add_argument("--world-model", required=True)
    p.add_argument("--world-model-config", default=None)
    p.add_argument("--value-function", required=True)
    p.add_argument("--value-function-config", default=None)
    p.add_argument(
        "--value-function-type",
        choices=["v", "qha"],
        default="v",
        help="'v' = legacy V(h) PairwiseValueFunction. "
             "'qha' = action-conditional ActionConditionalValueFunction Q(h, a).",
    )
    p.add_argument("--task-suite", default="libero_10")
    p.add_argument("--num-trials", type=int, default=5)
    p.add_argument("--num-steps-wait", type=int, default=10)
    p.add_argument("--replan-steps", type=int, default=5)
    p.add_argument("--perturbation-cm", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--max-tasks", type=int, default=None)
    p.add_argument("--episode-timeout", type=int, default=300)
    p.add_argument("--output-dir", default="data/contact_mpc/libero_pro_mcts")
    p.add_argument("--mcts-num-simulations", type=int, default=32)
    p.add_argument("--mcts-width-k", type=int, default=4)
    p.add_argument("--mcts-max-depth", type=int, default=1,
                   help="Depth 1 = K-sample MPC with value-scoring. Depth >1 = true tree search.")
    p.add_argument("--mcts-c-puct", type=float, default=1.4)
    p.add_argument("--mcts-everywhere", action="store_true",
                   help="Run MCTS at every decision, not just contact events (slower).")
    p.add_argument("--verbose-mcts", action="store_true",
                   help="Log MCTS diagnostics (visit counts, score spreads) at every "
                        "search call. Useful for smoke-testing whether search is "
                        "discriminating between candidates.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


# -------------------- LIBERO helpers (match run_libero_pro.py exactly) --------------------


def _quat2axisangle(quat):
    quat = quat.copy()
    if quat[3] > 1.0: quat[3] = 1.0
    elif quat[3] < -1.0: quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0): return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def perturb_object_positions(env, init_state, perturbation_m, rng):
    """Perturb free-joint object positions by Gaussian noise on xyz only.
    Matches run_libero_pro.py exactly so runs are comparable."""
    state = init_state.clone() if hasattr(init_state, "clone") else init_state.copy()
    sim = env.env.sim
    model = sim.model
    for joint_idx in range(model.njnt):
        if model.jnt_type[joint_idx] == 0:  # mjJNT_FREE
            pos_start = model.jnt_qposadr[joint_idx]
            state[pos_start:pos_start + 3] += rng.normal(0, perturbation_m, size=3)
    return state


def build_obs_element(obs, task_description):
    """Match run_libero_pro.py's obs dict construction exactly."""
    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, RESIZE_SIZE, RESIZE_SIZE))
    wrist = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist, RESIZE_SIZE, RESIZE_SIZE))
    state = np.concatenate((
        obs["robot0_eef_pos"],
        _quat2axisangle(obs["robot0_eef_quat"]),
        obs["robot0_gripper_qpos"],
    ))
    return {
        "observation/image": img,
        "observation/wrist_image": wrist,
        "observation/state": state,
        "prompt": str(task_description),
    }


def detect_contact_in_chunk(action_chunk: np.ndarray) -> bool:
    """Gripper sign-flip heuristic from run_mpc_experiment.py."""
    gripper_cmds = action_chunk[:, -1] if action_chunk.ndim == 2 else action_chunk[-1:]
    for i in range(1, len(gripper_cmds)):
        if (gripper_cmds[i] > 0) != (gripper_cmds[i - 1] > 0):
            return True
    return False


# -------------------- Adapters: Pi0.5 / torch -> MCTS callable interfaces --------------------


class Pi05PolicySampler:
    """Wraps Pi0.5's sample_actions as an MCTS policy prior.

    For depth >1 in the tree, we sample from the *same* Pi0.5 call on
    the root observation — the tree rolls those candidate chunks forward
    through the world model, but the action "grammar" is set by the VLA's
    view of the real world. This is the pragmatic Option-4 from the design
    doc; VLAPS-style learned-depth priors are a later upgrade.
    """

    def __init__(self, policy, obs_element, max_horizon: int):
        self.policy = policy
        self.obs = obs_element
        self.max_horizon = max_horizon
        self._cached_prior_features: np.ndarray | None = None

    def __call__(self, hidden_state, k, rng):
        actions = []
        for _ in range(k):
            result = self.policy.infer(dict(self.obs))
            action_chunk = np.asarray(
                result["actions"][: self.max_horizon], dtype=np.float32,
            )
            # Pad if shorter than max_horizon
            if action_chunk.shape[0] < self.max_horizon:
                pad = np.zeros(
                    (self.max_horizon - action_chunk.shape[0], action_chunk.shape[1]),
                    dtype=np.float32,
                )
                action_chunk = np.concatenate([action_chunk, pad], axis=0)
            actions.append(action_chunk)

            # Cache one hidden state from the first call (for root init)
            if self._cached_prior_features is None:
                feats = result.get("vlm_features")
                if feats is not None:
                    self._cached_prior_features = np.asarray(feats, dtype=np.float32)

        # Uniform prior: Pi0.5 doesn't expose per-sample likelihood
        priors = np.ones(k, dtype=np.float32) / k
        return np.stack(actions), priors

    @property
    def latest_root_features(self) -> np.ndarray | None:
        """Retrieve the VLM hidden state from the most recent Pi0.5 call."""
        return self._cached_prior_features

    def reset_cache(self):
        self._cached_prior_features = None


class TorchWorldModelAdapter:
    """Wraps a PyTorch LatentWorldModel as a numpy-in / numpy-out callable."""

    def __init__(self, world_model: LatentWorldModel, device):
        self.wm = world_model
        self.device = device
        self.max_horizon = world_model.config.max_horizon

    def __call__(self, hidden_state: np.ndarray, action_chunk: np.ndarray) -> np.ndarray:
        # Normalize action chunk length
        a = action_chunk
        if a.shape[0] < self.max_horizon:
            pad = np.zeros((self.max_horizon - a.shape[0], a.shape[1]), dtype=np.float32)
            a = np.concatenate([a, pad], axis=0)
        elif a.shape[0] > self.max_horizon:
            a = a[: self.max_horizon]

        h_t = torch.tensor(hidden_state, dtype=torch.float32, device=self.device).unsqueeze(0)
        a_t = torch.tensor(a, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            next_h = self.wm(h_t, a_t)
        return next_h.squeeze(0).cpu().numpy().astype(np.float32)


class TorchValueFnAdapter:
    """Wraps a PyTorch PairwiseValueFunction V(h) as a ValueFn callable.

    The widened ValueFn protocol passes ``action_chunk``, ``frame``, and
    ``task`` as extra args; this adapter ignores them (V(h) doesn't need them).
    """

    def __init__(self, value_fn: PairwiseValueFunction, device):
        self.vf = value_fn
        self.device = device

    def __call__(
        self,
        hidden_state: np.ndarray,
        action_chunk: np.ndarray | None = None,
        *,
        frame: np.ndarray | None = None,
        task: str | None = None,
    ) -> float:
        h = torch.tensor(hidden_state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            return float(self.vf(h).item())


class TorchQFnAdapter:
    """Wraps a PyTorch ActionConditionalValueFunction Q(h, a) as a ValueFn callable.

    Requires ``action_chunk`` to be passed by MCTS. If absent (e.g., a depth-0
    bootstrap call that we don't currently make), falls back to scoring with
    a zero action chunk so the call doesn't crash.
    """

    def __init__(self, q_fn: ActionConditionalValueFunction, device):
        self.q = q_fn
        self.device = device
        self.horizon = q_fn.action_chunk_horizon
        self.action_dim = q_fn.action_dim

    def __call__(
        self,
        hidden_state: np.ndarray,
        action_chunk: np.ndarray | None = None,
        *,
        frame: np.ndarray | None = None,
        task: str | None = None,
    ) -> float:
        if action_chunk is None:
            action_chunk = np.zeros((self.horizon, self.action_dim), dtype=np.float32)
        # Pad / truncate to expected horizon
        if action_chunk.shape[0] < self.horizon:
            pad = np.zeros(
                (self.horizon - action_chunk.shape[0], action_chunk.shape[1]),
                dtype=np.float32,
            )
            action_chunk = np.concatenate([action_chunk, pad], axis=0)
        elif action_chunk.shape[0] > self.horizon:
            action_chunk = action_chunk[: self.horizon]

        h = torch.tensor(hidden_state, dtype=torch.float32, device=self.device).unsqueeze(0)
        a = torch.tensor(action_chunk, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            return float(self.q(h, a).item())


# -------------------- Checkpoint loading --------------------


def _default_cfg_path(model_path: str) -> pathlib.Path:
    p = pathlib.Path(model_path)
    return p.parent / (p.stem + "_config.pt")


def load_world_model(ckpt_path: str, cfg_path: str | None, device) -> LatentWorldModel:
    cfg = torch.load(cfg_path or _default_cfg_path(ckpt_path), weights_only=False)
    wm = LatentWorldModel(cfg)
    wm.load_state_dict(torch.load(ckpt_path, weights_only=True))
    wm = wm.to(device).eval()
    return wm


def load_value_function(ckpt_path: str, cfg_path: str | None, device) -> PairwiseValueFunction:
    cfg = torch.load(cfg_path or _default_cfg_path(ckpt_path), weights_only=False)
    vf = PairwiseValueFunction(cfg["input_dim"], cfg["hidden_dim"])
    vf.load_state_dict(torch.load(ckpt_path, weights_only=True))
    vf = vf.to(device).eval()
    return vf


def load_q_function(
    ckpt_path: str, cfg_path: str | None, device,
) -> ActionConditionalValueFunction:
    cfg = torch.load(cfg_path or _default_cfg_path(ckpt_path), weights_only=False)
    if cfg.get("type") and cfg["type"] != "ActionConditionalValueFunction":
        raise ValueError(
            f"Expected ActionConditionalValueFunction config at {cfg_path}, "
            f"got type={cfg.get('type')!r}"
        )
    q = ActionConditionalValueFunction(
        hidden_state_dim=cfg["hidden_state_dim"],
        action_chunk_horizon=cfg["action_chunk_horizon"],
        action_dim=cfg["action_dim"],
        action_emb_dim=cfg.get("action_emb_dim", 128),
        hidden_dim=cfg.get("hidden_dim", 256),
        dropout=cfg.get("dropout", 0.0),  # eval-time: dropout off
    )
    q.load_state_dict(torch.load(ckpt_path, weights_only=True))
    q = q.to(device).eval()
    return q


# -------------------- Evaluation loops --------------------


def run_episode(
    *,
    policy,
    env,
    init_state,
    task_description: str,
    perturbation_m: float,
    perturb_rng: np.random.Generator,
    max_steps: int,
    num_steps_wait: int,
    replan_steps: int,
    episode_timeout: int,
    mode: str,                           # "baseline" or "mcts"
    mcts_planner: MCTSPlanner | None,
    mcts_everywhere: bool,
    policy_sampler: Pi05PolicySampler | None,
    verbose_mcts: bool = False,
) -> dict:
    """Run one LIBERO episode under one mode; return outcome + telemetry."""
    env.reset()

    # Apply perturbation identically across modes for comparable runs
    init_state_np = init_state.clone() if hasattr(init_state, "clone") else init_state.copy()
    if perturbation_m > 0:
        init_state_np = perturb_object_positions(env, init_state_np, perturbation_m, perturb_rng)
    obs = env.set_init_state(init_state_np)

    plan = collections.deque()
    done = False
    t = 0
    episode_start = time.time()
    last_action_chunk = None
    mcts_calls = 0
    contact_decisions = 0
    total_decisions = 0

    while t < max_steps + num_steps_wait:
        if time.time() - episode_start > episode_timeout:
            break

        if t < num_steps_wait:
            obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
            t += 1
            continue

        if not plan:
            total_decisions += 1
            obs_element = build_obs_element(obs, task_description)

            use_mcts = False
            if mode == "mcts" and mcts_planner is not None:
                if mcts_everywhere:
                    use_mcts = True
                elif last_action_chunk is not None and detect_contact_in_chunk(last_action_chunk):
                    use_mcts = True
                    contact_decisions += 1

            if use_mcts:
                # Prime the policy sampler with the current observation
                policy_sampler.obs = obs_element
                policy_sampler.reset_cache()
                # One Pi0.5 call to get root hidden state
                result = policy.infer(dict(obs_element))
                h_root = np.asarray(result["vlm_features"], dtype=np.float32)
                # Seed the sampler cache so we don't waste the first call
                policy_sampler._cached_prior_features = h_root  # noqa: SLF001

                action_chunk, diag = mcts_planner.plan(
                    h_root,
                    frame=obs_element["observation/image"],
                    task=obs_element["prompt"],
                )
                mcts_calls += 1

                if verbose_mcts:
                    visits = diag["child_visits"]
                    qs = diag["child_Q"]
                    q_spread = max(qs) - min(qs) if qs else 0.0
                    visit_concentration = max(visits) / max(sum(visits), 1)
                    print(
                        f"    [MCTS t={t}] visits={visits} "
                        f"Q_spread={q_spread:.4f} "
                        f"top_visit_frac={visit_concentration:.2f} "
                        f"chosen_idx={diag['chosen_idx']}",
                        flush=True,
                    )
            else:
                result = policy.infer(dict(obs_element))
                action_chunk = np.asarray(result["actions"], dtype=np.float32)

            last_action_chunk = action_chunk
            plan.extend(action_chunk[:replan_steps])

        action = plan.popleft()
        obs, _, done, _ = env.step(action.tolist())
        if done:
            break
        t += 1

    return {
        "success": bool(done),
        "steps": t,
        "mcts_calls": mcts_calls,
        "contact_decisions": contact_decisions,
        "total_decisions": total_decisions,
        "wall_seconds": time.time() - episode_start,
    }


def run_suite(*, policy, task_suite, args, mode, mcts_planner, policy_sampler, verbose_mcts=False):
    """Run (max_tasks × num_trials) episodes under one mode."""
    num_tasks = args.max_tasks or task_suite.n_tasks
    max_steps = MAX_STEPS[args.task_suite]
    perturbation_m = args.perturbation_cm / 100.0

    per_task: dict[int, dict] = {}
    total_episodes = 0
    total_successes = 0
    total_mcts_calls = 0

    for task_id in range(num_tasks):
        task = task_suite.get_task(task_id)
        init_states = task_suite.get_task_init_states(task_id)
        bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        env = OffScreenRenderEnv(
            bddl_file_name=str(bddl),
            camera_heights=LIBERO_ENV_RESOLUTION,
            camera_widths=LIBERO_ENV_RESOLUTION,
        )
        env.seed(args.seed)

        task_successes = 0
        trials = min(args.num_trials, len(init_states))
        perturb_rng = np.random.default_rng(args.seed + 1000)  # deterministic per task

        for trial_idx in range(trials):
            result = run_episode(
                policy=policy,
                env=env,
                init_state=init_states[trial_idx],
                task_description=task.language,
                perturbation_m=perturbation_m,
                perturb_rng=perturb_rng,
                max_steps=max_steps,
                num_steps_wait=args.num_steps_wait,
                replan_steps=args.replan_steps,
                episode_timeout=args.episode_timeout,
                mode=mode,
                mcts_planner=mcts_planner,
                mcts_everywhere=args.mcts_everywhere,
                policy_sampler=policy_sampler,
                verbose_mcts=verbose_mcts,
            )
            total_episodes += 1
            total_mcts_calls += result["mcts_calls"]
            if result["success"]:
                total_successes += 1
                task_successes += 1

        per_task[task_id] = {
            "task": task.language,
            "successes": task_successes,
            "trials": trials,
            "rate": task_successes / trials if trials else 0.0,
        }
        env.close()

        rate = total_successes / total_episodes * 100 if total_episodes else 0.0
        tag = f"[{mode.upper()}]"
        print(f"  {tag} Task {task_id+1}/{num_tasks}: "
              f"{task_successes}/{trials} ({per_task[task_id]['rate']:.0%}) | "
              f"running {total_successes}/{total_episodes} ({rate:.1f}%) | "
              f"mcts_calls={total_mcts_calls} | {task.language[:60]}",
              flush=True)

    return {
        "mode": mode,
        "total_episodes": total_episodes,
        "total_successes": total_successes,
        "success_rate": total_successes / total_episodes * 100 if total_episodes else 0.0,
        "total_mcts_calls": total_mcts_calls,
        "per_task": {str(k): v for k, v in per_task.items()},
    }


# -------------------- Main --------------------


def main():
    args = parse_args()
    np.random.seed(args.seed)
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.task_suite not in MAX_STEPS:
        raise ValueError(f"Unknown task suite: {args.task_suite}")

    device = torch.device(args.device)

    print(f"Loading world model from {args.world_model}", flush=True)
    wm = load_world_model(args.world_model, args.world_model_config, device)
    print(f"  {wm.param_count():,} params, hidden_dim={wm.config.hidden_dim}, H={wm.config.max_horizon}", flush=True)

    print(
        f"Loading value function from {args.value_function} "
        f"(type={args.value_function_type})",
        flush=True,
    )
    if args.value_function_type == "v":
        vf = load_value_function(args.value_function, args.value_function_config, device)
    elif args.value_function_type == "qha":
        vf = load_q_function(args.value_function, args.value_function_config, device)
    else:
        raise ValueError(f"Unknown --value-function-type: {args.value_function_type!r}")
    print(f"  {vf.param_count():,} params", flush=True)

    print("Loading Pi0.5 policy...", flush=True)
    from openpi.shared import download
    # Download BOTH assets and params on first run. The legacy pattern of
    # downloading only "/assets" works on warm caches but fails on a fresh
    # GPU box because restore_params then can't find /params locally.
    download.maybe_download(args.checkpoint + "/assets")
    download.maybe_download(args.checkpoint + "/params")
    train_cfg = _config.get_config(args.config_name)
    policy = _policy_config.create_trained_policy(train_cfg, args.checkpoint)
    print("Policy loaded.", flush=True)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite]()

    # Build MCTS components (adapter fns + planner)
    wm_fn = TorchWorldModelAdapter(wm, device)
    if args.value_function_type == "v":
        vf_fn = TorchValueFnAdapter(vf, device)
    else:
        vf_fn = TorchQFnAdapter(vf, device)
    # policy_sampler's obs is rebound per-decision; placeholder init
    placeholder_obs = {
        "observation/image": np.zeros((RESIZE_SIZE, RESIZE_SIZE, 3), dtype=np.uint8),
        "observation/wrist_image": np.zeros((RESIZE_SIZE, RESIZE_SIZE, 3), dtype=np.uint8),
        "observation/state": np.zeros(8, dtype=np.float32),
        "prompt": "placeholder",
    }
    sampler = Pi05PolicySampler(policy, placeholder_obs, max_horizon=wm.config.max_horizon)
    planner = MCTSPlanner(
        policy_sampler=sampler,
        world_model=wm_fn,
        value_fn=vf_fn,
        config=MCTSConfig(
            num_simulations=args.mcts_num_simulations,
            width_k=args.mcts_width_k,
            max_depth=args.mcts_max_depth,
            c_puct=args.mcts_c_puct,
        ),
        rng=np.random.default_rng(args.seed),
    )

    # Run baseline
    print(f"\n{'='*60}", flush=True)
    print(f"BASELINE (vanilla Pi0.5) on {args.task_suite} at {args.perturbation_cm}cm perturbation", flush=True)
    print(f"{'='*60}", flush=True)
    baseline = run_suite(
        policy=policy, task_suite=task_suite, args=args,
        mode="baseline", mcts_planner=None, policy_sampler=None,
    )

    # Run MCTS
    trigger = "everywhere" if args.mcts_everywhere else "contact-only"
    print(f"\n{'='*60}", flush=True)
    print(f"MCTS (sims={args.mcts_num_simulations}, K={args.mcts_width_k}, "
          f"depth={args.mcts_max_depth}, trigger={trigger})", flush=True)
    print(f"{'='*60}", flush=True)
    mcts_out = run_suite(
        policy=policy, task_suite=task_suite, args=args,
        mode="mcts", mcts_planner=planner, policy_sampler=sampler,
        verbose_mcts=args.verbose_mcts,
    )

    # Report
    print(f"\n{'='*60}", flush=True)
    print("FINAL RESULTS", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"Baseline: {baseline['success_rate']:.1f}% ({baseline['total_successes']}/{baseline['total_episodes']})", flush=True)
    print(f"MCTS:     {mcts_out['success_rate']:.1f}% ({mcts_out['total_successes']}/{mcts_out['total_episodes']})", flush=True)
    delta = mcts_out["success_rate"] - baseline["success_rate"]
    print(f"Delta:    {delta:+.1f} percentage points", flush=True)
    print(f"MCTS calls: {mcts_out['total_mcts_calls']}", flush=True)

    # Per-task comparison
    print(f"\nPer-task (tasks where modes differ):", flush=True)
    for tid in sorted(baseline["per_task"].keys()):
        b = baseline["per_task"][tid]
        m = mcts_out["per_task"][tid]
        if b["rate"] != m["rate"]:
            print(f"  Task {tid}: baseline={b['rate']:.0%} -> mcts={m['rate']:.0%} "
                  f"({m['rate']-b['rate']:+.0%}) | {b['task'][:60]}", flush=True)

    # Persist
    payload = {
        "args": vars(args),
        "baseline": baseline,
        "mcts": mcts_out,
        "delta_percentage_points": delta,
    }
    out_path = output_dir / f"libero_pro_mcts_{args.task_suite}_{args.perturbation_cm}cm.json"
    out_path.write_text(json.dumps(payload, indent=2, default=str))
    print(f"\nSaved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
