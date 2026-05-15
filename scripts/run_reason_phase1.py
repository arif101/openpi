"""REASON-VLA Phase 1: validate iterative action refinement.

The single gating experiment for REASON-VLA. Does gradient descent on
action chunks, guided by our learned WM + VF, improve VLA success?

At each decision point:
  1. Pi0.5 emits initial action chunk a_0
  2. For N iterations:
       loss = -VF(WM(hidden_state, a_t))
       a_{t+1} = a_t - lr · ∇_a loss
       clamp a_{t+1} to joint range
  3. Execute the refined action a_N

Compare success rate vs num_refinement_iterations N ∈ {0, 1, 2, 3, 5, 10}.

KILL CRITERION: if success(N=5) - success(N=0) < +3pp on LIBERO-10 5cm seed=7,
the iterative-refinement thesis dies and REASON-VLA is wrong-shaped.

PASS CRITERION: monotonic improvement with N → architectural premise validated;
proceed to add R-NCE energy critic, ACT router, and physics-grounded layers.

Usage:
    # Run a sweep over refinement iterations:
    bash scripts/run_reason_phase1_sweep.sh

    # Or a single N:
    PYTHONPATH=src:third_party/libero uv run python3 -u \\
      scripts/run_reason_phase1.py \\
        --world-model data/contact_mpc/mpc_results/world_model.pt \\
        --value-function data/contact_mpc/value_function/value_function.pt \\
        --task-suite libero_10 --num-trials 5 --perturbation-cm 5.0 --seed 7 \\
        --num-refinement-iterations 5 \\
        --refinement-lr 0.05 \\
        --output-dir data/contact_mpc/reason_phase1
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

from openpi.contact_mpc.value_function.architecture import PairwiseValueFunction
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
    p.add_argument("--task-suite", default="libero_10")
    p.add_argument("--num-trials", type=int, default=5)
    p.add_argument("--num-steps-wait", type=int, default=10)
    p.add_argument("--replan-steps", type=int, default=5)
    p.add_argument("--perturbation-cm", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--max-tasks", type=int, default=None)
    p.add_argument("--episode-timeout", type=int, default=300)
    p.add_argument("--output-dir", default="data/contact_mpc/reason_phase1")
    # --- refinement knobs ---
    p.add_argument("--num-refinement-iterations", type=int, default=5,
                   help="N gradient steps on the action chunk per decision. 0 = baseline Pi0.5.")
    p.add_argument("--refinement-lr", type=float, default=0.05,
                   help="Step size for action gradient descent.")
    p.add_argument("--refinement-clip", type=float, default=1.0,
                   help="Clamp refined actions to [-clip, +clip] (joint cmd range).")
    p.add_argument("--refinement-trigger", choices=["always", "contact"], default="always",
                   help="When to refine. 'always' = every decision. 'contact' = only after "
                        "contact events (mirrors Phase 5 MCTS trigger).")
    p.add_argument("--verbose-refinement", action="store_true",
                   help="Log per-iteration scores and gradient norms.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


# -------------------- LIBERO helpers (match run_libero_pro_mcts.py) --------------------


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
        if model.jnt_type[joint_idx] == 0:  # mjJNT_FREE
            pos_start = model.jnt_qposadr[joint_idx]
            state[pos_start:pos_start + 3] += rng.normal(0, perturbation_m, size=3)
    return state


def build_obs_element(obs, task_description):
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
    """Gripper sign-flip heuristic (same as run_libero_pro_mcts.py)."""
    gripper_cmds = action_chunk[:, -1] if action_chunk.ndim == 2 else action_chunk[-1:]
    for i in range(1, len(gripper_cmds)):
        if (gripper_cmds[i] > 0) != (gripper_cmds[i - 1] > 0):
            return True
    return False


# -------------------- The refinement engine --------------------


class IterativeRefiner:
    """Gradient-based action refinement on top of a frozen WM + VF.

    Given an initial action chunk a_0 from Pi0.5 and the current latent state h,
    take N gradient steps on the action via -VF(WM(h, a)). Returns the refined
    action chunk plus per-iteration diagnostics.
    """

    def __init__(
        self,
        wm: LatentWorldModel,
        vf: PairwiseValueFunction,
        num_iterations: int,
        lr: float,
        action_clip: float,
        device: str,
    ):
        self.wm = wm
        self.vf = vf
        self.N = num_iterations
        self.lr = lr
        self.action_clip = action_clip
        self.device = device
        self.max_horizon = wm.config.max_horizon

        # Freeze WM + VF; gradients flow only into the action tensor.
        for p in self.wm.parameters():
            p.requires_grad_(False)
        for p in self.vf.parameters():
            p.requires_grad_(False)

    def refine(self, hidden_state: np.ndarray, initial_action: np.ndarray) -> tuple[np.ndarray, dict]:
        """Refine `initial_action` against learned WM + VF for N steps.

        Returns:
            refined_action: numpy array same shape as initial_action
            diag: per-iteration diagnostics
        """
        if self.N == 0:
            return initial_action.copy(), {
                "scores": [],
                "grad_norms": [],
                "initial_score": None,
                "final_score": None,
            }

        # Normalize action length to WM's expected horizon
        a_np = initial_action.astype(np.float32)
        original_len = a_np.shape[0]
        if a_np.shape[0] < self.max_horizon:
            pad = np.zeros((self.max_horizon - a_np.shape[0], a_np.shape[1]), dtype=np.float32)
            a_np = np.concatenate([a_np, pad], axis=0)
        elif a_np.shape[0] > self.max_horizon:
            a_np = a_np[: self.max_horizon]

        h = torch.tensor(hidden_state, dtype=torch.float32, device=self.device).unsqueeze(0)
        a = torch.tensor(a_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        a.requires_grad_(True)

        scores: list[float] = []
        grad_norms: list[float] = []

        # Initial score for logging
        with torch.no_grad():
            initial_predicted = self.wm(h, a)
            initial_score = float(self.vf(initial_predicted).item())

        optimizer = torch.optim.SGD([a], lr=self.lr)
        for step_i in range(self.N):
            optimizer.zero_grad()
            predicted = self.wm(h, a)              # [1, hidden_dim]
            score = self.vf(predicted).squeeze()   # scalar
            loss = -score
            loss.backward()
            grad_norms.append(float(a.grad.norm().item()))
            optimizer.step()
            with torch.no_grad():
                a.clamp_(-self.action_clip, +self.action_clip)
                # Re-score post-update
                post_predicted = self.wm(h, a)
                scores.append(float(self.vf(post_predicted).item()))

        final_action = a.detach().squeeze(0).cpu().numpy()[:original_len].astype(np.float32)
        return final_action, {
            "scores": scores,
            "grad_norms": grad_norms,
            "initial_score": initial_score,
            "final_score": scores[-1] if scores else initial_score,
        }


# -------------------- Checkpoint loading --------------------


def _default_cfg_path(model_path: str) -> pathlib.Path:
    p = pathlib.Path(model_path)
    return p.parent / (p.stem + "_config.pt")


def load_world_model(ckpt_path: str, cfg_path: str | None, device) -> LatentWorldModel:
    cfg = torch.load(cfg_path or _default_cfg_path(ckpt_path), weights_only=False)
    wm = LatentWorldModel(cfg)
    wm.load_state_dict(torch.load(ckpt_path, weights_only=True))
    return wm.to(device).eval()


def load_value_function(ckpt_path: str, cfg_path: str | None, device) -> PairwiseValueFunction:
    cfg = torch.load(cfg_path or _default_cfg_path(ckpt_path), weights_only=False)
    vf = PairwiseValueFunction(cfg["input_dim"], cfg["hidden_dim"])
    vf.load_state_dict(torch.load(ckpt_path, weights_only=True))
    return vf.to(device).eval()


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
    refiner: IterativeRefiner | None,
    refinement_trigger: str,
    verbose: bool,
) -> dict:
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
    score_improvements = []  # list of (initial, final) per refined decision

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

            # Always get Pi0.5's initial action
            result = policy.infer(dict(obs_element))
            initial_action = np.asarray(result["actions"], dtype=np.float32)
            hidden_state = np.asarray(result["vlm_features"], dtype=np.float32)

            # Decide whether to refine
            should_refine = (
                refiner is not None and refiner.N > 0 and (
                    refinement_trigger == "always"
                    or (refinement_trigger == "contact"
                        and last_action_chunk is not None
                        and detect_contact_in_chunk(last_action_chunk))
                )
            )

            if should_refine:
                refined, diag = refiner.refine(hidden_state, initial_action)
                action_chunk = refined
                refinements_run += 1
                if diag["initial_score"] is not None:
                    score_improvements.append(
                        (diag["initial_score"], diag["final_score"])
                    )
                if verbose:
                    print(
                        f"    [refine t={t}] init={diag['initial_score']:.3f} "
                        f"final={diag['final_score']:.3f} "
                        f"delta={diag['final_score']-diag['initial_score']:+.3f} "
                        f"grad_norms={['{:.3f}'.format(g) for g in diag['grad_norms'][:3]]}...",
                        flush=True,
                    )
            else:
                action_chunk = initial_action

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
        "refinements_run": refinements_run,
        "total_decisions": total_decisions,
        "wall_seconds": time.time() - episode_start,
        "score_improvements": score_improvements,
    }


def run_suite(*, policy, task_suite, args, refiner, refinement_trigger, verbose):
    num_tasks = args.max_tasks or task_suite.n_tasks
    max_steps = MAX_STEPS[args.task_suite]
    perturbation_m = args.perturbation_cm / 100.0

    total_episodes = 0
    total_successes = 0
    total_refinements = 0
    all_improvements: list[tuple[float, float]] = []
    per_task = {}

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
        trials = min(args.num_trials, len(init_states))
        perturb_rng = np.random.default_rng(args.seed + 1000)
        task_successes = 0

        for trial_idx in range(trials):
            result = run_episode(
                policy=policy, env=env, init_state=init_states[trial_idx],
                task_description=task.language,
                perturbation_m=perturbation_m,
                perturb_rng=perturb_rng,
                max_steps=max_steps,
                num_steps_wait=args.num_steps_wait,
                replan_steps=args.replan_steps,
                episode_timeout=args.episode_timeout,
                refiner=refiner,
                refinement_trigger=refinement_trigger,
                verbose=verbose,
            )
            total_episodes += 1
            total_refinements += result["refinements_run"]
            all_improvements.extend(result["score_improvements"])
            if result["success"]:
                total_successes += 1
                task_successes += 1

        per_task[task_id] = {
            "task": task.language,
            "successes": task_successes,
            "trials": trials,
            "rate": task_successes / trials,
        }
        env.close()
        rate = total_successes / total_episodes * 100
        print(
            f"  Task {task_id+1}/{num_tasks}: {task_successes}/{trials} "
            f"({per_task[task_id]['rate']:.0%}) | running {total_successes}/{total_episodes} "
            f"({rate:.1f}%) | refinements={total_refinements} | "
            f"{task.language[:60]}",
            flush=True,
        )

    return {
        "total_episodes": total_episodes,
        "total_successes": total_successes,
        "success_rate": total_successes / total_episodes * 100,
        "total_refinements": total_refinements,
        "per_task": per_task,
        "mean_score_lift": (
            float(np.mean([f - i for i, f in all_improvements])) if all_improvements else None
        ),
        "median_score_lift": (
            float(np.median([f - i for i, f in all_improvements])) if all_improvements else None
        ),
    }


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.task_suite not in MAX_STEPS:
        raise ValueError(f"Unknown task suite: {args.task_suite}")

    device = torch.device(args.device)
    print(f"Device: {device}", flush=True)

    print(f"Loading world model from {args.world_model}", flush=True)
    wm = load_world_model(args.world_model, args.world_model_config, device)
    print(f"  {wm.param_count():,} params, hidden_dim={wm.config.hidden_dim}, H={wm.config.max_horizon}",
          flush=True)

    print(f"Loading value function from {args.value_function}", flush=True)
    vf = load_value_function(args.value_function, args.value_function_config, device)
    print(f"  {vf.param_count():,} params", flush=True)

    print("Loading Pi0.5 policy...", flush=True)
    from openpi.shared import download
    download.maybe_download(args.checkpoint + "/assets")
    download.maybe_download(args.checkpoint + "/params")
    train_cfg = _config.get_config(args.config_name)
    policy = _policy_config.create_trained_policy(train_cfg, args.checkpoint)
    print("Policy loaded.", flush=True)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite]()

    refiner = IterativeRefiner(
        wm=wm,
        vf=vf,
        num_iterations=args.num_refinement_iterations,
        lr=args.refinement_lr,
        action_clip=args.refinement_clip,
        device=device,
    )

    print(f"\n{'='*70}", flush=True)
    print(f"REASON-VLA Phase 1: N={args.num_refinement_iterations} refinement iterations",
          flush=True)
    print(f"  lr={args.refinement_lr}, clip={args.refinement_clip}, "
          f"trigger={args.refinement_trigger}", flush=True)
    print(f"  task_suite={args.task_suite}, perturbation={args.perturbation_cm}cm, seed={args.seed}",
          flush=True)
    print(f"{'='*70}", flush=True)

    result = run_suite(
        policy=policy, task_suite=task_suite, args=args,
        refiner=refiner,
        refinement_trigger=args.refinement_trigger,
        verbose=args.verbose_refinement,
    )

    print(f"\n{'='*70}", flush=True)
    print("REASON PHASE 1 RESULTS", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"N={args.num_refinement_iterations}: "
          f"{result['success_rate']:.1f}% ({result['total_successes']}/{result['total_episodes']})",
          flush=True)
    print(f"Refinements run: {result['total_refinements']}", flush=True)
    if result["mean_score_lift"] is not None:
        print(f"Mean score lift per refinement: {result['mean_score_lift']:+.4f}", flush=True)
        print(f"Median score lift per refinement: {result['median_score_lift']:+.4f}", flush=True)

    payload = {"args": vars(args), "result": result}
    out_path = output_dir / (
        f"reason_phase1_{args.task_suite}_{args.perturbation_cm}cm_"
        f"seed{args.seed}_N{args.num_refinement_iterations}.json"
    )
    out_path.write_text(json.dumps(payload, indent=2, default=str))
    print(f"\nSaved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
