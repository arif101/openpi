"""End-to-end MPC experiment with contact-triggered planning.

MPC only activates at contact events (gripper state change in the
previous action chunk). Free-space decisions use K=1 (baseline speed).
This makes MPC ~8x cheaper than running K=8 everywhere.

Contact detection: if any action in the previous chunk had a gripper
sign flip (action dim 6 crosses zero), the next decision is contact.

Usage:
    PYTHONPATH=/workspace/openpi/src:/workspace/openpi/third_party/libero \
    /workspace/openpi/.venv/bin/python -u scripts/run_mpc_experiment.py \
        --task-suite libero_90 --num-trials 5 --K 8
"""

import argparse
import collections
import math
import pathlib
import signal
import time

import jax
import jax.numpy as jnp
import numpy as np
import torch

# Patch torch.load for LIBERO init states
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
from huggingface_hub import hf_hub_download

from openpi.contact_mpc.features.dataset import FeatureDataset
from openpi.contact_mpc.world_model.architecture import LatentWorldModel, WorldModelConfig, SIZE_CONFIGS
from openpi.contact_mpc.world_model.train import train_world_model
from openpi.contact_mpc.value_function.architecture import PairwiseValueFunction
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-repo", default="arif101/libero90_vlm_features")
    parser.add_argument("--config-name", default="pi05_libero")
    parser.add_argument("--checkpoint", default="gs://openpi-assets/checkpoints/pi05_libero")
    parser.add_argument("--task-suite", default="libero_90")
    parser.add_argument("--num-trials", type=int, default=5)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--K", type=int, default=8)
    parser.add_argument("--wm-size", default="small")
    parser.add_argument("--wm-horizon", type=int, default=10)
    parser.add_argument("--output-dir", default="data/contact_mpc/mpc_results")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--skip-wm-train", action="store_true")
    parser.add_argument("--episode-timeout", type=int, default=120, help="Max seconds per episode")
    parser.add_argument("--mpc-everywhere", action="store_true", help="Run MPC at every decision, not just contact")
    parser.add_argument("--skip-baseline", action="store_true", help="Skip baseline run, use known 28.9% rate")
    return parser.parse_args()


def _quat2axisangle(quat):
    quat = quat.copy()
    if quat[3] > 1.0: quat[3] = 1.0
    elif quat[3] < -1.0: quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0): return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def detect_contact_in_chunk(action_chunk: np.ndarray) -> bool:
    """Check if an action chunk contains a gripper sign change.

    Gripper command is the last dimension (index 6). A sign change
    (positive → negative or vice versa) indicates a grasp or release.
    """
    gripper_cmds = action_chunk[:, -1] if action_chunk.ndim == 2 else action_chunk[-1:]
    for i in range(1, len(gripper_cmds)):
        if (gripper_cmds[i] > 0) != (gripper_cmds[i-1] > 0):
            return True
    return False


def mpc_select_action(policy, obs_element, world_model, value_fn, K=8):
    """Sample K action chunks, predict futures via world model, score, pick best."""
    candidates = []
    features_list = []
    for _ in range(K):
        result = policy.infer(dict(obs_element))
        candidates.append(result["actions"])
        if "vlm_features" in result and result["vlm_features"] is not None:
            features_list.append(result["vlm_features"])

    if features_list and world_model is not None:
        device = next(world_model.parameters()).device
        h_t = np.asarray(features_list[0], dtype=np.float32)
        h_t_tensor = torch.tensor(h_t, dtype=torch.float32).unsqueeze(0).to(device)

        scores = []
        for action_chunk in candidates:
            a = np.asarray(action_chunk[:world_model.config.max_horizon], dtype=np.float32)
            if a.shape[0] < world_model.config.max_horizon:
                a = np.pad(a, [(0, world_model.config.max_horizon - a.shape[0]), (0, 0)])
            a_tensor = torch.tensor(a, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                predicted_future = world_model(h_t_tensor, a_tensor)
                score = value_fn(predicted_future).item()
            scores.append(score)

        best_idx = int(np.argmax(scores))
        spread = max(scores) - min(scores)
    else:
        best_idx = 0
        scores = []
        spread = 0.0

    return candidates[best_idx], {
        "best_idx": best_idx,
        "score_spread": spread,
        "scores": scores,
    }


def run_episodes(policy, world_model, value_fn, task_suite, args, K, mode_name):
    """Run episodes with given K. Returns results dict."""
    num_tasks = args.max_tasks or task_suite.n_tasks
    max_steps = MAX_STEPS[args.task_suite]

    total_episodes = 0
    total_successes = 0
    task_results = {}
    all_score_spreads = []
    contact_decisions = 0
    total_decisions = 0

    for task_id in range(num_tasks):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        env = OffScreenRenderEnv(
            bddl_file_name=str(task_bddl_file),
            camera_heights=LIBERO_ENV_RESOLUTION,
            camera_widths=LIBERO_ENV_RESOLUTION,
        )
        env.seed(args.seed)
        task_description = task.language
        task_successes = 0

        for trial_idx in range(min(args.num_trials, len(initial_states))):
            env.reset()
            obs = env.set_init_state(initial_states[trial_idx])
            action_plan = collections.deque()
            done = False
            t = 0
            last_action_chunk = None
            episode_start = time.time()

            while t < max_steps + args.num_steps_wait:
                # Episode timeout
                if time.time() - episode_start > args.episode_timeout:
                    print(f"    TIMEOUT at t={t}", flush=True)
                    break

                if t < args.num_steps_wait:
                    obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                    t += 1
                    continue

                if not action_plan:
                    total_decisions += 1

                    # Build observation
                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, RESIZE_SIZE, RESIZE_SIZE))
                    wrist_img = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist_img, RESIZE_SIZE, RESIZE_SIZE))
                    state = np.concatenate((
                        obs["robot0_eef_pos"],
                        _quat2axisangle(obs["robot0_eef_quat"]),
                        obs["robot0_gripper_qpos"],
                    ))
                    element = {
                        "observation/image": img,
                        "observation/wrist_image": wrist_img,
                        "observation/state": state,
                        "prompt": str(task_description),
                    }

                    # Decide: use MPC or baseline?
                    use_mpc = False
                    if K > 1:
                        if args.mpc_everywhere:
                            use_mpc = True
                        elif last_action_chunk is not None and detect_contact_in_chunk(last_action_chunk):
                            use_mpc = True
                            contact_decisions += 1

                    if use_mpc:
                        action_chunk, mpc_info = mpc_select_action(
                            policy, element, world_model, value_fn, K=K
                        )
                        all_score_spreads.append(mpc_info["score_spread"])
                    else:
                        result = policy.infer(element)
                        action_chunk = result["actions"]

                    last_action_chunk = action_chunk
                    action_plan.extend(action_chunk[:args.replan_steps])

                action = action_plan.popleft()
                obs, reward, done, info = env.step(action.tolist())
                if done:
                    total_successes += 1
                    task_successes += 1
                    break
                t += 1

            total_episodes += 1

        task_results[task_id] = {
            "task": task_description,
            "successes": task_successes,
            "trials": min(args.num_trials, len(initial_states)),
            "rate": task_successes / min(args.num_trials, len(initial_states)),
        }
        env.close()

        if (task_id + 1) % 10 == 0 or task_id == num_tasks - 1:
            rate = total_successes / total_episodes * 100 if total_episodes > 0 else 0
            spread_str = f", mean_spread={np.mean(all_score_spreads):.4f}" if all_score_spreads else ""
            contact_str = f", contact_decisions={contact_decisions}/{total_decisions}" if K > 1 and not args.mpc_everywhere else ""
            print(f"  [{mode_name}] Tasks: {task_id+1}/{num_tasks}, "
                  f"Episodes: {total_episodes}, "
                  f"Successes: {total_successes} ({rate:.1f}%)"
                  f"{spread_str}{contact_str}", flush=True)

    return {
        "total_episodes": total_episodes,
        "total_successes": total_successes,
        "success_rate": total_successes / total_episodes * 100 if total_episodes > 0 else 0,
        "task_results": task_results,
        "mean_score_spread": float(np.mean(all_score_spreads)) if all_score_spreads else 0,
        "contact_decisions": contact_decisions,
        "total_decisions": total_decisions,
    }


def main():
    args = parse_args()
    np.random.seed(args.seed)
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.task_suite not in MAX_STEPS:
        raise ValueError(f"Unknown task suite: {args.task_suite}")

    # === Step 1: Train or load world model ===
    wm_path = output_dir / "world_model.pt"
    wm_config_path = output_dir / "world_model_config.pt"

    if args.skip_wm_train and wm_path.exists():
        print("Loading existing world model...", flush=True)
        wm_config = torch.load(str(wm_config_path), weights_only=False)
        world_model = LatentWorldModel(wm_config)
        world_model.load_state_dict(torch.load(str(wm_path), weights_only=True))
    else:
        print("=== Training World Model ===", flush=True)
        features_path = hf_hub_download(args.hf_repo, "libero90_features_H10.npz", repo_type="dataset")
        features = FeatureDataset.load(features_path)
        world_model, wm_metrics = train_world_model(
            features, horizon=args.wm_horizon, size=args.wm_size,
            num_epochs=100, batch_size=64, lr=1e-4,
        )
        torch.save(world_model.state_dict(), wm_path)
        torch.save(world_model.config, wm_config_path)
        print(f"World model: {wm_metrics['param_count']:,} params, val_loss={wm_metrics['val_loss_best']:.6f}", flush=True)

    world_model.eval()
    if torch.cuda.is_available():
        world_model = world_model.cuda()

    # === Step 2: Load value function ===
    print("\n=== Loading Value Function ===", flush=True)
    vf_candidates = [
        output_dir.parent / "value_function" / "value_function.pt",
        pathlib.Path("data/contact_mpc/value_function/value_function.pt"),
        output_dir / "value_function.pt",
    ]
    vf_path = None
    for p in vf_candidates:
        if p.exists():
            vf_path = p
            break
    if vf_path is None:
        print("ERROR: Value function not found.", flush=True)
        return

    vf_config = torch.load(str(vf_path).replace("value_function.pt", "value_function_config.pt"), weights_only=False)
    value_fn = PairwiseValueFunction(vf_config["input_dim"], vf_config["hidden_dim"])
    value_fn.load_state_dict(torch.load(str(vf_path), weights_only=True))
    value_fn.eval()
    if torch.cuda.is_available():
        value_fn = value_fn.cuda()
    print(f"Loaded value function: {value_fn.param_count():,} params", flush=True)

    # === Step 3: Load Pi0.5 policy ===
    print("\n=== Loading Pi0.5 Policy ===", flush=True)
    from openpi.shared import download
    download.maybe_download(args.checkpoint + "/assets")
    train_config = _config.get_config(args.config_name)
    policy = _policy_config.create_trained_policy(train_config, args.checkpoint)
    print("Policy loaded.", flush=True)

    # === Step 4: Initialize LIBERO ===
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite]()

    # === Step 5: Run baseline (or skip) ===
    if args.skip_baseline:
        print(f"\nSkipping baseline (using known rates: libero_90=28.9%, libero_10=93.3%)", flush=True)
        known_rates = {"libero_90": 28.9, "libero_10": 93.3}
        baseline = {
            "success_rate": known_rates.get(args.task_suite, 0),
            "total_successes": 0,
            "total_episodes": 0,
        }
    else:
        print(f"\n{'='*60}", flush=True)
        print(f"Running baseline (K=1) on {args.task_suite}", flush=True)
        print(f"{'='*60}", flush=True)
        baseline = run_episodes(policy, world_model, value_fn, task_suite, args, K=1, mode_name="baseline")

    # === Step 6: Run MPC ===
    trigger_mode = "everywhere" if args.mpc_everywhere else "contact-only"
    print(f"\n{'='*60}", flush=True)
    print(f"Running MPC (K={args.K}, trigger={trigger_mode}) on {args.task_suite}", flush=True)
    print(f"{'='*60}", flush=True)
    mpc = run_episodes(policy, world_model, value_fn, task_suite, args, K=args.K, mode_name="mpc")

    # === Step 7: Report ===
    print(f"\n{'='*60}", flush=True)
    print("FINAL RESULTS", flush=True)
    print(f"{'='*60}", flush=True)
    baseline_str = f"{baseline['success_rate']:.1f}%"
    if baseline['total_episodes'] > 0:
        baseline_str += f" ({baseline['total_successes']}/{baseline['total_episodes']})"
    else:
        baseline_str += " (known)"
    print(f"Baseline (K=1):           {baseline_str}", flush=True)
    print(f"MPC (K={args.K}, {trigger_mode}): {mpc['success_rate']:.1f}% ({mpc['total_successes']}/{mpc['total_episodes']})", flush=True)
    print(f"Improvement:              {mpc['success_rate'] - baseline['success_rate']:+.1f} percentage points", flush=True)
    if mpc["mean_score_spread"] > 0:
        print(f"Mean score spread:        {mpc['mean_score_spread']:.4f}", flush=True)
    if mpc["contact_decisions"] > 0:
        print(f"Contact decisions:        {mpc['contact_decisions']}/{mpc['total_decisions']} "
              f"({mpc['contact_decisions']/mpc['total_decisions']*100:.1f}%)", flush=True)

    # Per-task comparison
    print(f"\nPer-task breakdown (tasks where results differ):", flush=True)
    for tid in sorted(baseline["task_results"].keys()):
        b = baseline["task_results"][tid]
        m = mpc["task_results"][tid]
        if b["rate"] != m["rate"]:
            delta = m["rate"] - b["rate"]
            print(f"  Task {tid}: baseline={b['rate']:.0%} → mpc={m['rate']:.0%} ({delta:+.0%}) | {b['task'][:60]}", flush=True)

    # Save
    np.savez(
        output_dir / "mpc_results.npz",
        baseline_rate=baseline["success_rate"],
        mpc_rate=mpc["success_rate"],
        K=args.K,
        trigger_mode=trigger_mode,
        mean_score_spread=mpc["mean_score_spread"],
        contact_decisions=mpc["contact_decisions"],
        total_decisions=mpc["total_decisions"],
    )
    print(f"\nSaved to {output_dir}/mpc_results.npz", flush=True)


if __name__ == "__main__":
    main()
