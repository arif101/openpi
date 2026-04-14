"""End-to-end MPC experiment: train world model, evaluate with value function, run on LIBERO.

This script does everything needed for the MPC test:
1. Downloads features + rollouts from HF
2. Trains a world model (small, H=10)
3. Loads the trained value function
4. Re-evaluates KS3 with the real value function
5. Runs best-of-K MPC evaluation on LIBERO-90

Usage:
    PYTHONPATH=/workspace/openpi/src:/workspace/openpi/third_party/libero \
    /workspace/openpi/.venv/bin/python -u scripts/run_mpc_experiment.py \
        --task-suite libero_90 --num-trials 5 --K 8 \
        --output-dir data/contact_mpc/mpc_results
"""

import argparse
import collections
import math
import pathlib
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
from openpi.contact_mpc.world_model.train import train_world_model, build_dataset_for_horizon
from openpi.contact_mpc.value_function.architecture import PairwiseValueFunction
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config
from openpi.models import model as _model

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
    parser.add_argument("--K", type=int, default=8, help="Number of candidate action chunks")
    parser.add_argument("--wm-size", default="small", help="World model size: small, medium, large")
    parser.add_argument("--wm-horizon", type=int, default=10)
    parser.add_argument("--output-dir", default="data/contact_mpc/mpc_results")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--skip-wm-train", action="store_true", help="Skip world model training (load from output-dir)")
    parser.add_argument("--skip-eval", action="store_true", help="Only train world model, don't run LIBERO eval")
    return parser.parse_args()


def _quat2axisangle(quat):
    quat = quat.copy()
    if quat[3] > 1.0: quat[3] = 1.0
    elif quat[3] < -1.0: quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0): return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def mpc_select_action(
    policy,
    obs_element: dict,
    world_model: LatentWorldModel,
    value_fn: PairwiseValueFunction,
    K: int = 8,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Sample K action chunks, predict futures, score, pick best.

    Args:
        policy: The loaded Policy with transforms.
        obs_element: Raw observation dict (observation/image, etc.)
        world_model: Trained world model.
        value_fn: Trained value function.
        K: Number of candidates.

    Returns:
        Tuple of (best_action_chunk, best_features, info_dict).
    """
    # Generate K candidates by calling policy.infer K times
    # Each call uses a different internal RNG state
    candidates = []
    features_list = []
    for _ in range(K):
        result = policy.infer(dict(obs_element))  # copy to avoid mutation
        candidates.append(result["actions"])  # [action_horizon, 7]
        if "vlm_features" in result and result["vlm_features"] is not None:
            features_list.append(result["vlm_features"])  # [2048]

    # If we have features and a world model, predict futures and score
    if features_list and world_model is not None:
        h_t = features_list[0]  # All K share the same observation → same hidden state
        h_t_tensor = torch.tensor(h_t, dtype=torch.float32).unsqueeze(0)  # [1, 2048]

        scores = []
        for i, action_chunk in enumerate(candidates):
            # Pad action chunk to world model's expected dims
            a = action_chunk[:world_model.config.max_horizon]  # [H, 7]
            if a.shape[0] < world_model.config.max_horizon:
                a = np.pad(a, [(0, world_model.config.max_horizon - a.shape[0]), (0, 0)])
            a_tensor = torch.tensor(a, dtype=torch.float32).unsqueeze(0)  # [1, H, 7]

            with torch.no_grad():
                predicted_future = world_model(h_t_tensor, a_tensor)  # [1, 2048]
                score = value_fn(predicted_future).item()
            scores.append(score)

        best_idx = np.argmax(scores)
        info = {
            "scores": scores,
            "best_idx": best_idx,
            "best_score": scores[best_idx],
            "worst_score": min(scores),
            "score_spread": max(scores) - min(scores),
        }
    else:
        # Fallback: random selection
        best_idx = 0
        info = {"scores": [], "best_idx": 0, "score_spread": 0.0}

    return candidates[best_idx], features_list[0] if features_list else None, info


def main():
    args = parse_args()
    np.random.seed(args.seed)
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    max_steps = MAX_STEPS.get(args.task_suite)
    if max_steps is None:
        raise ValueError(f"Unknown task suite: {args.task_suite}")

    # === Step 1: Train world model on demo features ===
    wm_path = output_dir / "world_model.pt"
    wm_config_path = output_dir / "world_model_config.pt"

    if args.skip_wm_train and wm_path.exists():
        print("Loading existing world model...", flush=True)
        wm_config = torch.load(str(wm_config_path), weights_only=False)
        world_model = LatentWorldModel(wm_config)
        world_model.load_state_dict(torch.load(str(wm_path), weights_only=True))
        world_model.eval()
    else:
        print("=== Training World Model ===", flush=True)
        features_path = hf_hub_download(args.hf_repo, "libero90_features_H10.npz", repo_type="dataset")
        features = FeatureDataset.load(features_path)
        print(f"Loaded {features.hidden_states.shape[0]} demo triples", flush=True)

        world_model, wm_metrics = train_world_model(
            features, horizon=args.wm_horizon, size=args.wm_size,
            num_epochs=100, batch_size=64, lr=1e-4,
        )
        torch.save(world_model.state_dict(), wm_path)
        torch.save(world_model.config, wm_config_path)
        print(f"World model trained: {wm_metrics['param_count']:,} params, "
              f"val_loss={wm_metrics['val_loss_best']:.6f}", flush=True)

    # === Step 2: Load value function ===
    print("\n=== Loading Value Function ===", flush=True)
    vf_path = output_dir.parent / "value_function" / "value_function.pt"
    if not vf_path.exists():
        # Try to find it
        for candidate in [
            pathlib.Path("data/contact_mpc/value_function/value_function.pt"),
            output_dir / "value_function.pt",
        ]:
            if candidate.exists():
                vf_path = candidate
                break

    if not vf_path.exists():
        print("ERROR: Value function not found. Train it first with run_train_value_function.py", flush=True)
        return

    vf_config = torch.load(str(vf_path).replace("value_function.pt", "value_function_config.pt"), weights_only=False)
    value_fn = PairwiseValueFunction(vf_config["input_dim"], vf_config["hidden_dim"])
    value_fn.load_state_dict(torch.load(str(vf_path), weights_only=True))
    value_fn.eval()
    print(f"Loaded value function: {value_fn.param_count():,} params", flush=True)

    if args.skip_eval:
        print("Skipping LIBERO evaluation (--skip-eval)", flush=True)
        return

    # === Step 3: Load Pi0.5 policy ===
    print("\n=== Loading Pi0.5 Policy ===", flush=True)
    from openpi.shared import download
    download.maybe_download(args.checkpoint + "/assets")  # Ensure assets are downloaded
    train_config = _config.get_config(args.config_name)
    policy = _policy_config.create_trained_policy(train_config, args.checkpoint)
    print("Policy loaded.", flush=True)

    # === Step 4: Run MPC evaluation on LIBERO ===
    print(f"\n=== MPC Evaluation: K={args.K}, {args.task_suite} ===", flush=True)
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite]()
    num_tasks = args.max_tasks or task_suite.n_tasks

    # Run both baseline (K=1) and MPC (K=args.K) for comparison
    results = {}

    for mode, K in [("baseline", 1), ("mpc", args.K)]:
        print(f"\n--- Running {mode} (K={K}) ---", flush=True)
        total_episodes = 0
        total_successes = 0
        task_results = {}
        all_score_spreads = []

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

                while t < max_steps + args.num_steps_wait:
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    if not action_plan:
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

                        if K == 1:
                            # Baseline: single action chunk
                            result = policy.infer(element)
                            action_chunk = result["actions"]
                        else:
                            # MPC: sample K, score with world model + value function
                            action_chunk, _, mpc_info = mpc_select_action(
                                policy, element, world_model, value_fn, K=K
                            )
                            all_score_spreads.append(mpc_info["score_spread"])

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

            if (task_id + 1) % 10 == 0:
                print(f"  Tasks: {task_id+1}/{num_tasks}, "
                      f"Episodes: {total_episodes}, "
                      f"Successes: {total_successes} ({total_successes/total_episodes*100:.1f}%)",
                      flush=True)

        rate = total_successes / total_episodes * 100
        results[mode] = {
            "total_episodes": total_episodes,
            "total_successes": total_successes,
            "success_rate": rate,
            "task_results": task_results,
            "mean_score_spread": np.mean(all_score_spreads) if all_score_spreads else 0,
        }
        print(f"\n{mode.upper()}: {total_successes}/{total_episodes} ({rate:.1f}%)", flush=True)

    # === Step 5: Report ===
    print(f"\n{'='*60}", flush=True)
    print("FINAL RESULTS", flush=True)
    print(f"{'='*60}", flush=True)
    baseline_rate = results["baseline"]["success_rate"]
    mpc_rate = results["mpc"]["success_rate"]
    print(f"Baseline (K=1): {baseline_rate:.1f}%", flush=True)
    print(f"MPC (K={args.K}):     {mpc_rate:.1f}%", flush=True)
    print(f"Improvement:    {mpc_rate - baseline_rate:+.1f} percentage points", flush=True)
    if results["mpc"]["mean_score_spread"] > 0:
        print(f"Mean score spread across K={args.K} candidates: {results['mpc']['mean_score_spread']:.4f}", flush=True)

    # Per-task comparison
    print(f"\nPer-task breakdown (tasks where results differ):", flush=True)
    for tid in sorted(results["baseline"]["task_results"].keys()):
        b = results["baseline"]["task_results"][tid]
        m = results["mpc"]["task_results"][tid]
        if b["rate"] != m["rate"]:
            print(f"  Task {tid} ({b['task'][:50]}): baseline={b['rate']:.0%} → mpc={m['rate']:.0%}", flush=True)

    # Save results
    np.savez(
        output_dir / "mpc_results.npz",
        baseline_rate=baseline_rate,
        mpc_rate=mpc_rate,
        K=args.K,
    )
    print(f"\nSaved to {output_dir}/mpc_results.npz", flush=True)


if __name__ == "__main__":
    main()
