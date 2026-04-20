"""Per-candidate LIBERO evaluation that saves per-task success rates.

Simpler wrapper around the eval loop in run_libero_pro.py: runs a single
policy checkpoint on a (subset of) LIBERO tasks and writes a JSON file
with {task_id: {success_rate, successes, trials, task_description}}.

The correlation study pairs this per-task breakdown with the per-task
imagined scores produced by the offline evaluator.

Usage:
    PYTHONPATH=/workspace/openpi/src:/workspace/openpi/third_party/libero \
    /workspace/openpi/.venv/bin/python -u scripts/run_libero_candidate_eval.py \
        --checkpoint data/contact_mpc/candidates/planning_0 \
        --task-ids 0,1,2,3,4 \
        --num-trials 5 \
        --output data/contact_mpc/real_eval/planning_0.json
"""

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
    p.add_argument("--checkpoint", required=True,
                   help="Path to LoRA candidate checkpoint (directory).")
    p.add_argument("--candidate-name", default=None,
                   help="Label for this candidate. Defaults to checkpoint basename.")
    p.add_argument("--task-suite", default="libero_90")
    p.add_argument("--task-ids", default=None,
                   help="Comma-separated task IDs to eval (default: all tasks).")
    p.add_argument("--num-trials", type=int, default=5)
    p.add_argument("--trial-start-idx", type=int, default=0,
                   help="Offset into task init states (use held-out trials, e.g. 5 to avoid training trials 0-4).")
    p.add_argument("--num-steps-wait", type=int, default=10)
    p.add_argument("--replan-steps", type=int, default=5)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--episode-timeout", type=int, default=300)
    p.add_argument("--output", required=True)
    return p.parse_args()


def _quat2axisangle(quat):
    quat = quat.copy()
    if quat[3] > 1.0: quat[3] = 1.0
    elif quat[3] < -1.0: quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0): return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def run_one_candidate(policy, task_suite, task_ids, args):
    per_task: dict[int, dict] = {}
    for task_id in task_ids:
        task = task_suite.get_task(task_id)
        init_states = task_suite.get_task_init_states(task_id)
        start = args.trial_start_idx
        end = min(start + args.num_trials, len(init_states))
        if end <= start:
            print(f"  [SKIP] task {task_id}: not enough init states "
                  f"(have {len(init_states)}, need start={start})", flush=True)
            continue

        bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        env = OffScreenRenderEnv(
            bddl_file_name=str(bddl),
            camera_heights=LIBERO_ENV_RESOLUTION,
            camera_widths=LIBERO_ENV_RESOLUTION,
        )
        env.seed(args.seed)

        successes = 0
        trials = end - start
        for trial_idx in range(start, end):
            env.reset()
            obs = env.set_init_state(init_states[trial_idx])
            plan = collections.deque()
            t = 0
            done = False
            ep_start = time.time()

            while t < MAX_STEPS[args.task_suite] + args.num_steps_wait:
                if time.time() - ep_start > args.episode_timeout:
                    break
                if t < args.num_steps_wait:
                    obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
                    t += 1
                    continue
                if not plan:
                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, RESIZE_SIZE, RESIZE_SIZE))
                    wrist = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist, RESIZE_SIZE, RESIZE_SIZE))
                    state = np.concatenate((
                        obs["robot0_eef_pos"],
                        _quat2axisangle(obs["robot0_eef_quat"]),
                        obs["robot0_gripper_qpos"],
                    ))
                    element = {
                        "observation/image": img,
                        "observation/wrist_image": wrist,
                        "observation/state": state,
                        "prompt": str(task.language),
                    }
                    chunk = policy.infer(element)["actions"]
                    plan.extend(chunk[: args.replan_steps])
                action = plan.popleft()
                obs, _, done, _ = env.step(action.tolist())
                if done:
                    successes += 1
                    break
                t += 1

        per_task[task_id] = {
            "task_description": task.language,
            "successes": successes,
            "trials": trials,
            "success_rate": successes / trials if trials else 0.0,
        }
        print(f"  task {task_id}: {successes}/{trials} "
              f"({per_task[task_id]['success_rate']:.0%}) — {task.language[:70]}",
              flush=True)
        env.close()

    return per_task


def main():
    args = parse_args()
    np.random.seed(args.seed)

    candidate_name = args.candidate_name or pathlib.Path(args.checkpoint).name

    print(f"Loading Pi0.5 + LoRA from {args.checkpoint}", flush=True)
    train_config = _config.get_config(args.config_name)
    policy = _policy_config.create_trained_policy(train_config, args.checkpoint)
    print("Policy loaded.", flush=True)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite]()

    if args.task_ids:
        task_ids = [int(x) for x in args.task_ids.split(",")]
    else:
        task_ids = list(range(task_suite.n_tasks))
    print(f"Evaluating on {len(task_ids)} tasks, {args.num_trials} trials each "
          f"(trial_start_idx={args.trial_start_idx})", flush=True)

    per_task = run_one_candidate(policy, task_suite, task_ids, args)

    total_successes = sum(p["successes"] for p in per_task.values())
    total_trials = sum(p["trials"] for p in per_task.values())
    payload = {
        "candidate_name": candidate_name,
        "checkpoint": args.checkpoint,
        "task_suite": args.task_suite,
        "trial_start_idx": args.trial_start_idx,
        "num_trials_per_task": args.num_trials,
        "overall_success_rate": total_successes / total_trials if total_trials else 0.0,
        "total_successes": total_successes,
        "total_trials": total_trials,
        "per_task": {str(k): v for k, v in per_task.items()},
    }

    out_path = pathlib.Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nOverall: {total_successes}/{total_trials} "
          f"({payload['overall_success_rate']:.1%})", flush=True)
    print(f"Saved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
