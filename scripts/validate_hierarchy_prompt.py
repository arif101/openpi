"""Validate the hierarchy hypothesis with hand-written prompt overrides.

Sprint 1 confirmed the structural limit: V(h) and Q(h, a) both hover at ±1pp
on LIBERO-90 perturbed. Phase 2 hypothesis: hierarchy fixes this by changing
the *task* Pi0.5 sees, not the candidate selection.

This script tests the underlying mechanism — does prompt-rewriting help on
the structural-limit tasks? — without yet building a Sonnet-in-the-loop
pipeline. We hand-write 1-3 sub-goal/rephrase prompts per failing task and
compare success rates against the original task description, all on the same
perturbed init states (paired comparison).

Decision rule:
  - ≥3 of 5 tasks where any override lifts success ≥+40pp → hierarchy works,
    commit to Phase 2 with Sonnet auto-decomposition.
  - 1-2 tasks improve, others flat → task-dependent; do prompt-engineering
    scoping before automating.
  - 0 tasks improve → hierarchy doesn't escape the structural limit;
    different research direction needed.

Usage:
    PYTHONPATH=src:third_party/libero uv run python3 -u \\
        scripts/validate_hierarchy_prompt.py \\
        --task-suite libero_90 --perturbation-cm 5.0 --seed 7 \\
        --num-trials 5 \\
        --prompt-overrides-json scripts/hierarchy_overrides_libero90.json \\
        --output-dir data/contact_mpc/hierarchy_validation
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
    p.add_argument("--task-suite", default="libero_90")
    p.add_argument("--num-trials", type=int, default=5)
    p.add_argument("--num-steps-wait", type=int, default=10)
    p.add_argument("--replan-steps", type=int, default=5)
    p.add_argument("--perturbation-cm", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--episode-timeout", type=int, default=300)
    p.add_argument(
        "--prompt-overrides-json",
        required=True,
        help="JSON file mapping task_idx (str) → list[str] of override prompts.",
    )
    p.add_argument("--output-dir", default="data/contact_mpc/hierarchy_validation")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def _quat2axisangle(quat):
    quat = quat.copy()
    if quat[3] > 1.0: quat[3] = 1.0
    elif quat[3] < -1.0: quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0): return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def perturb_object_positions(env, init_state, perturbation_m, rng):
    """Same perturbation logic as run_libero_pro.py / run_libero_pro_mcts.py."""
    state = init_state.clone() if hasattr(init_state, "clone") else init_state.copy()
    sim = env.env.sim
    model = sim.model
    for joint_idx in range(model.njnt):
        if model.jnt_type[joint_idx] == 0:  # mjJNT_FREE
            pos_start = model.jnt_qposadr[joint_idx]
            state[pos_start:pos_start + 3] += rng.normal(0, perturbation_m, size=3)
    return state


def build_obs_element(obs, prompt: str) -> dict:
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
        "prompt": str(prompt),
    }


def run_episode(*, policy, env, perturbed_init_state, prompt: str, max_steps: int,
                num_steps_wait: int, replan_steps: int, episode_timeout: int) -> bool:
    """Run one Pi0.5 baseline episode with a specific prompt. Returns success bool."""
    env.reset()
    env.set_init_state(perturbed_init_state)

    plan = collections.deque()
    done = False
    t = 0
    episode_start = time.time()

    while t < max_steps + num_steps_wait:
        if time.time() - episode_start > episode_timeout:
            break
        if t < num_steps_wait:
            obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
            t += 1
            continue
        if not plan:
            obs_element = build_obs_element(obs, prompt)
            result = policy.infer(dict(obs_element))
            action_chunk = np.asarray(result["actions"], dtype=np.float32)
            plan.extend(action_chunk[:replan_steps])
        action = plan.popleft()
        obs, _, done, _ = env.step(action.tolist())
        if done:
            return True
        t += 1
    return bool(done)


def main():
    args = parse_args()
    np.random.seed(args.seed)
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.task_suite not in MAX_STEPS:
        raise ValueError(f"Unknown task suite: {args.task_suite}")

    # Load overrides
    overrides_path = pathlib.Path(args.prompt_overrides_json)
    overrides_raw = json.loads(overrides_path.read_text())
    # Coerce keys to int and values to list[str]; skip leading-underscore keys (comments).
    overrides: dict[int, list[str]] = {
        int(k): list(v) for k, v in overrides_raw.items()
        if not k.startswith("_")
    }
    target_task_ids = sorted(overrides.keys())
    print(f"Loaded overrides for {len(target_task_ids)} tasks: {target_task_ids}", flush=True)

    # Load policy
    print("Loading Pi0.5 policy...", flush=True)
    from openpi.shared import download
    download.maybe_download(args.checkpoint + "/assets")
    download.maybe_download(args.checkpoint + "/params")
    train_cfg = _config.get_config(args.config_name)
    policy = _policy_config.create_trained_policy(train_cfg, args.checkpoint)
    print("Policy loaded.", flush=True)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite]()
    max_steps = MAX_STEPS[args.task_suite]
    perturbation_m = args.perturbation_cm / 100.0

    results = []  # list of {task_id, original_prompt, override_prompt, original_rate, override_rate, lift}

    for task_id in target_task_ids:
        task = task_suite.get_task(task_id)
        original_prompt = task.language
        task_overrides = overrides[task_id]

        bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        env = OffScreenRenderEnv(
            bddl_file_name=str(bddl),
            camera_heights=LIBERO_ENV_RESOLUTION,
            camera_widths=LIBERO_ENV_RESOLUTION,
        )
        env.seed(args.seed)
        init_states = task_suite.get_task_init_states(task_id)
        trials = min(args.num_trials, len(init_states))

        # Pre-compute perturbed init states ONCE so original + every override see
        # the same physical scene. Same perturbation seed convention as
        # run_libero_pro_mcts.py: args.seed + 1000 per task.
        perturb_rng = np.random.default_rng(args.seed + 1000)
        perturbed_states = []
        for trial_idx in range(trials):
            env.reset()
            init = init_states[trial_idx]
            perturbed = perturb_object_positions(env, init, perturbation_m, perturb_rng)
            perturbed_states.append(perturbed)

        print(f"\n{'='*70}", flush=True)
        print(f"Task {task_id}: {original_prompt}", flush=True)
        print(f"{'='*70}", flush=True)

        # Run original prompt
        original_succ = 0
        for trial_idx in range(trials):
            ok = run_episode(
                policy=policy, env=env,
                perturbed_init_state=perturbed_states[trial_idx],
                prompt=original_prompt,
                max_steps=max_steps,
                num_steps_wait=args.num_steps_wait,
                replan_steps=args.replan_steps,
                episode_timeout=args.episode_timeout,
            )
            original_succ += int(ok)
            print(f"  [original t={trial_idx}] {'OK' if ok else 'FAIL'} | "
                  f"running {original_succ}/{trial_idx+1}", flush=True)
        original_rate = original_succ / trials

        # Run each override
        for ov_idx, ov_prompt in enumerate(task_overrides):
            ov_succ = 0
            for trial_idx in range(trials):
                ok = run_episode(
                    policy=policy, env=env,
                    perturbed_init_state=perturbed_states[trial_idx],
                    prompt=ov_prompt,
                    max_steps=max_steps,
                    num_steps_wait=args.num_steps_wait,
                    replan_steps=args.replan_steps,
                    episode_timeout=args.episode_timeout,
                )
                ov_succ += int(ok)
                print(f"  [override {ov_idx}={ov_prompt[:50]!r} t={trial_idx}] "
                      f"{'OK' if ok else 'FAIL'} | running {ov_succ}/{trial_idx+1}",
                      flush=True)
            ov_rate = ov_succ / trials
            results.append({
                "task_id": task_id,
                "original_prompt": original_prompt,
                "override_idx": ov_idx,
                "override_prompt": ov_prompt,
                "trials": trials,
                "original_successes": original_succ,
                "override_successes": ov_succ,
                "original_rate": original_rate,
                "override_rate": ov_rate,
                "lift_pp": (ov_rate - original_rate) * 100,
            })
            print(f"  → original {original_rate:.0%}  vs  "
                  f"override {ov_rate:.0%}  =  lift {(ov_rate-original_rate)*100:+.0f}pp",
                  flush=True)

        env.close()

    # Aggregate report
    print(f"\n{'='*70}", flush=True)
    print("HIERARCHY VALIDATION SUMMARY", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"{'task':>4}  {'orig%':>6}  {'ovrd%':>6}  {'lift':>5}  override_prompt", flush=True)
    best_lift_per_task: dict[int, float] = {}
    for r in results:
        print(f"{r['task_id']:>4}  {r['original_rate']:>5.0%}  {r['override_rate']:>5.0%}  "
              f"{r['lift_pp']:>+4.0f}  {r['override_prompt'][:60]}", flush=True)
        prev = best_lift_per_task.get(r["task_id"], -1e9)
        best_lift_per_task[r["task_id"]] = max(prev, r["lift_pp"])

    print(f"\nBest lift per task:", flush=True)
    big_lifts = 0
    for tid in sorted(best_lift_per_task.keys()):
        bl = best_lift_per_task[tid]
        marker = "***" if bl >= 40 else ("**" if bl >= 20 else "")
        print(f"  task {tid}: {bl:+.0f}pp {marker}", flush=True)
        if bl >= 40:
            big_lifts += 1

    print(f"\nDecision rule:", flush=True)
    print(f"  Tasks with ≥+40pp best lift: {big_lifts} / {len(best_lift_per_task)}", flush=True)
    if big_lifts >= 3:
        print("  → HIERARCHY WORKS. Commit to Phase 2 with Sonnet decomposition.", flush=True)
    elif big_lifts >= 1:
        print("  → TASK-DEPENDENT. Scoping study before automation.", flush=True)
    else:
        print("  → STRUCTURAL LIMIT EVEN WITH HIERARCHY. Different direction needed.", flush=True)

    # Persist
    out = output_dir / f"hierarchy_validation_{args.task_suite}_{args.perturbation_cm}cm.json"
    out.write_text(json.dumps({
        "args": vars(args),
        "results": results,
        "best_lift_per_task": best_lift_per_task,
        "decision": "commit" if big_lifts >= 3 else ("scoping" if big_lifts >= 1 else "structural_limit"),
    }, indent=2, default=str))
    print(f"\nSaved to {out}", flush=True)


if __name__ == "__main__":
    main()
