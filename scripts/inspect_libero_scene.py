"""Inspect a LIBERO task's scene + BDDL goal to extract real EE targets.

Run this BEFORE running MPPI experiments so the YAML targets correspond
to actual scene objects, not educated guesses.

For each task:
  1. Load the BDDL file (the task spec)
  2. Initialize the env at trial_idx=0
  3. Print the BDDL goal predicates verbatim
  4. Print all body names + world-frame xyz positions
  5. Suggest a reasonable EE target based on the goal

Usage:
    # Single task
    PYTHONPATH=src:third_party/libero uv run python3 -u \\
        scripts/inspect_libero_scene.py \\
        --task-suite libero_10 --task-idx 0

    # All tasks in a suite — dumps to YAML you can manually clean up
    PYTHONPATH=src:third_party/libero uv run python3 -u \\
        scripts/inspect_libero_scene.py \\
        --task-suite libero_10 --all \\
        --output scripts/reason_v3_targets_libero_10_inspected.yaml
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import re
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

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task-suite", default="libero_10")
    p.add_argument("--task-idx", type=int, default=None,
                   help="Single task index. Use --all to inspect all.")
    p.add_argument("--all", action="store_true", help="Inspect every task in the suite")
    p.add_argument("--output", default=None,
                   help="If set, write a YAML target file at this path")
    return p.parse_args()


def parse_bddl_goal(bddl_path: pathlib.Path) -> tuple[str, list[tuple[str, list[str]]]]:
    """Extract the goal block + parsed predicates from a BDDL file.

    Returns:
        (raw_goal_text, [(predicate_name, [args]), ...])
    """
    text = bddl_path.read_text()

    # Goal block: (:goal (and (P1 args) (P2 args) ...))
    m = re.search(r"\(:goal\s+(\(.*?\)\s*\))\s*\)", text, re.DOTALL)
    if not m:
        return "(no goal found)", []
    raw_goal = m.group(1)

    # Naive predicate extraction: match (predname arg1 arg2 ...) inside the goal
    predicates = []
    # Strip outer (and ...) if present
    inner = raw_goal
    if inner.lstrip().startswith("(and"):
        inner = inner.strip()[4:-1]
    for pred_match in re.finditer(r"\(([\w-]+)\s+([^()]*?)\)", inner):
        pred = pred_match.group(1)
        args = pred_match.group(2).split()
        predicates.append((pred, args))
    return raw_goal.strip(), predicates


def suggest_target(predicates, body_positions: dict[str, np.ndarray]) -> tuple[np.ndarray | None, str]:
    """Heuristic: derive a single EE target from the BDDL goal predicates.

    For each predicate type, suggest where the gripper would end up at success:
      (on X Y), (in X Y), (in-container X Y) → top/center of Y
      (open X), (closed X) → near X's handle (X position is a fallback)
      (turn-on X) → near X

    Returns (xyz, rationale) or (None, "no target") if we can't derive one.
    """
    if not predicates:
        return None, "no predicates"

    # BDDL uses capitalized predicates (In, On, Close, Turnon) — lowercase for matching.
    # Pick the FIRST predicate that names a known body — use Y if it's a relational predicate
    relational = {"on", "in", "in-container"}
    unary = {"open", "close", "closed", "turnon", "turnoff", "turn-on", "turn-off"}
    for pred, args in predicates:
        p = pred.lower()
        if p in relational and len(args) >= 2:
            target_obj = args[1]
            for body_name, xyz in body_positions.items():
                if target_obj.lower() in body_name.lower() or body_name.lower() in target_obj.lower():
                    return xyz.copy(), f"derived from ({pred} {' '.join(args)}) → body '{body_name}'"
        elif p in unary and len(args) >= 1:
            target_obj = args[0]
            for body_name, xyz in body_positions.items():
                if target_obj.lower() in body_name.lower() or body_name.lower() in target_obj.lower():
                    return xyz.copy(), f"derived from ({pred} {target_obj}) → body '{body_name}'"

    return None, f"no body match for predicates {[p[0] for p in predicates]}"


def inspect_task(task_suite_name: str, task_idx: int, verbose: bool = True) -> dict:
    """Inspect a single task and return its structure as a dict."""
    bm = benchmark.get_benchmark_dict()[task_suite_name]()
    task = bm.get_task(task_idx)
    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file

    if verbose:
        print(f"\n{'='*70}")
        print(f"Task {task_idx}: {task.language}")
        print(f"{'='*70}")
        print(f"BDDL: {bddl}")

    raw_goal, predicates = parse_bddl_goal(bddl)
    if verbose:
        print(f"\n--- BDDL goal ---")
        print(f"  {raw_goal}")
        if predicates:
            print(f"\n--- parsed predicates ---")
            for pred, args in predicates:
                print(f"  ({pred} {' '.join(args)})")

    # Initialize the env at trial 0 to extract body positions
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl), camera_heights=128, camera_widths=128,
    )
    env.seed(0)
    init_states = bm.get_task_init_states(task_idx)
    env.reset()
    env.set_init_state(init_states[0])

    sim = env.env.sim
    model = sim.model

    # Collect body name → xyz, skipping the world body and unnamed
    body_positions = {}
    for b in range(1, model.nbody):
        name = model.body_id2name(b) if hasattr(model, "body_id2name") else f"body_{b}"
        if not name:
            continue
        body_positions[name] = sim.data.body_xpos[b].copy()

    # Identify potentially-relevant bodies (objects, not robot links)
    robot_prefixes = ("robot0_", "gripper0_", "panda", "link", "world")
    object_bodies = {
        n: p for n, p in body_positions.items()
        if not any(n.lower().startswith(p) for p in robot_prefixes)
    }

    if verbose:
        print(f"\n--- non-robot bodies (n={len(object_bodies)}) ---")
        for name, xyz in sorted(object_bodies.items()):
            print(f"  {name:35s} → [{xyz[0]:+.3f}, {xyz[1]:+.3f}, {xyz[2]:+.3f}]")

    suggested_xyz, rationale = suggest_target(predicates, object_bodies)
    if verbose:
        print(f"\n--- suggested EE target ---")
        if suggested_xyz is not None:
            print(f"  xyz = [{suggested_xyz[0]:+.3f}, {suggested_xyz[1]:+.3f}, {suggested_xyz[2]:+.3f}]")
            print(f"  rationale: {rationale}")
        else:
            print(f"  (could not derive automatically: {rationale})")
            print(f"  Inspect the bodies above and pick the right target manually.")

    env.close()

    return {
        "task_idx": task_idx,
        "description": task.language,
        "bddl_path": str(bddl),
        "goal_text": raw_goal,
        "predicates": [{"pred": p, "args": a} for p, a in predicates],
        "object_positions": {n: p.tolist() for n, p in object_bodies.items()},
        "suggested_target_xyz": suggested_xyz.tolist() if suggested_xyz is not None else None,
        "rationale": rationale,
    }


def main() -> int:
    args = parse_args()
    if args.all and args.task_idx is not None:
        print("Use --task-idx OR --all, not both", file=sys.stderr)
        return 1
    if not args.all and args.task_idx is None:
        print("Provide --task-idx <N> or --all", file=sys.stderr)
        return 1

    bm = benchmark.get_benchmark_dict()[args.task_suite]()
    if args.all:
        indices = list(range(bm.n_tasks))
    else:
        indices = [args.task_idx]

    inspections = {}
    for idx in indices:
        info = inspect_task(args.task_suite, idx, verbose=True)
        inspections[idx] = info

    if args.output:
        # Write a YAML in the format reason_v3_targets.yaml expects
        out_dict = {args.task_suite: {}}
        for idx, info in inspections.items():
            entry = {
                "description": info["description"],
                "target_xyz": info["suggested_target_xyz"] or [0.0, 0.0, 0.85],
                "rationale": info["rationale"],
                # Keep raw inspection data so a human can verify / override
                "goal_text": info["goal_text"],
                "candidate_objects": info["object_positions"],
            }
            out_dict[args.task_suite][idx] = entry
        pathlib.Path(args.output).write_text(yaml.safe_dump(out_dict, sort_keys=False))
        print(f"\n→ Wrote inspection YAML to {args.output}")
        print(f"  Review each task; if the suggested target is wrong, pick the correct")
        print(f"  body's xyz from candidate_objects and set target_xyz accordingly.")
        print(f"  Then copy the cleaned-up file to scripts/reason_v3_targets.yaml")

    return 0


if __name__ == "__main__":
    sys.exit(main())
