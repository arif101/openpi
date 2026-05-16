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


def _match_body(query: str, body_positions: dict) -> tuple[str, "np.ndarray"] | None:
    q = query.lower()
    for body_name, xyz in body_positions.items():
        n = body_name.lower()
        if q in n or n in q:
            return body_name, xyz
    return None


def suggest_target(predicates, body_positions: dict[str, np.ndarray]) -> tuple[dict | None, str]:
    """Derive MPPI cost target entry (mode + track_bodies + goal_xyz) from BDDL goal.

    Schema returned:
      {"mode": "object", "track_bodies": [name1, name2, ...], "goal_xyz": [x, y, z]}
      {"mode": "ee", "goal_xyz": [x, y, z]}
      None if no body match found.

    For relational predicates (In/On X Y): X is tracked, Y is the goal. Multiple
    relational predicates with the same Y → all Xs added to track_bodies (the
    runtime picks the farthest). Unary predicates (Close/Turnon Y): EE mode,
    Y is the goal.
    """
    if not predicates:
        return None, "no predicates"

    relational = {"on", "in", "in-container"}
    unary = {"open", "close", "closed", "turnon", "turnoff", "turn-on", "turn-off"}

    track_bodies: list[str] = []
    goal_body: tuple[str, np.ndarray] | None = None
    fired: list[str] = []
    mode = None

    for pred, args in predicates:
        p = pred.lower()
        if p in relational and len(args) >= 2:
            obj_hit = _match_body(args[0], body_positions)
            goal_hit = _match_body(args[1], body_positions)
            if obj_hit and goal_hit:
                if goal_body is None:
                    goal_body = goal_hit
                    mode = "object"
                if obj_hit[0] not in track_bodies:
                    track_bodies.append(obj_hit[0])
                fired.append(f"({pred} {args[0]} {args[1]}) → track {obj_hit[0]}, goal {goal_hit[0]}")
        elif p in unary and len(args) >= 1:
            goal_hit = _match_body(args[0], body_positions)
            if goal_hit:
                if goal_body is None:
                    goal_body = goal_hit
                    mode = "ee"
                fired.append(f"({pred} {args[0]}) → ee mode, goal {goal_hit[0]}")

    if goal_body is None:
        return None, f"no body match for predicates {[p[0] for p in predicates]}"

    out = {
        "mode": mode,
        "goal_xyz": [float(x) for x in goal_body[1]],
    }
    if mode == "object":
        out["track_bodies"] = track_bodies
    return out, "; ".join(fired)


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

    suggested, rationale = suggest_target(predicates, object_bodies)
    if verbose:
        print(f"\n--- suggested MPPI target ---")
        if suggested is not None:
            xyz = suggested["goal_xyz"]
            print(f"  mode: {suggested['mode']}")
            print(f"  goal_xyz: [{xyz[0]:+.3f}, {xyz[1]:+.3f}, {xyz[2]:+.3f}]")
            if "track_bodies" in suggested:
                print(f"  track_bodies: {suggested['track_bodies']}")
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
        "suggested": suggested,
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
            sugg = info.get("suggested") or {}
            entry = {
                "description": info["description"],
                "mode": sugg.get("mode", "object"),
                "goal_xyz": sugg.get("goal_xyz", [0.0, 0.0, 0.85]),
                "rationale": info["rationale"],
                # Raw inspection data preserved for manual review / override
                "goal_text": info["goal_text"],
                "candidate_objects": info["object_positions"],
            }
            if "track_bodies" in sugg:
                entry["track_bodies"] = sugg["track_bodies"]
            out_dict[args.task_suite][idx] = entry
        pathlib.Path(args.output).write_text(yaml.safe_dump(out_dict, sort_keys=False))
        print(f"\n→ Wrote inspection YAML to {args.output}")
        print(f"  Review each task; if the suggestion is wrong, edit goal_xyz /")
        print(f"  track_bodies using bodies from candidate_objects. Then copy the")
        print(f"  cleaned-up file to scripts/reason_v3_targets.yaml")

    return 0


if __name__ == "__main__":
    sys.exit(main())
