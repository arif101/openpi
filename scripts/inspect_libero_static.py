"""Mac-friendly LIBERO scene inspection — pure file parsing, no env needed.

The full inspector (scripts/inspect_libero_scene.py) needs to spin up
OffScreenRenderEnv to read sim.data.body_xpos. That requires LIBERO +
robosuite + mujoco + OpenGL rendering — finicky on Mac.

This version reads the same information statically from:
  1. The BDDL task file → goal predicates
  2. The MJCF scene file → body positions from <body pos="x y z"> attrs

Limitations vs the full inspector:
  - Positions are the MJCF DEFAULT poses, not after init_state randomization.
    For LIBERO this is fine: init_state mostly randomizes object positions
    by a few cm, which doesn't change the rough EE target.
  - We can't see qpos-derived positions (kinematic chains).
  - LIBERO sometimes wraps the scene in a meta-XML; we follow <include> tags.

Dependencies: only stdlib + pyyaml. Works on Mac with no GPU.

Usage:
    python3 scripts/inspect_libero_static.py \\
        --bddl-root third_party/libero/libero/libero/bddl_files \\
        --task-suite libero_10 \\
        --output scripts/inspected_libero_10.yaml
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys
import xml.etree.ElementTree as ET

import yaml


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--bddl-root", required=True,
                   help="Path to LIBERO bddl_files directory")
    p.add_argument("--task-suite", default="libero_10",
                   help="LIBERO task suite name (matches a subdirectory under bddl-root)")
    p.add_argument("--task-idx", type=int, default=None,
                   help="Single task index. Omit to process all tasks in the suite.")
    p.add_argument("--output", default=None,
                   help="Output YAML path (if not set, just prints).")
    return p.parse_args()


def parse_bddl_goal(bddl_text: str) -> tuple[str, list[tuple[str, list[str]]]]:
    """Extract the goal block + parsed predicates from BDDL text."""
    m = re.search(r"\(:goal\s+(\(.*?\)\s*\))\s*\)", bddl_text, re.DOTALL)
    if not m:
        return "(no goal found)", []
    raw_goal = m.group(1)

    inner = raw_goal
    if inner.lstrip().startswith("(and"):
        inner = inner.strip()[4:-1]

    predicates = []
    for pred_match in re.finditer(r"\(([\w-]+)\s+([^()]*?)\)", inner):
        pred = pred_match.group(1)
        args = pred_match.group(2).split()
        predicates.append((pred, args))
    return raw_goal.strip(), predicates


def find_mjcf_for_bddl(bddl_path: pathlib.Path, libero_root: pathlib.Path) -> pathlib.Path | None:
    """Locate the MJCF / problem XML referenced by a BDDL file.

    LIBERO BDDL files reference (:domain ...) and (:problem ...) lines. The
    actual scene XML is built dynamically by robosuite from a base + object
    set. We approximate by scanning for `xml` references inside the BDDL.
    """
    text = bddl_path.read_text()
    # BDDL files in LIBERO sometimes have (:domain "<path>.xml")
    m = re.search(r":domain\s+\"([^\"]+\.xml)\"", text)
    if m:
        candidate = libero_root / m.group(1)
        if candidate.exists():
            return candidate
    # Fall back to a sibling "_scene.xml"
    sibling = bddl_path.with_suffix(".xml")
    if sibling.exists():
        return sibling
    return None


def walk_mjcf_for_bodies(mjcf_path: pathlib.Path,
                        bodies: dict[str, list[float]],
                        visited: set[pathlib.Path] = None) -> None:
    """Recursively extract <body name="..." pos="..."> from an MJCF + includes."""
    if visited is None:
        visited = set()
    p = mjcf_path.resolve()
    if p in visited or not p.exists():
        return
    visited.add(p)

    try:
        tree = ET.parse(p)
    except ET.ParseError as e:
        print(f"  [warn] could not parse {p}: {e}", file=sys.stderr)
        return
    root = tree.getroot()

    for body in root.iter("body"):
        name = body.get("name")
        pos = body.get("pos")
        if name and pos:
            xyz = [float(x) for x in pos.split()]
            if len(xyz) == 3:
                bodies[name] = xyz

    # Follow <include file="..."> directives
    for inc in root.iter("include"):
        f = inc.get("file")
        if f:
            sub_path = (p.parent / f).resolve()
            walk_mjcf_for_bodies(sub_path, bodies, visited)


def suggest_target(predicates, body_positions: dict[str, list[float]]) -> tuple[list[float] | None, str]:
    """Heuristic: pick a target xyz based on the first matchable predicate."""
    if not predicates:
        return None, "no predicates"

    def find_body(query: str) -> tuple[str, list[float]] | None:
        q = query.lower().rstrip("_0123456789")
        # Strip trailing numeric IDs (e.g., basket_1 → basket)
        for name, xyz in body_positions.items():
            n_lower = name.lower()
            if q == n_lower or q in n_lower or n_lower in q:
                return name, xyz
        return None

    relational = {"on", "in", "in-container"}
    unary = {"open", "close", "closed", "turnon", "turnoff", "turn-on", "turn-off"}
    for pred, args in predicates:
        p = pred.lower()
        if p in relational and len(args) >= 2:
            hit = find_body(args[1])
            if hit:
                return hit[1].copy() if hasattr(hit[1], "copy") else list(hit[1]), \
                    f"derived from ({pred} {' '.join(args)}) → body '{hit[0]}'"
        elif p in unary and len(args) >= 1:
            hit = find_body(args[0])
            if hit:
                return list(hit[1]), f"derived from ({pred} {args[0]}) → body '{hit[0]}'"

    return None, f"no body match for predicates {[p[0] for p in predicates]}"


def inspect_task(bddl_path: pathlib.Path, libero_root: pathlib.Path,
                 verbose: bool = True) -> dict:
    """Inspect a single task statically."""
    bddl_text = bddl_path.read_text()
    raw_goal, predicates = parse_bddl_goal(bddl_text)

    if verbose:
        print(f"\n--- {bddl_path.name} ---")
        print(f"goal: {raw_goal}")
        for pred, args in predicates:
            print(f"  ({pred} {' '.join(args)})")

    # Try to find the associated MJCF
    bodies: dict[str, list[float]] = {}
    mjcf = find_mjcf_for_bddl(bddl_path, libero_root)
    if mjcf is not None:
        walk_mjcf_for_bodies(mjcf, bodies)
        if verbose:
            print(f"mjcf: {mjcf} ({len(bodies)} bodies)")
    else:
        if verbose:
            print(f"mjcf: NOT FOUND (BDDL doesn't reference one statically)")

    # Filter out robot bodies
    robot_prefixes = ("robot0_", "gripper0_", "panda", "link", "world")
    object_bodies = {
        n: p for n, p in bodies.items()
        if not any(n.lower().startswith(rp) for rp in robot_prefixes)
    }

    if verbose:
        print(f"non-robot bodies (n={len(object_bodies)}):")
        for name, xyz in sorted(object_bodies.items()):
            print(f"  {name:30s} → [{xyz[0]:+.3f}, {xyz[1]:+.3f}, {xyz[2]:+.3f}]")

    suggested, rationale = suggest_target(predicates, object_bodies)
    if verbose:
        if suggested is not None:
            print(f"suggested: [{suggested[0]:+.3f}, {suggested[1]:+.3f}, {suggested[2]:+.3f}]")
            print(f"  rationale: {rationale}")
        else:
            print(f"suggested: NONE — {rationale}")

    return {
        "bddl": bddl_path.name,
        "goal_text": raw_goal,
        "predicates": [{"pred": p, "args": a} for p, a in predicates],
        "candidate_objects": object_bodies,
        "suggested_target_xyz": suggested,
        "rationale": rationale,
    }


def main() -> int:
    args = parse_args()
    bddl_root = pathlib.Path(args.bddl_root)
    if not bddl_root.exists():
        print(f"BDDL root not found: {bddl_root}", file=sys.stderr)
        return 1
    libero_root = bddl_root.parent  # one level up from bddl_files

    # Discover BDDL files for the suite. LIBERO's bddl_files structure has
    # subdirectories per suite (libero_10, libero_90, etc.) holding *.bddl files.
    suite_root = bddl_root / args.task_suite
    if not suite_root.exists():
        # Fallback: scan all .bddl recursively, filter by name prefix
        bddls = sorted(p for p in bddl_root.rglob("*.bddl")
                       if args.task_suite.lower() in str(p).lower())
    else:
        bddls = sorted(suite_root.rglob("*.bddl"))

    if not bddls:
        print(f"No BDDL files found for suite {args.task_suite}", file=sys.stderr)
        return 2

    print(f"Found {len(bddls)} BDDL files for {args.task_suite}", file=sys.stderr)

    if args.task_idx is not None:
        if args.task_idx >= len(bddls):
            print(f"task_idx {args.task_idx} out of range (n={len(bddls)})", file=sys.stderr)
            return 3
        bddls = [bddls[args.task_idx]]

    results = {}
    for idx, bddl in enumerate(bddls):
        info = inspect_task(bddl, libero_root, verbose=True)
        results[args.task_idx if args.task_idx is not None else idx] = info

    if args.output:
        out = {args.task_suite: {}}
        for idx, info in results.items():
            out[args.task_suite][idx] = {
                "description": info["bddl"],
                "target_xyz": info["suggested_target_xyz"] or [0.0, 0.0, 0.85],
                "rationale": info["rationale"],
                "goal_text": info["goal_text"],
                "candidate_objects": info["candidate_objects"],
            }
        pathlib.Path(args.output).write_text(yaml.safe_dump(out, sort_keys=False))
        print(f"\n→ Wrote {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
