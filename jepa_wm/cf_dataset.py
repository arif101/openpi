"""Counterfactual instruction construction for LIBERO grounding fine-tune.

For each (image, state, L_correct, action) we build L_cf by swapping ONE referent to a
different object/relation that is plausible in the SAME scene:
  - object suite : swap the named object -> another graspable object PRESENT in that scene
                   (hard, in-scene minimal pair; verified from BDDL).
  - goal suite   : swap to another goal instruction (shared scene across the 10 goal tasks).
  - spatial/long : swap to another same-suite instruction (soft negative; flagged).

Use: build_task_lookup() once, then make_counterfactual(instruction, rng) per sample.
Wrap a LeRobotDataset to emit `task_cf` alongside `task`.
"""
from __future__ import annotations

import os
import re

from libero.libero import benchmark, get_libero_path

_NONGRASP = {
    "basket", "plate", "stove", "cabinet", "rack", "microwave", "ramekin",
    "glazed_rim_porcelain_ramekin", "caddy", "wooden_cabinet", "wine_rack",
    "cookies", "cookie_box", "flat_stove", "short_cabinet", "white_cabinet",
    "main_table", "kitchen_table", "study_table", "living_room_table",
}
_SUITES = ["libero_object", "libero_spatial", "libero_goal", "libero_10"]


def _scene_graspables(bddl_path: str) -> list[str]:
    txt = open(bddl_path).read()
    m = re.search(r"\(:objects(.*?)\)", txt, re.S)
    out = []
    if m:
        for line in m.group(1).strip().splitlines():
            parts = line.strip().split(" - ")
            if len(parts) == 2:
                t = parts[1].strip()
                if t not in _NONGRASP:
                    out.append(t)
    return sorted(set(out))


def build_task_lookup() -> dict:
    """instruction -> {suite, graspable_names(present in scene), bddl}."""
    lut = {}
    for suite in _SUITES:
        bd = benchmark.get_benchmark_dict()[suite]()
        for i in range(bd.n_tasks):
            t = bd.get_task(i)
            bddl = os.path.join(get_libero_path("bddl_files"), t.problem_folder, t.bddl_file)
            graspables = _scene_graspables(bddl)
            lut[t.language] = {
                "suite": suite,
                "graspable_names": [g.replace("_", " ") for g in graspables],
                "bddl": bddl,
            }
    return lut


def make_counterfactual(instruction: str, meta: dict, lut: dict, rng) -> tuple[str, str]:
    """Return (L_cf, kind). kind in {hard_object, swap_goal, soft_suite, none}."""
    suite = meta["suite"]
    if suite == "libero_object":
        present = meta["graspable_names"]
        cur = [nm for nm in present if nm in instruction]
        others = [nm for nm in present if nm not in instruction]
        if cur and others:
            tgt = others[rng.integers(len(others))]
            return instruction.replace(cur[0], tgt), "hard_object"
    # goal / spatial / long: swap to another instruction from the same suite
    pool = [k for k, v in lut.items() if v["suite"] == suite and k != instruction]
    if pool:
        cf = pool[rng.integers(len(pool))]
        return cf, ("swap_goal" if suite == "libero_goal" else "soft_suite")
    return instruction, "none"


if __name__ == "__main__":
    import numpy as np
    rng = np.random.default_rng(0)
    lut = build_task_lookup()
    print(f"built lookup for {len(lut)} instructions across {len(_SUITES)} suites\n")
    # show example counterfactuals per suite
    seen = set()
    for instr, meta in lut.items():
        s = meta["suite"]
        if s in seen:
            continue
        seen.add(s)
        cf, kind = make_counterfactual(instr, meta, lut, rng)
        print(f"[{s}] ({kind})")
        print(f"   L     : {instr!r}")
        print(f"   L_cf  : {cf!r}")
        print(f"   scene : {meta['graspable_names']}\n")
    # count how many object-suite instructions get a HARD in-scene swap
    n_hard = 0
    for instr, meta in lut.items():
        if meta["suite"] == "libero_object":
            _, kind = make_counterfactual(instr, meta, lut, rng)
            n_hard += kind == "hard_object"
    print(f"object-suite hard in-scene counterfactuals available: {n_hard}/10")
