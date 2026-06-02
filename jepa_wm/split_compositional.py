"""Leave-one-referent-out compositional split (Design 1 of COMPOSITION_SPLIT_DESIGN.md).

For each held-out target object in libero_object, define:
  STORE episodes = all object-suite episodes whose target != held-out object
  QUERY episodes = episodes whose target == held-out object
The held-out (skill x object) combination is absent from the store, but the SKILL
(pick->basket) is present (9 other objects) and the OBJECT is present in scenes/pretraining.

Parsing logic is self-testable offline (no dataset). Episode mapping needs the dataset (box).
"""
from __future__ import annotations

import re

# libero_object instructions (task_index 20-29), from dataset inspection 2026-05-31
OBJECT_INSTRUCTIONS = [
    "pick up the orange juice and place it in the basket",
    "pick up the ketchup and place it in the basket",
    "pick up the cream cheese and place it in the basket",
    "pick up the bbq sauce and place it in the basket",
    "pick up the alphabet soup and place it in the basket",
    "pick up the milk and place it in the basket",
    "pick up the salad dressing and place it in the basket",
    "pick up the butter and place it in the basket",
    "pick up the tomato sauce and place it in the basket",
    "pick up the chocolate pudding and place it in the basket",
]

_OBJ_RE = re.compile(r"pick up the (.+?) and place it in the basket")


def target_object(instruction: str) -> str | None:
    m = _OBJ_RE.search(instruction)
    return m.group(1) if m else None


def leave_one_out_splits(instructions=None):
    """Return list of dicts: {held_out_object, held_out_instruction, store_instructions}."""
    instructions = instructions or OBJECT_INSTRUCTIONS
    objs = [(target_object(i), i) for i in instructions]
    assert all(o for o, _ in objs), "failed to parse a target object"
    splits = []
    for obj, instr in objs:
        store = [i for o, i in objs if o != obj]
        splits.append({"held_out_object": obj, "held_out_instruction": instr, "store_instructions": store})
    return splits


def build_episode_split(held_out_instruction, ds_meta, suite="libero_object"):
    """Map a leave-one-out split to STORE vs QUERY episode indices on the loaded dataset.
    ds_meta = LeRobotDataset(...).meta. Returns (store_eps, query_eps, frame_ranges)."""
    eps = ds_meta.episodes
    n = len(eps)
    ep_task = eps["tasks"]
    frm, to = eps["dataset_from_index"], eps["dataset_to_index"]
    store, query, ranges = [], [], {}
    for e in range(n):
        t = ep_task[e][0] if isinstance(ep_task[e], list) else ep_task[e]
        if target_object(t) is None:          # not an object-suite task
            continue
        ranges[e] = (int(frm[e]), int(to[e]))
        (query if t == held_out_instruction else store).append(e)
    return store, query, ranges


if __name__ == "__main__":
    # offline self-test of the parsing + split logic (no dataset)
    splits = leave_one_out_splits()
    print(f"built {len(splits)} leave-one-out splits")
    objs = [s["held_out_object"] for s in splits]
    print("held-out objects:", objs)
    assert len(set(objs)) == 10, "expected 10 distinct held-out objects"
    s0 = splits[0]
    assert len(s0["store_instructions"]) == 9, "store should hold 9 instructions"
    assert s0["held_out_instruction"] not in s0["store_instructions"], "held-out leaked into store"
    print(f"\nexample split: held_out={s0['held_out_object']!r}")
    print(f"  store has {len(s0['store_instructions'])} other-object skills")
    print("  parse check:", all(target_object(i) for i in OBJECT_INSTRUCTIONS))
    print("SELF-TEST OK")
