"""Synthetic training data for the SUB-GOAL EMITTER (design module A).

The emitter maps a free-form instruction -> a (noun-phrase, gripper-event) ladder that the OWLv2 binder
then grounds. Its job is STRUCTURAL (extract the object/container noun spans and emit them with the right
gripper polarity + ordering), NOT to memorize bindings -- so it must generalize to UNSEEN objects and
containers by COPYING spans from the input. We therefore train on templates x broad vocabulary and HOLD
OUT a disjoint set of objects+containers for validation: success on held-out nouns == "container is just
another open-vocab noun" by construction (kills the hardcoded basket).

Output ladder token format:
  pick+place : <sub> {obj} <grip> close <sub> {cont} <grip> open <eos>
  pick-only  : <sub> {obj} <grip> close <eos>

Writes data/emitter/{train,val}.jsonl  (each line: {"instr":..., "ladder":...})
No GPU, no demos -- pure structure.  Usage: python motor_distill/gen_emitter_data.py
"""
from __future__ import annotations
import argparse, itertools, json, pathlib, random

# Broad object vocab (LIBERO object-suite + common manipulables for span-copy generalization).
OBJECTS = [
    "alphabet soup", "bbq sauce", "butter", "chocolate pudding", "cream cheese", "ketchup", "milk",
    "orange juice", "salad dressing", "tomato sauce",                      # LIBERO object suite (seen)
    "red block", "blue cup", "green bottle", "soup can", "cereal box", "banana", "apple", "lemon",
    "wine bottle", "coffee mug", "sponge", "screwdriver", "remote", "toy car", "spoon", "fork",
    "mustard bottle", "soda can", "tea box", "salt shaker", "pepper grinder", "egg carton",
]
CONTAINERS = [
    "basket",                                                              # LIBERO seen container
    "bowl", "plate", "drawer", "box", "bin", "tray", "pot", "pan", "crate", "cup", "mug", "caddy", "dish",
]
PICK_PLACE_TPL = [
    "pick up the {o} and place it in the {c}", "put the {o} in the {c}", "place the {o} into the {c}",
    "move the {o} to the {c}", "grab the {o} and drop it in the {c}", "set the {o} down in the {c}",
    "take the {o} and put it in the {c}", "drop the {o} into the {c}", "load the {o} into the {c}",
    "put the {o} on the {c}", "place the {o} on top of the {c}",
]
PICK_TPL = ["pick up the {o}", "grab the {o}", "lift the {o}", "take the {o}", "pick the {o} up"]


def ladder_pick(o):       return f"<sub> {o} <grip> close <eos>"
def ladder_place(o, c):   return f"<sub> {o} <grip> close <sub> {c} <grip> open <eos>"


def gen(objs, conts, seed):
    rng = random.Random(seed)
    rows = []
    for o, t in itertools.product(objs, PICK_TPL):
        rows.append({"instr": t.format(o=o), "ladder": ladder_pick(o)})
    for o, c in itertools.product(objs, conts):
        for t in rng.sample(PICK_PLACE_TPL, k=min(5, len(PICK_PLACE_TPL))):  # subsample templates per pair
            rows.append({"instr": t.format(o=o, c=c), "ladder": ladder_place(o, c)})
    rng.shuffle(rows)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/emitter")
    ap.add_argument("--n-holdout-obj", type=int, default=8)
    ap.add_argument("--n-holdout-cont", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    ho_obj = set(rng.sample(OBJECTS, args.n_holdout_obj))
    ho_cont = set(rng.sample(CONTAINERS, args.n_holdout_cont))
    tr_obj = [o for o in OBJECTS if o not in ho_obj]
    tr_cont = [c for c in CONTAINERS if c not in ho_cont]

    train = gen(tr_obj, tr_cont, args.seed)
    # val: held-out objects with seen containers, held-out containers with seen objects, and both-held-out
    val = (gen(list(ho_obj), tr_cont, args.seed + 1)
           + gen(tr_obj, list(ho_cont), args.seed + 2)
           + gen(list(ho_obj), list(ho_cont), args.seed + 3))
    out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train), ("val", val)):
        with open(out / f"{name}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    json.dump({"holdout_objects": sorted(ho_obj), "holdout_containers": sorted(ho_cont)},
              open(out / "split.json", "w"), indent=2)
    print(f"train={len(train)}  val={len(val)} (held-out objs={len(ho_obj)} conts={len(ho_cont)})")
    print(f"held-out objects   : {sorted(ho_obj)}")
    print(f"held-out containers: {sorted(ho_cont)}")
    print(f"-> {out}/  e.g.  {train[0]['instr']!r} => {train[0]['ladder']!r}")


if __name__ == "__main__":
    main()
