"""LANGUAGE -> ordered SUBGOAL SEQUENCE (general, no BDDL-goal / no oracle).

De-hardcodes two things at once (directive 2026-06-17: general/learned, no hardcoded/privileged):
  (1) the target/container comes from the LANGUAGE INSTRUCTION, not the sim's (:goal (On X Y)) predicate;
  (2) replaces the fixed DUAL-goal with N subgoals -> a sequencer can execute long-horizon (libero-10).

Output: list of dicts, each a subgoal:
    {"skill": grasp|place|open|turnon|close|push, "obj": <noun or None>, "target": <noun or None>,
     "relation": <relational clause that disambiguates obj, e.g. 'between the plate and the ramekin'> or None}

The noun strings ('alphabet soup','basket','plate','drawer','stove','moka pot',...) are resolved to scene
objects DOWNSTREAM by the open-vocab binder (DINOv2/SAM) — this module is pure language understanding,
general across LIBERO-PRO templates (NOT a per-task lookup).

Self-test:  python3 motor_distill/language_planner.py            (parses sample instructions, prints plans)
"""
from __future__ import annotations
import re

_PLACE_PREP = r"(?:on top of|on|in|into|onto|inside)"
_NOUN = r"(.+?)"


def _noun(s: str) -> str:
    """Normalize an object/target noun phrase for the binder (strip articles, layers, descriptors)."""
    s = s.strip().lower()
    s = re.sub(r"\b(the|a|an)\b", " ", s)
    s = re.sub(r"\b(top|bottom|middle|front|back|left|right)\s+(layer|drawer|region|of)?\b", " ", s)
    s = re.sub(r"\b(layer|region|box|of)\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    # cabinet/drawer are the same fixture in LIBERO
    s = s.replace("drawer", "cabinet") if "cabinet" not in s else s
    return s


def _pickplace(obj: str, target: str, relation: str | None = None) -> list[dict]:
    o = _noun(obj)
    return [{"skill": "grasp", "obj": o, "target": None, "relation": relation},
            {"skill": "place", "obj": o, "target": _noun(target), "relation": relation}]


def plan(instruction: str) -> list[dict]:
    """Parse a LIBERO-PRO instruction into an ordered subgoal list. General templated parse."""
    t = re.sub(r"\s+", " ", instruction.strip().lower())
    subs: list[dict] = []
    last_fixture: str | None = None

    # ---- "put both the A and the B <prep> the Z" (two distinct objects) ----
    m = re.match(rf"^put both (?:the )?{_NOUN} and (?:the )?{_NOUN} {_PLACE_PREP} the {_NOUN}$", t)
    if m:
        z = m.group(3)
        return _pickplace(m.group(1), z) + _pickplace(m.group(2), z)
    # ---- "put both <obj>s <prep> the Z" (two instances of same object) ----
    m = re.match(rf"^put both {_NOUN} {_PLACE_PREP} the {_NOUN}$", t)
    if m:
        o = _noun(m.group(1)); o = o[:-1] if o.endswith("s") else o; z = m.group(2)
        return [{"skill": "grasp", "obj": o, "target": None, "relation": "instance-0"},
                {"skill": "place", "obj": o, "target": _noun(z), "relation": "instance-0"},
                {"skill": "grasp", "obj": o, "target": None, "relation": "instance-1"},
                {"skill": "place", "obj": o, "target": _noun(z), "relation": "instance-1"}]

    # ---- split into top-level clauses on " and " that introduce a NEW verb ----
    VERBS = ("pick ", "put ", "place ", "open ", "turn on ", "turn ", "close ", "push ")
    parts = re.split(r"\s+and\s+", t)
    clauses: list[str] = []
    for p in parts:
        if clauses and not p.startswith(VERBS) and not p.startswith("place "):
            clauses[-1] = clauses[-1] + " and " + p   # 'and' was inside a noun/relation clause
        else:
            clauses.append(p)

    for c in clauses:
        c = c.strip()
        # pick the X [<relation>] and place it <prep> the Y   (object/spatial) — when 'place' kept in same clause
        m = re.match(rf"^(?:pick|pick up) (?:the )?{_NOUN} {_PLACE_PREP} the {_NOUN}$", c)
        # the above is ambiguous; handle the canonical "pick the X ... place it ..." that got split:
        m = re.match(rf"^(?:pick|pick up) (?:the )?{_NOUN}$", c)
        if m:
            # relation may be embedded: "akita black bowl between the plate and the ramekin"
            raw = m.group(1)
            rel = None
            rm = re.match(rf"^(.+?) (between .+|next to .+|from .+|in .+|on .+)$", raw)
            if rm: raw, rel = rm.group(1), rm.group(2)
            subs.append({"skill": "grasp", "obj": _noun(raw), "target": None, "relation": rel})
            subs.append({"skill": "place", "obj": _noun(raw), "target": None, "relation": rel, "_pending_target": True})
            continue
        m = re.match(rf"^place(?: it)? {_PLACE_PREP} the {_NOUN}$", c)
        if m:
            tgt = _noun(m.group(1)); last_fixture = tgt
            for s in reversed(subs):
                if s.get("_pending_target"): s["target"] = tgt; s.pop("_pending_target"); break
            continue
        # put the X <prep> the Y
        m = re.match(rf"^put (?:the )?{_NOUN} {_PLACE_PREP} the {_NOUN}$", c)
        if m:
            subs += _pickplace(m.group(1), m.group(2)); last_fixture = _noun(m.group(2)); continue
        # open the [..] X [and put ...]
        m = re.match(rf"^open (?:the )?{_NOUN}$", c)
        if m:
            tgt = _noun(m.group(1)); subs.append({"skill": "open", "obj": tgt, "target": None, "relation": None}); last_fixture = tgt; continue
        # turn on the X
        m = re.match(rf"^turn on (?:the )?{_NOUN}$", c)
        if m:
            tgt = _noun(m.group(1)); subs.append({"skill": "turnon", "obj": tgt, "target": None, "relation": None}); last_fixture = tgt; continue
        # push the X to the Y
        m = re.match(rf"^push (?:the )?{_NOUN} to (?:the )?{_NOUN}$", c)
        if m:
            subs.append({"skill": "push", "obj": _noun(m.group(1)), "target": _noun(m.group(2)), "relation": None}); continue
        # close it / close the X
        m = re.match(rf"^close(?: it| the {_NOUN})?$", c)
        if m:
            tgt = _noun(m.group(1)) if m.group(1) else last_fixture
            subs.append({"skill": "close", "obj": tgt, "target": None, "relation": None}); continue
        # "put the X inside/in it" (after an open) -> place into last fixture
        m = re.match(rf"^put (?:the )?{_NOUN} (?:inside|in it|on it)$", c)
        if m:
            o = _noun(m.group(1))
            subs.append({"skill": "grasp", "obj": o, "target": None, "relation": None})
            subs.append({"skill": "place", "obj": o, "target": last_fixture, "relation": None}); continue
        subs.append({"skill": "UNPARSED", "obj": c, "target": None, "relation": None})

    # resolve any place subgoals whose target is still pending (use last_fixture)
    for s in subs:
        if s.get("_pending_target"): s["target"] = last_fixture; s.pop("_pending_target")
    return subs


def parse_relation(rel: str):
    """Parse a relational clause -> (kind, [reference nouns]). General; no per-task lookup."""
    if not rel:
        return (None, [])
    r = rel.strip().lower()
    m = re.match(r"between (?:the )?(.+?) and (?:the )?(.+)$", r)
    if m: return ("between", [_noun(m.group(1)), _noun(m.group(2))])
    m = re.match(r"next to (?:the )?(.+)$", r)
    if m: return ("next_to", [_noun(m.group(1))])
    m = re.match(r"(?:on|in) (?:the )?(.+)$", r)
    if m: return ("near", [_noun(m.group(1))])
    m = re.match(r"from (.+)$", r)
    if m: return ("from", [m.group(1).strip()])   # e.g. 'table center'
    return (None, [])


def _xy(p):
    return (float(p[0]), float(p[1]))


def resolve_relation(relation: str, candidates: list, ref_pos: dict, table_center=(0.0, 0.0)):
    """Pick the candidate instance satisfying the relation, by GEOMETRY vs DETECTED reference positions.
    candidates: [(key, pos3)]; ref_pos: {noun: pos3} (detected, honest — NOT body_pos/oracle).
    Returns the chosen key (or candidates[0] key if unresolved). General geometric rule, no oracle."""
    if not candidates:
        return None
    kind, refs = parse_relation(relation)
    def d2(a, b):
        return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2
    if kind == "between" and len(refs) >= 2 and all(r in ref_pos for r in refs[:2]):
        a, b = _xy(ref_pos[refs[0]]), _xy(ref_pos[refs[1]])
        mid = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
        return min(candidates, key=lambda c: d2(_xy(c[1]), mid))[0]
    if kind in ("next_to", "near") and refs and refs[0] in ref_pos:
        r = _xy(ref_pos[refs[0]])
        return min(candidates, key=lambda c: d2(_xy(c[1]), r))[0]
    if kind == "from" and "center" in (refs[0] if refs else ""):
        return min(candidates, key=lambda c: d2(_xy(c[1]), tuple(table_center)))[0]
    return candidates[0][0]


if __name__ == "__main__":
    # relational resolver unit test (geometry only)
    print("=== resolve_relation unit test ===")
    cands = [("bowl_1", (0.0, 0.2, 0.97)), ("bowl_2", (-0.18, 0.32, 0.97))]
    refs = {"plate": (0.08, 0.04, 0.97), "ramekin": (-0.21, 0.19, 0.97), "cookies": (0.05, 0.19, 0.97)}
    for rel, exp in [("between the plate and the ramekin", None), ("next to the cookies box", None), ("from table center", None)]:
        print(f"  {rel:38s} -> {resolve_relation(rel, cands, refs)}")
    samples = [
        "Pick the alphabet soup and place it in the basket",
        "Pick the akita black bowl between the plate and the ramekin and place it on the plate",
        "Pick the akita black bowl next to the cookies box and place it on the plate",
        "Put the bowl on the plate",
        "Put the bowl on the stove",
        "Open the middle layer of the drawer",
        "Open the top layer of the drawer and put the bowl inside",
        "Turn on the stove",
        "Push the plate to the front of the stove",
        "turn on the stove and put the moka pot on it",
        "put the black bowl in the bottom drawer of the cabinet and close it",
        "put the yellow and white mug in the microwave and close it",
        "put both moka pots on the stove",
        "put both the alphabet soup and the cream cheese box in the basket",
    ]
    for s in samples:
        print(f"\n{s}")
        for g in plan(s):
            print(f"    {g['skill']:7s} obj={g['obj']!r:28s} target={g['target']!r:14s} rel={g.get('relation')!r}")
