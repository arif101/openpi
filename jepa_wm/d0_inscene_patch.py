"""D0 counterfactual baseline for libero_object: swap the instruction's object to a
DIFFERENT object that is present in the SAME scene (in-scene, hard counterfactual).

Success is scored against the ORIGINAL task's goal, so:
  high cf success  = policy ignored the swap, did the original object = BLIND (the gap).
  low  cf success  = policy followed the swapped instruction = grounded.
"""
import os
import re
import lerobot.envs.libero as L

_MODE = os.environ.get("D0_PROMPT_MODE", "correct")
_NONGRASP = {"basket", "plate", "stove", "cabinet", "rack", "microwave",
             "glazed_rim_porcelain_ramekin", "ramekin", "caddy", "wooden_cabinet",
             "wine_rack", "cookies", "cookie_box", "flat_stove"}
_orig = L.LiberoEnv.__init__
_pr = {"n": 0}


def _types_from_bddl(path):
    txt = open(path).read()
    m = re.search(r"\(:objects(.*?)\)", txt, re.S)
    types = []
    if m:
        for line in m.group(1).strip().splitlines():
            parts = line.strip().split(" - ")
            if len(parts) == 2:
                types.append(parts[1].strip())
    return types


def _patched(self, *a, **k):
    _orig(self, *a, **k)
    instr = self.task_description
    if _MODE == "null":
        self.task_description = ""
    elif _MODE == "cf":
        try:
            types = [t for t in _types_from_bddl(self._task_bddl_file) if t not in _NONGRASP]
            names = {t: t.replace("_", " ") for t in types}
            cur = [t for t, nm in names.items() if nm in instr]
            others = sorted(t for t, nm in names.items() if nm not in instr)
            if cur and others:
                tgt = others[self.task_id % len(others)]  # deterministic in-scene swap
                self.task_description = instr.replace(names[cur[0]], names[tgt])
        except Exception as e:  # noqa: BLE001
            print(f"[D0obj] cf fail: {e}")
    if _pr["n"] < 12:
        print(f"[D0obj] mode={_MODE} task={self.task_id} orig={instr!r} -> used={self.task_description!r}")
        _pr["n"] += 1


L.LiberoEnv.__init__ = _patched
print(f"[D0obj] installed mode={_MODE}")
