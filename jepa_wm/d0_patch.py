"""D0 linguistic-blindness diagnostic: override the LIBERO task instruction per condition,
then run the STANDARD lerobot eval (faithful preprocessing + success detection unchanged).

Conditions (env var D0_PROMPT_MODE):
  correct : the real instruction (baseline).
  null    : empty instruction "" (does the policy act without any language?).
  cf      : a DIFFERENT goal's instruction from the SAME suite/scene (counterfactual).
            Success is still scored against the ORIGINAL task's goal — so high success under
            cf = the policy IGNORED the (wrong) instruction = linguistic blindness.

Language-Grounding-Score = SR(correct) - SR(cf). ~0 => blind; large => grounded.
"""
import os
import lerobot.envs.libero as L

_MODE = os.environ.get("D0_PROMPT_MODE", "correct")
_orig_init = L.LiberoEnv.__init__
_printed = {"n": 0}


def _patched_init(self, *args, **kwargs):
    _orig_init(self, *args, **kwargs)
    task_suite = args[0] if args else kwargs.get("task_suite")
    correct = self.task_description
    if _MODE == "null":
        self.task_description = ""
    elif _MODE == "cf":
        try:
            n = getattr(task_suite, "n_tasks", None) or len(task_suite.tasks)
            cf_id = (self.task_id + 1) % n
            self.task_description = task_suite.get_task(cf_id).language
        except Exception as e:  # noqa: BLE001
            print(f"[D0 PATCH] cf override failed: {e}")
    if _printed["n"] < 12:
        print(f"[D0 PATCH] mode={_MODE} task_id={self.task_id} "
              f"correct={correct!r} -> used={self.task_description!r}")
        _printed["n"] += 1


L.LiberoEnv.__init__ = _patched_init
print(f"[D0 PATCH] installed mode={_MODE}")
