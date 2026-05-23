---
name: BDDL re-stratification result (2026-05-18, task 3)
description: Actual atomic-predicate failure distribution across N=26 task-3 PHYS_FAIL traces, with comparison to pre-run predictions. Identifies the legitimate drawer-close-only failure stratum (8 traces).
type: project
originSessionId: 53e17618-aeb4-4007-9d29-d56bfc1900c3
---
## Distribution (N=26 LIBERO-10 task-3 5cm-perturbation failures)

| Label | Count | % | Meaning |
|---|---|---|---|
| `IN_FALSE_CLOSE_FALSE` | 11 | 42% | bowl never reached the drawer AND drawer never closed |
| `IN_TRUE_CLOSE_FALSE`  |  8 | 30% | **bowl IS inside drawer, drawer NOT closed** — legitimate subpred failure |
| `IN_FALSE_CLOSE_TRUE`  |  6 | 23% | drawer closed but bowl somewhere else (often near init pose) |
| `IN_TRUE_CLOSE_TRUE`   |  1 |  3% | corpus says PHYS_FAIL, BDDL re-eval says success → **replay-diverged** |

## Comparison to predictions in `project_in_predicate_failure_modes.md`

- `In=False ∧ Close=True`: predicted 5–8, actual 6 ✓
- `In=False ∧ Close=False`: predicted 10–15, actual 11 ✓
- `In=True ∧ Close=False`: **predicted 2–5, actual 8 — above range**
- `WRONG_DRAWER`: predicted 0–2, actual 0 ✓
- `IN_TRUE_CLOSE_TRUE`: not predicted as a category. New.

**Why:** The "drawer-close-only" failure mode is much more common than the
predictions assumed. This is the actual signal we care about for recovery
diagnostics — the policy demonstrates competence (gets the bowl in the drawer)
but doesn't complete the final close step. 8 traces is a real N.

**How to apply:**
- The original Euclidean stratifier mislabeled most of these. real-P3 v2's
  0/5 recovery test was likely run on a mix of IN_FALSE and IN_TRUE traces;
  the result tells us nothing about recovery on the legitimate IN_TRUE_CLOSE_FALSE
  stratum.
- The 8 IN_TRUE_CLOSE_FALSE traces (binomial CI ≈ ±34pp at N=8) is the clean
  denominator for the recovery question "can a planner close the drawer from
  a bowl-already-placed state?" Worth running real-P3 v2 on, with the caveat
  that the CI is still wide.
- The 1 IN_TRUE_CLOSE_TRUE trace is a **must-investigate** before any
  downstream recovery claim. If on-line BDDL eval and offline replay BDDL eval
  disagree on a single trace, that questions every PHYS_FAIL/PHYS_OK label
  in the corpus. Could be JAX-RNG sampling nondeterminism (sprint-1 finding)
  or a one-frame timing mismatch in `_check_success`.

## Files

- Output: `data/contact_mpc/failure_strata_bddl.json` on GPU box
- Run log: `logs/restratify_bddl_task3.log`
- Script: `scripts/restratify_failures_bddl.py` (commit 9ff77e4)
