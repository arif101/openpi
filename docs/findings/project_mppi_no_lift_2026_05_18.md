---
name: MPPI no overall lift on task 3 5cm (no-replay diagnostic, 2026-05-18)
description: Across N=40 paired-seed trials per mode, MPPI augmentation produced 0pp overall lift and 0pp improvement on the dominant failure mode (IN_TRUE_CLOSE_FALSE = drawer-close subpred fail). MPPI does not touch the late-trajectory failure mode.
type: project
originSessionId: 53e17618-aeb4-4007-9d29-d56bfc1900c3
---
## Result

LIBERO-10 task 3, 5cm perturbation, seeds {7, 21, 42, 99}, 10 trials/seed/mode.
BDDL atomic evaluation via direct state-jump to recorded final qpos (no replay,
no drift). Failures stratified by which atomic was False.

| Stratum | baseline | mppi | Δ failure-rate |
|---|---|---|---|
| `IN_TRUE_CLOSE_FALSE` (bowl in drawer, drawer open) | 7 (17.5%) | 7 (17.5%) | **0pp** |
| `IN_FALSE_CLOSE_FALSE` (bowl never reached drawer) | 5 (12.5%) | 6 (15.0%) | +2.5pp |
| `IN_FALSE_CLOSE_TRUE` (drawer closed, bowl elsewhere) | 1 (2.5%) | 0 (0%) | −2.5pp |
| **Overall success** | **27/40 (67.5%)** | **27/40 (67.5%)** | **0pp** |

## Why this matters

1. Contradicts the +20pp claim in `project_mppi_first_signal.md` —
   averaged across seeds {7, 21, 42, 99}, MPPI breaks even. The +20pp
   was seed-specific (best on seed=42, worst on seed=99 and seed=7).
2. The dominant failure mode (IN_TRUE_CLOSE_FALSE, drawer-close
   subpredicate) is **identical** between baseline and MPPI: 7/40 each.
   This isn't variance — it's MPPI failing to touch the failure mode that
   accounts for 50%+ of failures.

## What this implies about MPPI

MPPI is not generating or re-weighting the drawer-close action sequence:

- **Possibility 1**: Action chunk horizon too short to plan the close motion
  after the bowl is placed (~step 350 of 520).
- **Possibility 2**: Privileged cost doesn't reward drawer-close progress
  (cost is on bowl-to-goal position, not on drawer qpos delta).
- **Possibility 3**: Pi0.5 base policy itself doesn't sample any
  drawer-close trajectories from this state — MPPI re-weights samples it
  receives but can't generate fundamentally new behavior.

If (3) is true, MPPI on Pi0.5 has a structural ceiling: it cannot do
anything Pi0.5 doesn't already sample. The only fix is to make Pi0.5
sample drawer-close candidates, which requires either fine-tuning
(distillation) or a non-MPPI augmentation that generates novel actions
(e.g. a hand-scripted close primitive).

## How to apply

- **Stop claiming MPPI gives +20pp on task 3 5cm in writeups / pitches**
  until reproduced under controlled paired-seed conditions with N≥20/seed.
- Earlier "MPPI first signal" memory should be marked superseded.
- Sprint redirection candidates, in priority order:
  1. **Diagnose which of (1)/(2)/(3) is causing MPPI no-op on IN_TRUE_CLOSE_FALSE.**
     Inspect the MPPI candidate set on a failing rollout: do any candidates
     have positive drawer-close qpos delta? If yes, problem is cost (2).
     If no, problem is base-policy sampling diversity (3) or horizon (1).
  2. Build a **drawer-close detector + hand-scripted close primitive** as
     a comparison baseline. If hand-scripted closes succeed where MPPI
     doesn't, it confirms the failure mode is purely about action diversity.
  3. Skip MPPI variants for task 3 — pivot to a task where the dominant
     failure mode is closer to MPPI's sweet spot (e.g. trajectory
     refinement during transport rather than discrete close action).

## Files

- Source corpus: 80 task-3 traces in `data/contact_mpc/recovery_source_traces`
- v2 re-stratifier output: `data/contact_mpc/failure_strata_bddl_v2.json`
- Per-mode split script: `/tmp/split_by_mode.py` (not committed)
- v2 re-stratifier (state-jump): `scripts/restratify_failures_bddl.py` (commit 85829d0)
