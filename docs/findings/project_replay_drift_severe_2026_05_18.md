---
name: Replay-drift confound NOT ruled out — severe across task-3 corpus (2026-05-18)
description: init_sim_state + replay drifts up to 40cm from recorded trajectory on contact-rich rollouts. Compromises every state-restore diagnostic in the project.
type: project
originSessionId: 53e17618-aeb4-4007-9d29-d56bfc1900c3
---
## Evidence

Took 26 task-3 PHYS_FAIL traces. Replayed each from its saved `init_sim_state`
through the recorded action sequence. Compared replayed final bowl position to
recorded final bowl position from the npz.

| Statistic | Value |
|---|---|
| N | 26 |
| Mean drift | 9.62 cm |
| Median drift | 4.68 cm |
| Max drift | **39.53 cm** |
| drift > 2 cm | 17 / 26 (65%) |
| drift > 5 cm | 12 / 26 (46%) |
| drift > 10 cm | 7 / 26 (27%) |
| drift = 0 cm | 5 / 26 (these are GRIP_LOSS-style: bowl barely moved, no contact) |

The "1/26 replay-diverged" trace I initially treated as a one-off (where corpus
PHYS_FAIL ≠ offline BDDL eval) was **the median, not an outlier**. Bigger drifts
exist but happened to stay on the same side of the success boundary.

## Why

`sim.get_state().flatten()` saves qpos + qvel + time. It does NOT save:
- OSC controller internal state (action filter, last-target-pose, integrator)
- Contact warm-start info
- Actuator state

The env wrapper's `env.reset() + set_state_from_flattened()` restores physics
state but the controller starts fresh. The 10 no-op wait-steps at the start of
replay run through the controller, which converges to a slightly different
joint trajectory than the original — and this divergence compounds through
~500 action steps of contact-rich manipulation.

## What this invalidates

| Diagnostic | Status |
|---|---|
| Phase A +6.7pp / +20pp result | **Probably clean** — no state restore mid-trial. |
| real-P3 v2 0/5 recovery test | **Compromised.** Restored mid-state ≠ original mid-state. |
| dead_state_check verdicts (at any restore_frac) | **Compromised.** Restore point isn't where we think. |
| Controlled-diagnostic / failure-injection (task #53) | **Compromised** if it used state restore. |
| Sprint-1 cross-box JAX-RNG finding | Orthogonal — not affected by this. |
| BDDL re-stratifier label distribution (8 IN_TRUE_CLOSE_FALSE) | **Partially compromised.** The labels reflect post-replay state, not original-rollout state. The "real failure mode" of each trace as originally rolled out may differ. |

## Why task #51 was marked completed prematurely

Task #51 ("Rule out replay-drift confound") was closed after adding
`init_sim_state` save (task #52). The save was necessary but not sufficient:
the saved state doesn't include controller/integrator state, so determinism
breaks even when the recorded `init_sim_state` is restored exactly.

## How to apply

- Stop trusting any diagnostic that relies on init_sim_state + replay alone.
- Before running real-P3 v2 again, decide which fix to use:
  1. **Capture controller state too** (save OSC integrator/filter/last-pose
     state alongside sim state; restore on replay).
  2. **Sub-sample replay** — restore at very early restore_frac (e.g. 0.05–0.10)
     where controller state hasn't diverged yet, and accept that this changes
     the experimental question to "given an early-trajectory state, can a
     planner recover?" Lose the "mid-failure state" framing.
  3. **Don't restore — fresh rollouts.** Run baseline+recovery side-by-side from
     time 0 with paired seeds and perturbations. Lose state-restore comparison
     but gain determinism.
- Re-open task #51 as not actually complete.
- The IN_TRUE_CLOSE_FALSE label distribution remains useful as a *category
  diagnostic* — the BDDL atomic eval semantics are correct, even if the
  per-trace label may not match the original rollout. Use distributions, not
  trace identities.

## Files

- Drift analysis script (one-shot): `/tmp/replay_drift.py` (not committed)
- Source: `data/contact_mpc/failure_strata_bddl.json` (26 traces re-stratified)
- Run log: `logs/restratify_bddl_task3.log`
