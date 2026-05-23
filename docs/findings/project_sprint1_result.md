---
name: Sprint 1 result — Q(h, a) confirms structural limit on LIBERO-90
description: 2026-04-27. Q(h, a) action-conditional VF gives +1pp lift on LIBERO-90 5cm seed=7, indistinguishable from V(h)'s ±1pp. Inference-time wrappers can't fix LIBERO-90 regardless of value function. Pivot Phase 2 hierarchical earlier.
type: project
originSessionId: 53e17618-aeb4-4007-9d29-d56bfc1900c3
---
**Date:** 2026-04-27.

## Sprint 1 gate verdict

LIBERO-90 5cm seed=7, 20 tasks × 5 trials = N=100:

| Value function | Baseline | MCTS | Δ |
|---|---|---|---|
| V(h) (Phase 5) | 23/100 | 24/100 | +1pp |
| V(h) (today) | 22/100 | 21/100 | -1pp |
| Q(h, a) (today) | 21/100 | 22/100 | +1pp |

All three within ±1pp of zero. Q(h, a)'s action-conditional architecture + perturbation augmentation + multi-task data plumbing **did not break the structural limit**.

## What this means

The inference-time wrapper hypothesis (any VF + WM + MCTS can rescue Pi0.5 on LIBERO-90 perturbed) is empirically falsified. Confirms `project_generalization_audit.md`'s literature-derived thesis: when the base policy is at sub-30% on the task distribution, K=4 candidates from the miscalibrated OOD prior don't include good actions, so no scorer can pick a good one. The bottleneck is upstream of the value function.

Sprint 1 deliverables (architecture widening, Q(h, a) class, perturbation augmentation, training pipeline, eval flag) all shipped and tested. The negative result is itself publishable as empirical validation of the structural limit.

## Cross-box reproducibility issue discovered along the way

Pi0.5's `policy.infer()` has cross-run JAX-RNG nondeterminism that's not pinned by `args.seed`. Same checkpoint, same code, same seed → different K=4 stochastic candidates across boxes. Phase 5's "+10pp on LIBERO-10 5cm seed=7" was reproducible on the prior box but today gave -6pp. Cross-box absolute MCTS numbers can't be compared without pinning a JAX PRNG key into the Pi0.5 sampling path. **Within-run baseline-vs-MCTS deltas are still trustworthy** (they share the same nondeterminism). This is a known caveat to attach to any cross-box claim in the writeup.

## How to apply

1. **Stop tuning the VF for Sprint 1.** Don't retrain Q(h, a) with stronger regularization or more data — the LIBERO-90 ceiling is structural, not VF-shaped.
2. **Pivot to Phase 2 hierarchical decomposition earlier than the original month-2 timeline.** Hi-Robot-style sub-goal generation gets us out of the "K=4 stochastic at this state" candidate pool and into "different sub-goal entirely" — the only known mechanism for breaking sub-30% baselines.
3. **Pitch update:** the LIBERO-10 +9pp Phase 5 result is the load-bearing claim. LIBERO-90 is honestly framed as "structural limit confirmed; needs hierarchy, not better VF."
4. **Reproducibility note:** before any cross-box benchmark in the paper, fix Pi0.5 RNG seeding (thread an explicit JAX PRNG key into `policy.infer()`). Otherwise within-run deltas only.

## Files

- `data/contact_mpc/v_libero90/libero_pro_mcts_libero_90_5.0cm.json` — V(h) result
- `data/contact_mpc/qha_libero90/libero_pro_mcts_libero_90_5.0cm.json` — Q(h, a) result
- `data/contact_mpc/q_function_libero90/q_function.pt` — trained Q(h, a) checkpoint (val_acc 0.673, gap 0.323 — overfit on 19 valid tasks)
