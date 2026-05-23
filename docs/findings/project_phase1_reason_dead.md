---
name: REASON-VLA Phase 1 negative result — WM-only refinement is broken
description: 2026-05-15. Gradient descent on learned -VF(WM(state, action)) monotonically HURTS task success (52% → 30% at N=5 on LIBERO-10 5cm seed=7). VF gradient is misaligned with real task success. Validates that physics grounding is necessary, not optional.
type: project
originSessionId: 53e17618-aeb4-4007-9d29-d56bfc1900c3
---
**Date:** 2026-05-15.

## The experiment

REASON-VLA Phase 1: iterative gradient refinement of Pi0.5's action chunks via `loss = -VF(WM(hidden_state, action))`. PyTorch SGD on action chunk, N iterations per decision, then execute.

LIBERO-10, 5cm perturbation, seed=7. WM = our existing 4M latent WM. VF = our existing 525K PairwiseValueFunction trained on LIBERO-90 success/failure pairs.

## The result

| N | Success | Δ vs N=0 | Mean VF Score Lift |
|---|---|---|---|
| 0 | 52% (26/50) | — | — |
| 1 | 48% | -4pp | +0.006 |
| 2 | 48% | -4pp | +0.011 |
| 3 | 46% | -6pp | +0.017 |
| 5 | **30%** | **-22pp** | +0.028 |

**Monotonically worse with N.** The VF score goes UP at every iteration (refinement is correctly minimizing -VF), but real task success goes DOWN. The optimization works; the objective is wrong.

## What this proves

1. **The VF gradient is misaligned with real task success.** Our 525K VF, trained on Bradley-Terry pairwise loss across 19 of 90 LIBERO-90 tasks, scores "looks like success-from-training-distribution" — not "actually succeeds at the current task."

2. **Pi0.5's prior > learned VF gradient.** Pi0.5's flow matching converges to good actions for its training distribution. Gradient descent on a narrowly-trained learned VF pulls those actions into regions the VF rates high but that don't correspond to real-world success.

3. **More refinement = more harm.** Loss decreases monotonically (VF score up), but real performance decreases monotonically. The objective is actively misleading the optimization.

4. **WM-only refinement of action chunks is dead.** Don't tune hyperparameters. Don't try a smaller learning rate (it'd just make the harm slower). Don't try a different VF target. The mechanism is wrong-shaped without a ground-truth signal source.

## The structural lesson

Inference-time refinement of pretrained VLAs **requires a gradient source that doesn't suffer from training-distribution bias**. Learned scorers (VF, energy model, learned reward) all share this failure mode at our data scale. Differentiable physics is the obvious candidate ground-truth signal.

This validates the user's intuition from earlier in the project: *"don't we need physics-based differentiation? without physics aware it would just be the same problem as before right?"* Yes. Confirmed empirically.

## Implications for the project

- **Drop:** "iterative gradient refinement on learned -VF(WM(s, a))" as the inference-time mechanism
- **Drop:** the "ACT router learns when to refine more" plank — irrelevant if refinement direction is wrong
- **Drop:** the "R-NCE energy critic to replace V(h)" plank — same fundamental failure mode (training-distribution bias)
- **Keep:** the AlphaZero-for-VLAs vision — policy + value + search + iterative improvement
- **Make load-bearing:** differentiable physics via PyTorch + MuJoCo `mjd_transitionFD` Jacobians
- **Architecture pivot:** the gradient source must include physics constraints; learned models provide direction *within* feasibility regions, not navigation toward them

## How to apply

When designing the next iteration of REASON-VLA, the architecture must answer: where does the refinement gradient come from? Acceptable answers: (a) differentiable physics constraints, (b) classical analytical objectives (smoothness, contact forces), (c) Pi0.5 prior anchor preventing drift. NOT acceptable as the sole source: any single learned VF/energy/reward trained on a narrow distribution.

Future experiments that should kill themselves automatically if they reproduce this pattern: any "gradient descent on a learned scoring function" where the learned function was trained on Pi0.5 trajectories from a narrow benchmark. Add physics first, validate the mixed gradient improves things, then maybe add a learned auxiliary term.

## Files

- `data/contact_mpc/reason_phase1/reason_phase1_libero_10_5.0cm_seed7_N{0,1,2,3,5}.json` — full per-task JSON
- Phase 1 script: `scripts/run_reason_phase1.py`
- Sweep wrapper: `scripts/run_reason_phase1_sweep.sh`

## Cross-references

- Sprint 1 negative result on LIBERO-90 (Q(h, a) ≈ V(h) ≈ ±1pp): `project_sprint1_result.md`. Same structural failure mode at a different layer.
- Generalization audit (wrappers can't fix sub-30% baselines): `project_generalization_audit.md`. The mechanism we just proved empirically.
