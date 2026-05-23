---
name: Project roadmap — Sprint 1 pivoted to Q(h, a) VF
description: Sprint 1 changed 2026-04-25 mid-session. Original (Robometer + evo diffusion) had two design errors. New Sprint 1 = build Q(h, a) action-conditional VF with multi-suite + perturbation aug, then decide whether evo diffusion adds anything.
type: project
originSessionId: 53e17618-aeb4-4007-9d29-d56bfc1900c3
---
**Date:** 2026-04-25.

The full root-level `ROADMAP.md` still describes the original Sprint 1 (evo diffusion + Robometer drop-in + perturbation curve). That plan was pivoted mid-session because two of its premises broke under closer inspection.

## Why Sprint 1 was pivoted

**1. Robometer is a video model, not a latent-state scorer.** It expects (N frames + task) → per-frame outputs and architecturally cannot replace our `ValueFn(hidden_state)`. The "1-day drop-in" estimate from `project_generalization_audit.md` was based on a misreading of the abstract.

**2. Sonnet-as-PRM doesn't reason about raw joint deltas.** Claude has strong vision + language reasoning but a 7×5 matrix of joint values is opaque without semantic conversion (forward-kinematics → end-effector deltas → natural-language description). VLA-Reasoner does this conversion; their abstract doesn't say so.

**3. Evolutionary diffusion amplifies whatever signal the PRM gives.** Run it on a memorizing VF → amplifies memorization → likely *worse* on LIBERO-90, not better. The original plan assumed a working PRM that we don't have.

## New Sprint 1 (current)

Build a better value function first, then re-evaluate everything else.

| Step | Work | Status |
|---|---|---|
| 1 | Widen `ValueFn` protocol; add `frame=`, `task=` to `MCTSPlanner.plan()` | **Done** — 12/12 MCTS tests pass |
| 2 | New `ActionConditionalValueFunction` Q(h, a) — action encoder + scorer with dropout | **Done** — local smoke test converges |
| 3 | Multi-suite data loader + feature-space perturbation augmentation | **Done** — `build_within_task_pairs_qha`, `feature_perturbation_augment` |
| 4 | `train_q_function` with AdamW + weight decay + per-batch perturbation aug | **Done** — synthetic smoke test |
| 5 | `scripts/run_train_q_function.py` orchestration | **Done** |
| 6 | `--value-function-type {v,qha}` flag + `TorchQFnAdapter` in eval harness | **Done** |
| 7 | GPU: train Q(h, a) on libero_90 features | **Pending** (needs GPU) |
| 8 | GPU: eval Q(h, a) + MCTS on LIBERO-10 5cm seed=7 | **Pending** |
| 9 | GPU: eval Q(h, a) + MCTS on LIBERO-90 5cm seed=7 (decision gate) | **Pending** |

## Decision gate after Step 9

| Q(h, a) lift on LIBERO-90 5cm | Verdict |
|---|---|
| ≥ +5pp | VF was the bottleneck. Pitch becomes "task-agnostic Q-function for any open VLA." Run evo diffusion (Tasks 12, 13) for additional lift. |
| +2 to +5pp | Modest improvement. Skip evo diffusion. Run perturbation curve + seed=21 for clean writeup. |
| < +2pp | Inference-time fix is structurally bounded. Confirms `project_generalization_audit`. Pivot Phase 2 (hierarchical decomposition) earlier. Negative result is still publishable. |

## Deferred (originally Sprint 1)

- seed=21 + perturbation curve (3/7/10cm) — pure CLI sweep, run after VF decision gate.
- Evolutionary diffusion sampler — only if Q(h, a) ≥+5pp on LIBERO-90.
- Multi-suite data collection (libero_10, spatial, object, goal) — only if libero_90-only training shows partial lift (+2–5pp).
- Robometer-4B integration — abandoned for Sprint 1; revisit only if we build a video-buffer + frame-rendering pipeline.
- Sonnet-PRM — abandoned for Sprint 1; revisit only with action-to-text converter (~1 day).

## How to apply

When future sessions ask "what's the next experiment to run?" the answer is **train Q(h, a) on GPU and run the decision gate**. Do NOT propose Robometer drop-in or evo diffusion as Sprint 1 work — both are deferred behind the VF decision. The root `ROADMAP.md` is stale on this point and needs an in-place update next time we touch it.
