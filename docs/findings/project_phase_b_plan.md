---
name: Phase B distillation plan (2026-05-17)
description: Concrete LoRA fine-tune plan that turns Pi0.5 into a physics-aware policy using the refined-rollout corpus accumulated during Phase A. Bridges MPPI from inference-time wrapper to training-time capability fixer.
type: project
originSessionId: 53e17618-aeb4-4007-9d29-d56bfc1900c3
---
**Why Phase B is now load-bearing.** Phase A results show MPPI is a robustness amplifier only — at 10cm perturbation on multi-object tasks, failures shift to deep policy disorientation that no inference-time search can fix. The capability gain at higher perturbations requires the policy itself to learn the physics-grounded correction patterns. That's what Phase B does.

**The data.** Every successful MPPI episode logged via `--log-refined-rollouts` produces a per-episode NPZ with arrays:
- `image`: [N, 224, 224, 3] uint8 (agent view at each MPPI decision)
- `wrist_image`: [N, 224, 224, 3] uint8
- `state`: [N, ?] joint+EE+gripper state
- `prompt`: [N] task language strings
- `prior_action`: [N, H, 7] Pi0.5's original chunk
- `refined_action`: [N, H, 7] MPPI-refined chunk (= the training target)
- per-decision cost diagnostics

The supervised objective: train Pi0.5 (frozen base + LoRA adapters) to emit `refined_action` given the same obs. This teaches the policy to anticipate the physics-grounded correction without needing the search loop at inference time.

**Architecture choice — LoRA, not full fine-tune.**
- openpi has LoRA support at src/openpi/models/lora.py
- Apply LoRA to the action decoder + last few VLM layers. Backbone (vision encoder) frozen.
- Why LoRA: limited data (~hundreds of episodes × ~50 decisions = ~5K-20K samples), full fine-tune overfits and breaks Pi0.5's vanilla capability. LoRA preserves the base, adds physics-aware deltas.
- Rank: 16-32 to start. ~10-30M trainable params on the 3B base.

**Training script path.** Build scripts/run_phase_b_distill.py that:
1. Globs the refined-rollout NPZs, builds a torch Dataset returning (obs_dict, target_action_chunk).
2. Loads Pi0.5 base + applies LoRA via openpi's existing infra.
3. Loss: flow-matching loss matching Pi0.5's original training objective, but with `refined_action` as the target instead of demo data.
4. Standard AdamW, cosine schedule, ~5-20 epochs, eval periodically by running MPPI loop and reporting nominal_cost trajectories.
5. Saves LoRA-only checkpoint (~100MB vs 12GB for the full base).

**Acceptance gate for Phase B.** Two metrics, before and after fine-tuning:
1. **Baseline (no MPPI) success rate on LIBERO-PRO 5cm seed=7 task 3 must improve by ≥+5pp.** This is the "model itself got better" gate.
2. **With MPPI on top, success rate doesn't regress below original-Pi0.5-baseline + Phase A's +20pp.** Phase B should not break the inference-time wrapper.

If gate 1 passes: capability gain validated. Distillation works.
If gate 2 fails: LoRA harmed the model's generality. Reduce LoRA rank or training epochs.

**Eval matrix for the paper after Phase B trains:**
- Pi0.5 baseline (no MPPI, no LoRA) — original numbers
- Pi0.5 + MPPI (no LoRA) — Phase A numbers, already collected
- Pi0.5 + LoRA (no MPPI) — Phase B alone
- Pi0.5 + LoRA + MPPI — best combo
On LIBERO-10 task 3, 5, 7 × 5cm × 3 seeds × N=10. Plus task 7 at 10cm to show LoRA recovers what MPPI alone couldn't.

**Capital math.** LoRA fine-tune on ~20K samples × 5 epochs on the 3B base ≈ 12-24 GPU hours on an A100/H100. ~$50-200 of compute. Negligible vs the Phase A data-generation cost we've already paid.

**Open questions to resolve before training:**
1. Does openpi's existing LoRA wiring expose the action decoder layers we want, or do we need to plumb new adapters?
2. Pi0.5 was trained with flow matching — do refined_action chunks need to be reformatted (e.g. normalized, action-chunked at H=50)?
3. Should we filter the refined-rollout corpus by `refined_cost < threshold` to avoid distilling marginal rescues?

**How to apply:**
- Don't start Phase B until Phase A matrix is in (the data sample size is the input). Current state: have data from 2 trials × 2 seeds on task 3 plus task 7 noise. Need scripts/run_phase_a_matrix.sh to complete.
- When data is ready, write run_phase_b_distill.py against openpi's LoRA infra. Iterate locally first on a small subset.
- The acceptance gates above are what distinguishes a real result from "we trained something."
