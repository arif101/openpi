---
name: Inference-time generalization audit — pure wrappers don't fix LIBERO-90-class collapse
description: Literature-confirmed: no published wrapper has lifted a sub-30% base policy by ≥5pp on a held-out benchmark. Pivot the pitch to "robustness amplifier for competent baselines" (Cortex 2.0 model), not "generalization fixer."
type: project
originSessionId: 53e17618-aeb4-4007-9d29-d56bfc1900c3
---
**Date:** 2026-04-25.

## The brutal finding

Every clean wrapper-only lift in the 2026 VLA literature requires one of:
- Base policy already 50%+ (VLA-Reasoner: +5pp on a 76% baseline)
- Test-time RL that mutates the model (TT-VLA)
- Per-task PRO/VF training (Cortex 2.0: 60–70% → 95%+ requires 160–8,700 episodes per task)
- Hierarchy where the high-level module is materially smarter than the VLA

**No published wrapper has lifted a 23%-baseline VLA on a held-out benchmark by ≥5pp.** Our LIBERO-90 +1pp result matches the structural pattern, not a bug.

LIBERO-Plus authors (who looked hardest at this) recommend training-time data diversity. LIBERO-PRO authors propose the benchmark and *no fix*.

## Pitch implication — pivot from "generalization fixer" to "robustness amplifier"

Old (untenable): "Universal robustness wrapper that generalizes any VLA."
New (defensible): **"Cortex 2.0 for any open VLA. Turn 60–70% production policies into 90%+ reliability on the tasks customers actually run."**

Customer market: teams deploying open VLAs (Weave, Ultra, Servo7, Remy) who need reliability on a fixed task set, NOT generalization to unseen tasks.

## Three highest-EV interventions for sprint

**1. Evolutionary diffusion sampler (VLA-Pilot, arXiv 2511.14178).** K=32 candidates, 10 evolution steps, mutation by Pi0.5's flow denoiser. Strongest published OOD-wrapper result: MSR 0.50 OOD vs 0.19 FOREWARN. Replaces our K=4 stochastic sampling. ~2 days.

**2. Replace our 525K-param VF with Robometer-4B-LIBERO** (HuggingFace `aliangdw/Robometer-4B-LIBERO`). Trained on 1M trajectories across 24 sources. Drop-in scorer. Skips our 19-task memorization problem. ~1 day.

**3. Drop the 4M-param WM entirely.** Replace with foundation-VLM-direct scoring on either Pi0.5's executed first-step frames or FOREWARN-style VLM-on-latents. Our WM is two orders of magnitude under the smallest published WM that generalizes (VLA-Reasoner uses 600M iVideoGPT). ~1-2 days.

## What NOT to chase

- Scaling our WM to 50M+ (foundation video WMs are better, can't ship in 2 weeks)
- Multi-LoRA ensembles (breaks SDK shape, requires Pi0.5 retraining)
- MCTS depth/exploration tuning (candidate quality is the bottleneck, not search)
- RoboAlign-style RL fine-tuning (works at +17.5pp on real-world but breaks SDK shape)

## Key evidence quotes

- VLA-Reasoner (2509.22643): 600M iVideoGPT WM, +5pp on LIBERO with 76% baseline. No LIBERO-90 result.
- VLA-Pilot (2511.14178): evolutionary diffusion holds MSR 0.50 OOD when prior wrappers collapse to 0.19.
- LIBERO-PRO (2510.03827): Pi0.5 collapses 0.93 → 0.08 on 10-task suite under perturbation. No fix proposed.
- LIBERO-Plus (2510.13626): training-time diversity (20k trajectories) lifts OpenVLA-OFT 68.1 → 79.6 (+11.5pp). Authors test 20+ baselines, ZERO inference-time wrappers — field signal that wrappers don't fix this.
- Cortex 2.0 (2604.20246): Pi0.5 60–70% → 95–98% on industrial tasks BUT uses 160–8,700 task-specific episodes for fine-tuning the PRO/WM. Not zero-shot.

## How to apply

When framing the YC application, paper, or any external pitch: lead with "robustness amplifier for competent VLAs" (not "generalization fixer"). Use Cortex 2.0 as the comparable archetype (commercial validation exists). Honest scope: we close the OOD reliability gap on tasks the base VLA already half-knows; we don't rescue confused policies on truly novel tasks. Pi ecosystem customers need this exactly — they deploy on fixed task sets and need reliability, not generalization.

For technical work: the three sprint interventions (evolutionary diffusion, Robometer-4B drop-in, drop the trained WM) are independent, testable, well-evidenced. Each is 1-2 days. Combined, they replace the architecture's three weakest pieces with foundation-model-grade alternatives.
