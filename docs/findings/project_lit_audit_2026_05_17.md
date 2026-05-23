---
name: Lit audit — physics-grounded VLAs + test-time refinement (2026-05-17)
description: Three-stream literature audit. Verdict: don't rebuild VLA from scratch; physics-grounded MPPI on π0.5 evaluated on LIBERO-PRO is the defensible wedge. Phase 1 negative result becomes positioning argument vs ~8 learned-scorer competitors.
type: project
originSessionId: 53e17618-aeb4-4007-9d29-d56bfc1900c3
---
Three parallel research streams: physics-grounded VLA training, inference-time refinement landscape, generalization literature. All three converge.

**Don't rebuild the VLA from scratch.**
- No published result shows physics-loss training beats matched-compute imitation.
- Hyperscalers (PI, Skild, Figure, DeepMind, NVIDIA) own data + co-training + RL + diff-sim distillation.
- π*0.6 + RECAP (arxiv 2511.14759, Nov 2025) just moved the base-policy frontier. Chasing a moving target with no data moat is suicide.
- The biggest published deltas on LIBERO/SimplerEnv come from reasoning-augmented imitation (ECoT, ThinkAct, InternVLA-M1: +11-28pp), not physics losses.

**Test-time refinement on VLAs is now a crowded lane — but our specific slice is empty.**

Competing methods (2025-2026):
- VLAPS (2508.12211): MCTS + learned env model on Octo, +67pp on hardest LIBERO tasks
- VLA-Reasoner (ICRA 2026): MCTS over learned WM on OpenVLA/π0-FAST
- SITCOM (2510.04041): learned-WM rollouts + shaped reward on OpenVLA, +28pp SIMPLER, 1.67-7.6x slowdown
- CoVer (2602.12281): contrastive verifier on π0/π0.5, +45pp real-world avg, **2% overhead** (gold standard)
- ProgressVLA (2603.27670): classifier guidance via WM, +12.5pp long-horizon LIBERO
- GPC (2510.01068): policy-score composition, +7-10pp Robomimic
- TACO/LITEN (2510.19752): test-time scaling as anti-exploration

**What nobody has done:**
1. Physics-engine-grounded MPPI scoring on a flow/diffusion VLA (every competitor uses LEARNED WM/verifier).
2. Test-time deliberation evaluated on LIBERO-PRO (2510.03827, Oct 2025) or LIBERO-Plus (2510.13626, Oct 2025) — the benchmarks that show π0/π0.5 collapse from 95% to <30% under perturbation.

**Phase 1 negative result is the wedge.** We already showed learned-WM gradient refinement REDUCED real LIBERO success (52% → 30% at N=5). That's the empirical case for physics-grounded scoring. None of the 8 competitors above can match this evidence because they didn't measure it.

**Concrete paper:** "Physics-grounded MPPI refinement on π0.5/π0.6, first test-time deliberation method evaluated on LIBERO-PRO, with head-to-head ablation: physics scorer vs learned-WM scorer."

**Publication bar from literature:**
- ≥3 seeds x ≥2 task suites (single-seed +20pp = desk reject)
- Strong base (π0.5 fine-tuned, not weak OpenVLA)
- Speed honesty (our ~5x slowdown is in SITCOM's 1.67-7.6x band — fine if disclosed)
- Lift target: +5-10pp on LIBERO-PRO is CoVer/GPC/ProgressVLA range and publishable

**Why:** crowded lane → must differentiate on scorer (physics vs learned) and eval (LIBERO-PRO vs vanilla LIBERO). Our existing assets (Phase-1 negative result, MuJoCo autograd wrapper already shipped, MPPI integration already working) all point at this exact contribution.

**How to apply:**
- Stop saying "wrapper for openpi" externally.
- New YC/paper framing: "Inference-time deliberation for any open VLA; robustness-amplifier benchmark (LIBERO-PRO/Plus) as our category."
- Plan a head-to-head ablation slot: same MPPI loop, swap physics scorer for learned-WM (Q(h,a) from Sprint 1) — confirms which signal generalizes.
- Do NOT pivot to from-scratch foundation model. The data and compute math does not work.
- Phase B distillation (LoRA from physics-refined rollouts back into π0.5) is the upside narrative — it makes physics live in the weights without retraining from scratch.

**Open citations to track for the related-work section:**
- LIBERO-PRO 2510.03827
- LIBERO-Plus 2510.13626
- π*0.6 / RECAP 2511.14759
- VLAPS 2508.12211
- VLA-Reasoner project page vla-reasoner.github.io
- SITCOM 2510.04041
- CoVer 2602.12281
- ProgressVLA 2603.27670
- GPC 2510.01068
- Generative Predictive Control 2502.00622
- Unified Generation-Refinement Planning 2508.01192
- V-JEPA-2-AC 2506.09985
- DreamVLA arXiv 2507.04447 (NeurIPS 2025)
- Embodied CoT 2407.08693
- Helix / Helix-02 (Figure blog)
- Gemini Robotics 1.5 2510.03342

---

**Update 2026-05-17 — SANA-WM (NVLabs, May 2026): adjacent, not competitor.**

NVIDIA SANA-WM is a 2.6B-param camera-conditioned video world model, not a robotics WM. Architecture: hybrid linear diffusion transformer + Gated DeltaNet linear attention + softmax attention + dual-branch camera control. Generates 60s 720p video from one image + 6-DoF camera trajectory. 34s inference on single RTX 5090 (NVFP4 quantized). 15-day training on 64 H100s with 213K public clips. 36x throughput vs industrial baselines.

**Overlap with our work: essentially none.**
- No action conditioning (camera-pose only)
- No physics (no MuJoCo, no contact losses, no rigid-body constraints)
- No test-time refinement / MPC / search
- Inference = forward diffusion denoising, no policy on top

**Why this matters anyway:**
- Even NVLabs is shipping "world models" that are camera-conditioned pixel-video generators, not control substrates. Reinforces that "world model for VLAs" remains under-explored, consistent with our Phase 1 finding that learned-WM rollouts hurt task success.
- Their efficiency math (~$1-3M training cost equivalent, 2.6B params, distilled inference) is informative for our Path 2 capital model — 2-3B foundation models at the video scale are now Series-A-sized programs, not hyperscaler-only.

**How to apply:**
- Don't pivot toward SANA-WM as a learned WM substitute for physics — Phase 1 already ruled that out.
- Keep as a prior-art citation in the related-work section: "generative video world models (SANA-WM, Genie-2, Cosmos) target visual rollouts and lack action grounding."
- Use SANA-WM's compute math as a benchmark for Path 2 capital pitches.

Sources:
- arXiv 2605.15178
- https://nvlabs.github.io/Sana/WM/
- https://huggingface.co/papers/2605.15178
