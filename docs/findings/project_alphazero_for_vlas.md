---
name: AlphaZero for VLAs roadmap (2026-05-17)
description: Reframed vision: physics-grounded self-improvement system for VLAs. Wrapper is Phase A; end-state (Phase D) is our own physics-grounded foundation model trained on self-play data from earlier phases.
type: project
originSessionId: 53e17618-aeb4-4007-9d29-d56bfc1900c3
---
User pushed back on "wrapper for openpi" framing as too small. The actual vision is the AlphaZero recipe for VLAs:

| Phase | Contribution | Why it has to come first |
|---|---|---|
| A (current) | Physics-grounded MPPI search at inference. Better action chunks than Pi0.5 prior alone. | The search has to work before downstream phases can use it. |
| B | Distill physics-refined rollouts back into VLA via LoRA. Fast inference; model becomes physics-aware. | Phase A is too slow for production. Distillation collapses search cost into model weights. |
| C | Differentiable physics losses baked into VLA training (autograd MuJoCo wrapper already shipped src/openpi/contact_mpc/refinement/mujoco_autograd.py). | Physics grounding lives in weights, not just inference. Smaller model becomes capable. |
| D (12-18 months) | Pretrain a VLA from scratch with physics-grounded losses + self-improvement loop. A/B/C are the data and reward pipeline. | This is the "our own foundation model" outcome — possible only because A/B/C exist. |

**Why:** "Wrapper" framing was risk-mitigated and undersold the work. The MPPI inference-time refinement is *the data engine* for an eventually-physics-grounded VLA, not an end product. AlphaZero analogy: the MCTS search didn't matter as a product; it mattered because it produced training data the policy network couldn't generate alone.

**Capital model:**
- Pretraining a 3B VLA from scratch costs $5-10M (PI raised $400M, Figure/Skild similar).
- Phase A → B → C is staged: each phase financially smaller than D, each produces a publishable / pitchable artifact, each builds the engine for the next.

**How to apply:**
- YC pitch language: "physics-grounded self-improvement system for VLAs," not "robustness wrapper."
- Paper framing: same code path, but the long-term story is the foundation model, with Phase A as the proof point.
- Do NOT skip ahead to D. We need A → B working before C/D are even sensible.
- The immediate gate (seed=21 on task 3 ≥+10pp) is what makes Phase A real. Without that, the whole stack is theoretical.
