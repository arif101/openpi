---
name: Phase roadmap for the VLA continuous-improvement company
description: Three-phase scope — method validation (now), memorization (follow-up), reasoning (follow-up). What each phase claims and does not claim.
type: project
originSessionId: 53e17618-aeb4-4007-9d29-d56bfc1900c3
---
Three-phase scope, locked 2026-04-21:

**Phase 1 — Method validation (current MVP, targeting YC application).**
- Claim: "Given a deployed VLA's failure logs, we attribute root causes, propose cluster-targeted LoRA patches, rank them via offline latent-world-model evaluation, and validate imagined-vs-real correlation — all without additional real-robot rollouts."
- Pipeline: attribution (Claude Sonnet 4.6) → cluster → LoRA training (DPO preferred, SFT-on-matched-successes acceptable fallback) → offline evaluator → real LIBERO eval → Pearson r correlation.
- Does NOT claim: solving the LIBERO-PRO memorization gap, adding reasoning to the VLA, or long-horizon planning.

**Phase 2 — Memorization (follow-up, 2–4 weeks).**
- Domain randomization during training (perturb scene state, break absolute-position shortcuts).
- Counterfactual data augmentation via video diffusion / physics sim (the "reverse VOID" thread).
- Targets LIBERO-PRO 46% → 70%+ on position perturbations.

**Phase 3 — Reasoning (follow-up, 2–3 months).**
- Architectural changes to Pi0.5: ECoT-style text reasoning tokens or LaRA-VLA latent reasoning tokens before action tokens.
- GRPO-style online RL with reasoning rewards (post-R1 recipe).
- Needs reasoning-annotated demos; much larger data effort than Phase 1.

**Why:** User explicitly asked to scope Phase 1 narrowly so the MVP validates on its own merits. Research-paper-grade contributions (memorization, reasoning) are deferred to Phases 2 and 3, each treated as a separate commitment.

**How to apply:** When planning work on this repo, default to Phase 1 scope. Do not let Phase 2/3 concerns derail MVP shipping decisions. If the user brings up memorization or reasoning, acknowledge they are on the roadmap but distinct from current scope.
