---
name: Path to our own physics-grounded foundation model (2026-05-17)
description: Three end-states (Pi0.5+LoRA / from-scratch pretrain / diff-physics-native pretrain). Recommendation: Path 1 then optional Path 2 if traction. Capability fixer emerges from distilling MPPI-refined rollouts back into the policy.
type: project
originSessionId: 53e17618-aeb4-4007-9d29-d56bfc1900c3
---
User confirmed "eventually own foundation model" as a goal. Not now, but it's the end-state. This file pins how to get there without throwing away the Pi0.5 gift.

**Capability fixer vs robustness amplifier — the bridge.**
MPPI today is robustness-only because it samples around the policy prior; if the prior is hopeless (out of object workspace), perturbations stay hopeless. Capability gain comes from distillation: the MPPI-refined rollouts in successful trials ARE doing things the base policy couldn't, so LoRA-fine-tuning on those rollouts moves the prior into the rescued region. Then MPPI rescues a *new* failure mode further out. Repeat. AlphaZero ratchet.

Same physics signal feeds two loops:
- Inference loop (MPPI on top of policy) → robustness today
- Training loop (distill refined rollouts into policy) → capability tomorrow

**Three forms of physics-at-training, ranked by ambition:**
(a) Distillation — LoRA fine-tune on MPPI-refined rollouts. Physics enters via data. $5-20K, low risk.
(b) Auxiliary physics losses — prediction heads for next-step contact/collision/pose, penalize unphysical predictions. $30-100K, medium risk. DreamVLA precedent (+12pp).
(c) Differentiable physics gradients — backprop through MuJoCo via the autograd wrapper already shipped at src/openpi/contact_mpc/refinement/mujoco_autograd.py. $100-500K, high risk. Per lit audit nobody has done this for a manipulation VLA at scale.

Path forward: Phase B = (a), Phase C = (a)+(b), Phase D = (a)+(b)+(c). Each builds on previous. None requires throwing Pi0.5 away.

**Three plausible end-states for "our own foundation model":**
- Path 1: Pi0.5 + physics LoRA fine-tunes. 12-18mo, $50-200K. Pi0.5 derivative, weights ours, architecture not. Recommended starting point.
- Path 2: From-scratch pretrain on physics-refined data. 18-30mo, $500K-3M. Truly our model, requires raising. Pursue if Path 1 ships traction.
- Path 3: Diff-physics-native pretraining. 3-5yr, $10M+. NVIDIA / PI territory. Not our lane.

**The Pi0.5 gift.** PI gave us 5-10M of pretraining compute for free. Throwing it away to start from scratch now ships a worse model 2 years later. Compound on the gift instead — Phase B/C/D produce physics-grounded capability gains in months, not years.

**Capital math note:** When we eventually do Path 2, the dataset is the physics-refined-rollout corpus we'll have generated during Phase A → B → C. So the wrapper isn't just the inference product. It's the data engine for the foundation model.

**Why not start from scratch now:**
- π*0.6 / RECAP (Nov 2025) just shifted the base policy frontier. Chasing a moving target with no data moat = suicide.
- Hyperscalers (PI, Figure, Skild, NVIDIA, Google) own the data/co-training/RL/diff-sim lanes.
- The published literature has zero evidence that physics-loss training beats matched-compute imitation.
- A small team has no chance at Path 3 economics. Path 1+2 is the actual addressable plan.

**How to apply:**
- When pitching: "Inference-time physics-grounded deliberation today; physics-grounded foundation model derived via Path 1 → Path 2 in 18-30 months." Don't say "wrapper." Don't say "from scratch." Say "physics-grounded self-improvement system, end-state foundation model."
- When building: don't pre-optimize for Path 3. Stay in Phase A (current) until N=10 reproduces, then start Phase B distillation immediately.
- When raising: $500K-3M is the right Series-A range (Path 2 capital). Pitching above that with no data is not credible.
