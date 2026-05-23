---
name: MVP standalone claim and what it validates
description: The exact, defensible claim the Phase 1 MVP makes on its own. Used to judge whether a proposed change belongs in scope.
type: project
originSessionId: 53e17618-aeb4-4007-9d29-d56bfc1900c3
---
The Phase 1 MVP stands alone as a **method-validation claim**, not a capability claim. Specifically:

**What we validate:**
1. VLM-based failure attribution decomposes Pi0.5 rollouts into actionable clusters (measured by sensible cluster counts + coherent root causes).
2. Cluster-targeted LoRA training on (success, failure) pairs produces candidates with differentiable behavior (measured by non-trivial offline scores across candidates).
3. A latent-space world model over Pi0.5's pooled VLM hidden state ranks candidates whose rank correlates with real LIBERO improvement (Pearson r ≥ 0.5 target; r ≥ 0.3 acceptable with caveat; r < 0.3 means method pipeline works but signal is weak).

**What we do NOT validate in Phase 1:**
- That the trained LoRAs generalize to unseen scenes or OOD positions (that's Phase 2).
- That the policy now reasons about tasks (that's Phase 3).
- That real-robot transfer from sim training works (that's outside LIBERO entirely).

**Kill criteria (demo outcomes):**
- r ≥ 0.5: full-strength demo with validation claim.
- 0.3 ≤ r < 0.5: demo ships with "early signal, pilot pending" framing.
- r < 0.3: demo reframes to "attribution + ranking works; promotion gated by real A/B."

All three outcomes produce a shippable YC demo because the pipeline itself is the product.

**Why:** User locked this scope 2026-04-21 after repeated push-back on bolting memorization/reasoning claims onto the MVP. Scope discipline is load-bearing — trying to validate three claims at once makes all three weaker.

**How to apply:** When proposing pipeline changes, check that they serve one of the three Phase 1 validations above. If a change attacks memorization or reasoning, push it to Phase 2/3 even if the code is small.
