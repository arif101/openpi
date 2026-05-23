---
name: Reachability field v1 — mean R² = 0.512 (passes gate, barely)
description: First reachability field network for Franka + OSC_POSE: f(q ∈ R^7, a ∈ R^7) → realized_EE_delta ∈ R^3. Held-out R² 0.51, peaked at 0.54 epoch 40 with overfit after. Cleared acceptance gate; not yet shown to be USEFUL as conditioning input.
type: project
originSessionId: 53e17618-aeb4-4007-9d29-d56bfc1900c3
---
## Result

Trained MLP (14→128→128→128→3) on 10k (config, action) → realized_EE_xyz
samples. 90/10 train/val split.

| Metric | Value |
|---|---|
| Best val mean R² | +0.542 (epoch 40) |
| Final mean R² | +0.512 (epoch 79) |
| Per-axis at best | x +0.49 / y +0.53 / z +0.52 |
| Acceptance gate (>0.5) | **PASS** |

Overfit signature: train_loss keeps dropping (0.017 → 0.003) while val_loss
bottoms at ep 40 (0.0078) and rises slightly to 0.0083. Either need more
data or lighter capacity.

## Two data-gen bugs discovered along the way

1. **env.reset() per sample was 1000x too slow.** First attempt: 0.6 samples/sec
   (projects 4.6 hours). Fixed by direct qpos write + OSC controller reset_goal()
   per iteration, with periodic env.reset every 50 samples. Final rate: 580/sec
   for the clean version (without periodic reset) or 3.5/sec (with reset).
2. **Silent sim-state corruption** after random extreme joint configs. Sharp
   transition at i=50: all subsequent samples had EXACTLY zero EE motion. The
   try/except silently swallowed env.step exceptions, and post-NaN sim state
   returns the same position forever. Fixed by `np.isfinite()` check + reset
   on detection, plus the periodic reset above as a safety belt.

Both bugs only manifested at scale. First (clean-without-reset) run reported
"median EE displacement = 0mm" — a single statistic that, if anyone hadn't
inspected it, would have produced a trained model with negative R².

## What the field captures

Field input: (joint config q ∈ R^7, action a ∈ R^7). Output: realized EE
xyz displacement after K=20 OSC steps.

R² ≈ 0.5 means the field explains roughly half the variance in realized
motion from the (q, a) inputs alone. The other half is presumably:
- Object collisions (objects pinned out of workspace in data gen, but
  deployment has objects — this is a distribution shift risk)
- Stochastic controller dynamics
- Higher-order effects (joint accelerations, contact forces)

## What the field does NOT yet validate

R² = 0.5 passes the gate but does not validate the field is USEFUL as a
conditioning input for Pi0.5. A field with R²=0.5 might be too noisy for
the policy to extract reliable feasibility information. The next decision
gate (before committing weeks to fine-tuning):

- **Option A (heavy):** Commit to task #65 directly — design Pi0.5
  conditioning pathway, LoRA fine-tune, eval on task 3 + transfer task.
  Several weeks. If field is too weak, weeks lost.
- **Option B (light):** Use the field as an MPPI cost term first (no
  fine-tuning). Penalize candidates whose predicted realized motion is
  small relative to commanded. Test on task 3's 14 IN_TRUE_CLOSE_FALSE
  traces. If MPPI with field cost recovers any of those 14, the field
  signal is real and conditioning is worth committing to. If not, the
  field isn't strong enough yet — improve before fine-tuning.
- **Option C:** Improve the field first — 50k samples (still cheap),
  regularization (weight decay, dropout), explicit object-position
  features to handle deployment distribution. Push R² past 0.7. Then
  commit to (A).

User direction was (A) — "the deep version isn't a binary flag; it's
a learned reachability field" — explicit policy conditioning, not an
MPPI cost. But (B) is a one-day insurance check before committing
multi-week to (A).

## How to apply

- Field path: `data/contact_mpc/reachability_field/reachability_field.pt`
  on the GPU box. State dict for the MLP. 14-dim input.
- Field normalization: configs and actions are NOT normalized in the
  dataset (raw float32). Predictions are realized EE delta in meters.
- Generator + trainer: `scripts/build_reachability_field.py`
  (commit 28f7d90 + 4aa75dd).
- For ANY downstream use (MPPI cost term or Pi0.5 conditioning), need
  to handle the distribution shift: training data had objects pinned
  far away; deployment has objects in workspace. The field will be
  WRONG when EE motion is blocked by an object (predicts free motion;
  actual is zero). This is a known gap.
