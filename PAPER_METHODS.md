# Methods (paper draft)

## Notation

- Policy: π(a | o, ℓ) emits 7-dim end-effector delta actions a_t given observation o_t and language instruction ℓ. Pi0.5 emits action chunks of 10 actions per inference call.
- Controller: OSC_POSE in LIBERO. Maps commanded EE delta a_t[:3] to joint torques.
- Realized motion: δ_t = ee_pos[t+k] − ee_pos[t] for lookahead k.
- Commanded motion: c_t = a_t[:3] (the policy's 3-dim translational delta).
- Execution ratio: r_t = ‖δ_t‖ / ‖c_t‖.

## Method 1 — ε-Signature

The prediction-error signature ε at time t is the gap between the policy's commanded motion and the controller's realized motion. We compute a low-execution-ratio indicator:

L_t = 1 if r_t < τ_exec else 0

with τ_exec = 0.01 chosen empirically for OSC_POSE controller scale.

**Per-trace summary statistics** for clustering (8 features):
1-3. corr_x, corr_y, corr_z: Pearson correlation between c_t[axis] and δ_t[axis] across the trace.
4. mean_cos_align: mean of cos∠(c_t, δ_t).
5. median_exec_ratio: median of r_t over the trace.
6. frac_low_exec_ratio: fraction of timesteps with L_t = 1.
7. max_consecutive_low_exec: longest run of consecutive L_t = 1 timesteps.
8. persistent_dir_after_high_eps: count of chunk boundaries (every 10 steps) where the average commanded direction in chunk k+1 is within 0.9 cos of chunk k's direction, while chunk k had mean(1 − cos∠(c, δ)) > 0.5 (i.e., the policy issues the same commanded direction across boundaries despite high prediction error in chunk k — the active-inference "fails to use feedback" signature).

## Method 2 — Unsupervised mechanism stratification

We fit KMeans (K=2) on standardized per-trace ε-features from a calibration corpus of failure traces. Result: two clusters interpretable as:
- C0 (generation failure): low frac_low_exec_ratio, moderate corr_y. Policy emits varied commands that are realized but accomplish the wrong task.
- C1 (execution-stuck): high frac_low_exec_ratio, low corr_y. Policy commands sustained motion that doesn't get realized (controller saturation, kinematic singularity, contact-jamming).

**Stability**: leave-one-trace-out ARI = 1.000. **Cleanliness vs outcome labels**: silhouette score 0.50 (ε-clusters) vs 0.10 (BDDL atomic-derived outcome labels) — 5× improvement.

**Bimodality finding**: outcome-defined BDDL labels mix mechanisms. IN_FALSE_CLOSE_FALSE label (combined task-3 failures) splits 6 to 5 between C0 and C1 — not one mechanism, two.

## Method 3 — Streaming metacognitive gate

At each rollout step t, compute the windowed feature:

f_t = (1 / W) · Σ_{s=t-W}^{t-1} L_{s,k}

where L_{s,k} uses lookahead k for OSC controller convergence. Locked five-parameter gate:

| Parameter | Value | Justification |
|---|---|---|
| W | 50 steps | Window for feature averaging |
| k | 5 steps (~250ms) | OSC convergence wait |
| τ_exec | 0.01 | Low-execution-ratio cutoff |
| τ_gate | 0.160 | Calibrated FPR≤0.1 at f=0.5 on task 3 |
| N_sustained | 5 chunks (50 steps) | Debounce against transient threshold crossings on PHYS_OK traces |

**Gate fires** at step t iff f_s > τ_gate for s ∈ [t − N_sustained + 1, t] (i.e., feature has exceeded threshold for N_sustained consecutive steps).

**Decision latency**: k + (N_sustained × 10) = 550ms after the first stuck moment.

## Method 4 — Mechanism dispatcher

At gate-fire step t, compute the 8-dim ε-feature vector over the partial trace [0, t]. Standardize using the calibration corpus's fit scaler. Compute distance to each KMeans cluster center; assign to nearest. **Predicted cluster ∈ {C0, C1}.**

The dispatcher gates intervention:
- If predicted C1 → trigger retract-and-reapproach primitive.
- If predicted C0 → no intervention; policy continues unimpeded.

This is the architectural innovation: gate detection is permissive (high recall on failures), dispatcher is selective (high precision on mechanism class).

## Method 5 — Retract-and-reapproach primitive

When triggered (gate fires AND dispatcher predicts C1):
1. Compute retract direction d ← −mean({a_t[:3] : t ∈ last 30 commanded steps}), normalized to unit vector.
2. Override commanded action with a* = (d · 0.4, 0, 0, 0, gripper_state) for 30 consecutive steps. Gripper state is preserved (held objects don't drop).
3. After 30 retract steps, return control to the policy.

The primitive is **task-agnostic**: no task-specific waypoints, no canonical pose. Anti-mean-recent-direction works for any execution-stuck failure where the policy has been issuing sustained motion in one direction without realization.

## Method 6 — Architecture deployment math

Two intrinsics determine net effect:
- recovery_rate R = P(intervention=OK | baseline=FAIL): fraction of failures the intervention recovers.
- damage_rate D = P(intervention=FAIL | baseline=OK): fraction of successes the intervention damages.

Given base failure rate P(fail):
- Net effect Δ = R · P(fail) − D · (1 − P(fail))
- Break-even threshold P*(fail) = D / (R + D)

The architecture is net-positive in deployment when the application's base failure rate exceeds P*(fail).

**Empirical anchors** (5cm perturbation on tasks 0/6/9, paired matched-seed comparison):
- Phase 2c (no dispatch): R = 0.70, D = 0.10 → P*(fail) = 12.5%
- Phase 2f (with dispatch): R = 1.00, D = 0.09 → P*(fail) = 8.3%

Mechanism dispatch lowers the break-even threshold by 4.2pp, broadening the deployment regime in which the architecture is net-positive.

## Setup

- VLA: Pi0.5 from `sunshk/pi05_libero_pytorch` (PyTorch port of the JAX OpenPI checkpoint, fine-tuned on LIBERO).
- Controller: OSC_POSE (default LIBERO controller).
- Environment: LIBERO `libero_10` task suite, 30 trials per task on tasks 0 (multi-pick), 6 (mug-plate-pudding), 9 (microwave-close).
- Perturbation: validated Gaussian σ=5cm Gaussian noise on free-joint xyz, with settle-check (20 dummy steps, reject if any movable body has speed > 2cm/s or z < 0.4m, up to 30 retries).
- Calibration corpus for cluster model: 80 task-3 traces collected earlier (54 PHYS_OK + 26 PHYS_FAIL).

## Evaluation protocol

- Pre-registered acceptance criteria (locked before each phase):
  - Phase 2b: TPR ≥ 0.70 AND FPR ≤ 0.20 per task with the fixed 5-parameter gate
  - Phase 2c: combined recovery rate ≥ 20% on C1 traces; PHYS_OK survival ≥ 80%
  - Phase 2f: net effect > Phase 2c net effect (positive delta from dispatch)
- Trial-by-trial matched-seed comparison: both baseline and intervention runs use identical perturbation seed → comparable initial states.
- Bootstrap 95% confidence intervals on all rate quantities (n_boot = 2000-5000).
