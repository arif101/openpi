---
name: Reachability field v1 fails deployment validation (2026-05-19)
description: Field passed training-distribution gate (R²=0.51) but on the failure-trace deployment distribution mean R²=0.10, y-axis R²=−0.37, inflection-zone correct calls 0/14, AUROC 0.47. The training distribution excluded scene objects; deployment failures depend on scene contact. Kills the "fine-tune Pi0.5 on this field" path.
type: project
originSessionId: 53e17618-aeb4-4007-9d29-d56bfc1900c3
---
## Result

Validated reachability field on the 14 IN_TRUE_CLOSE_FALSE failure traces +
14 matched PHYS_OK traces, 1997 total prediction points.

| Test | Result | What it means |
|---|---|---|
| R² across all trace points (mean of xyz) | **+0.10** | Vs. +0.51 on training distribution — massive shift |
| R² y-axis specifically | **−0.37** | Worse than predicting mean; field is anti-correlated on y |
| Per-axis correlation | x=+0.78, y=+0.43, z=+0.53 | x captures the cup, y and z don't |
| Inflection-zone correct (stuck = predicted low) | **0/14** | Field never identifies the actual failure pattern |
| AUROC pred_norm separating moved>2cm vs static | **0.47** | Worse than random |

At the stuck-inflection step of every failing trace, the field predicts
+5cm to +10cm of +y motion. Actual realized motion: 0.0–0.7cm. The
field has no idea the EE is blocked.

## Why

Training data: 10k (q, a) pairs with scene objects pinned far away. The
field learned "free-space motion given (q, a)" — a clean kinematic
question with reasonable R² in that regime.

Deployment data: same robot + controller, but scene objects in their
task positions. Failures happen because EE contacts cabinet exterior,
hovers near drawer geometry, or runs into wine rack proximity. The
field has no scene representation, so it predicts free-space motion in
situations where motion is blocked.

The "cheap and unlimited" training data is cheap because it excludes
the variable that actually drives the failure mode. Distribution
shift, by construction.

## Implications

1. **Option A (fine-tune Pi0.5 on this field) is dead.** Conditioning on
   R²=0.10 + AUROC=0.47 on the deployment distribution would teach Pi0.5
   nothing about the failure mode. Weeks of fine-tuning would yield zero
   transfer.

2. **The field-as-MPPI-cost insurance check (option B from yesterday) was
   the right move.** It surfaced the negative result in one day vs. several
   weeks of LoRA fine-tuning.

3. **The "embodiment-feasibility" framing was correct in spirit, incomplete
   in formulation.** Embodiment alone is insufficient; the failure mode
   depends on (embodiment, scene). A reachability field for this project
   must be scene-aware.

4. **Three concrete paths to a scene-aware field:**
   - **B1**: Sample (q, a) pairs WITH scene objects in their canonical
     task-3 initial positions. Training data is now task-specific.
     R² should jump on task 3 but won't transfer across tasks.
   - **B2**: Add scene-state to the input vector — object xyz + quat for
     up to N objects. Train across tasks; field becomes scene-conditional.
     More expensive to train, generalizes better.
   - **B3**: Precompute a signed-distance-field (SDF) for each task's
     workspace, and use SDF features as input. Cleanest geometric
     conditioning. Most expensive infrastructure-wise.

5. **None of those produce a generalizable feasibility signal without
   facing the harder question: how does scene representation get into
   the policy?** The original "cheap and unlimited" appeal of the field
   was that it abstracted scene away. Once we put scene back in, we're
   back to "the policy needs to model the scene to know what's
   feasible." That's not a small feasibility-conditioning patch; it's
   a question about what scene representation the policy operates over.

## How to apply

- **Stop pursuing the current field** for the policy-conditioning use case.
- Before committing to B1/B2/B3, decide whether feasibility-conditioning
  is still the right research direction given that the cheap version
  doesn't work. The user's framing pivoted to architectural intervention
  because that's a research-shaped artifact rather than a band-aid; if
  the scene-aware version is significantly more expensive and less
  generalizable, the cost-benefit changes.
- The validation script (`scripts/test_field_on_failure_traces.py`) is a
  reusable testing harness for any future field — pass an updated `.pt`
  file and a label-json and it reports the same metrics.

## Files

- Field model (will-not-be-used): `data/contact_mpc/reachability_field/reachability_field.pt`
- Validation result: `data/contact_mpc/field_validation.json`
- Validation script: `scripts/test_field_on_failure_traces.py` (commit 87692c4)
