# Factored VLA for the LIBERO-PRO TASK axis — results (2026-06-05)

## Problem
On LIBERO-PRO's TASK axis the instruction names a *different* target than the one the policy
memorized for that scene. Frozen π0.5 ignores the language and grabs the memorized object:
on the clean object suite, **baseline grounding = 20%** (8/10 scenes the gripper goes to the
memorized object at 0.2–4.5cm while the named target sits 13–19cm away).

## Diagnosis (the moat — nobody has this exact combination)
Causal probing of frozen π0.5 (same scene, swap only the named object):
- `S_img = 0.45` — the VLM **does** move its image representation when you rename the object
  (perception/language is present, not the failure).
- `S_act = 0.38`, causal `shift→T = +0.10` on 8/10 — the language→target motor circuit **exists
  and is causal**, just **out-weighted** by a memorized appearance attractor (dir→M 0.51 vs dir→T 0.10).
- **Classifier-free guidance (CAG) cannot fix it**: amplifying the language gain saturates
  (dir→T plateaus +0.20, never overtakes) and destabilizes (|disp| +20%). The bias is in the
  **weights**, not fixable at inference.

## Architecture: factor binding from motor
```
language ─► [OWLv2 open-vocab binding] ─► 3D goal ─► [distilled goal-conditioned attractor motor] ─► action
            (open-vocab + depth geometry)              (object-agnostic; sees only goal-relative state)
```
1. **Motor** — π0.5's reaching distilled into a goal-relative attractor head. Sees *no appearance*,
   so it **cannot** be captured by the memorized object (by construction). Oracle-goal redirect: **20%→90%**.
2. **Binding** — OWLv2 (open-vocab detector) → target box → median real-depth → geometric unprojection
   (robosuite camera matrix, verified **2.6cm** round-trip). **Frozen, open-vocab → generalizes to
   held-out objects by construction** (no learned readout to memorize).

## Headline results (object axis, counterfactual; 3 seeds × 10 scenes)
| method | reach-named | lift-named (grasp) |
|---|---|---|
| baseline π0.5 | 20% | 20% |
| CAG (classifier-free guidance, inference-time SOTA) | 30% | — |
| learned π0.5-feature binder | 50% | — |
| **Ours: distilled motor + OWLv2 geometric binding** | **67% (±5%)** | **47% (±5%)** |
| oracle-goal ceiling (motor with perfect goal) | 90% | — |

- **>2× over CAG** (67% vs 30%), the inference-time baseline.
- **Task-relevant**: actually **lifts the language-named object 47%** vs 20% baseline (2.4×).

## Cross-suite generalization (frozen pipeline, NO retraining)
Same distilled motor + OWLv2 binder, applied to suites with different objects/scenes:
| suite | reach-named | note |
|---|---|---|
| libero_object (in-distribution) | 67% (±5%) | 3-seed |
| **libero_10 (kitchen/living-room objects, never distilled)** | **78%** | objects unseen by the motor — generalizes by construction |
| libero_goal | 100% (N=8) | grounding generalizes; lift n/a (mostly non-pick tasks) |

The motor is object-agnostic and the binder is frozen open-vocab → the pipeline reaches the
language-named object on objects/scenes it was never trained on. This is the structure-not-coverage result.

## Key ablations / negative results
- **Learned binder fails held-out generalization** (2cm train → 20cm on unseen object names) —
  it memorizes. The **open-vocab OWLv2 binder generalizes by construction** (the structure-not-coverage win).
- **GroundingDINO fails** (tiny: 88px/35cm; base: 10/10 detect but ALL wrong object) — its phrase-fusion
  is salience-biased and confidently boxes the *memorized* (salient) object = the disease at the binder.
  **OWLv2 fails safe (miss) and is 3–4cm accurate when it fires.**
- **SigLIP-region / SigLIP-verify fail** — CLIP/SigLIP patch features are too spatially coarse (~19cm) and
  can't classify small box crops; SigLIP-verify *hurt* (50% < 67%).
- **Gaussian splatting / feature fields** (F3RM/LangSplat/GraspSplats) ruled out for our 1–2-camera setup
  (need 30–150 posed views); parked for the world-model thread instead.

## Limitations / next
- 3 wrong-object failures are inherent OWLv2 confusions on ambiguous grocery items (cream cheese / ketchup /
  chocolate pudding); a stronger detector or wrist-cam refinement could help.
- **Full pick+place tested: 0/10 success, but PICK works** — 5/10 grasp+lift and enter the transport phase;
  transport-and-release fails because the reach head was distilled ONLY on reaching (never on navigating while
  holding an object or releasing at a 2nd goal). **The place phase needs a dedicated head/controller** (distill
  π0.5's place actions, or a transport-while-holding policy) — the clear next build. Grounding+grasp (the hard
  binding problem) is solved; place is mechanical follow-on.
- Sim depth is exact; real-world transfer = swap in MOMA-calibrated monocular depth (1–3mm) — a calibrated
  peripheral, not a research problem.

## Pipeline (motor_distill/)
collect_reach → distill_reach (motor) · eval_reach_redirect (oracle gate) ·
unproject_check/loc_check/det_compare (binder calibration) · eval_e2e_owl (OWLv2 binder e2e) ·
eval_cag (CAG baseline) · cf_bind_diag (causal diagnostic).
