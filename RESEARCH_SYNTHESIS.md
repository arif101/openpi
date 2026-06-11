# Research Synthesis — Toward a General Robotic Foundation Model

**Goal.** A general, *learned* robot-manipulation foundation model that reaches SOTA across all four
LIBERO-PRO suites (object / spatial / goal / long-horizon-10) and beats the prior overall SOTA
(**VLS 36.81%**, π0.5 23.69%). The hard axis is **relocation / object-swap**: every current VLA,
including π0.5, collapses when a known object is moved to a new position.

This document synthesizes the diagnostic arc, the validated results, the dead ends (reported
honestly), the committed architecture, and the open research as of 2026-06-11.

---

## 1. The core diagnosis: VLAs memorize positions; precision lives in the wrist

Three measurements, on the official LIBERO-PRO metric, pin the failure mode:

| Probe | Result | Reading |
|---|---|---|
| π0.5 per-axis (object suite) | lan/paraphrase **1.00**, object-variant/same-spot **0.83**, **swap (moved) 0.17** | Robust to object & language variation; **fails on relocation** → memorized *positions*, not perception. |
| π0.5 camera ablation (memorized-pos, isolates motor) | full **1.00**, base-cam-zeroed **0.63**, **wrist-cam-zeroed 0.00** | π0.5's precision is **eye-in-hand visual servoing** — collapse is wrist-specific, not generic input shock. |
| Binding under resolution | full-frame CLIP/OWLv2 ≈ 0.40 disambiguation; **crop@1024 + CLIP = 1.00** | The binding failure is a **resolution wall**; foveation (hi-res crop) solves disambiguation zero-shot. |

**Conclusion that organizes everything below:** π0.5's *motor* is near-perfect given the correct goal;
the task-axis failure is **binding + position memorization**. The lever is not a better monolith — it is
to **factor** the policy so position can't be memorized and binding can't be weight-baked.

---

## 2. Validated building blocks (honest, measured)

- **Factored wrist-cam motor** (distilled from π0.5). Inputs = wrist cam + **relative** goal + proprio;
  **no** base image, **no** absolute pose, **no** object identity. Privileged-goal eval:
  standard `libero_object` **0.833** (vs object-blind 0.167, 5×), **swap 0.500** (vs π0.5 0.17, vs
  steering 0.30). A *general* motor made *precise* by the wrist view, and **position-robust because it
  only ever sees a relative goal** — invariance by omission, by construction.
- **Foveated binder.** Region propose → crop@1024 → CLIP disambiguate → mask → depth → 3D. 1.00
  disambiguation on cluttered tabletops where full-frame detectors plateau at 0.40.
- **Hide-distractors result.** Hiding non-target objects lifts π0.5 swap **0.17 → 0.53–0.63**,
  zero-shot, no LoRA. Diagnoses the swap collapse as **distractor confusion**, not pure memorization.
- **ε prediction-error self-model** (our strongest reusable asset). The commanded-vs-realized EE/action
  gap detects mid-episode failure at **AUROC 0.97**; the gate **generalizes across tasks** without
  refit (TPR 1.0 on contact-rich, FPR ≈ 0.1); a runtime cluster-**dispatcher** that fires recovery only
  on the right failure cluster turned a net −1.1pp intervention into **net +3.3pp**. No VLA in the
  literature uses an internal ε signal as a closed-loop replanning trigger.
- **Grounded-physics place (analytic).** Object pose = EE + measured grasp offset; release from
  container geometry + gravity. Object-suite **0.60**, swap **0.63** under oracle perception — *kept only
  as a validated insight and warm-start* (see §4: rejected as the contribution).

---

## 3. Dead ends (reported, not buried)

- **Learned world-model gradient refinement** *hurts* task success (52% → 30%): the learned objective
  is misaligned with truth. Physics grounding via a *learned* WM is unreliable.
- **Reachability field** (MLP f(q,a)→realized EE-delta) passed in-distribution (R²≈0.51) but **failed
  deployment** (R²≈0.10, inflection-zone 0/14): training excluded scene contact; the failure mode
  depends on (embodiment, scene), not embodiment alone.
- **MPPI "+20pp"** was seed-specific; on a proper no-replay BDDL eval it gave **no overall lift**.
- **De-attractor graying / inpainting variants** went out-of-distribution; **hiding** works, graying
  doesn't.
- **Analytic place as a contribution** — rejected by design review (see §4).

---

## 4. The pivot: a *learned* architecture, not an analytic result

The analytic grounded-physics place beats VLS on paper, but it is **scripted** (analytic place,
hard-coded container goal, discrete grasp/place threshold, privileged detection pixel). Per the
explicit research bar:

> *"Analytic SOTA doesn't mean anything in deep-learning research. We need position invariance built
> into the model, so that we can re-orient ourselves when needed."* … *"A novel architectural change
> that changes the way VLAs think."*

So the analytic result is demoted to an *insight + warm-start*, and the program is now a **learned**
model whose novelty is architectural.

---

## 5. Committed architecture — the Predictive-Sensorimotor VLA

Current VLAs **react** (appearance → action, feedforward). The rethink: the model **predicts** its
egocentric sensory future, **acts** to fulfill the prediction, and **replans when surprised** —
with **ε prediction-error as the single unified currency** for perception, subgoal selection, control,
and replanning.

**Pillars:**
1. **Invariance by omission.** The motor never represents absolute pose — only relative/egocentric
   goals. Can't perceive position ⇒ can't memorize it ⇒ relocation-invariant by construction.
   (Validated: relative-goal motor swap 0.50 vs absolute/π0.5 0.17.)
2. **ε as unified currency.** The validated ε self-model (AUROC 0.97) defines *subgoal-achieved* and
   **self-triggers replanning** — the trigger no reasoning-VLA has.
3. **Reasoning as a first-class loop** (hybrid): a **relational** scene representation (object IDs +
   *relative* relations — invariance-by-omission *and* what spatial reasoning needs) → emit a **latent
   subgoal** to the fast motor → language only for high-level decomposition.

**Novelty wedge (validated unoccupied by three independent deep-research sweeps):** the **unification at
VLA scale** — ε prediction-error as the one signal that *selects an egocentric subgoal, servos toward
it, and autonomously triggers replanning.* Cite-and-beat: **SAFE** (detection-only), **DoReMi / RACER /
AHA** (external LLM/VLM critics), **R-AIF** (active inference, small scale only), **SuSIE / GHIL-Glue**
(open-loop visual subgoals); reasoning-VLAs **ECoT / Hi-Robot / CoT-VLA / Emma-X / LCB / ThinkAct /
SAGE** are all feedforward or timer/user-triggered — **none self-replans on prediction error.**

---

## 6. The decisive experiment this session — a falsified pillar (honest negative)

A fourth candidate pillar — *"spatial intelligence emerges for free from predicting egocentric
wrist-cam perception"* — was tested before any multi-week build (`motor_distill/probe_spatial_emergence.py`).
We trained a spatial-autoencoder + latent-dynamics predictor on wrist RGB (no depth, no VLM), froze it,
and linear-probed the latents to decode object position, evaluating on **held-out relocated (swap)**
positions, stratified to the late approach phase where the object is actually in the wrist view.

| arm (late phase) | std decode | **swap decode** | swap/std |
|---|---|---|---|
| random conv features | 9.0 cm | 25.9 cm | 2.88 |
| reconstruction-only autoencoder | 8.6 cm | 32.0 cm | 3.70 |
| **predict (the pillar)** | 13.5 cm | **26.3 cm** | 1.95 |

**Prediction did not beat random conv features on relocation (26 cm), and nothing approached the
~2–4 cm needed for servoing.** Predicting egocentric perception does **not** yield relocation-invariant
spatial grounding "for free"; it drifts toward the same learned-WM failure mode that already burned us.

**→ Pillar cut.** Spatial grounding must come from the components that *work* (foveated binder + wrist
servo), not from an emergent predictive module. A cheap negative (hours) that saved weeks — exactly the
diagnose-before-iterating discipline the program runs on.

---

## 7. Open research & next experiments

- **Running (`wf_7bc912d6-d5c`)** — the *mechanism* research to lock the method: which internal ε
  signal to use in a flow-matching policy; what the replan action should do per failure mode; the
  post-probe spatial-grounding source (learned, relocation-invariant, no frozen VLM); how to train
  ε-triggered replanning without collapse; the decisive experiment + cite-and-beat.
- **The three decisive isolation experiments** (cheap, reuse our machinery):
  1. **Omission-invariance** — absolute-coord motor memorizes and collapses on swap vs relative
     generalizes.
  2. **Reasoning-necessity** — reactive single-pass fails spatial/goal/10 vs reason→subgoal loop.
  3. **ε-replanning value** — no-replan / timer-replan vs ε-self-trigger on long-horizon-10 + OOD
     (guard against the break-even where false-trigger damage exceeds recovery, < ~8% failure rate).

---

## 8. Reproducibility

- **Data + checkpoints** (2.99k rollout npz, 28 motor checkpoints, scripts/specs) backed up to the
  private HF dataset `arif101/openpi-motordistill-backup-20260611`.
- Key code: `motor_distill/train_wristcam_motor.py` (factored motor), `eval_wristcam_motor.py`,
  `collect_motor_data_hide.py` (position-diverse data engine), `eval_physics_place.py` (analytic place,
  warm-start only), `probe_spatial_emergence.py` (this session's falsification probe).
- Detailed, dated findings live in the project memory index; this file is the narrative synthesis.

---

*Standing principles: brutal honesty (no inflated SOTA); every component learned and as general as
possible; self-contained at deployment (π0.5 is an offline teacher only); validate novelty via deep
research before building; do the right thing, not the cheap thing.*
