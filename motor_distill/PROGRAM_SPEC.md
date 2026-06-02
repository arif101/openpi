# COMMITTED PROGRAM — make the action head general (not a lookup table)

**One direction. Kill-criterion set. No pivot until answered yes/no.**

## The thesis (from our own evidence + the scale-vs-structure read)
VLA action heads memorize (appearance -> action lookup) because: (1) imitation training has
no generalization pressure, (2) they condition on appearance, not physical state, (3) no
structural inductive bias forces physically-equivalent scenes to equivalent actions. Cosmos 3
scales this (bigger/denser lookup, pixel-space "appearance-physics") but does NOT cure it —
no object-centric factorization, no equivariance, generalizes by coverage not construction.

## What our OWN experiments established (don't re-litigate)
- Plain distilled flow head (goal-relative conditioned): demo-mimic, 0% grasp with a correct-but-OOD
  target. MEMORIZES. [[project_keystone_manifold_separates]]
- Equivariant head (SE(2) canonicalization): ~= plain, also memorized. => EQUIVARIANCE-CONDITIONING
  ALONE IS NOT THE CURE. (Do not bet the program on equivariance.)
- Hand-coded DMP (goal-ATTRACTOR + learned forcing): re-targets BY CONSTRUCTION, grasps 87% in-dist,
  degrades 40pp vs Pi0.5's 73pp under perturbation, BEATS Pi0.5 +20pp at 10cm (task 3).
  => THE ATTRACTOR STRUCTURE IS THE VALIDATED CURE. [[project_dmp_retarget_beats_pi05]]

## The committed architecture: learned attractor head (NDP) + learned goal
Replace Pi0.5's appearance-conditioned flow action expert with a **Neural Dynamic Policy** head:
- a net (conditioned on PHYSICAL/object state + proprio) outputs DMP **forcing weights** (the learned,
  general "shape"), and
- a **differentiable goal-attractor (DMP) layer** integrates them to an action trajectory that
  CONVERGES to a goal by construction -> generalization-by-construction (the DMP property, now learned).
- the GOAL is predicted by a learned perception/affordance head from physical state (so re-targeting is
  to a LEARNED goal, not a hand-coded one -> answers "don't hand-design primitives").
- object-centric conditioning (relational over objects) for multi-object / novel arrangements (Rung 1).
Physics-conditioning (object state) is the input; the attractor is the structure; the net is the only
learned-from-demos part. Generalization comes from structure, not from memorized appearance.

## Staged plan + KILL-CRITERION

### Stage 1 (DECISIVE — does the cure survive being LEARNED?): NDP vs plain, object-state-conditioned
- Build the NDP head (differentiable torch DMP + forcing net). Build a plain head baseline (net:
  state -> action chunk, the memorizer).
- Train both on pert0 grasp demos, conditioned on object state (privileged from sim).
- Closed-loop grasp-lift eval, in-dist (pert0) + OOD (pert5/10), structured vs plain.
- HYPOTHESIS: NDP degrades gracefully (like the hand-coded DMP); plain collapses (like the distilled head).
- **KILL: if the LEARNED NDP head does NOT generalize OOD clearly better than the plain head (graceful
  degradation), the attractor structure does not survive learning -> STOP, the lookup table isn't cured
  by structure-in-the-head, reconsider the whole thesis.**

### Stage 2 (learned goal): perception/affordance head predicts the goal from physical state
- Replace the hand-derived grasp goal with a learned head: physical state -> goal pose. Does the
  predicted goal transfer to NOVEL objects (LIBERO-PRO object-swap)? KILL if it can't.

### Stage 3 (graft on Pi0.5 + the real benchmark): integrate NDP head into Pi0.5, eval LIBERO-PRO PER-AXIS
- Keep frozen VLM; replace action expert with NDP head reading VLM + physical-state features.
- Eval per-axis vs stock Pi0.5 (honest: combined "0%" is worst-case; pi0.5 more robust than peers).

### Stage 4 (the hard half, DEFERRED): physics-aware perception from PIXELS
- Extract physical state / affordances / contact from images (the intuitive-physics problem). Only
  attempt if Stages 1-3 validate. Until then, Stages 1-3 use privileged sim state to ISOLATE the
  head-generalization question from the perception question.

## Discipline
- ONE direction (above). The kill-criteria are pre-set. No pivoting to a fresh deep-research "wedge"
  until Stage 1 answers yes/no. We have pivoted ~6 times by chasing novelty over driving to ground;
  not again.
- Benchmark commit: LIBERO-PRO (reuses harness, current, shows the memorization failure) + VLABench.
- Assets: HF corpus arif101/libero10-pi05-perturbation-traces; code motor_distill/ on branch
  factored-policy; DMP validated (dmp.py); harness (run_reason_v3_mppi.py).

## First build (GPU-independent): the NDP layer
ndp.py = differentiable torch DMP integration + forcing net, unit-tested for (a) it reproduces a demo
when forcing fit, (b) it RE-TARGETS to an unseen goal by construction (the property), (c) gradients flow
(trainable). Then Stage 1 train+eval (GPU).
