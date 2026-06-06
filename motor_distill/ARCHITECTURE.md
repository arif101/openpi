# Factored VLA for LIBERO-PRO — architecture, components, contributions

## What it is (one line)
A **factored, mostly-frozen pipeline**: open-vocab foundation models bind the *named* object to a 3D goal,
and one **tiny learned object-agnostic motor** (distilled from π0.5 + DART) executes it. **Not** a monolithic
model, **not** an ensemble (no voting), **not** an MoE (no learned inference-time router over experts).
π0.5 is the **offline teacher only** — it is NOT run at inference.

## Inference-time data flow

```
 INSTRUCTION                              RGB-D @1024  +  camera matrix K, extrinsics
 "put the cream cheese                          │
       in the basket"                           │
        │                                        ▼
   parse nouns                    ┌──────────────── BINDER  (FROZEN / zero-shot) ───────────────┐
   {object:"cream cheese",        │  OWLv2  : propose candidate regions (low thr, high recall)  │
    container:"basket"}  ───────► │  CLIP   : adaptive-crop each region @hi-res, pick the crop   │
        │                         │           matching the NAMED noun  (resolution = the fix)    │
        │                         │  depth+geometry: median near-surface depth → unproject       │
        │                         └───────────────────────────┬─────────────────────────────────┘
        │                                                       │  3D goals: g_obj (xyz), g_cont (xyz)
        │                                                       ▼
        │                         ┌──────────── MOTOR  (LEARNED, ours, ~0.3M params) ────────────┐
        └───── relay plan ──────► │  INPUT (object-AGNOSTIC):  [ ee − active_goal , quat, grip ]  │
              [obj→close,         │     NO image, NO object identity  →  cannot memorize          │
               cont→open]         │  RELAY: active_goal = g_obj (approach/grasp) → g_cont (carry) │
                                  │         switched by a grasp latch (held = closed & lifted)    │
                                  │  OUTPUT: K=10 action chunk  → TEMPORAL ENSEMBLE over chunks   │
                                  └───────────────────────────┬─────────────────────────────────┘
                                                               │  6-DOF EE-delta + gripper
                                                               ▼
                                            robosuite/LIBERO  OSC_POSE controller  →  robot
```

## Offline (training only — NOT at inference)

```
 π0.5 (3B VLM, TEACHER) ──► 40 clean pick-place demos        ┐
                        └──► 51 DART trajectories            ├──► distill MOTOR (L1 pose + BCE gripper, K-chunk)
   (DART = π0.5 rollout w/ injected action noise → visits    ┘     student never sees image / object identity
    drifted states → log π0.5's CORRECTION; ONLY in scenes
    where π0.5 binds correctly; stored GOAL-RELATIVE
    → anti-memorization preserved)
```

## Components & where they come from
| Component | Status | Role | Memorize? |
|---|---|---|---|
| OWLv2 (open-vocab detector) | **frozen, off-the-shelf** | propose candidate object regions | n/a |
| CLIP ViT-L/14 | **frozen, off-the-shelf** | disambiguate the NAMED object among crops | n/a |
| depth + camera geometry | **deterministic** | crop pixel → 3D goal | n/a |
| Motor head (~0.3M MLP, K-chunk) | **LEARNED (ours)** | goal-relative state → action chunk | **no — blind to object** |
| π0.5 (3B VLM) | **offline teacher only** | demos + DART corrective labels | (not deployed) |
| OSC_POSE | env controller | execute 6-DOF deltas | n/a |

Deployed footprint ≈ OWLv2 (~150M) + CLIP (~300M) + ~0.3M motor — **no 3B VLM at inference.**

## Contributions (what's actually novel)
1. **Diagnosis (finding):** the LIBERO-PRO TASK-axis binding failure is a **RESOLUTION** problem — the named
   objects are sub-detectable blobs at the VLM's 224px input — *not* a semantic/language failure. Proven by
   foveation recovering it (0.40→0.80 object, 0→1.00 container).
2. **Foveation binder:** render-high-res → propose → adaptive-crop → CLIP-disambiguate → geometry → 3D goal.
   Open-vocab + object-agnostic crops → generalizes to **unseen objects and containers by construction**.
3. **Object-agnostic motor + DART-in-goal-relative-frame:** a motor that sees only goal-relative state
   (anti-memorization **by construction** — it literally cannot key on object identity), made reliable by
   distilling π0.5 with action-chunking + L1/BCE + temporal ensembling + **DART corrections harvested only in
   correct-binding scenes and stored goal-relative** (launders the teacher's labels without re-introducing
   memorization).
4. **Result:** first **non-hardcoded** factored policy to clear the LIBERO-PRO TASK-axis full pick-place bar
   (30% vs CAG 21.7%, seed 7; multi-seed pending), with a deployed footprint far smaller than the teacher VLA.

## Model type — the honest label
- NOT a new monolithic VLA. NOT an ensemble (no vote). NOT an MoE (the relay is a 2-state latch, not a learned
  soft router over experts — we explored MoE/gate variants; the reliable system collapsed to a single motor).
- It is a **compositional / factored policy**: frozen open-vocab foundation models produce a symbolic 3D goal
  that conditions one small learned, appearance-blind motor distilled from a VLA teacher. Neuro-symbolic-flavored
  (geometry is deterministic; binder + motor are neural).

## What borrows vs what's ours
Borrowed/frozen: OWLv2, CLIP, SAM, π0.5, ACT chunking, DART, temporal ensembling. **Ours:** the resolution
diagnosis, the foveation binder for VLA grounding, and the anti-memorization-preserving DART-in-goal-relative
recipe — and the demonstration that this composition clears the unsolved TASK axis without hardcoding.
