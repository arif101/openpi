---
name: PhysVLA architecture + roadmap (2026-05-18)
description: System diagram, what's working, what to improve, and the 12-month roadmap from Phase A through paper + YC W27 application.
type: project
originSessionId: 53e17618-aeb4-4007-9d29-d56bfc1900c3
---
## System architecture (current state, 2026-05-18 01:30 UTC)

```
                LIBERO env (robosuite + MuJoCo)
                  Perturbed init via reject-sampled validity check:
                    - settle 20 steps with zero action
                    - reject if any movable body's final speed > 2cm/s
                    - reject if any movable body's z < 0.40m (fell through)
                    - reject if any movable body drifted >15cm from intended qpos
                    - up to 40 retries before falling back to last sample
                  Continuous physics ground truth: contact wrenches (cfrc_ext),
                  6-DoF object poses, gripper qpos, full qpos/qvel, contact list

obs --> Pi0.5 frozen backbone (vlm_features) --> action chunk a_0 (prior)

                        Two refinement layers:

  MPPI (Phase A, shipped)                  PhysVLA (in progress)
  K candidates around a_0                  encode obs -> z_t
  forward-sim in env                       dynamics(z_t, a_t) -> z_{t+1}
  cost = target+approach+collision+anchor  aux heads (training only):
  softmin -> refined a*                       z -> contact_wrench (6)
  Result: +6.7pp task3 5cm                    z -> object_pose (n*7)
                                              z -> slip_velocity (3)
                                           Refinement: gradient-step a
                                              through trained dynamics
                                              toward goal latent
```

## What's shipped + working

- LIBERO env wiring + per-step physics-trace logger (--log-physics-traces)
- Perturbation system with end-state validity check (reject-sample bad spawns)
- MPPI refinement (Phase A): published-bar result +6.7pp on task 3 3-seed N=10
- Phase A→B handoff: refined-rollout logger + LeRobot converter + LoRA TrainConfig
- PhysVLA model code: encoder (3M params), dynamics transformer (1M), aux heads (200k). Smoke test passes locally.
- PhysVLA dataset class + training script + episode-aware train/val split
- Failure-trace analyzer (offline diagnosis from NPZs)
- 4-way matrix comparison utility (gate-off / gate-on / grip-v1 / grip-v2)
- Three sets of trial data: Phase A no-fix (50 trials, pooled +2pp), trust-gate (50 trials, 0pp), grip-v1 broken (40 trials, -13pp), grip-v2 fixed (50 trials, 0pp)
- 33 successful refined rollouts logged for Phase B distillation
- 192 partial physics traces (v0_partial, archived for ablation)

## What we're doing well

- Diagnostic rigor: caught our own +20pp small-sample artifact at N=10; threw out grip-v1 immediately when its pooled delta dropped; caught perturbation pathology via audit not blind trust.
- Modular code: every component independently smoke-testable on Mac.
- AlphaZero recipe properly scoped: search produces data, distillation produces capability, iterate.
- Honest framing: retracted overclaims; documented every regression; paper will have negative-result ablations.
- The wedge is real: lit audit confirmed no published method does physics-engine MPPI on flow VLAs; no published method evaluates test-time refinement on LIBERO-PRO; no published method does continuous-physics-supervised latent dynamics.

## What to improve

| Issue | Impact | Fix |
|---|---|---|
| Phase A 5cm matrix has pathological spawns (audit showed 50% rate before fix) | inflates variance | Re-run with v2.1 validity check now committed |
| Training data narrow (5 tasks x 5 seeds x 5 perturbations) | dynamics won't generalize | Expand to LIBERO-90 + sim-augmentation (mass/friction randomization) |
| No force/torque in observation | model can't see grip state | Add F/T sensor read from MuJoCo to obs vector |
| Pi0.5 backbone frozen | doesn't shape toward physics | Phase 2: LoRA fine-tune backbone jointly with PhysVLA |
| No real-robot eval | sim-only results limited | Phase 3: deploy on a Franka with force sensing |
| Single perturbation regime (init only) | doesn't test online disturbance | Add mid-trajectory pushes |
| EE-delta action representation | can't express compliance | Phase 3: impedance/admittance action layer |

## 12-month roadmap

**Months 1-2 (now-July 2026) - Phase 1: PhysVLA v1 paper**
- Finish clean data collection (~6h GPU)
- Train dynamics + aux heads (~5h GPU)
- Acceptance gate: latent linearly decodes physics quantities at R² > 0.7 on held-out trajectories
- Build action-refinement layer (gradient step through trained dynamics toward goal latent)
- Eval on LIBERO-PRO matrix: Pi0.5 vs Pi0.5+MPPI vs Pi0.5+PhysVLA
- Target: +10pp pooled vs Pi0.5+MPPI on the matrix, cleaner variance
- Paper: "PhysVLA: Continuous physics supervision for latent dynamics in VLA refinement"
- Submit to CoRL 2026 (~Jul deadline) or RSS 2027 (~Jan)

**Months 3-4 (Aug-Sep 2026) - Phase 2: scale + ablations**
- Expand training data to LIBERO-90 + sim-augmented physics randomization
- LoRA fine-tune Pi0.5 jointly with PhysVLA encoder
- Compare against learned-WM scorers (V-JEPA-2-AC if available; our own Q(h,a) baseline)
- Ablations: wrench-only vs wrench+pose+slip; effect of physics randomization

**Months 5-6 (Oct-Nov 2026) - Phase 3: real robot**
- Single Franka with F/T sensor + impedance controller
- Implement physics-native action layer (impedance + compliance) on top of PhysVLA
- Contact-rich demo task (e.g. wiping, deformable-bag pick-place, peg-in-hole)
- Video for the YC pitch
- **YC W27 application Sep 2026.** Pitch: "physics-grounded VLA for contact-rich manipulation; replace brittle learned policies in factory/lab settings; aligned with PI's published think-revise roadmap; complement to OneRobot's WM-for-eval play."

**Months 7-12 (Dec 2026-May 2027) - Phase 4: scale + raise**
- Series A pitch: paper accepted + demo video + one design-partner pilot
- Target raise: $3-8M, 18-month runway
- Hire 2-3 robotics+ML engineers
- Sign first commercial pilots (manufacturing, lab automation)

**Year 2: own foundation model**
- Drop Pi0.5 backbone
- Train physics-aware visual-language encoder from scratch on physics-rich data we've generated
- Become fully ours architecturally

## Key gates / decision points

| Decision | Trigger | Action |
|---|---|---|
| Continue PhysVLA v1 | latent R² > 0.7 on physics decode | proceed |
| Pivot if PhysVLA flat on LIBERO-PRO | matrix delta < +3pp pooled | reframe paper as "physics aux supervision improves representations but not test-time refinement," ship anyway |
| Apply to YC | one strong demo video + paper accepted/under review | apply W27 |
| Raise Series A | YC accept + 1 design partner LOI | raise $3-8M |
| Build own backbone | $3M+ runway + 6 months of data | start from-scratch training |

## Capital math sanity check

- Phase 1 (PhysVLA v1): <$100 GPU compute. Tractable on a single rented RTX 6000 Ada.
- Phase 2 (scale + ablations): ~$5K compute. Self-funded.
- Phase 3 (real robot): one Franka research arm rental: ~$3K/mo. F/T sensor: $5K capex. Total ~$15K over 3 months.
- Phase 4 (own backbone training): $500K-3M. Series A territory.

## Honest risks

- PhysVLA v1 might not give +10pp. If +2-5pp, still publishable as "physics aux supervision improves test-time refinement modestly," but weaker.
- Sim-to-real transfer is hard. Phase 3 may not work on first attempt.
- Hyperscalers may release something similar (V-JEPA-2-AC v2, GR00T-Manipulation, etc.) that obviates the wedge. Watch announcements.
- Real robot demo dependent on hardware availability.
- The +6.7pp Phase A result has high variance across seeds. Need clean re-run with validity check before claiming.
