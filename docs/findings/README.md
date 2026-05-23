# Research findings — openpi contact-MPC / PhysVLA project

Working notes accumulated across experiments on Pi0.5 + LIBERO. These are
research-state documents, not polished reports. Read the consolidated
findings first; everything else is detail.

## Start here

- **[Consolidated findings, 2026-04-25 → 2026-05-23](project_consolidated_findings_2026_05_23.md)** — single-document synthesis of what we have learned. Diagnostic apparatus, validated and falsified hypotheses, strategic position. Reading this should make the rest of the documents optional, not required.

## Phase / roadmap docs

- [Phase roadmap](project_phase_roadmap.md) — original 3-phase plan
- [MVP standalone claim](project_mvp_standalone_claim.md) — scope of what Phase 1 validates on its own
- [Pivoted roadmap (2026-04-25)](project_roadmap.md) — Sprint 1 pivot after generalization audit
- [Phase B plan (2026-05-17)](project_phase_b_plan.md) — LoRA distillation plan (paused)
- [PhysVLA architecture + roadmap](project_physvla_architecture_roadmap.md) — 12-month roadmap from Phase A onward
- [AlphaZero-for-VLAs roadmap](project_alphazero_for_vlas.md) — 4-phase path from wrapper to foundation model
- [Path to foundation model](project_path_to_foundation_model.md) — Path 1/2/3 framework

## Major findings

### Validated empirical results

- [Experiment A result](project_experiment_a_result.md) — Pi0.5 + WM + MCTS gave +8–10 pp on LIBERO-PRO 5cm in initial sprint
- [MPPI first signal](project_mppi_first_signal.md) — +20pp on LIBERO-10 task 3, seed=7 and seed=21 (later revised — see MPPI no-lift)

### Negative results that shaped the path

- [Sprint 1 result](project_sprint1_result.md) — Q(h,a) ≈ V(h) ±1pp on LIBERO-90; inference-time ceiling
- [Phase 1 dead](project_phase1_reason_dead.md) — WM-only gradient refinement hurts task success
- [Generalization audit](project_generalization_audit.md) — Pure wrappers can't fix sub-30% baselines
- [Recovery regime day-1 diagnostic](project_recovery_regime_diagnostic_day1.md) — Hand-scripted primitives 0/5
- [Replay drift severe](project_replay_drift_severe_2026_05_18.md) — `init_sim_state+replay` drifts up to 40cm; compromises every state-restore diagnostic
- [Dead-state finding](project_dead_state_finding.md) — Original Euclidean stratifier was 80% misclassified
- [MPPI no overall lift](project_mppi_no_lift_2026_05_18.md) — Across N=40/mode paired-seed: baseline = MPPI; earlier +20pp was seed-specific
- [Reachability field v1 fails validation](project_reachability_field_v1_failed_validation_2026_05_19.md) — Field passes training-distribution gate but fails on deployment

### Failure-mode diagnostics

- [BDDL `In()` predicate failure modes](project_in_predicate_failure_modes.md) — pre-experiment predictions before re-stratification
- [BDDL re-stratification result](project_bddl_restratification_result.md) — actual distribution: 53% IN_TRUE_CLOSE_FALSE
- [Drawer-close failure root cause](project_drawer_close_failure_root_cause_2026_05_18.md) — Pi0.5 commands drawer-close in 14/14 failing traces; EE doesn't move in 11/14; execution failure, not generation failure
- [IN_FALSE failure diagnostic](project_in_false_diagnostic_2026_05_23.md) — 83% of IN_FALSE failures are NO_APPROACH (Pi0.5 never gets EE near bowl, never commands grasp). Structurally different from IN_TRUE; needs a different reward component
- [Recovery primitive v1 partial result](project_recovery_primitive_v1_partial_2026_05_18.md) — joint-teleport + push closes drawer in 2/14 (mechanism validated, naive impl rejected)
- [Reachability field v1 (training-distribution pass)](project_reachability_field_v1_2026_05_19.md) — R² = 0.51 on its own validation set

## Strategic context

- [YC competitive intel](project_yc_competitive_intel.md) — robotics-startup landscape (W26 batch)
- [PI blog intel](project_pi_blog_intel.md) — Physical Intelligence's published roadmap
- [Lit audit (2026-05-17)](project_lit_audit_2026_05_17.md) — competitor positioning at time of MPPI signal

## Conventions

- Filenames suffixed with `_YYYY_MM_DD` are dated to indicate when the finding
  was made. Files without dates are typically standing documents
  (roadmaps, plans, conventions).
- Each finding doc opens with a YAML frontmatter block (`name`,
  `description`, `type`) — leftover from the source memory system, harmless
  to read.
- When findings conflict, prefer the more recent file. The consolidated
  findings doc is the most recent and explicitly notes which earlier
  claims it supersedes.
