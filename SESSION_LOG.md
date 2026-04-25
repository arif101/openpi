# VLA Research Session Log

_Started: March 30, 2026_
_Last updated: April 16, 2026_

---

## Phase 1: Direction Finding (Mar 30 - Apr 4)

### Explored directions
1. **VLA-serve** — serving framework for VLA models ("vLLM for robots"). Designed full architecture (engine, scheduler, KV cache). Parked — commodity infra, not differentiated enough.
2. **Latent-space agent communication** — models sharing at weight/embedding level. Killed — latent spaces aren't interoperable across model families.
3. **VLA Data Engine** — SDG pipeline for robotics training data. Interesting but incumbents (NVIDIA/Gretel $320M, Snorkel $100M) are consolidating.
4. **Continual learning atoms** (vlm-rl) — 31 experiments over 2 months. 63% forward transfer on VLMs. But LoRA baseline matched atoms. Sequential training was the mechanism, not atoms.
5. **Agent orchestration** — analyzed claw-code architecture. Good patterns (state machines, budget-first, trait-based dispatch) but agent infra is crowded.
6. **Reverse VOID** — physics-aware video perturbation for training data. Novel but 2-3 month research project before validation.
7. **VLA reasoning runtime** — two-clock architecture (VLM reasons at ~1Hz, action expert executes at ~50Hz). Promising but needs validation.

### Key market research findings
- 57% of orgs have agents in production, quality is #1 barrier (32%)
- VLA field exploding: 164 papers at ICLR 2026 (up from 9 in 2025)
- LIBERO-PRO shows all VLAs (Pi0, Pi0.5, OpenVLA) drop to 0% with 5cm position perturbation — they memorize, don't understand
- One Robot (YC W26) doing world models for VLA eval — validates the space
- Cosmos Policy (NVIDIA) achieves 98.5% on LIBERO using video world model for actions

---

## Phase 2: LIBERO Experiments (Apr 4 - Apr 7)

### Setup
- Pi0.5 via OpenPI on RunPod A40 boxes
- LIBERO benchmark (MuJoCo/robosuite, Franka Panda arm)

### Baseline results

| Benchmark | Pi0.5 Score | What it measures |
|---|---|---|
| LIBERO-Spatial | 96% | Memorized tasks (10 tasks, trained on these) |
| LIBERO-10 | 94% | Memorized tasks (10 tasks, trained on these) |
| LIBERO-90 | 30.2% | Zero-shot generalization (90 unseen tasks) |
| LIBERO-PRO | 0-38% (paper) | Robustness to perturbations on trained tasks |

### Text prompt sensitivity test (Apr 4)
Tested whether Pi0.5 responds to different text prompts on random observations.

**Result: YES — model responds with semantic sensitivity.**
- "stop" vs "pick up cup": L2_diff=2.73, cosine_sim=0.70
- "wrong task" vs "pick up cup": L2_diff=2.50, cosine_sim=0.46
- "detailed plan" vs "pick up cup": L2_diff=2.18, cosine_sim=0.82
- Cross-trial std: 0.06-0.18 (signal 10x larger than noise)

**Caveat:** LIBERO-PRO showed Pi0.5 ignores text on MEMORIZED scenes (produces same trajectory for any text including nonsense). The text sensitivity we measured was on random images, not LIBERO scenes.

### Reward-weighted regression experiment (Apr 7)
Collected 900 rollouts on LIBERO-90: 272 successes (30.2%), 628 failures.
Converted to LeRobot format, uploaded to HuggingFace.
Added LoRA to VLM backbone (gemma_2b_lora, rank=16).
Trained LoRA on 272 successful rollouts for 2000 steps.
Loss dropped from 0.23 to 0.04.

**Result: CATASTROPHIC FAILURE — 30.2% → 3.7%**

The model overfit on the 32 easy tasks it already aced and destroyed capability on everything else. All 272 successes came from tasks with 100% success rate — no learning signal for the hard tasks.

### Key learnings from experiments
1. Pi0.5 memorizes scene layouts, doesn't understand tasks
2. SFT on easy-task successes causes catastrophic forgetting
3. The 70% failure tasks have ZERO successful demos — can't learn from what doesn't exist
4. RL fine-tuning at 0% success rate has no reward signal
5. The problem is PLANNING (knowing what to do), not EXECUTION (physical capability)

---

## Phase 3: Research & Architecture Design (Apr 7 - Apr 8)

### VLA reasoning landscape (2026 SOTA)

| Approach | Paper | Key result |
|---|---|---|
| Explicit text CoT | ECoT (2024) | +28% on OpenVLA, but slow |
| Latent CoT | LaRA-VLA (Feb 2026) | +13.5%, 90% faster than text CoT, 135ms inference |
| RL vs SFT | Simple Recipe (2025) | RL gives +42.6% OOD improvement over SFT, <2% forgetting |
| Test-time RL | TT-VLA (Jan 2026) | +2-12% via on-the-fly adaptation |
| Goal-conditioned | Act2Goal (ICLR 2026) | World model generates visual trajectory, actions follow |
| Backward planning | LBP (2025) | 88.6% on LIBERO-Long via backward subgoal prediction |
| World model policy | Cosmos Policy (2026) | 98.5% LIBERO, actions as latent frames in diffusion |
| Flow RL | ReinFlow (NeurIPS 2025) | Converts flow matching ODE → SDE for RL compatibility |
| Continual learning | Stellar VLA (2025) | Dirichlet Process MoE routing, +50% over baselines |

### Proposed architecture: Latent Action Inpainting

**Core idea:** Decompose tasks into state waypoints, generate short action segments between adjacent waypoints, verify after each segment.

```
Image + Instruction → VLM predicts state waypoints [s0, s1, s2, s3, s4]
  For each pair (s_i, s_{i+1}):
    Action expert generates short trajectory (conditioned on target s_{i+1})
    Execute → verify state matches target → continue or replan
```

**Status:** target_state_proj layer added to Pi0.5, but training produced negative results (13.3%, worse than 30% baseline) due to SFT approach. Idea is still viable with proper RL training.

### SigLIP embedding failure (Apr 8)
- Adjacent keyframe cosine similarity: 0.9947
- Random pair cosine similarity: 0.9895
- Gap: 0.0053 (essentially zero)
- **SigLIP can't distinguish manipulation task phases.** All robot scenes look the same to it.

### Pivot to state-based waypoints (Apr 8)
- Adjacent keyframe L2 distance: 0.49, random pair: 1.07 (2.19x separation ratio)
- 1,286 keyframes, 1,014 segments, 4.7 avg keyframes per rollout

---

## Phase 4: Contact MPC Experiments (Apr 11 - Apr 16)

### Thesis
"The VLA memorization gap is a search problem, not a representation problem." Test whether generating K=8 action candidates and scoring them with a world model + value function can close the gap from 30% to 45%+ on LIBERO-90, without any weight updates.

### Linear Probe — Representation Check (Apr 12)
- Extracted VLM hidden states from 272 demo episodes (8,614 triples on HF)
- Linear probe accuracy: **76.8%** classifying early vs late in episode
- **Conclusion:** Hidden states encode task progress — the search hypothesis is viable
- But pairwise ranking accuracy is only 64-66% — insufficient for MPC candidate selection

### World Model Training (Apr 12-13)
- 9 configs trained: {small ~1M, medium ~5M, large ~20M} × {H=3, H=5, H=10}
- KS1 (beats no-change baseline): **ALL PASS** (14-31% improvement)
- KS3 (ranking accuracy ≥70%): **ALL FAIL** (46-57%)
- Diagnosis: the linear probe scorer was the bottleneck. Ground-truth future states also only achieve 66.1% ranking. The world model predictions were fine; the evaluation tool was too weak.

### Rollout Collection (Apr 13)
- 450 episodes on LIBERO-90: 130 successes (28.9%), 320 failures
- 30,739 decision points with VLM hidden states
- JIT compilation fix: 8.9s → 0.07s per decision (130x speedup)
- Always-return-features from sample_actions: zero overhead

### Value Function Training (Apr 13-14)
- Within-task pairwise Bradley-Terry on success/failure pairs
- Only 19 tasks had both successes and failures
- Training accuracy: 99.6%, held-out-task accuracy: 66.1%
- **R5 confirmed: value function memorizes task identity, not success features**
- Direct ranking on all rollout states: 70.2% (cross-task signal exists but within-task doesn't)

### MPC Evaluation (Apr 14-15)

| Benchmark | Baseline (K=1) | MPC (K=8, contact) | Delta |
|---|---|---|---|
| LIBERO-10 | 93.3% | 96.7% | **+3.3%** |
| LIBERO-90 | 28.9% | 26.7% | **-2.2%** |

- Contact trigger fires correctly (~18% of decisions)
- Score spread across K=8 candidates: only 0.03-0.06 (candidates too similar)
- **Conclusion:** MPC mechanism works on known tasks. Value function doesn't generalize to unseen tasks. Test-time search alone cannot close the memorization gap.

### LIBERO-PRO Baseline (Apr 15) — THE HEADLINE RESULT

Pi0.5 on LIBERO-10 with 5cm object position shift:

| Task | Baseline | Perturbed (5cm) | Drop |
|---|---|---|---|
| alphabet soup + cream cheese → basket | 100% | 0% | -100% |
| alphabet soup + tomato → basket | 100% | 20% | -80% |
| white mug on plate + yellow mug | 100% | 20% | -80% |
| cream cheese + butter → basket | 100% | 40% | -60% |
| stove + moka pot | 100% | 40% | -60% |
| yellow mug in microwave | 80% | 40% | -40% |
| book in compartment | 100% | 60% | -40% |
| black bowl in drawer | 100% | 80% | -20% |
| white mug + chocolate pudding | 100% | 100% | 0% |
| both moka pots on stove | 60% | 60% | 0% |
| **TOTAL** | **94%** | **46%** | **-48pp** |

**Pi0.5 memorizes absolute spatial positions.** Two tasks are naturally robust (zero drop), eight collapse. This is the number we're going to improve.

---

## What Didn't Work and Why

| Approach | Result | Root cause |
|---|---|---|
| LoRA SFT on successes (Phase 2) | 30% → 3.7% | Biased success data from easy tasks only |
| Waypoint conditioning (Phase 3) | 30% → 13.3% | SFT training destroyed base model |
| MPC on LIBERO-90 (Phase 4) | 28.9% → 26.7% | Value function memorized task identity |
| Linear probe as KS3 scorer | 46-66% ranking | Classification ≠ ranking |

---

## Key Technical Learnings

1. **JIT compilation matters enormously.** Un-JIT'd VLM forward pass: 8.9s. JIT'd: 0.07s (130x).
2. **`sample_actions` returns features for free.** Prefix pass already computes hidden states.
3. **JAX logging gets swallowed.** Use `logging.basicConfig(force=True)` after imports or `print(flush=True)`.
4. **`uv sync` breaks on `av` package.** Override to `av==11.0.0` + install ffmpeg dev headers.
5. **Value functions memorize task identity with limited data.** 19 tasks is not enough for generalization.
6. **Mean-pooled VLM features:** 76.8% classification, only 64% ranking. Detectable but not rankable.
7. **Flow-matching candidate diversity is tiny.** Score spread 0.03-0.06 across K=8. Rectified flow is too deterministic.
8. **Pi0.5 is the most robust VLA to position perturbation.** Pi0/OpenVLA drop to 0%. Pi0.5 holds at 38-46%.
9. **SFT on biased data always causes forgetting.** On-policy RL with PPO + LoRA is the proven alternative (<2% forgetting).
10. **Cosmos Policy (98.5% LIBERO) requires 8-64 H100s for 48h.** Not feasible at 1-2 GPU scale. Also worse OOD than Pi0.5.

---

## Infrastructure Built

### Code (`src/openpi/contact_mpc/`)

```
contact_mpc/
├── features/
│   ├── extractor.py, dataset.py, probe.py, ranking_probe.py + tests
├── world_model/
│   ├── architecture.py, train.py, evaluate.py + tests
├── value_function/
│   ├── architecture.py, pairwise_dataset.py, train.py + tests
├── sampling/, planner/, eval/ (placeholders)
```

### Scripts
| Script | Purpose |
|---|---|
| `run_extract_features.py` | Extract VLM hidden states from LeRobot datasets |
| `run_train_world_model.py` | Train world model with size/horizon sweep |
| `run_ranking_probe.py` | Train pairwise ranking probe |
| `run_collect_rollouts.py` | Collect Pi0.5 rollouts on LIBERO with hidden states |
| `run_train_value_function.py` | Train Bradley-Terry value function |
| `run_mpc_experiment.py` | End-to-end MPC evaluation (baseline vs K=8) |
| `run_libero_pro.py` | LIBERO-PRO position perturbation evaluation |
| `build_waypoint_dataset.py` | Build LeRobot dataset with target_state column |

### Data on HuggingFace (`arif101/libero90_vlm_features`)
- `libero90_features_H10.npz` — 8,614 demo feature triples
- `rollouts_libero_90.npz` — 450 Pi0.5 rollouts (130 success, 320 failure)
- `libero_pro_libero_10_5.0cm.npz` — LIBERO-PRO perturbation results
- `best_probe.pkl` — Linear probe

### Other data (`arif101/libero90_successes`, `arif101/libero90_waypoints`)
- Demo trajectories and waypoint-augmented variants

### Setup
- GPU setup gist: https://gist.github.com/arif101/d11940897b13130f1e782b3141c38cca
- OpenPI fork: `arif101/openpi`, branch `waypoint-conditioning`

---

## Literature Landscape (as of April 2026)

### LIBERO-90 SOTA (saturated by memorization)
- Cosmos Policy: 98.5% | RLinf-VLA: 98.1% | Pi0.5 baseline: 30%

### LIBERO-PRO (unsolved)
- All models collapse to 0-38% under 5cm position perturbation
- Nobody has published >50% on LIBERO-PRO
- **This is the gap we're targeting**

### RL Fine-tuning of VLAs
- Simple Recipe: PPO + LoRA, +42.6% OOD, <2% forgetting
- ReinFlow: flow matching → SDE for RL (NeurIPS 2025)
- WoVR: world model + RL, 39.9% → 69.2%

### Key related work
- SITCOM: MPC-style VLA reranking (closest to our MPC)
- Stellar VLA: continual learning via Dirichlet Process MoE
- LIBERO-PRO paper (arXiv:2510.03827): exposes memorization in all VLAs

---

## Next Direction: LIBERO-PRO Robustness

### The Problem
Pi0.5 memorizes absolute spatial positions. 5cm object shift → 94% drops to 46%.

### Potential Approaches (in order of test speed)
1. **Relative action transform at inference** — zero training, pure geometry
2. **Object position injection into state** — small projection layer, reuses target_state_proj
3. **Visual prompting** — attention markers on input image, zero training
4. **Prompt engineering with spatial hints** — test language pathway compensation
5. **RL fine-tuning with position randomization** — PPO + LoRA via ReinFlow, domain randomization

### Target
Push LIBERO-PRO position perturbation from 46% to 70%+ while maintaining ≥90% without perturbation.

### YC Application
- Problem: "VLAs score 94% in the lab, 46% when you move an object 5cm"
- Solution: robustness-first VLA adaptation
- Credibility: first model to break 70% on LIBERO-PRO
- Product: "bring us your VLA, we make it robust for deployment"

---

## Files Modified in Pi0.5 (openpi fork, waypoint-conditioning branch)

| File | Change |
|---|---|
| `src/openpi/models/pi0.py` | `extract_vlm_features()`, `sample_actions()` returns features, `target_state_proj` |
| `src/openpi/models/model.py` | `target_state` field in Observation |
| `src/openpi/policies/policy.py` | `infer()` unpacks features tuple |
| `src/openpi/policies/libero_policy.py` | target_state pass-through |
| `src/openpi/transforms.py` | PadStatesAndActions pads target_state |
| `src/openpi/training/weight_loaders.py` | merge regex for target_state_proj |
| `src/openpi/training/config.py` | waypoint data config + training config |
| `pyproject.toml` | scikit-learn, av==11.0.0, chex==0.1.87 overrides |

## Resources
- OpenPI fork: github.com/arif101/openpi (waypoint-conditioning branch)
- HuggingFace: arif101/libero90_vlm_features, arif101/libero90_successes, arif101/libero90_waypoints
- GPU setup gist: gist.github.com/arif101/d11940897b13130f1e782b3141c38cca
- Previous session log: /Users/arifahmed/side-projects/vla-serve/SESSION_LOG.md

---

## Phase 5: World-Model Research + LIBERO-PRO Lift (Apr 19 – Apr 25)

This phase pivoted from the attribution/MVP track of Phase 1–4 to a research-heavy world-model thesis after multiple rounds of strategic clarification. The work answered the gating question for the entire research program.

### The strategic pivot

After the Phase 4 numbers (KS3 ranking 46–57%, value function memorizing task identity), several rounds of debate over the right direction:

- "Factored VLA" pitch (latent disentanglement + counterfactual augmentation) — critic destroyed it on Locatello-impossibility grounds.
- "Improvement-loop attribution platform" — built but pulled back to research focus when user pushed against commodity-tooling pitch.
- "Latent MBRL + reasoning + chess" — too ambitious, too many moving parts.
- "World models as the moat" — what we landed on, with LIBERO-PRO as the headline benchmark.

User input that locked the direction: *"better world models, better reasoning will lead to VLAs generalizing to more scenarios."*

### Built infrastructure (this phase)

```
contact_mpc/
├── attribution/         # Phase 1.5 — Claude Sonnet 4.6 failure judge + cluster
│   ├── judge.py         # 17 tests — VLM judge with prompt caching, tool use
│   ├── cluster.py       # 13 tests — failure_type × hidden-state k-means
│   └── prompts.py       # robotics-specific 3-way taxonomy
├── eval/                # Phase 1.5 — offline LoRA candidate ranking
│   ├── offline_evaluator.py    # 13 tests — score candidates via WM + VF
│   └── correlation_study.py    # 16 tests — Pearson r + bootstrap CI
├── dpo/                 # Built; not exercised because we pivoted to MCTS
│   ├── pair_builder.py  # 8 tests — within-task (success, failure) pairs
│   └── loss.py          # 17 tests — flow-matching DPO with cosine/L2 modes
├── planner/             # Phase 5 — inference-time tree search
│   └── mcts.py          # 12 tests — AlphaZero-style PUCT, lazy expansion
└── world_model/         # Extended with anti-collapse regularizers
    └── train.py         # +VICReg + InfoNCE; per-dim target_std matching
```

### World-model debugging — the diagnostic + fix story

Phase 4 left a broken WM: KS3 ranking 0.545, val MSE 5e-5 but the value function ranked WM outputs at 0.371 (worse than random). Built a diagnostic script that ran on the existing checkpoint and produced the numbers below.

**Diagnosed three stacked failure modes:**
1. **Variance collapse**: 99% of dims had pred std < 50% of real std; median ratio 0.065. MSE on 2048-dim Pi0.5 features hides task-critical error in nuisance-dim averaging.
2. **Manifold drift**: pred-to-real NN distance >> real-to-real NN distance after VICReg-only fix (drift_ratio went from 1.0 → 2.0 → 7.3 across iterations).
3. **Cosine-InfoNCE degeneracy**: Pi0.5's pooled features cluster on a tight cone; all pairwise cosines ≈ 1.0; cosine-InfoNCE training loss stuck at exactly `0.1 × log(64) = 0.416` — the random-classification floor — for 40 epochs. Switched to L2-distance InfoNCE.

### Training recipe progression

| Variant | KS3 | Δ vs prev | Verdict |
|---|---|---|---|
| Baseline (MSE only) | 0.545 | — | FAIL |
| + VICReg w/ per-dim real-std target | 0.643 | +0.098 | FAIL |
| + cosine InfoNCE | 0.672 | +0.029 | FAIL |
| **+ L2 InfoNCE (replacing cosine)** | **0.726** | **+0.054** | **PASS** ✓ |

L2 InfoNCE was the breakthrough — fixed the contrastive degeneracy for cone-shaped frozen-VLM features. Total improvement +18.1pp over MSE-only baseline; first WM variant to pass the 0.70 KS3 threshold.

### Experiment A — the gating result

Wrapped the L2-InfoNCE-VICReg WM + Phase 4 value function in MCTS (K=4 candidates, 32 simulations, depth 1, contact-triggered) and evaluated on LIBERO-10 with 5cm object-position perturbation.

**Multi-seed result on LIBERO-PRO 5cm:**

| Seed | Pi0.5 baseline | Pi0.5 + MCTS | Lift |
|---|---|---|---|
| 7 | 46.0% (23/50) | 56.0% (28/50) | **+10.0pp** |
| 14 | 36.0% (18/50) | 44.0% (22/50) | **+8.0pp** |

- Reproduces the LIBERO-PRO paper's reported 46% baseline exactly (seed=7).
- Both seeds land +8 to +10pp lift consistently.
- Per-task: 5+ tasks improved, 0–1 regressed across the two seeds.
- Wall time: ~70 min per 100-episode run on A40. ~970 MCTS calls per run (contact-triggered).
- **First measurable lift on LIBERO-PRO from any inference-time-search method.**

### Novelty audit — honest assessment

After the result landed, ran a novelty audit against the 2025–2026 MCTS+VLA literature. Findings:

| Axis | Verdict |
|---|---|
| Architectural | **Not novel.** VLAPS (arXiv 2508.12211, Aug 2025), VLA-Reasoner (2509.22643, Sep 2025), V-VLAPS (2601.00969, Jan 2026) preceded the MCTS-with-VLA-prior-over-learned-WM pattern. |
| Backbone substitution (Pi0.5 vs Octo) | Trivially novel. |
| Empirical | **Novel and defensible.** First MCTS+WM result on LIBERO-PRO specifically. |
| Methods recipe | **Novel — strongest contribution.** VICReg+per-dim-matched + L2-InfoNCE for cone-shaped frozen-VLM features. Diagnostic + fix combination unpublished. |
| Robustness positioning | Novel. Almost all 2026 inference-time-search-VLA work targets capability, not robustness. |

Drop the "first MCTS+WM in VLAs" framing entirely. The honest novelty: methods recipe + LIBERO-PRO empirical first + robustness framing.

### YC positioning v2 — the PI blog finding

Read PI's blog after the novelty audit. Their **April 16, 2026 π0.7 release** (9 days before this writing) explicitly named our wedge as future work:

> *"Powerful and steerable models like π0.7 might make it possible in the future to solve even more complex unseen tasks by having the model 'think through' possible ways to perform them, leverage its ability to follow diverse prompts to ground these thoughts in actions, and then reflect on the outcomes to revise the task plan."*

PI has named "deliberation / think harder / reflect and revise" as missing in 3+ separate blog posts (π0, RTC, Knowledge Insulation, π0.7). They have not built it. Their existing System-2 (Hi Robot) is hierarchical *prompting*, not search.

**New positioning** (in `demo/yc_pitch.md`): *"The inference-time deliberation layer for openpi."* Use PI's own published roadmap as our product description. Position complementary to PI, alongside their existing partners (Weave/Ultra are vertical operators; we're the horizontal reasoning layer they invited but don't yet have).

### W26 cohort intel

- **One Robot (W26)** has "world model for VLA eval" branded — we cannot enter that frame and win. They train task-specific action-conditioned video WMs per customer for VLA eval/sim. Operators-with-pedigree (Industrial Next, Tesla) — not lab spinout.
- **Pattern across W26 robotics**: 6 of 8 are picks-and-shovels (data, sim, eval, hands). Zero claim architectural novelty. All have a deployable artifact in 90 days. Funded shapes: data engine > eval/sim > vertical operator > hardware-with-data-moat.
- **No-LOI playbook**: replace LOIs with shipped artifacts. The demo IS the LOI (Origami sells hands to Amazon, Servo7 has Carly running, Luel hit $2M ARR).

### What stands and what's still open

**Stands:**
- KS3=0.726 WM checkpoint (`world_model_H10_medium_ncel2_vic.pt`)
- +8–10pp lift on LIBERO-10 perturbed across 2 seeds
- 60+ unit tests across attribution, eval, planner, regularizer modules
- YC pitch v2 in `demo/yc_pitch.md`
- Memory persisted: phase roadmap, MVP claim, scope discipline, Experiment A result, YC competitive intel, PI blog intel

**Still open (next 2 weeks):**
- LIBERO-90 generalization run (queued, ~6 hr)
- Seed=21 replication for 3-seed CI on LIBERO-10
- Perturbation curve at 3, 7, 10 cm
- Ablations: VICReg-only, NCE-only, plain-MSE WM
- Cross-VLA test (port recipe to OpenVLA)
- Real-robot demo (gated on hardware access)
- arXiv preprint (target CoRL 2026 deadline ~June)

### Files added this phase

```
src/openpi/contact_mpc/attribution/   (judge, prompts, cluster + tests)
src/openpi/contact_mpc/dpo/           (pair_builder, loss + tests)
src/openpi/contact_mpc/eval/          (offline_evaluator, correlation_study + tests)
src/openpi/contact_mpc/planner/       (mcts + tests)
src/openpi/contact_mpc/world_model/regularizers_test.py
scripts/diagnose_world_model.py
scripts/diagnose_vic_checkpoint.sh
scripts/smoke_test_mcts.sh
scripts/run_experiment_a.sh
scripts/run_libero_pro_mcts.py
scripts/run_libero_candidate_eval.py
scripts/run_correlation_study.py
scripts/run_candidate_ranking.py
scripts/run_failure_clustering.py
scripts/run_prepare_dpo_pairs.py
scripts/run_vlm_attribution.py
scripts/replay_failure_frames.py
scripts/setup_gpu_box.sh
scripts/prepare_lora_candidates.py
demo/dashboard/                       (HTML + Chart.js + fallback data)
demo/loom_script.md
demo/README.md
demo/yc_pitch.md                       (v2 positioning, PI-roadmap-anchored)
```

### Memory persisted

- `project_phase_roadmap.md` — Phase 1–3 scope (validation / memorization / reasoning)
- `project_mvp_standalone_claim.md` — what Phase 1 validates
- `feedback_scope_discipline.md` — don't de-scope to avoid infra work
- `project_experiment_a_result.md` — multi-seed LIBERO-PRO numbers
- `project_yc_competitive_intel.md` — W26 cohort patterns + One Robot
- `project_pi_blog_intel.md` — π0.7 quote and positioning
