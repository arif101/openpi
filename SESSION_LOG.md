# Contact MPC Research Session Log

_Started: April 11, 2026_
_Last updated: April 16, 2026_

---

## Mission

Make frozen VLAs work on tasks they weren't trained on, without catastrophic forgetting. Specifically: close the Pi0.5 memorization gap (96% on LIBERO-10, 30% on LIBERO-90) and improve robustness under perturbation (LIBERO-PRO).

---

## Key Results

### LIBERO-PRO Position Perturbation (the headline finding)

Pi0.5 on LIBERO-10 with 5cm object position shift:
- **Baseline (no perturbation): 94%**
- **Perturbed (5cm shift): 46%**
- **Drop: 48 percentage points**

Per-task breakdown:
| Task | Baseline | Perturbed | Drop |
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

**Conclusion:** Pi0.5 memorizes absolute spatial positions. Two tasks are naturally robust (zero drop), eight collapse. Tasks involving precise placement (mug on plate, items in basket) are most fragile.

### MPC Experiment Results

| Benchmark | Baseline (K=1) | MPC (K=8, contact) | Delta |
|---|---|---|---|
| LIBERO-10 | 93.3% | 96.7% | +3.3% |
| LIBERO-90 | 28.9% | 26.7% | -2.2% |

**Conclusion:** MPC with a learned value function helps on memorized tasks but hurts on unseen tasks. The value function memorized task identity (99.6% train accuracy, 66.1% held-out) rather than learning generalizable success features. Test-time search alone cannot close the memorization gap.

### Linear Probe (Representation Check)

- Linear probe on frozen VLM hidden states: **76.8% accuracy** classifying early vs late in episode
- Signal is linearly separable — hidden states encode task progress
- But pairwise ranking accuracy is much lower (64-66%) — insufficient for MPC candidate selection

### World Model

- Trained across 9 configs (3 sizes x 3 horizons)
- KS1 passes: 14-31% MSE improvement over no-change baseline
- KS3 fails: ranking accuracy 46-57% with linear probe scorer
- Diagnosis: the proxy scorer was the bottleneck, not the world model. Ground-truth future states also rank poorly with the linear probe.

---

## Infrastructure Built

### Code (in `src/openpi/contact_mpc/`)

```
contact_mpc/
├── features/
│   ├── extractor.py          # VLM feature extraction wrapper
│   ├── dataset.py            # (h_t, action_chunk, h_{t+H}) dataset builder
│   ├── probe.py              # Linear + MLP probes for representation check
│   ├── ranking_probe.py      # Bradley-Terry pairwise ranking probe
│   └── *_test.py             # Tests for each module
├── world_model/
│   ├── architecture.py       # Transformer with per-timestep action tokens
│   ├── train.py              # Training loop with horizon/size sweep
│   ├── evaluate.py           # KS1/KS2/KS3 kill switch evaluation
│   └── architecture_test.py  # 12 tests including permutation sensitivity
├── value_function/
│   ├── architecture.py       # 2-layer MLP with Bradley-Terry scoring
│   ├── pairwise_dataset.py   # Within-task pair construction
│   ├── train.py              # Training with early stopping
│   └── pairwise_dataset_test.py
├── sampling/                 # (placeholder for future diversity experiments)
├── planner/                  # (placeholder for MPC planner)
└── eval/                     # (placeholder for evaluation runner)
```

### Scripts

| Script | Purpose |
|---|---|
| `scripts/run_extract_features.py` | Extract VLM hidden states from LeRobot datasets |
| `scripts/run_train_world_model.py` | Train world model with size/horizon sweep |
| `scripts/run_ranking_probe.py` | Train pairwise ranking probe, re-evaluate world models |
| `scripts/run_collect_rollouts.py` | Collect Pi0.5 rollouts on LIBERO with hidden states |
| `scripts/run_train_value_function.py` | Train Bradley-Terry value function on rollout pairs |
| `scripts/run_mpc_experiment.py` | End-to-end MPC evaluation (baseline vs K=8) |
| `scripts/run_libero_pro.py` | LIBERO-PRO position perturbation evaluation |

### Model Changes

- `src/openpi/models/pi0.py`: Added `extract_vlm_features()` method and modified `sample_actions()` to always return `(actions, pooled_features)` from the prefix pass
- `src/openpi/policies/policy.py`: Updated `infer()` to unpack features tuple and return `vlm_features` in output dict

### Data on HuggingFace (`arif101/libero90_vlm_features`)

| File | Contents |
|---|---|
| `libero90_features_H10.npz` | 8,614 (h_t, action_chunk, h_{t+H}) triples from demo data |
| `best_probe.pkl` | Linear probe trained on demo features |
| `rollouts_libero_90.npz` | 450 Pi0.5 rollouts: 130 successes, 320 failures, 30,739 decision points |
| `libero_pro_libero_10_5.0cm.npz` | LIBERO-PRO results (94% → 46%) |

### Setup

- GPU setup gist: https://gist.github.com/arif101/d11940897b13130f1e782b3141c38cca
- `pyproject.toml` overrides: `av==11.0.0`, `chex==0.1.87` for compatibility
- LIBERO config: `~/.libero/config.yaml` auto-created by setup script
- `torch.load` monkey-patch for PyTorch 2.7 `weights_only` compatibility with LIBERO init states

---

## Key Technical Learnings

1. **JIT compilation matters enormously.** Un-JIT'd VLM forward pass: 8.9s. JIT'd: 0.07s. A 130x speedup from one `nnx_utils.module_jit()` call. This was the difference between a 1-hour rollout collection and a 90-hour one.

2. **`sample_actions` returns both actions and features for free.** The prefix VLM pass already computes hidden states — we just captured them instead of discarding. Zero overhead for feature extraction during rollout collection.

3. **JAX logging gets swallowed.** JAX/absl reconfigures the root logger on import. Use `logging.basicConfig(force=True)` after all imports, or use `print(flush=True)` for critical output.

4. **`uv sync` breaks on `av` package.** RunPod images ship ffmpeg 4-5, `av>=14` requires ffmpeg 7. Override to `av==11.0.0` in `pyproject.toml`. Also need `apt-get install pkg-config libavformat-dev ...` for source build.

5. **Value functions memorize task identity with limited data.** 19 tasks with both success and failure is not enough for generalization. Training accuracy 99.6%, held-out-task accuracy 66.1%. The within-task pair constraint prevents suite-level shortcuts but doesn't prevent task-level memorization.

6. **Mean-pooled VLM features encode task progress but not fine-grained ranking.** Linear probe at 76.8% (classification) but only 64% (pairwise ranking). Temporal progress within successful demos is detectable but not strongly rankable.

7. **Score spread across K=8 flow-matching candidates is tiny (0.03-0.06).** Rectified flow produces near-identical candidates from different noise seeds. World model predictions barely diverge. This limits MPC's ability to select meaningfully different candidates.

8. **Pi0.5 is more robust than Pi0 and OpenVLA on position perturbation.** Pi0 and OpenVLA drop to 0% on position shifts. Pi0.5 holds at 38-46%. This makes Pi0.5 the best starting point for robustness work.

---

## Literature Landscape (as of April 2026)

### LIBERO-90 SOTA
- Cosmos Policy: 98.5% (video diffusion world model)
- RLinf-VLA: 98.1% (PPO + process reward model)
- Pi0.5 baseline: 30% (without fine-tuning)

### LIBERO-PRO (robustness)
- All models collapse to 0-38% under 5cm position perturbation
- Nobody has published >50% on LIBERO-PRO
- This is the unsolved benchmark

### RL Fine-tuning of VLAs
- Simple Recipe (UT-Austin): PPO + LoRA, +42.6% OOD, <2% forgetting
- ReinFlow (NeurIPS 2025): converts flow matching ODE → SDE for RL compatibility
- πRL: flow-based VLA fine-tuning with Flow-Noise and Flow-SDE
- TGRPO: trajectory-wise GRPO for VLAs
- WoVR: world model + RL, LIBERO 39.9% → 69.2%

### Key Related Work
- SITCOM: MPC-style reranking of VLA outputs (closest to our MPC approach)
- Stellar VLA: continual learning via Dirichlet Process MoE routing
- FLARE (GR00T N1.5): training-time future latent prediction (not inference-time)
- LIBERO-PRO paper (arXiv:2510.03827): exposes memorization in all VLAs

---

## What Didn't Work and Why

| Approach | Result | Why it failed |
|---|---|---|
| Waypoint conditioning (earlier session) | 13.3% (worse than 30% baseline) | SFT on biased success data caused catastrophic forgetting |
| LoRA SFT on successes (earlier session) | 3.7% (down from 30%) | Trained on easy-task successes only, destroyed hard-task capability |
| MPC with world model + value function | 26.7% (down from 28.9%) | Value function memorized task identity, hurts on unseen tasks |
| Linear/MLP probe as KS3 scorer | 46-66% ranking accuracy | Probe trained for classification, not ranking; insufficient for MPC |

---

## Next Direction: LIBERO-PRO Robustness

### The Problem
Pi0.5 memorizes absolute spatial positions. 5cm object shift → 94% drops to 46%.

### Potential Approaches (in order of test speed)

1. **Relative action transform at inference** — Detect object position shift, translate all generated actions by the offset. Zero training. Tests whether the model's trajectories are correct but spatially offset.

2. **Object position injection into state** — Append target object xyz to the 7-dim state vector. Train a small projection layer (reuse `target_state_proj` infrastructure). Gives the model explicit spatial information it currently infers poorly from images.

3. **Visual prompting** — Draw attention markers on the input image at the target object's actual position. Zero training. Tests whether the VLM can be guided to attend to shifted objects.

4. **Prompt engineering with spatial hints** — Add "the object has moved left" to the text prompt. Tests whether the language pathway can compensate for the visual memorization.

5. **RL fine-tuning with position randomization** — PPO + LoRA via ReinFlow, training on LIBERO with randomized object positions. Most principled but most compute-intensive. Uses domain randomization to force learning of relative spatial reasoning.

### Target
Push LIBERO-PRO position perturbation from 46% to 70%+ while maintaining ≥90% without perturbation.

### YC Application
- Problem: "VLAs memorize — they score 94% in the lab, 46% when you move an object 5cm"
- Solution: robustness-first VLA adaptation
- Credibility: first model to break 70% on LIBERO-PRO position perturbation
- Product: "bring us your VLA, we make it robust for deployment"

---

## Files Modified in Pi0.5 (openpi fork, waypoint-conditioning branch)

| File | Change |
|---|---|
| `src/openpi/models/pi0.py` | Added `extract_vlm_features()`, modified `sample_actions()` to return features, added `target_state_proj` layer |
| `src/openpi/models/model.py` | Added `target_state` field to Observation dataclass |
| `src/openpi/policies/policy.py` | Updated `infer()` to unpack features tuple |
| `src/openpi/policies/libero_policy.py` | Added target_state pass-through |
| `src/openpi/transforms.py` | PadStatesAndActions also pads target_state |
| `src/openpi/training/weight_loaders.py` | Updated merge regex for target_state_proj |
| `src/openpi/training/config.py` | Added waypoint data config and training config |
| `pyproject.toml` | Added scikit-learn, pinned av==11.0.0, chex==0.1.87 |
