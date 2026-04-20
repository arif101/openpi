# VLA Continuous Improvement Demo

End-to-end pipeline: Pi0.5 failure logs → attribution → cluster-targeted LoRA
candidates → latent-world-model offline evaluation → imagined-vs-real
validation → static dashboard for the Loom video.

## Quickstart — fastest path to a recordable demo

The dashboard at `demo/dashboard/index.html` renders with illustrative
fallback data when no real artifacts exist, so you can record a demo video
immediately while the pipeline runs in parallel. Swap in real data as each
stage completes.

```bash
# Open directly (Chrome handles local fetch() best)
open demo/dashboard/index.html

# Or serve with a local HTTP server (avoids some file:// fetch restrictions)
cd demo/dashboard && python3 -m http.server 8000
```

## Full pipeline (real artifacts)

All commands assume:
- You are on a GPU box with the openpi env activated (`uv sync` + LIBERO installed).
- `ANTHROPIC_API_KEY` is set.
- The baseline rollouts exist at `data/contact_mpc/rollouts/rollouts_libero_90.npz`
  (from a prior run of `scripts/run_collect_rollouts.py`).
- A trained world model and value function exist at
  `data/contact_mpc/mpc_results/world_model.pt` and
  `data/contact_mpc/value_function/value_function.pt` (from Phase 4).

### Step 1 — Replay failure frames (~1 hr)

```bash
PYTHONPATH=/workspace/openpi/src:/workspace/openpi/third_party/libero \
/workspace/openpi/.venv/bin/python -u scripts/replay_failure_frames.py \
    --rollouts data/contact_mpc/rollouts/rollouts_libero_90.npz \
    --output-dir data/contact_mpc/frames
```

Produces `data/contact_mpc/frames/ep_<id>/t<step>.jpg` + `metadata.json`.

### Step 2 — Run VLM attribution (~$3, ~15 min)

```bash
PYTHONPATH=src python3 scripts/run_vlm_attribution.py \
    --frames-dir data/contact_mpc/frames \
    --output data/contact_mpc/failure_taxonomy.json
```

### Step 3 — Cluster failures (seconds)

```bash
PYTHONPATH=src python3 scripts/run_failure_clustering.py \
    --taxonomy data/contact_mpc/failure_taxonomy.json \
    --rollouts data/contact_mpc/rollouts/rollouts_libero_90.npz \
    --output data/contact_mpc/clusters.json
```

### Step 4 — Prepare LoRA candidate manifests (seconds)

```bash
PYTHONPATH=src python3 scripts/prepare_lora_candidates.py \
    --clusters data/contact_mpc/clusters.json \
    --success-rollouts data/contact_mpc/rollouts/rollouts_libero_90.npz \
    --output-root data/contact_mpc/candidates
```

### Step 5 — Train LoRA candidates (3–4 hrs per candidate on A40)

Each candidate directory contains a `README.md` with the suggested
training command. Run them sequentially:

```bash
bash data/contact_mpc/candidates/train_all_candidates.sh
```

Outputs checkpoints at
`data/checkpoints/pi05_libero_waypoint_lora/lora_<cluster_id>/`.

### Step 6 — Collect held-out rollouts per candidate (~30 min each)

For each candidate, run `run_collect_rollouts.py` pointed at that LoRA's
checkpoint, restricted to a shared held-out task set (e.g., trial_idx
5–9 across all tasks):

```bash
for cid in planning_0 skill_0 perception_0; do
  PYTHONPATH=/workspace/openpi/src:/workspace/openpi/third_party/libero \
  /workspace/openpi/.venv/bin/python -u scripts/run_collect_rollouts.py \
      --checkpoint data/checkpoints/pi05_libero_waypoint_lora/lora_$cid \
      --config-name pi05_libero \
      --output-dir data/contact_mpc/candidates/$cid \
      --num-trials 5
done
```

### Step 7 — Rank candidates offline (~5 min)

```bash
PYTHONPATH=src python3 scripts/run_candidate_ranking.py \
    --world-model data/contact_mpc/mpc_results/world_model.pt \
    --value-function data/contact_mpc/value_function/value_function.pt \
    --candidate-rollouts \
        planning_0:data/contact_mpc/candidates/planning_0/rollouts_libero_90.npz \
        skill_0:data/contact_mpc/candidates/skill_0/rollouts_libero_90.npz \
        perception_0:data/contact_mpc/candidates/perception_0/rollouts_libero_90.npz \
    --output demo/dashboard/data/ranked_candidates.json
```

### Step 8 — Real LIBERO eval per candidate (~1 hr each)

```bash
for cid in planning_0 skill_0 perception_0; do
  PYTHONPATH=/workspace/openpi/src:/workspace/openpi/third_party/libero \
  /workspace/openpi/.venv/bin/python -u scripts/run_libero_candidate_eval.py \
      --checkpoint data/checkpoints/pi05_libero_waypoint_lora/lora_$cid \
      --candidate-name $cid \
      --trial-start-idx 5 \
      --task-ids 0,1,2,3,4,5,6,7,8,9 \
      --output data/contact_mpc/real_eval/$cid.json
done
```

### Step 9 — Correlation study (seconds)

```bash
PYTHONPATH=src python3 scripts/run_correlation_study.py \
    --world-model data/contact_mpc/mpc_results/world_model.pt \
    --value-function data/contact_mpc/value_function/value_function.pt \
    --candidate planning_0=data/contact_mpc/candidates/planning_0/rollouts_libero_90.npz=data/contact_mpc/real_eval/planning_0.json \
    --candidate skill_0=data/contact_mpc/candidates/skill_0/rollouts_libero_90.npz=data/contact_mpc/real_eval/skill_0.json \
    --candidate perception_0=data/contact_mpc/candidates/perception_0/rollouts_libero_90.npz=data/contact_mpc/real_eval/perception_0.json \
    --output-json demo/dashboard/data/correlation.json \
    --output-plot demo/dashboard/data/scatter_plot.png
```

### Step 10 — Copy final artifacts into dashboard/data

```bash
cp data/contact_mpc/failure_taxonomy.json demo/dashboard/data/
cp data/contact_mpc/clusters.json demo/dashboard/data/
# ranked_candidates.json and correlation.json are already there from steps 7/9.
```

### Step 11 — Record Loom

Follow `demo/loom_script.md`.

## File map

```
demo/
├── README.md              # this file
├── loom_script.md         # 2-min recording script
└── dashboard/
    ├── index.html         # single-page dashboard (opens in any browser)
    ├── app.js             # Chart.js rendering, fallback data
    └── data/              # pipeline artifacts (populated by scripts above)

scripts/
├── replay_failure_frames.py      # Step 1 — LIBERO frame replay
├── run_vlm_attribution.py        # Step 2 — Claude attribution judge
├── run_failure_clustering.py     # Step 3 — taxonomy + k-means clustering
├── prepare_lora_candidates.py    # Step 4 — training manifests
├── run_candidate_ranking.py      # Step 7 — offline world-model scoring
├── run_libero_candidate_eval.py  # Step 8 — per-candidate real eval
└── run_correlation_study.py      # Step 9 — imagined vs real Pearson r

src/openpi/contact_mpc/
├── attribution/
│   ├── judge.py           # Claude API wrapper with prompt caching
│   ├── prompts.py         # taxonomy system prompt + tool schema
│   ├── cluster.py         # failure clustering module
│   └── *_test.py          # unit tests
└── eval/
    ├── offline_evaluator.py    # score LoRA candidate via world model
    ├── correlation_study.py    # Pearson r + bootstrap CI
    └── *_test.py               # unit tests
```

## Running tests

```bash
PYTHONPATH=src python3 -m pytest \
    src/openpi/contact_mpc/attribution/ \
    src/openpi/contact_mpc/eval/ \
    --no-header --noconftest -v
```

60 tests covering attribution judge, cluster logic, offline evaluator, and
correlation study. No GPU, no API keys required.
