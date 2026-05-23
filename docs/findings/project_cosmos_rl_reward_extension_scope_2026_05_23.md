---
name: Cosmos-RL reward-interface extension — scoping doc (2026-05-23)
description: Concrete engineering scope for extending Cosmos-RL's reward path to accept VLA rollout trajectory information. Code-traced against actual Cosmos-RL source. ~2-4 weeks for the 4-arm dense-reward GRPO ablation experiment.
type: project
originSessionId: 53e17618-aeb4-4007-9d29-d56bfc1900c3
---
## Current state of Cosmos-RL's VLA reward path (after code-trace)

The text-shaped `to_be_evaluated: str` interface in `cosmos_rl/dispatcher/algo/reward.py` is the LLM RL path — irrelevant for VLA. The VLA path is structurally separate:

```
cosmos_rl/rollout/vla_rollout/vla_rollout.py
  ├─ self.obs_keys = ["full_images", "wrist_images", "states"]
  └─ task_records[i] = {
       "complete": bool,            ← THE ONLY REWARD SIGNAL TODAY
       "finish_step": int,
       "input_ids": [...],          ← for re-tokenization
       "actions": [...],            ← commanded actions per step
       "full_images": [...],
       "states": [...]
     }
```

The reward used by GRPO is the binary `complete` flag from `env.check_success()` in `cosmos_rl/simulators/libero/venv.py:115`. Sparse, end-of-rollout, no per-step shaping.

**The architectural gap:** `task_records` does not capture privileged sim state (qpos, qvel, ee_pos, object_pos, contacts). The LIBERO env *has* this info — it just isn't logged through the rollout pipeline.

## Code-traced extension points

To plug in our 4-component dense reward, four files need changes:

### 1. `cosmos_rl/simulators/libero/env_wrapper.py` — expose privileged state per step

`LiberoEnvWrapper.step()` already returns `obs` dict via the venv. Extend the obs dict to include:

```python
obs["qpos"]        = sim.data.qpos.copy()        # (n_q,)
obs["qvel"]        = sim.data.qvel.copy()        # (n_qvel,)
obs["ee_pos"]      = sim.data.site_xpos[eef_id]  # (3,)
obs["object_pos"]  = ... per object              # (n_obj, 3)
obs["contact_flags"] = check_contact(robot, obj) per object  # (n_obj,)
```

Same for `venv.py`'s step handler (subprocess wrapper).

**Effort: 0.5 day.** Mechanical, copy from our own analysis scripts which already do this.

### 2. `cosmos_rl/rollout/vla_rollout/vla_rollout.py` — log privileged keys

```python
# line 137, currently:
self.obs_keys = ["full_images", "wrist_images", "states"]
# change to:
self.obs_keys = ["full_images", "wrist_images", "states"]
self.privileged_keys = ["qpos", "ee_pos", "object_pos", "contact_flags",
                        "commanded_action", "realized_ee_delta"]
```

Then in `task_records` init (line 365) and step-update loop (line 488), iterate over both obs and privileged keys.

The `commanded_action` and `realized_ee_delta` are derived (not directly from env):
- `commanded_action = vla_output["actions"][i]` — already in flow
- `realized_ee_delta = obs["ee_pos"][step] - obs["ee_pos"][step-1]` — derive at log time

**Effort: 0.5 day.**

### 3. `cosmos_rl/reward/dense_vla_rewards.py` (NEW) — 4 reward components

```python
def reward_tracking_error(task_record, lambda_=1.0):
    """Penalize commanded EE motion not realized by controller.
    Targets IN_TRUE_CLOSE_FALSE (execution stuck)."""
    cmd = np.array(task_record["commanded_action"])  # (T, 7)
    realized = np.array(task_record["realized_ee_delta"])  # (T, 3)
    cmd_mag = np.linalg.norm(cmd[:, :3], axis=1)
    real_mag = np.linalg.norm(realized, axis=1)
    gap = np.maximum(0.0, cmd_mag - real_mag * 10)  # scale factor empirical
    return -lambda_ * float(gap.sum())

def reward_grasp_in_contact(task_record, lambda_=1.0):
    """Penalize gripper-closed commands without object contact.
    Targets IN_FALSE / GRASP_TOO_HIGH (depth-perception error)."""
    grip = np.array(task_record["commanded_action"])[:, 6]  # (T,)
    contacts = np.array(task_record["contact_flags"])  # (T, n_obj) — robot-object contact
    any_contact = contacts.any(axis=1)
    grip_closed = grip > 0.5
    violations = grip_closed & ~any_contact
    return -lambda_ * float(violations.sum())

def reward_approach_progress(task_record, lambda_=1.0, target_obj_idx=0):
    """Reward EE getting closer to target object over rollout.
    Targets IN_FALSE / APPROACH_WILDLY_OFF (perception lost)."""
    ee = np.array(task_record["ee_pos"])              # (T, 3)
    target = np.array(task_record["object_pos"])[:, target_obj_idx]  # (T, 3)
    dist = np.linalg.norm(ee - target, axis=1)
    initial = dist[0]
    minimum = dist.min()
    progress = max(0.0, initial - minimum)
    return lambda_ * float(progress)

def reward_task_success(task_record):
    """Sparse: 1.0 if complete else 0.0."""
    return 1.0 if task_record["complete"] else 0.0
```

**Effort: 1 day** including unit tests on existing trace data.

### 4. `cosmos_rl/rollout/vla_rollout/vla_rollout.py` — invoke dense rewards in `pack_trajectory`

Currently `pack_trajectory` (line 617) uses `success_rates[payload_idx]` derived from binary `complete`. Replace with the composed reward:

```python
from cosmos_rl.reward.dense_vla_rewards import (
    reward_task_success, reward_tracking_error,
    reward_grasp_in_contact, reward_approach_progress,
)

def compose_reward(record, variant):
    r = reward_task_success(record)
    if variant in ("B", "C", "D"):
        r += reward_tracking_error(record, lambda_=self.config.reward.lambda_tracking)
    if variant in ("C", "D"):
        r += reward_grasp_in_contact(record, lambda_=self.config.reward.lambda_grasp)
    if variant == "D":
        r += reward_approach_progress(record, lambda_=self.config.reward.lambda_approach)
    return r
```

Wire `variant` and `lambda_*` through `config.reward.*`. GRPO advantage computation in `RolloutGroup.compute_rollouts` operates on the composed scalar reward unchanged.

**Effort: 1 day.**

## Total scope estimate

| Component | Effort | Risk |
|---|---|---|
| Env-wrapper privileged state | 0.5 day | Low |
| Rollout pipeline logging | 0.5 day | Low |
| Dense reward functions | 1 day | Low (unit-testable offline) |
| GRPO integration | 1 day | Medium (config wiring) |
| End-to-end test on small N | 2 days | Medium (debugging cross-process issues) |
| 4-arm ablation run | 5-7 days GPU | High (compute cost) |
| Per-stratum eval + writeup | 2-3 days | Low |

**Total: 2-3 weeks of engineer time + 1 week of GPU time.** Realistic with debugging: 4 weeks.

## Risks and unknowns

1. **GRPO with dense reward might be less stable than with sparse.** Dense reward has different variance characteristics; GRPO's group-normalization may need re-tuning (temperature, epsilon bounds). Mitigation: start with arm B only, verify stability, then add C/D.

2. **The `task_records` dict could grow large** if we log full qpos/qvel arrays for 512-step rollouts × N parallel envs. ~MB per record × 100 records × N envs. Possibly memory-bound. Mitigation: store only needed fields, downsample if necessary.

3. **`λ` weights need tuning.** Each reward component has a scale; without tuning, one component dominates. Mitigation: brief grid search (3 values × 3 components × small N) before main ablation. 1 extra day.

4. **The grasp_in_contact reward depends on `check_contact()` being cheap to call per step.** MuJoCo contact queries are cheap but the multiprocessing subprocess wrapper adds overhead. Mitigation: profile early; if expensive, batch contact checks at the end of rollout (post-hoc trajectory analysis) rather than per-step.

5. **Upstream contribution back to NVIDIA.** The extension is generic (privileged state logging + dense reward composition) and useful beyond our specific reward. Worth filing as a PR to nvidia-cosmos/cosmos-rl. Adds 2-3 days for cleanup + PR process but gets us the credibility signal and gives back to the ecosystem.

## Sequencing recommendation

1. **Week 1:** Local fork. Extend env-wrapper + rollout logging. Smoke-test that logging works without changing reward semantics.
2. **Week 2:** Add dense reward functions. Validate offline on our existing 80-trace corpus (we have qpos, ee_pos, object_pos in the npz files; same data, different source).
3. **Week 3:** GRPO integration. Run arm A baseline; verify identical to vanilla Cosmos-RL behavior. Then arm B in single seed.
4. **Week 4:** Full 4-arm ablation, 3 seeds × 4 arms = 12 runs. Per-stratum evaluation. Writeup.

## What we need before week 1

- **GPU access reinstated.** Same box or new.
- **Confirmed Pi0.5 base checkpoint** — published openpi or NVIDIA-fork? Need to match what Cosmos-RL uses.
- **A decision on framing** (YC pitch vs paper vs collaborator engagement) — affects writeup target.
- **Cosmos-RL fork pushed** — branch on arif101/cosmos-rl or similar, so changes are reversible.

## What doesn't need solving before week 1

- Transfer-test target task identification. Can be done in parallel during weeks 2-3 (cheap GPU rollouts on task 9 or other tasks). Result feeds into week 4 evaluation.
