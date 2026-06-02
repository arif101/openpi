# Our VLA — architecture spec

**One line:** a goal-conditioned action policy wrapped in a surprise-gated control loop
driven by a two-channel predictive head. Not a bigger feedforward map — a reactive policy
that knows when it is failing and invokes a remedy.

## Components

| Part | Toy instantiation | Real-VLA counterpart |
|---|---|---|
| **Perception** | JEPA encoder: image → z (shared, frozen) | SigLIP/DINO + tokenizer |
| **Goal** | target position g (2-d) | language → goal embedding |
| **Memory** | belief state b; surprise-gated writes; episodic k-NN | recurrent/SSM state + retrieval |
| **Policy (System 1)** | π(state, g) → action; small BC net | flow-matching action head |
| **Metacognition** | two predictive heads (below) | joint predictive-coding head |
| **Control flow** | surprise-gated tool-calls | tool tokens emitted by policy |

## The two-channel predictive head (proven complementary in M3)

- **World head** `ẑ' = P(z, a)` → world-surprise `‖ẑ' − z'‖`. Catches "the world did something
  impossible" (object anomalies). AUROC 1.000 on ball_jump; 0.556 (chance) on effector-stuck.
- **Self head** `d̂ = g(proprio, a)` → self-residual `‖d̂ − realized_effector_delta‖`. Catches
  "my action had no effect" (execution-stuck). AUROC 0.997 on pusher_stuck; 0.530 on ball_jump.

Neither subsumes the other. An action-conditioned world model alone MISSES execution-stuck.

## Control flow (surprise-gated)

```
each step:
  a = π(state, goal)
  step; observe realized effector delta + next obs
  s_self  = ‖self_head(proprio,a) − realized_delta‖
  s_world = ‖world_head(z,a) − z'‖
  if s_self sustained-high   → TOOL: retract-and-reapproach   (effector stuck)
  if s_world high            → memory.write(z,a,outcome); re-plan / re-perceive
```

## Discipline carried from prior results
- JEPA is a **monitor**, never optimized through (REASON 52→30 exploitability). Planning-through-
  the-model gated behind M5.
- Self-model conditioned only on what it can predict (the directly-actuated effector). The
  real-robot reachability field FAILED by excluding scene objects — (embodiment, scene) coupling.
- Tool-calls are scaffolding now; long-term they become learned policy outputs (tool tokens).
