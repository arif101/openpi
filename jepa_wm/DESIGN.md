# JEPA-WM: a metacognitive world-model substrate for action

## Thesis

VLAs are System-1 feedforward maps (`obs → action`). System-2 deliberation requires a
world model. **But our own strongest negative result (REASON Phase 1: optimizing actions
through a learned model dropped task success 52%→30%) says a learned differentiable model
is exploitable by the optimizer.** So we adopt the world model in the role where it is
*not* optimized through — as a metacognitive **monitor** — and gate any planning use behind
an explicit non-exploitability test.

LeCun's argument, graded against our evidence:

| Claim | Verdict | Evidence |
|---|---|---|
| VLAs are System-1, need System-2 + world model | ✅ correct diagnosis | our feedforward-failure corpus |
| Generative pixel prediction → blurry averages → use JEPA latent space | ✅ correct | our generative contact-WM gave no clean signal |
| Inference = optimize actions through the model (MPC) | ⚠️ exploitable | REASON 52%→30% |

## Core design decision

The world model is a **JEPA**: an action-conditioned predictor of the *predictable,
relevant* part of the next observation, in an abstract representation space (not pixels).
We use it in two strictly separated roles:

- **Monitor (now):** predict ẑ_{t+1} = P(z_t, a_t); surprise ε = ‖ẑ_{t+1} − z_{t+1}‖.
  Never optimized through. Drives the three first-class citizens: memory writes, tool-calls.
  This generalizes our hand-coded ε-signature (AUROC 0.97) into a *learned* one — and is
  exactly V-JEPA's "surprise spikes at impossible events."
- **Planner (gated, later):** optimize a_{t:H} to minimize predicted cost. Only enabled
  after M5 proves *this* model is not exploitable. Default: OFF.

### Three first-class citizens, all riding the learned surprise signal
- **Memory** — working belief state written *only when surprised* (surprise-gated writes);
  episodic k-NN retrieval over (z, a, outcome) across rollouts.
- **Metacognition** — the JEPA surprise ε itself. Grounded in observable prediction error,
  NOT a self-confidence head reading hidden state (that compounds hallucination).
- **Tool-calling** — policy may emit a tool token instead of an action when surprise is
  sustained-high: recover / re-perceive / re-plan / ask. Folds the mechanism-dispatch
  wrapper INTO the model as a learned output.

## Milestone ladder (each gated by evidence, smallest-first)

- **M0** Mac feasibility: toy MuJoCo sim (low-DOF, 64×64) + random-policy collector.
  Gate: renders + collects at usable speed.
- **M1** JEPA learns without collapse. Gate: latent variance/rank high AND linear probe
  recovers object position from latents (V-JEPA-2 "3D from video" in miniature).
- **M2** Surprise detects injected anomalies (teleport / pass-through / frozen gravity).
  Gate: prediction-error AUROC ≫ 0.5.
- **M3** Action-conditioned surprise separates feasible vs infeasible actions.
- **M4** Closed loop: surprise → memory writes + tool-calls.
- **M5** Exploitability diagnostic (the REASON gate). Only then consider System-2 MPC.

## Hardware reality (M5 Pro, 24 GB unified)
- Sim: MuJoCo native CPU, offscreen CGL rendering — confirmed working.
- Model: PyTorch MPS. Keep trainable params tiny (≤ ~20 M for M0–M3). 24 GB caps us;
  freeze any pretrained encoder so it carries no optimizer state.
- JAX here is CPU-only (no Metal) — MJX parallel envs won't accelerate; use single env.
