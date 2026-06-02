# JEPA-WM prototype — results log

Hardware: Apple M5 Pro, 24 GB. All runs on-device (MuJoCo CPU + PyTorch MPS).
Model: 1.12 M-param action-conditioned JEPA (frame-stacked, VICReg + EMA target).

## M0 — Mac feasibility ✅
Planar pusher + free ball, 64×64 top-down. ~1900 env-steps/s with rendering.
500 train + 100 val rollouts (×24 steps); 66% carry real ball motion (contact present).

## M1 — JEPA learns without collapse ✅
40 epochs, ~70 s on MPS.
- Anti-collapse: embedding std 1.04, effective rank 13→72 / 128 (collapse would be ~1). VICReg held.
- Structure: linear probe recovers ball position at **R² = 0.83–0.87** — the latent learned
  object location from pixels, unsupervised. V-JEPA-2's "structure from observation" in miniature.

## M2 — surprise vs injected anomalies — PARTIAL, and the partial is the point

Per-type AUROC (anomaly vs legal-contact, the hard test — not just "big motion"):

| Anomaly | What it violates | Observable top-down? | AUROC | Verdict |
|---|---|---|---|---|
| teleport | spatial continuity (in-plane jump) | yes | **0.997** | ✅ detected |
| ghost_push | motion **without** contact | yes | **0.888** | ✅ detected |
| pass_through | expected contact → **no** motion | yes (near no-op) | **0.043** | ❌ missed |

Also found and rejected as ill-posed for this sensor/dynamics:
- **levitate** (z-motion): invisible from a top-down camera. Sensor limit, not model limit.
- **freeze** (sudden stop): a non-event in quasi-static pushing — the ball stops anyway.

### The finding: world-surprise is asymmetric, and self-ε is its missing complement

The JEPA's observation-prediction surprise robustly flags **unexpected motion** (teleport, ghost)
but is **blind to expected-effect-absent** anomalies (pass_through). Verified physically: the
pass_through before/after frames are a near no-op, and the model — which only weakly predicts
contact→motion (that outcome is aleatorically uncertain in the data) — is unsurprised when the
ball stays put.

This independently re-derives, from a pure observation world-model, the necessity of the
**action-grounded ε-signal** we validated on the real robot (commanded-vs-realized end-effector
motion, AUROC 0.97 on the execution-stuck `IN_TRUE_CLOSE_FALSE` mode). pass_through *is* the toy
analog of execution-stuck, and the world model misses it for the same structural reason.

**Conclusion — a metacognitive VLA needs two orthogonal channels:**
1. **World-surprise** (JEPA observation prediction error): "is the world doing something impossible?"
   → catches teleport / ghost. Misses execution-stuck.
2. **Self-residual** (action-conditioned effector prediction error, our ε): "did my action have its
   intended effect?" → catches execution-stuck. Misses world-anomalies.

Neither subsumes the other. The predictive-coding head should predict **both** next-observation-latent
(world) and next-effector-state (self); surprise on either channel triggers metacognition.

## M3 — complementarity proven ✅

Two surgically-clean anomalies (each perturbs exactly one thing):
- **ball_jump** (OBJECT): pusher steps normally, then ball jumps.
- **pusher_stuck** (EFFECTOR): pusher commanded but doesn't move; ball normal. Faithful toy
  analog of execution-stuck (`IN_TRUE_CLOSE_FALSE`).

Two channels: **world-surprise** = JEPA latent prediction error; **self-residual** =
‖predicted pusher delta − realized pusher delta‖ (self-model fits normal data to 0.64mm).

AUROC (anomaly vs normal), 300 rollouts:

| channel | ball_jump (OBJECT) | pusher_stuck (EFFECTOR) |
|---|---|---|
| world-surprise (JEPA) | **1.000** | 0.556 |
| self-residual (ε) | 0.530 | **0.997** |

**Perfect diagonal.** Each channel fires on its own anomaly and is at chance on the other.

Key answer to the open question: **the action-conditioned world model does NOT catch
effector-stuck (0.556 ≈ chance).** A stationary pusher is a high-probability observation, so
the JEPA isn't surprised; the self-model, predicting only its own high-SNR motion, gets 0.997.
The self-channel is **necessary, not redundant** — and this reproduces, in a controlled toy,
the real-robot result (action-grounded ε caught execution-stuck at 0.97; a full observation
world-model would have missed it).

## Standing architecture claim (M0–M3 evidence)
A metacognitive VLA carries a **two-channel predictive head**: next-obs-latent (world) +
next-effector-state (self). Surprise on either triggers metacognition (memory write / tool-call).
World-surprise covers "the world did something impossible"; self-residual covers "my action had
no effect." Empirically orthogonal; jointly cover the anomaly space.

## M4 — closed loop built; first task mis-posed (honest null)

Built the full VLA: `SPEC.md` (architecture), `policy.py` (BC action head, our System-1 VLA,
trained MSE 3e-4), `eval_m4.py` (closed loop: self-residual stall detector → retract tool-call).
Mechanics work — recovery fired in 57/60 OOD episodes.

But the chosen task (precise ball-push-to-goal) has **no competent baseline**: the get-behind
expert itself scores 22% (min-dist) / 0% (final), median final distance 0.32 m — repositioning
"behind" the ball knocks it to a wall. Precise pushing is a manipulation-planning problem, not a
P-controller task. Its failures are **imprecision, not retract-recoverable stuck states**, so the
task cannot isolate the recovery claim. Bare = meta = 0% on OOD; correctly, retract doesn't fix
imprecision. **This is a task-design null, not an architecture result.** No number was manufactured.

### Fix: M4 needs a task whose dominant failure is a retract-recoverable effector-stuck
with a competent baseline. Candidate: **reach-with-obstacle** — pusher reaches a target
(baseline ~100%, trivial position control); an OOD immovable peg on the path causes an
effector-jam (the exact self-residual signal from M3); retract-and-go-around recovers. Cleanly
tests "acting on the self-residual recovers an effector-stuck a reactive policy can't" — our
real-robot claim. Trades object-manipulation flavor for a clean mechanism isolation.

## Vision encoder wired into the action head ✅
`vision_policy.py`: action head on JEPA latent z(image)+goal vs privileged state, held-out
action MSE 0.00048 vs 0.00037 (ratio 1.30). The frozen vision encoder is a sufficient
perceptual front-end — the policy acts from pixels, not ground-truth state. The 'V' is real.

## V-L-A accounting (honest)
- **V** (vision): ✅ JEPA encoder, now feeding the action head (above).
- **A** (action): toy single-step BC head. Real version = flow-matching chunk head (π0.5/SmolVLA).
- **L** (language/VLM): ❌ absent — "goal" is a coordinate. Comes from grafting onto a real small
  VLA (SmolVLA/PaliGemma) on GPU. We REUSE perception+language; we BUILD the metacognitive control
  architecture (two-channel predictive head + surprise-gated loop). That is the contribution.

## M4' — acting on surprise recovers failures ✅ (the closed loop works)

Reach-with-obstacle: reactive policy drives straight to a goal across an immovable box peg.

| condition | success |
|---|---|
| sanity (off-peg reach) | 48/48 = 100% (competent baseline) |
| obstacle, bare reactive | 0/48 = 0% (jams on peg every time) |
| obstacle, metacognitive | 44/48 = **92%** (recovery fired 48/48) |
| **net** | **+92pp** |

The full loop: self-residual detects the jam (the M3 signal, now in closed loop) → triggers
retract-and-go-around tool-call → recovers. This is our real-robot retract-and-reapproach claim
demonstrated end-to-end. 4 residual failures = detour side/horizon edge cases.

Debugging that mattered (recorded so it isn't re-hit): (1) self-model false-fires if the policy
commands far targets — OOD per-step motion; fix = bounded-velocity commands matching training.
(2) cylinder-cylinder coaxial contacts silently don't generate in MuJoCo — use a box. (3) a
non-origin body `pos` makes qpos ≠ world — keep the pusher body at the origin so the self-model
(trained in qpos=world) stays consistent.

## Status: metacognition validated end-to-end on the toy
Detect (M2/M3) → act (M4') closed. Two channels proven complementary; recovery works.
The architecture is de-risked enough to scale.

## Next
- Graft metacognition onto SmolVLA/π0.5 on GPU → the full V-L-A with language (where L comes from).
- M5: exploitability diagnostic before any planning-through-the-JEPA (the REASON gate).
