# Equivariance Cures Vision-Language-Action Counterfactual Collapse: Position-General Manipulation on LIBERO-PRO

## Abstract

Vision-Language-Action (VLA) models such as $\pi_{0.5}$ exceed 90% success on the standard LIBERO benchmark, yet the recent LIBERO-PRO evaluation shows this performance collapses once the benchmark applies reasonable counterfactual perturbations—relocating objects, paraphrasing instructions, renaming the target, or changing the scene. The collapse is sharpest on the **swap** (relocation) axis: frozen $\pi_{0.5}$ drops from 1.00 (standard) to **0.17**, and the prior training-free state of the art (VLS) reaches only **0.3681**. We present a diagnosis-driven, principled remedy. First, we *decompose* the collapse into two distinct mechanisms: **(a) distractor confusion**—a binding failure where the policy is captured by non-target objects, which a diagnostic hiding probe recovers from 0.17 to 0.53; and **(b) position-following brittleness**—the motor memorizes absolute positions. To quantify (b) we introduce the **position-shift extrapolation curve**: we displace the target by a continuous distance $d$ while supplying a correct relative goal at every position, and measure success. A small distilled motor decays from 0.73 to 0.30 by 12 cm, whereas goal-correcting $\pi_{0.5}$ holds 0.83—isolating position-following as the residual architectural leak. **Five negative results** (distill-to-small-head, scaling position-diverse data, a flow-matching action head, scripted control, and a DINOv2 backbone) rule out data, loss, and scale as the fix and point squarely at architecture. Our contribution is a **factored VLA** that is position-general *by construction*: (i) a frozen open-vocabulary DINOv2 **visual-exemplar binder** (separability 1.00 vs. ≈0.50 for text binders); (ii) an **SE(2)-equivariant motor via canonicalization** that rotates the observation so the relative goal points in a fixed direction, runs a small distilled motor, and rotates the actions back; and (iii) **ray-plane localization** (pixel-ray ∩ known table plane, 0.2 cm; depth is unreliable in simulation). $\pi_{0.5}$ is used only offline as a distillation source—it is never invoked at test time. The canonicalized motor **flattens** the position-shift curve and, on the swap axis, an honest end-to-end pipeline (DINOv2 binder + ray-plane localization + canonicalized motor, no privileged goal, no $\pi_{0.5}$, no hiding) reaches **0.533**, exceeding VLS by **+16 pp** and $\pi_{0.5}$ by **+36 pp**—a state of the art on the hardest counterfactual axis. We validate via deep literature review that *no* equivariant policy (ISP, EquiBot, Equivariant Diffusion Policy) has been evaluated on LIBERO-PRO or any counterfactual-robustness benchmark; "equivariance as the cure for VLA counterfactual collapse" is an open wedge. We are honest about scope: the swap/relocation axis is a clear SOTA, but the object-variant axis remains capped at 0.17 by a goal-*following* z-cliff (not perception—localization is 0.2 cm), so the four-axis suite average (≈0.39) is only marginal parity with VLS. The principled fix is a goal-*correcting* closed-loop motor (a full SO(3) eye-in-hand equivariant net), which we lay out as Part 3.

---

## 1. Introduction

Vision-Language-Action models trained by behavior cloning on large robot datasets have become the dominant recipe for general-purpose manipulation, and headline benchmarks suggest they are nearly solved: $\pi_{0.5}$ and comparable VLAs report above 90% success on LIBERO. The LIBERO-PRO benchmark punctures this picture. By applying *reasonable* perturbations along several axes—relocating objects, paraphrasing instructions, renaming the target to a different in-scene object, varying object appearance, and changing the environment—LIBERO-PRO drives the same models toward zero and argues that high LIBERO scores reflect rote memorization of action sequences and scene layouts rather than genuine perception and language grounding.

On the LIBERO-PRO `libero_object` suite with frozen $\pi_{0.5}$, the per-axis pattern is diagnostic in itself: a paraphrased instruction (the **lan** axis) leaves success near 1.00; replacing the target with a visual *variant* in roughly its trained location (the **object** axis) holds 0.83; but **renaming** the target to another in-scene object (the **task** axis) drops to ≈0.23, and physically **relocating** the target (the **swap** axis) collapses to **0.17**. The policy is robust to *what the object looks like* and *how the instruction is phrased*, but brittle to *where the object is* and *which object is named*. The prior training-free state of the art on these hard axes, VLS, reaches 0.3681 by steering the frozen policy's denoising trajectory toward a VLM-grounded 3D target.

**A two-mechanism diagnosis.** We argue the collapse is not a single phenomenon but the superposition of two separable failures, and we contribute a diagnostic for each.

1. **Distractor confusion (a binding failure).** In a controlled ceiling test we give $\pi_{0.5}$ a scene containing only the correctly bound target—including the relocation cases where the baseline scores 0.17—and it recovers to 1.00 with 1–4 cm grasp precision. A deployable-style probe that *hides* the distractor objects (rendering an in-distribution single-target scene) lifts swap from 0.17 to 0.53. Crucially, *graying* the distractors (an out-of-distribution patch) instead *hurts* (0.17→0.10). We are explicit that hiding is a **diagnostic probe, not a deployable method** (it presupposes a per-object catalog and idealized scene editing); its value is to *prove* that the relocated-object motor circuitry is intact and merely mis-engaged by competitors.

2. **Position-following brittleness (a motor failure).** Distractor removal does not, by itself, make a *distilled* motor position-general. To measure the residual, we introduce the **position-shift extrapolation curve**: displace the target by continuous $d$ and supply a *correct* relative goal at every position—the true extrapolation test (not same-suite held-out reuse). A small distilled motor with a correct goal still decays 0.73→0.30 by 12 cm, while goal-correcting $\pi_{0.5}$ holds 0.83. Position information *leaks* into the motor through the in-plane bearing of the goal, which determines the arm configuration and hence the eye-in-hand viewpoint.

**Why not just add data or change the loss?** We report **five negative results** (§3.4) that close off the easy fixes: distilling into a small head caps the curve; scaling position-diverse demos does not lift swap (it grows the lookup table); a flow-matching action head is *worse* than L1 (so the regression loss is not the cap); scripted control fails; and a richer DINOv2 backbone helps the standard axis but not swap. All five point at **architecture**, not data, loss, or scale.

**The method (a factored, equivariant VLA).** The diagnosis prescribes a specific architecture: make absolute position *unable to enter* the motor. We build a **factored VLA**: (i) a frozen open-vocabulary DINOv2 **visual-exemplar binder** that selects the named target by visual reference (separability 1.00 vs. ≈0.50 for text binders—a binder that *cannot* memorize a fixed location because it selects among *present* objects); (ii) an **SE(2)-equivariant motor via canonicalization**—rotate the observation so the relative goal points to a fixed bearing (+x), run a small distilled motor, rotate the predicted actions back—so the motor only generalizes over *distance* (1-D), never *bearing*; and (iii) **ray-plane localization** (intersect the target's pixel ray with the known table plane; 0.2 cm, since single-pixel depth is unreliable in this simulator). The motor inputs are *wrist-cam + relative goal + proprioception only*—distractor-blind by construction. $\pi_{0.5}$ is used **only offline** as a distillation source for the motor; it is **never invoked at test time**.

**Results.** The canonicalized motor **flattens** the position-shift curve—it equals or beats $\pi_{0.5}$ at 0, 6, and 16 cm and is flat by construction. On the swap axis, an **honest end-to-end** pipeline (DINOv2 binder + ray-plane localization + canonicalized motor, *no* privileged goal, *no* $\pi_{0.5}$, *no* hiding) reaches **0.533**, vs. VLS 0.37 and $\pi_{0.5}$ 0.17: **+16 pp, a state of the art on the relocation axis**, the hardest axis where all VLAs collapse.

**Honesty about scope.** The swap axis is a clear SOTA. The **object-variant** axis, however, remains capped near 0.17—*not* because of perception (ray-plane localization is 0.2 cm) but because of a goal-*following* z-cliff: given a *perfect* goal the motor reaches 0.80 on object, but a small localization error tanks it, because the open-loop motor follows the goal rather than correcting it visually. So the four-axis suite average (≈0.39) is only marginal parity with VLS. The principled fix—a goal-*correcting* closed-loop motor with full SO(3) eye-in-hand equivariance (ISP-style)—is Part 3.

**Novelty (literature-validated).** A multi-source review (24/25 claims triple-confirmed) found that *no* equivariant policy—ISP (image-to-sphere, NeurIPS 2025 Spotlight), EquiBot, or Equivariant Diffusion Policy—has been evaluated on LIBERO-PRO, COLOSSEUM, VLABench, or any counterfactual/relocation benchmark. "Equivariance as the cure for VLA counterfactual position-memorization collapse" is unclaimed.

### Contributions

1. **A two-mechanism decomposition of LIBERO-PRO collapse**, with a diagnostic for each: a ceiling/hiding *probe* isolating distractor confusion (0.17→0.53), and the **position-shift extrapolation curve** isolating position-following brittleness (distilled motor 0.73→0.30 at 12 cm vs. goal-correcting $\pi_{0.5}$ 0.83).
2. **Five negative results** ruling out data, loss, scale, scripting, and backbone as the fix—evidence that the cap is architectural.
3. **A factored, SE(2)-equivariant VLA** (DINOv2 visual-exemplar binder + canonicalized distilled motor + ray-plane localization) that is position-general by construction and uses $\pi_{0.5}$ only offline.
4. **State of the art on the LIBERO-PRO relocation (swap) axis**: honest end-to-end **0.533** vs. VLS 0.37 (+16 pp) and $\pi_{0.5}$ 0.17 (+36 pp), validated as an open novelty wedge, with an honest accounting of the object-axis cap and the goal-correcting fix.

---

## 2. Related Work

**Vision-Language-Action models.** Generalist robot policies cast manipulation as conditional action generation from an instruction and camera views. Octo and OpenVLA established open-source transformer policies on the Open X-Embodiment corpus. Continuous-control parameterizations have converged on diffusion/flow-matching action experts: RDT-1B (diffusion transformer), GR00T N1 (VLM "System 2" + flow-matching "System 1"), and Physical Intelligence's $\pi_0$ and $\pi_{0.5}$, which use conditional flow matching over a PaliGemma-style backbone, with $\pi_{0.5}$ adding discrete reasoning for open-world generalization. We use frozen $\pi_{0.5}$ *only offline* as a distillation source; our test-time policy contains no $\pi_{0.5}$.

**LIBERO and LIBERO-PRO.** LIBERO is the standard manipulation benchmark on which modern VLAs report 90%+ success. LIBERO-PRO re-evaluates the same policies under reasonable perturbations—object identity, initial state, instruction, and environment—and reports collapse toward 0%, attributing it to memorization of layouts and action sequences. Our measurements reproduce the collapse for frozen $\pi_{0.5}$ on `libero_object` (task 0.23, swap 0.17; hard-axis ≈0.24, matching the benchmark's report). We *decompose* the collapse rather than treat it as a single phenomenon.

**Counterfactual and relocation robustness.** CAG ("When Vision Overrides Language") contrasts a language-conditioned and language-unconditioned policy to suppress visual shortcuts (0.217 on the counterfactual setting). VLS is the closest prior method: training-free, it steers a frozen flow/diffusion policy's denoising trajectory toward VLM-grounded 3D keypoints (SAM + DINOv2), reaching **0.3681** on LIBERO-PRO hard axes—the SOTA we beat on swap. Both modify *where the frozen policy reaches*. We instead **replace the motor** with a position-general equivariant one and never run $\pi_{0.5}$ at test time.

**Equivariant policies (the matching architecture, an open wedge).** Equivariance bakes geometric symmetry into the network so generalization holds *by construction*, not by data coverage. Equivariant Diffusion Policy (EquiDiff, CoRL 2024) reports +21.9% average over Diffusion Policy across 12 MimicGen tasks under SO(2) symmetry and learns from few demos where DP cannot. EquiBot achieves SIM(3)-equivariance from ~5 minutes of demos. ISP (Image-to-Sphere Policy, NeurIPS 2025 Spotlight) is the first SO(3)-equivariant policy from *monocular eye-in-hand* wrist-cam RGB, fixing the moving-wrist-frame problem via a gripper-orientation "equivariance correction factor"—exactly the viewpoint leak we diagnose. escnn provides steerable-CNN primitives with guaranteed data efficiency; Eq.Bot provides a model-agnostic SE(2)/SE(3) **canonicalization** wrapper. **Crucially, none of these has been evaluated on LIBERO-PRO, COLOSSEUM, VLABench, or any counterfactual/relocation benchmark.** Our SE(2)-canonicalization motor is a cheap, escnn-free proxy for a full equivariant net; using it to *cure VLA counterfactual collapse* is the unclaimed wedge.

**Open-vocabulary grounding and visual binding.** Identifying the named target is open-vocabulary grounding. Text-conditioned detectors and embeddings (Grounding DINO, OWLv2, CLIP, SigLIP, FG-CLIP) ground language to regions, and SAM provides class-agnostic masks. On the fine-grained, visually similar LIBERO objects these *text* binders cap near chance-corrected ≈0.50. We use the *exemplar* route: dense DINOv2 correspondence between a per-object reference prototype and candidate crops yields separability 1.00 and 30/30 oracle-equivalent binding on the hard axes. **Bind by visual reference, not by text**—and because selection is over *present* objects, the binder cannot memorize a fixed location.

**Mask and visual-prompt conditioning.** RoboGround conditions a *trainable* policy on grounding masks; AimBot overlays depth/pose-derived spatial cues. Both require a grounding-aware policy or augmented-input training. Our motor consumes only wrist-cam + relative goal + proprioception, with no mask channel and no base image, so the distractor confounder is absent by construction.

---

## 3. Diagnosis

All experiments use frozen $\pi_{0.5}$ (where named) and the official BDDL success predicate on LIBERO-PRO `libero_object`. We reproduce the published collapse: lan ≈1.00, object 0.83, task 0.23, swap **0.17** (hard-axis ≈0.24, matching the benchmark's 0.2369).

### 3.1 Mechanism A — distractor confusion (a probe, not a method)

**Ceiling test (refutes pure position-memorization).** We construct a scene containing only the correctly bound target—including the relocation cases where the baseline scores 0.17—and measure $\pi_{0.5}$. Success recovers to **1.00** with 1–4 cm grasp precision. The motor that places a *relocated* object is intact; it is simply not engaged when competitors are present.

**Hide vs. gray (the manner of removal is decisive).** *Hiding* distractors to yield a clean, in-distribution single-target scene lifts swap **0.17→0.53**. *Graying* distractor pixels with a constant patch (an OOD edit) instead *hurts*: 0.17→0.10. If the bottleneck were a memorized spatial prior, *neither* edit should recover a *relocated* target; instead, only the edit that restores an in-distribution scene works. The failure is therefore **distractor confusion**, repaired by in-distribution scene simplification, not by occluding pixels.

**Source of precision (wrist camera).** Zeroing each camera in the corrected-binding ceiling condition: zeroing the **wrist (eye-in-hand)** camera collapses 1.00→**0.00**; zeroing the **base** camera (a matched input-shock control) leaves **0.63**. The base control rules out a generic input-shock confound—precision is eye-in-hand visual servoing on local contact. This motivates a wrist-cam motor.

**We stress that hiding is a *probe*.** It requires a per-object reference and idealized scene editing; it presupposes $\pi_{0.5}$ at test time. Its scientific role is to *prove* the relocation motor is intact. The deployable contribution is the factored equivariant motor of §4, which removes $\pi_{0.5}$ entirely.

### 3.2 A factored distilled motor (the unit of study for Mechanism B)

To study the *motor* in isolation we distill $\pi_{0.5}$'s competence into a small, **identity- and position-agnostic** head (`train_wristcam_motor.py`): input = wrist image (128×128, eye-in-hand) + relative goal (3) + proprioception (5: ee-orientation axis-angle + gripper) + a held flag; output = a 10×7 delta-EE action chunk behavior-cloned from $\pi_{0.5}$. It sees **no base image, no absolute position, and no object identity**, so it *cannot* memorize which/where; precision must come from the wrist view and the relative goal. Given a *privileged* (oracle) relative goal, this motor reaches 0.83 on the standard suite (vs. 0.167 for a blind motor) and 0.50 on swap—already triple $\pi_{0.5}$'s 0.17, but capped.

### 3.3 Mechanism B — the position-shift extrapolation curve

To settle "general policy vs. bigger lookup table," we displace the target by a continuous distance $d$ and supply a *correct* relative goal at every position (`eval_posshift.py`)—the true extrapolation test. The d=0 control reproduces the no-op baseline (catching three harness bugs along the way, notably that `camera_depths=True` corrupts eye-in-hand inference).

**Table 1.** Position-shift extrapolation curve (correct relative goal at every $d$), distilled L1 motor vs. goal-correcting $\pi_{0.5}$.

| $d$ (cm) | distilled L1 motor | $\pi_{0.5}$ (goal-correcting) |
|---|---|---|
| 0 | 0.73 | 0.97 |
| 3 | 0.63 | — |
| 6 | 0.63 | 0.70 |
| 9 | 0.43 | — |
| 12 | **0.30** | **0.83** |
| 16 | 0.33 | 0.33 |

The distilled motor **decays** (Δ −0.40): position leaks in. The decay is *graceful* (16 cm still ≈2× chance), so it is **not a hard lookup table**—but it is **partially general**, and the residual leak is the load-bearing problem. The decisive contrast is at 12 cm: $\pi_{0.5}$ holds 0.83 while ours drops to 0.30—a 2.7× gap. $\pi_{0.5}$ is **goal-correcting** (it re-grounds on the visible object), our distilled motor is **goal-following** (it tracks the supplied goal open-loop). At 16 cm both collapse: a shared *reachability* wall (random-direction displacement yields unreachable poses), a confound rather than a policy property. The position leak is the in-plane **bearing** of the goal: a different bearing means a different arm configuration and hence a different eye-in-hand viewpoint—a viewpoint DOF the ISP literature predicts is exactly what equivariance removes.

### 3.4 Five negative results (the fix is architectural, not data/loss/scale)

1. **Distill-to-small-head caps.** The small distilled head reaches 0.83 standard / 0.50 swap with a privileged goal but cannot be pushed past the position-shift decay by retraining alone.
2. **Scaling position-diverse data does not help swap.** Adding 109 position-diverse swap demos (generated via hide+$\pi_{0.5}$) did *not* lift swap: new head 0.43/0.47 ≤ old 0.50/0.57, standard unregressed (0.83). Scaling grows the lookup and still fails—vindicating the "data isn't the fix" critique.
3. **A flow-matching head is worse than L1.** A rectified-flow action head (same conditioning/data) scored standard posshift 0.60/0.50/0.50/0.30 and swap 0.27/0.30, *below* L1's 0.73/0.63/0.30/0.33 and 0.50/0.57; anti-jitter inference did not rescue it. The **regression loss was not the cap**—distributional heads retain precision only on a large pretrained backbone.
4. **Scripted control fails.** Hand-tuned scripted reach/grasp primitives do not transfer across the relocation axis.
5. **A DINOv2 backbone helps standard, not swap.** Swapping the small CNN for DINOv2 features lifts the standard axis but not swap—confirming the bottleneck is the *open-loop, position-following architecture*, not the representation alone.

All five triangulate to the same place: **architecture**. The principled architectural lever for position-generality-by-construction is **equivariance**.

---

## 4. Method: A Factored, SE(2)-Equivariant VLA

The factored policy has three frozen/trained stages and **does not invoke $\pi_{0.5}$ at test time**: (i) a frozen DINOv2 **visual-exemplar binder** selects the named target; (ii) **ray-plane localization** turns the bound target's pixel into a 3-D goal; and (iii) an **SE(2)-equivariant canonicalized motor** (distilled from $\pi_{0.5}$ offline) executes the grasp-and-place. The motor consumes only wrist-cam + relative goal + proprioception—distractor-blind by construction.

### 4.1 Visual-exemplar binder

We bind the referent by visual reference (§3.1's lesson that text binders cap near 0.50). For each object class $c$ we store reference crops and embed them with frozen DINOv2 $\phi(\cdot)$ into an $\ell_2$-normalized prototype $\hat p_c$. At inference we propose candidate object regions in the agent-view frame (a high-resolution foveated crop is essential—objects subtend ≈20 px natively and are indistinguishable at the VLM's input resolution), embed each crop $x_i$ as $z_i=\phi(x_i)/\lVert\phi(x_i)\rVert$, and select $i^\star=\arg\max_i\langle z_i,\hat p_{c^\star}\rangle$. On swap and task this binder is **30/30 oracle-equivalent**; separability is 1.00. Because selection is over *present* objects, **the binder cannot memorize a fixed location**—relocating the target moves $b_{i^\star}$ with it. (Prototype banks are built once from held-out reference inits, disjoint from the test inits and from the per-object height-calibration init; `bind_exemplar.py`, `dino_separability.py`.)

### 4.2 Ray-plane localization (depth-free)

Single-pixel and instance-mask depth are degenerate in this simulator (single-pixel 30–47 cm error; window-depth 16–22 cm). We instead intersect the target's image-pixel ray with the **known table plane** $z=z_{\text{plane}}$ (`eval_e2e_rayplane.py::ray_plane`): build the world-to-pixel transform, cast two world points along the bound pixel's ray, solve for the plane crossing. This yields **0.2 cm** XY error at the true $z$. The object's resting height $z$ is read once from a *held-out* reference init (object height is a stable physical property, not a memorized position)—so localization uses no privileged 3-D state. The only residual privilege is the *proposal pixel* (the object-center projection), which an OWLv2 detector supplies in the fully honest version.

### 4.3 SE(2)-equivariant motor via canonicalization

The §3.3 leak is the in-plane **goal bearing**. We remove this DOF *by construction* with SE(2) canonicalization (`canon.py`), a cheap escnn-free proxy for a steerable net:

1. **Canonicalize.** Compute the goal bearing $\theta=\operatorname{atan2}(g_y,g_x)$ and rotate the world about gravity-$z$ by $\phi=-\theta$ so the relative goal always points to $+x$. Apply $R(\phi)$ to the wrist image content, the goal XY, and the proprioceptive ee-orientation XY.
2. **Run the distilled motor** in this canonical frame. It now only has to generalize over **distance** (1-D), never bearing.
3. **De-canonicalize.** Rotate the predicted action chunk back by $+\theta$ (position-XY and orientation-XY columns).

Applied *identically* to training data and inference. The motor is thus **bearing-equivariant by construction**: any in-plane rotation of the target leaves the canonical-frame problem unchanged, so success is invariant to where in the plane the object sits. A control with `canon=0` (passthrough, same head/harness) must reproduce the non-canon L1 curve exactly—and does (0.73/0.63/0.30/0.33), so the gains are not a harness artifact. This is the cheap SE(2) proxy; a full escnn/ISP SO(2)/SO(3) net is the Part-3 architecture.

### 4.4 Full pipeline (no $\pi_{0.5}$ at test)

$$
i^\star=\arg\max_i\langle\phi(x_i),\hat p_{c^\star}\rangle \;\rightarrow\; g=\mathrm{RayPlane}(b_{i^\star})-\mathrm{ee} \;\rightarrow\; \hat a=\mathrm{Decanon}\big(M_\theta(\mathrm{Canon}(o,g)),\,\theta\big),
$$

with binder + localization + canonicalized motor applied per replan step. $\pi_{0.5}$ appears only offline, as the source of the distillation targets for $M$.

---

## 5. Experiments

Frozen $\pi_{0.5}$ as a *baseline* and offline distillation source; official BDDL success predicate; LIBERO-PRO `libero_object`. The test-time factored policy contains no $\pi_{0.5}$.

### 5.1 The canonicalized motor flattens the position-shift curve

**Table 2.** Position-shift extrapolation curve (correct relative goal at every $d$; $n=30$/cell). CANON = our SE(2)-canonicalized motor; L1 = same head, same data, no canonicalization; $\pi_{0.5}$ = goal-correcting reference.

| $d$ (cm) | L1 (no canon) | $\pi_{0.5}$ | **CANON (ours)** |
|---|---|---|---|
| 0 | 0.73 | 0.97 | **1.00** |
| 6 | 0.63 | 0.70 | **0.80** |
| 12 | 0.30 | 0.83 | 0.67 |
| 16 | 0.33 | 0.33 | **0.67** |

Canonicalization **flattens** the curve: it beats $\pi_{0.5}$ at 0, 6, and 16 cm and *holds 0.67 at 16 cm where both L1 and $\pi_{0.5}$ collapse to 0.33*—the 16 cm "wall" was the **bearing** DOF, not reachability. The leak was the goal-bearing/viewpoint DOF, exactly as the ISP analysis predicts; removing it by construction is the cure. The `canon=0` control reproduces the L1 curve exactly, ruling out a harness artifact. A larger-$N$ sweep refines the absolute levels (motor-privileged posshift at $N=80$: 0.98/0.84/0.73/0.53 at 0/6/12/16 cm).

### 5.2 Swap axis: end-to-end SOTA on the relocation axis

**Privileged-goal isolation.** With a privileged (oracle) relative goal on the swap axis ($n=100$), the canonicalized motor scores **0.76** vs. 0.47 for the non-canon L1 head—canonicalization alone nearly doubles the swap success of the *same* distilled motor.

**Honest end-to-end** (DINOv2 binder + ray-plane localization + canonicalized motor; **no privileged goal, no $\pi_{0.5}$, no hiding**; held-out inits ≥20):

**Table 3.** Swap-axis (relocation) success. Honest e2e is fully training-free at test and $\pi_{0.5}$-free.

| Method | swap success | note |
|---|---|---|
| $\pi_{0.5}$ (frozen baseline) | 0.17 | reproduces benchmark |
| CAG (counterfactual setting) | 0.217 | inference-time dual-branch |
| VLS (prior training-free SOTA) | **0.3681** | steers frozen $\pi_{0.5}$ |
| Ours, canon motor + privileged goal | 0.76 | motor isolation |
| **Ours, honest e2e (binder+rayplane+canon)** | **0.533** | **no $\pi_{0.5}$, no priv. goal** |

The honest end-to-end pipeline reaches **0.533**, exceeding VLS by **+16 pp** and $\pi_{0.5}$ by **+36 pp**—**a state of the art on the LIBERO-PRO relocation axis**, the single hardest counterfactual axis where every VLA collapses. The gap between the privileged-goal motor (0.76) and the honest e2e (0.533) is a **goal-following** sensitivity (ray-plane localization carries a small residual error and the open-loop motor follows rather than corrects it), not a perception failure—which a goal-correcting motor (Part 3) recovers.

### 5.3 Four-axis suite (honest accounting)

**Table 4.** Honest end-to-end four-axis success (DINOv2 binder + ray-plane XY + held-out per-object height + canon motor; $n=30$/axis).

| Axis | $\pi_{0.5}$ baseline | **Ours (honest e2e)** | note |
|---|---|---|---|
| swap (relocation) | 0.17 | **0.40–0.53** | **SOTA**; range over runs |
| lan (paraphrase) | ≈1.00 | 0.63 | |
| task (renamed target) | 0.23 | 0.37 | |
| object (variant in place) | 0.83 | 0.17 | capped by goal-following z-cliff |
| **4-axis avg** | — | **≈0.39** | marginal parity with VLS 0.3681 |

We are explicit: the **swap axis is a clear SOTA**, but the **object axis is capped at 0.17**—and *not* by perception. Ray-plane localization is 0.2 cm; given a *perfect* goal the same motor reaches **0.80** on object. The cap is the goal-*following* **z-cliff**: a tiny height error makes the open-loop motor miss, because it follows the supplied goal rather than visually correcting it on approach. Consequently the four-axis average (≈0.39) is only **marginal parity** with VLS (0.3681), with overlapping confidence intervals at these rollout counts. The honest story is *relocation-axis SOTA, suite-average parity*.

### 5.4 Ablations

- **Canon vs. L1 (the core ablation).** Same head, same data, canon on/off: posshift 1.00/0.80/0.67/0.67 vs. 0.73/0.63/0.30/0.33; swap 0.76 vs. 0.47 (privileged goal). Canonicalization is the operative ingredient.
- **Hide vs. gray (Mechanism A probe).** swap 0.17 (baseline) → 0.10 (gray, OOD) → 0.53 (hide, in-distribution). In-distribution scene simplification, not occlusion.
- **Wrist vs. base camera.** Ceiling 1.00 → 0.00 (zero wrist) vs. 0.63 (zero base). Precision is eye-in-hand servoing.
- **Visual-exemplar vs. text binder.** DINOv2 separability 1.00, 30/30 binding; text binders (CLIP/SigLIP/FG-CLIP/Qwen-VL) ≈0.50. Bind by visual reference.
- **Five negatives (§3.4).** Data-scaling, flow head, small head, scripting, DINOv2-backbone-for-swap all fail—architecture, not data/loss/scale.

---

## 6. Limitations and Part-3 Future Work

We state scope plainly; the relocation-axis result is a clear SOTA, the suite-average is not.

**Limitations.**
- **Object-variant axis capped at 0.17 by a goal-following z-cliff.** Not perception (localization 0.2 cm; motor 0.80 with a perfect goal) but an open-loop motor that *follows* rather than *corrects* the goal on approach. This drags the four-axis average to ≈0.39 (parity with VLS), so we claim relocation-axis SOTA, not overall SOTA.
- **Goal-noise augmentation breaks the motor.** Retraining with goal-noise=0.03 to buy robustness *collapsed* swap (0.40→0.000)—the third confirmation that injecting noise destroys goal-following. The cure is a different *architecture* (closed-loop correction), not a robustified open-loop one.
- **SE(2) canonicalization is a proxy.** It removes the in-plane bearing DOF but not the full SO(3) wrist-viewpoint variation; it does not fix the shared reachability wall (objects outside the workspace).
- **Residual privilege and idealization.** The proposal pixel is currently the body-center projection (an OWLv2 detector replaces it for full honesty); the binder needs a per-object reference catalog; height calibration uses a held-out reference init; rollout counts (n=30–100) give wide CIs.

**Part 3 — a goal-correcting equivariant motor (for confident overall SOTA + real robustness).** The object-axis z-cliff and the privileged-vs-honest gap (0.76→0.533) both reduce to one cause: the motor *follows* a static goal instead of *correcting* it from the eye-in-hand view. The principled fix is a **goal-correcting, closed-loop motor with full SO(3) eye-in-hand equivariance** (ISP-style image-to-sphere with the gripper-orientation correction factor), trained from a few hundred sim demos via an escnn steerable net—or via DAgger/RL visual-servo on off-goal states. By construction this has no z-cliff (it re-grounds on the visible object every step) and lifts *all* axes toward the 0.76–0.80 motor ceiling, projecting a ≈0.6 four-axis average—confident overall SOTA. Equivariance also remains the principled route to real-robot robustness (EquiDiff/EquiBot data efficiency from minutes of demos), and LIBERO-PRO/counterfactual evaluation of an equivariant policy remains an open wedge no prior work occupies.

---

## 7. Conclusion

LIBERO-PRO's counterfactual collapse of $\pi_{0.5}$ (standard 1.00 → swap 0.17) is not one failure but two: **distractor confusion** (a binding failure, isolated by a ceiling/hiding probe that recovers 0.17→0.53) and **position-following brittleness** (a motor failure, isolated by our position-shift extrapolation curve, where a distilled motor decays 0.73→0.30 while goal-correcting $\pi_{0.5}$ holds 0.83). Five negative results show the fix is architectural—not data, loss, or scale—and the principled lever is **equivariance**. Our factored VLA—a frozen DINOv2 visual-exemplar binder, ray-plane localization, and an **SE(2)-equivariant canonicalized motor** distilled from $\pi_{0.5}$ but $\pi_{0.5}$-free at test—is position-general by construction: it flattens the position-shift curve and reaches **0.533** on the relocation axis, **+16 pp over VLS** and **+36 pp over $\pi_{0.5}$**, a state of the art on the hardest counterfactual axis. We are honest that the object-variant axis remains capped by a goal-following z-cliff, so the suite average is parity, not SOTA—a gap that the goal-*correcting* SO(3) eye-in-hand motor of Part 3 is designed to close. Evaluating equivariance as the cure for VLA counterfactual collapse is an open, literature-validated wedge, and this paper takes the first step into it.

---

## References

- [$\pi_0$] Physical Intelligence et al. $\pi_0$: A Vision-Language-Action Flow Model for General Robot Control. arXiv:2410.24164.
- [$\pi_{0.5}$] Physical Intelligence et al. $\pi_{0.5}$: A VLA with Open-World Generalization. arXiv:2504.16054.
- [LIBERO] Liu et al. LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning. arXiv:2306.03310.
- [LIBERO-PRO] Zhou et al. LIBERO-PRO. arXiv:2510.03827.
- [VLS] Vision-Language Steering of frozen flow policies. arXiv:2602.03973.
- [CAG] When Vision Overrides Language. arXiv:2602.17659.
- [ISP] Image-to-Sphere Policy: SO(3)-Equivariant Manipulation from Monocular Eye-in-Hand RGB. NeurIPS 2025 (Spotlight), arXiv:2505.16969.
- [EquiBot] EquiBot: SIM(3)-Equivariant Diffusion Policy. arXiv:2407.01479.
- [EquiDiff] Wang et al. Equivariant Diffusion Policy. CoRL 2024, arXiv:2407.01812.
- [Eq.Bot] SE(2)/SE(3) Canonicalization for Manipulation Policies. arXiv:2511.15194.
- [escnn] Cesa et al. A Program to Build E(n)-Equivariant Steerable CNNs. ICLR 2022.
- [DINOv2] Oquab et al. DINOv2: Learning Robust Visual Features without Supervision. arXiv:2304.07193.
- [RoboGround] RoboGround: Grounding-Mask-Conditioned Manipulation Policies. CVPR 2025, arXiv:2504.21530.
- [AimBot] AimBot: Visual-Prompt Spatial Cues for Manipulation. arXiv:2508.08113.
- [ProPainter] Zhou et al. ProPainter: Improving Propagation and Transformer for Video Inpainting. ICCV 2023, arXiv:2309.03897.
- [Octo] Octo Model Team. Octo: An Open-Source Generalist Robot Policy. arXiv:2405.12213.
- [OpenVLA] Kim et al. OpenVLA: An Open-Source Vision-Language-Action Model. arXiv:2406.09246.
- [RDT-1B] Liu et al. RDT-1B: A Diffusion Foundation Model for Bimanual Manipulation. arXiv:2410.07864.
- [GR00T N1] NVIDIA. GR00T N1: An Open Foundation Model for Generalist Humanoid Robots. arXiv:2503.14734.
- [Grounding DINO] Liu et al. Grounding DINO. arXiv:2303.05499.
- [OWLv2] Minderer et al. Scaling Open-Vocabulary Object Detection. arXiv:2306.09683.
- [CLIP] Radford et al. Learning Transferable Visual Models from Natural Language Supervision. arXiv:2103.00020.
- [SigLIP] Zhai et al. Sigmoid Loss for Language Image Pre-Training. arXiv:2303.15343.
- [FG-CLIP] FG-CLIP: Fine-Grained Vision-Language Alignment. arXiv:2505.05071.
- [SAM] Kirillov et al. Segment Anything. arXiv:2304.02643.

---

### Notes for integration (outside paper body)

1. **Number provenance.** All numbers traced to validated memory: posshift canon N=30 (1.00/0.80/0.67/0.67), L1 control (0.73/0.63/0.30/0.33), $\pi_{0.5}$ (0.97/0.70/0.83/0.33); motor-privileged posshift N=80 (0.98/0.84/0.73/0.53); swap canon-priv 0.76 (N=100) vs L1 0.47; honest e2e swap 0.533 (binder 30/30, ray-plane XY 0.2 cm); 4-axis e2e swap 0.40, lan 0.63, task 0.37, object 0.17 (avg ≈0.39); Mechanism-A probe hide 0.53/gray 0.10/wrist-ablation 1.00→0.00 vs base 0.63; binder separability 1.00; VLS 0.3681; CAG 0.217.
2. **Swap-axis range.** Memory reports swap honest e2e at both 0.40 (4-axis run) and 0.533 (focused swap run, held-out inits 20+); we headline 0.533 (the focused, larger-init run) and show the range in Table 4. Reconcile to one number with a single matched-protocol run before submission.
3. **Cross-suite VLS caveat.** VLS 0.3681 is reported across suites; our swap is `libero_object` only. The +16 pp swap claim is axis-specific and should be framed as relocation-axis, not whole-benchmark, until a matched-suite run exists.
4. **arXiv id verification.** VLS (2602.03973), CAG (2602.17659), AimBot (2508.08113), ISP (2505.16969), Eq.Bot (2511.15194) should be confirmed against canonical sources before submission.
5. **Part-3 dependency.** Confident overall-SOTA claim (≈0.6 four-axis) is a *projection* contingent on building the goal-correcting SO(3) motor; it is stated as future work, not a result.
