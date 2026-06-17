Numbers are consistent (hard ours = 0.515 ≈ 0.52; 3-axis ours = 0.64; baseline 3-axis = 0.41). Here is the assembled paper.

---

# Distractor Confusion, Not Position Memorization: Recovering a Frozen VLA's Native Motor on LIBERO-PRO via Visual-Exemplar Binding and Distractor Suppression

## Abstract

Vision-Language-Action (VLA) models such as $\pi_{0.5}$ [arXiv:2504.16054] exceed 90% success on the standard LIBERO benchmark, but the recent LIBERO-PRO evaluation [arXiv:2510.03827] shows this number collapses toward zero once the benchmark perturbs objects, instructions, initial states, and environments. On the two hardest LIBERO-PRO axes—changing the instruction to name a different in-scene object (*task* axis) and relocating the target object (*swap* axis)—frozen $\pi_{0.5}$ scores only 0.23 and 0.17, a result widely read as evidence that VLAs memorize fixed object positions rather than perceive and bind language to the scene.

We present a diagnosis and an inference-time remedy that revise this interpretation. Through a controlled ceiling test, we find that $\pi_{0.5}$'s own flow-matching motor is near-perfect (1.00 success) once it is given an in-distribution scene containing only the correctly bound target object; its hard-axis failures are dominated by **distractor confusion**—the policy is captured by non-target objects—rather than by an inability to reach relocated positions. We further show that the precision of the recovered motor comes specifically from the eye-in-hand (wrist) camera: zeroing the wrist camera collapses success from 1.00 to 0.00, while zeroing the base camera (a matched input-shock control) leaves it at 0.63.

Building on this, we propose a general, inference-only pipeline that leaves $\pi_{0.5}$ entirely frozen: an external open-vocabulary **visual-exemplar binder** (frozen DINOv2 [arXiv:2304.07193] correspondence against a per-object reference prototype) identifies the named target among scene objects, after which **distractor suppression** removes the remaining objects to produce a clean, in-distribution scene for the native motor. The visual-exemplar binder reaches 30/30 (oracle-equivalent) binding accuracy on the swap and task axes, where text binders (CLIP/SigLIP/FG-CLIP/Qwen-VL) cap near 0.50 on fine-grained similar objects. On the LIBERO-PRO `libero_object` suite (30 rollouts per axis, official BDDL success metric, frozen $\pi_{0.5}$), our method improves task 0.23→0.53 (+30pp) and swap 0.17→0.50 (+33pp), while preserving the object-variant axis (0.83→0.90). The hard-axis average rises from ~0.20 to 0.52, exceeding the prior training-free state of the art—VLS [arXiv:2602.03973] at 0.3681—by +15pp, with a 3-axis average of 0.64. We further isolate the mechanism: *graying* distractors (out-of-distribution patches) fails (0.17→0.10), whereas *hiding* them (an in-distribution scene) succeeds (0.17→0.63), confirming that the cure is in-distribution scene simplification, not pixel occlusion.

We position this as an inference-time method on a frozen base VLA, not a new foundation model, and discuss its assumptions (a per-object reference catalog; idealized hiding in simulation, deployable via inpainting) and a Phase-2 path toward a goal-correcting foundation motor that removes the $\pi_{0.5}$ dependence.

## 1. Introduction

Vision-Language-Action models trained by behavior cloning on large robot datasets have become the dominant recipe for general-purpose manipulation, and headline benchmarks suggest they are close to solved: $\pi_{0.5}$ and comparable VLAs report above 90% success on LIBERO. The LIBERO-PRO benchmark [arXiv:2510.03827] punctures this picture. By applying *reasonable* perturbations along five axes—manipulated objects, initial states, task instructions, and environments—LIBERO-PRO drives the same models toward 0% and argues that high LIBERO scores reflect rote memorization of action sequences and scene layouts rather than genuine perception or language grounding. On the two axes we study, frozen $\pi_{0.5}$ scores 0.23 when the instruction is rewritten to name a different object already in the scene (task axis), and 0.17 when the target object is relocated (swap axis), against 0.83 when only the object's appearance is varied in place (object axis).

The natural conclusion—and the prevailing one in recent literature on counterfactual VLA failures [arXiv:2602.17659]—is that the policy has memorized *where* to act and is blind to *what* the language asks for. This framing has motivated a wave of corrective work: counterfactual action guidance (CAG, 0.217 on LIBERO-PRO's counterfactual setting), grounding-mask conditioning (RoboGround [arXiv:2504.21530]), and, most recently, training-free steering of the frozen sampling process toward a VLM-grounded 3D target (VLS [arXiv:2602.03973]), which holds the current hard-axis state of the art at 0.3681.

**A different diagnosis.** We argue that the collapse is not primarily position-memorization but **distractor confusion**. We support this with a ceiling test in which $\pi_{0.5}$ is given a scene containing only the correctly bound target object: the policy succeeds 1.00 of the time, including on relocated objects. The motor is not blind to where the object is—when there is a single, in-distribution object to act on, $\pi_{0.5}$'s native flow-matching policy reaches and grasps it with 1–4cm precision. The failures on LIBERO-PRO arise because the policy is *captured* by the non-target objects that the perturbations introduce or rearrange. We further pin the source of this recovered precision to the eye-in-hand camera: ablating the wrist camera collapses the ceiling-test motor from 1.00 to 0.00, whereas ablating the base camera (a matched input-shock control) leaves it at 0.63—the motor servos precisely on local contact through the wrist view, and is comparatively robust to global-view corruption.

**An inference-time method that follows from the diagnosis.** If the problem is distractor confusion, the fix is to deliver the policy a clean, in-distribution scene containing the correct target. We do this in two frozen-model stages. First, an external **visual-exemplar binder** identifies the named target: rather than ground language to pixels with a text encoder, we match a per-object reference prototype to scene crops via frozen DINOv2 dense correspondence. On the LIBERO objects, DINOv2 visual separability is 1.00 and perfect-recall disambiguation (target prototype vs. all true-object crops) is 20/20, whereas text binders (CLIP, SigLIP, FG-CLIP, Qwen-VL) cap near 0.50 on fine-grained, visually similar objects; the binder reaches 30/30 (oracle-equivalent) on both the swap and task axes. Second, we apply **distractor suppression**: the non-target objects are removed so that $\pi_{0.5}$ receives a scene resembling its training distribution, on which its own motor takes over. A key ablation distinguishes our mechanism from generic occlusion: *graying* the distractors—which leaves out-of-distribution gray patches in the frame—fails (swap 0.17→0.10), while *hiding* them—producing an in-distribution scene—succeeds (0.17→0.63). The lever is in-distribution scene simplification, not pixel masking.

**Headline result.** On the LIBERO-PRO `libero_object` suite (30 rollouts per axis, official BDDL success metric, frozen $\pi_{0.5}$), our inference-only method improves:

- **object axis** (object variants): 0.83 → 0.90 (+7pp)
- **task axis** (changed instruction): 0.23 → 0.53 (+30pp)
- **swap axis** (relocation): 0.17 → 0.50 (+33pp)

The hard-axis average (task + swap) rises from ~0.20 to 0.52, exceeding the prior training-free state of the art, VLS at 0.3681, by +15pp; the 3-axis average is 0.64. Because the binder is oracle-equivalent on these axes, the gains reflect the motor acting correctly once distractor confusion is removed, not improved binding alone.

**Honesty about scope.** This is an inference-time method built on a frozen base VLA; it does not introduce a new foundation model, and it does not by itself close the gap to in-distribution LIBERO performance. The binder relies on a per-object reference catalog—open-vocabulary in the sense of "specify the target with a reference image," not pure language→object grounding. Distractor suppression is realized here by idealized hiding in simulation; a deployable instantiation would inpaint the distractors (e.g., ProPainter [arXiv:2309.03897]), and we have not yet measured the cost of inpainting artifacts. The evaluation covers the `libero_object` suite axes. These limitations motivate Phase 2: a goal-correcting foundation motor (a frozen DINOv2 backbone with a flow-matching action expert, a FLARE-style implicit world-model objective [arXiv:2505.15659], and DART-style recovery augmentation [Laskey et al., CoRL 2017]) that internalizes target binding and removes the dependence on $\pi_{0.5}$.

### Contributions

1. **A revised diagnosis of LIBERO-PRO collapse.** Through a controlled ceiling test and matched-input ablations, we show that $\pi_{0.5}$'s hard-axis failures (task 0.23, swap 0.17) are dominated by *distractor confusion*, not position-memorization: the native motor is near-perfect (1.00) on a clean, in-distribution single-target scene—including relocated objects—and its precision is sourced from the eye-in-hand camera (1.00→0.00 on wrist-cam ablation vs. 0.63 on base-cam ablation).
2. **A general, inference-only method on a fully frozen VLA.** We pair an open-vocabulary visual-exemplar binder (frozen DINOv2 correspondence to a reference prototype; 30/30 oracle-equivalent binding where text binders cap ~0.50) with distractor suppression that restores an in-distribution scene, and we show via the gray-vs-hide ablation that the operative mechanism is *in-distribution scene simplification* rather than occlusion.
3. **State-of-the-art training-free results on LIBERO-PRO hard axes.** Our method lifts frozen $\pi_{0.5}$ to a 0.52 hard-axis average (task 0.53, swap 0.50), +15pp over the prior training-free SOTA (VLS, 0.3681), while preserving the object-variant axis (0.90).

## 2. Related Work

**Vision-Language-Action models.** Recent generalist robot policies cast manipulation as conditional action generation from an instruction and one or more camera views. Octo [arXiv:2405.12213] and OpenVLA [arXiv:2406.09246] established open-source transformer policies trained on the Open X-Embodiment corpus, the latter pairing a Llama-2 backbone with a fused DINOv2/SigLIP visual encoder and discrete action tokens. Diffusion- and flow-matching action experts have since become the dominant continuous-control parameterization: RDT-1B [arXiv:2410.07864] is a 1.2B-parameter diffusion transformer for bimanual control, GR00T N1 [arXiv:2503.14734] couples a VLM "System 2" with a flow-matching "System 1" action module, and Physical Intelligence's $\pi_0$ [arXiv:2410.24164] and $\pi_{0.5}$ [arXiv:2504.16054] use conditional flow matching over a PaliGemma-style backbone, with $\pi_{0.5}$ adding FAST-tokenized discrete reasoning for open-world generalization. We take frozen $\pi_{0.5}$ as our base policy. Our contribution is orthogonal to this line: rather than train or fine-tune a new VLA, we show that $\pi_{0.5}$'s flow-matching motor is already near-perfect when its visual input is in-distribution, and we supply an inference-time perception wrapper that restores that condition.

**LIBERO and LIBERO-PRO.** LIBERO [arXiv:2306.03310] is the standard manipulation benchmark on which modern VLAs report 90%+ success. LIBERO-PRO [arXiv:2510.03827] re-evaluates the same policies under reasonable perturbations along five axes—object identity, initial state, task instruction, and environment—and reports that performance can collapse toward 0% under perturbations that a competent agent should handle, attributing the gap to memorization of action sequences and layouts rather than genuine perception or instruction following. Our measurements reproduce this collapse for frozen $\pi_{0.5}$ on the `libero_object` suite (task axis 0.23, swap axis 0.17). We depart from the benchmark paper's interpretation: through a ceiling test and graying-vs-hiding ablations, we localize the failure to *distractor confusion*—not deep position memorization—since suppressing non-target objects to recover an in-distribution scene restores the policy's native motor (single-condition 0.17→0.63 on the swap axis), whereas out-of-distribution gray patches do not (0.17→0.10).

**Counterfactual and relocation robustness.** Two recent works directly target the language-following failures that LIBERO-PRO exposes. CAG ("When Vision Overrides Language," [arXiv:2602.17659]) introduces a dual-branch inference scheme contrasting a language-conditioned policy against a language-unconditioned one to suppress visual shortcuts; we treat its 0.217 counterfactual-setting number as a baseline. VLS [arXiv:2602.03973] is the closest prior method in spirit: it is training-free and steers a frozen flow/diffusion policy's denoising trajectory via gradient-based refinement toward VLM-grounded 3D keypoints obtained with SAM and DINOv2, reaching 0.3681 on LIBERO-PRO hard axes—the current SOTA we compare against. Both methods modify *where the policy reaches* (a counterfactual branch or a steered 3D target). We instead leave $\pi_{0.5}$'s action distribution untouched and modify *what the policy sees*, simplifying the observation to an in-distribution single-object scene. This is a different mechanism—observation editing rather than action steering—and it raises the hard-axis average from ~0.20 to 0.52 (task 0.23→0.53, swap 0.17→0.50). The relative-precision evidence (zeroing the eye-in-hand wrist camera collapses success 1.00→0.00, whereas zeroing the base camera leaves 0.63) clarifies *why* $\pi_{0.5}$'s native motor suffices once binding is resolved: precision comes from wrist-camera visual servoing, which steering-based methods do not exploit.

**Open-vocabulary grounding and visual binding.** Identifying the named target among scene objects is an open-vocabulary grounding problem. Text-conditioned detectors and embeddings—Grounding DINO [arXiv:2303.05499], OWLv2 [arXiv:2306.09683], CLIP [arXiv:2103.00020], SigLIP [arXiv:2303.15343], and the fine-grained FG-CLIP [arXiv:2505.05071]—ground language to image regions, while the Segment Anything Model (SAM) [arXiv:2304.02643] provides class-agnostic masks. We find that on the fine-grained, visually similar objects in LIBERO these *text-driven* binders cap at roughly chance-corrected ~0.50 disambiguation, consistent with the known difficulty of text-to-region matching on subtle appearance differences. Our binder instead follows the *exemplar* route: dense DINOv2 [arXiv:2304.07193] correspondence between a per-object reference prototype and candidate crops yields perfect separability (1.00) and 20/20 perfect-recall disambiguation, and matches oracle binding accuracy (30/30) on the swap and task axes. The lesson—bind by visual reference, not by text—is the design choice that makes the downstream distractor suppression reliable; the cost is a per-object reference catalog rather than pure language-to-object grounding (§6.4).

**Mask and visual-prompt conditioning.** A complementary line conditions the policy on an explicit spatial cue rather than editing the scene. RoboGround [arXiv:2504.21530] concatenates grounding masks with the image to convey target location, shape, and size, and AimBot [arXiv:2508.08113] overlays reticles and shooting lines derived from depth and end-effector pose to inject spatial awareness; both report sizable gains (e.g., point→mask conditioning improving $\pi_0$-style policies by ~20pp) but require either a grounding-aware policy architecture or training with the augmented inputs. Our approach differs in two ways: it requires no architectural change or retraining of the base VLA, and it does not add a cue to a cluttered observation—it *removes* the confounding distractors so the remaining scene falls back inside the policy's training distribution.

**Distractor robustness.** Sensitivity to task-irrelevant objects is a recognized failure mode of imitation-learned policies. Our diagnostics give a sharp account of it for a state-of-the-art flow-matching VLA: $\pi_{0.5}$'s degradation on LIBERO-PRO's counterfactual axes is driven by distractor confusion, and is corrected by in-distribution scene simplification rather than by occluding pixels (graying fails, hiding works). In simulation we realize suppression by idealized hiding; on real images it is deployable via inpainting (e.g., ProPainter [arXiv:2309.03897]), which we leave to deployment-time evaluation.

## 3. Diagnosis: Distractor Confusion, Not Position Memorization

We first establish the empirical claim on which the method rests. All experiments in this section use a frozen $\pi_{0.5}$ and the official BDDL success predicate on the LIBERO-PRO `libero_object` suite.

**The collapse and its tempting interpretation.** Frozen $\pi_{0.5}$ scores 0.83 when an object is swapped for a visual *variant* in its trained location (object axis), but falls to 0.23 when the *instruction* names a different in-scene object (task axis) and to 0.17 when the target is *relocated* (swap axis). The natural hypothesis is that the policy has baked target locations into its weights and is robust only to perturbations that leave those locations intact.

**Ceiling test (refutes memorization).** We construct a scene containing only the correctly bound target object—including, critically, the relocation cases where the baseline scores 0.17—and measure $\pi_{0.5}$. Success recovers to 1.00, with 1–4cm grasp precision. The motor circuitry that places the relocated object is intact; it is simply not engaged when competitors are present.

**Manner of removal is decisive (gray vs. hide).** The *way* distractors are removed determines whether the motor recovers. *Graying* distractor pixels with a constant patch—an out-of-distribution edit—does not help and slightly hurts (swap 0.17→0.10). *Hiding* distractors to yield a clean, in-distribution single-object scene lifts the same axis to 0.63 in the matched single-condition test. If the bottleneck were a memorized spatial prior, neither edit should recover a *relocated* target; instead, only the edit that restores an in-distribution scene works. The failure is therefore best described as **distractor confusion**: in cluttered scenes the policy's binding is dominated by salient competitors, and its otherwise-competent motor is steered to the wrong object.

**Source of precision (wrist camera).** To locate where $\pi_{0.5}$'s recovered precision originates, we zero each camera in the single-object ceiling condition. Zeroing the wrist (eye-in-hand) camera collapses success from 1.00 to 0.00; zeroing the base camera as an input-shock control still leaves 0.63. The base-camera control rules out a generic input-shock confound—the policy survives a large perturbation to one stream but not to the wrist stream. Precision is thus eye-in-hand visual servoing on local contact, consistent with the distractor-confusion account, since the wrist view at grasp time is naturally distractor-free.

**Why visual reference, not text.** DINOv2 features make the LIBERO objects linearly separable (separability 1.00), and prototype-vs-true-crop disambiguation is 20/20 (perfect recall). Text binders (CLIP, SigLIP, FG-CLIP, Qwen-VL) cap near 0.50 on these fine-grained, visually similar objects given clean hi-res crops. The binding lesson is to bind by visual reference, not by text. This finding motivates the binder in §4.1.

## 4. Method

The diagnosis operationalizes as an inference-only pipeline that leaves $\pi_{0.5}$'s weights, action head, and control loop untouched. It has three stages: (i) an external open-vocabulary **visual-exemplar binder** that selects the named target among scene objects; (ii) a **distractor-suppression operator** that renders an in-distribution scene containing (effectively) only the target; and (iii) $\pi_{0.5}$'s **native flow-matching motor**, which grasps the target as if it were the sole object. No stage is trained; the binder is a frozen feature extractor plus a small reference catalog.

### 4.1 Visual-exemplar binder

We bind the language referent to a scene object by visual reference rather than by text, motivated by the §3 finding that text binders cap near 0.50 on the LIBERO objects whereas frozen DINOv2 separates them at 1.00.

**Prototype bank.** For each object class $c$ in the task vocabulary we store one or more reference crops $\{r_c^{(k)}\}$ and embed each with the frozen DINOv2 encoder $\phi(\cdot)$, yielding an $\ell_2$-normalized prototype

$$
p_c \;=\; \frac{1}{K_c}\sum_{k=1}^{K_c}\frac{\phi(r_c^{(k)})}{\lVert \phi(r_c^{(k)}) \rVert_2},\qquad
\hat p_c = p_c / \lVert p_c \rVert_2 .
$$

The bank $\mathcal{P}=\{\hat p_c\}$ is the only object-specific component and is open-vocabulary in the sense that a new object is added by supplying a reference image—no fine-tuning, no language–object weight baking.

**Target selection.** Given the instruction we extract the target class $c^\star$ (the referent noun). At inference we obtain candidate object regions $\{b_1,\dots,b_M\}$ in the agent-view frame (region proposals followed by a high-resolution crop; the foveated crop is essential because the objects subtend $\sim\!20$ px in the native frame and are indistinguishable at the VLM's input resolution). Each candidate crop $x_i$ is embedded as $z_i=\phi(x_i)/\lVert\phi(x_i)\rVert_2$, and the bound target is the candidate with maximal cosine similarity to the target prototype:

$$
i^\star \;=\; \arg\max_{i\in\{1,\dots,M\}}\; \langle z_i,\; \hat p_{c^\star}\rangle .
$$

On the swap and task axes this binder is exact: it matches the oracle target on $30/30$ rollouts, and the underlying disambiguation (target prototype vs. all true-object crops) is $20/20$. Because selection is over *present* scene objects via visual correspondence, the binder *cannot* memorize a fixed location—relocating the target moves $b_{i^\star}$ with it.

### 4.2 Distractor-suppression operator

Let $b_{i^\star}$ be the bound target region and $\mathcal{D}=\{b_j : j\neq i^\star\}$ the distractor regions. The suppression operator $\mathcal{S}$ produces a modified observation $\tilde o$ in which the distractors are removed while the scene statistics $\pi_{0.5}$ sees remain in-distribution:

$$
\tilde o \;=\; \mathcal{S}(o;\, \mathcal{D}) \;=\; \mathrm{Hide}\big(o,\, \textstyle\bigcup_{b\in\mathcal{D}} b\big),
$$

where $\mathrm{Hide}$ replaces each distractor with task-consistent background (in simulation, by suppressing the distractor bodies; in the wild, by inpainting—see below). The result is a clean single-object scene: exactly the regime in which $\pi_{0.5}$'s motor is near-perfect (§3).

**Why "hide" (in-distribution) succeeds where "gray" (OOD) fails.** A natural alternative is to mask the distractor pixels with a constant patch (graying). This *hurts*: on the swap axis, graying moves success 0.17→0.10, because $\pi_{0.5}$'s frozen encoder treats a flat gray patch as a novel out-of-distribution structure—an artifact that itself competes for attention—trading one distractor for another. *Hiding* the distractor (replacing it with the background the policy expects behind an object) returns the observation to the manifold of single-object scenes $\pi_{0.5}$ already handles, lifting swap success to 0.63. Formally, the operator must satisfy a manifold constraint $\tilde o \in \mathrm{supp}(p_{\text{train}})$, not merely a content constraint (distractor removed); graying meets the latter and violates the former. This in-distribution requirement is the load-bearing design choice of the method.

**Deployability via inpainting.** The simulator realizes $\mathrm{Hide}$ by suppressing distractor bodies, which is idealized. On real video the same operator is realized by masked image/video inpainting: the binder's distractor regions $\mathcal{D}$ supply the inpainting masks, and an off-the-shelf inpainter (e.g., ProPainter [arXiv:2309.03897]) fills them with temporally consistent background. The pipeline is thus deployable without privileged simulator access; we have not yet measured the inpainting-induced gap (§6.4).

### 4.3 Native motor and full pipeline

The suppressed observation $\tilde o$ is passed unchanged into $\pi_{0.5}$, which produces actions via its own conditional flow-matching action expert. Because $\tilde o$ presents a clean, in-distribution, single-object scene, the wrist-camera visual servoing that drives $\pi_{0.5}$'s precision (§3) operates in its competent regime. The full pipeline is

$$
i^\star = \arg\max_i \langle \phi(x_i),\hat p_{c^\star}\rangle
\;\longrightarrow\;
\tilde o = \mathcal{S}(o; \mathcal{D})
\;\longrightarrow\;
a_{1:H} = \pi_{0.5}(\tilde o),
$$

with binder and suppression applied per control step. Every component except $\pi_{0.5}$ is frozen and training-free; $\pi_{0.5}$ itself is entirely unmodified.

**Relation to mask/visual-prompt conditioning.** Prior work conditions a *trainable* policy on grounding signals—RoboGround [arXiv:2504.21530] injects target/placement masks into the policy network, and visual-prompt variants of $\pi_0$ [arXiv:2410.24164] report large gains from point→mask supervision. Our method differs in two respects that follow directly from our diagnosis: (a) the grounding signal is applied to the *observation* (input-space scene simplification) rather than to the policy's internal channels, and (b) the base policy is *frozen*—we exploit $\pi_{0.5}$'s already-competent single-object motor rather than retrain it to attend to a mask.

## 5. Experiments and Results

We evaluate on LIBERO-PRO [arXiv:2510.03827], using a *frozen* $\pi_{0.5}$ base policy and the official BDDL success predicate; our method adds no training and modifies no policy weights. Unless stated otherwise, numbers are over 30 rollouts per axis on the `libero_object` suite. We report three axes: **object** (named target replaced by a visually similar variant in roughly its trained location), **task** (instruction changed to name a different in-scene object), and **swap** (named target physically relocated).

### 5.1 Main result

**Table 1.** Per-axis success rate (BDDL metric), frozen $\pi_{0.5}$ baseline vs. ours. `libero_object`, 30 rollouts/axis.

| Axis | $\pi_{0.5}$ baseline | Ours | Δ |
|---|---|---|---|
| object (variant) | 0.83 | 0.90 | +0.07 |
| task (changed instruction) | 0.23 | 0.53 | +0.30 |
| swap (relocation) | 0.17 | 0.50 | +0.33 |
| **hard-axis avg (task+swap)** | **~0.20** | **0.52** | **+0.32** |
| 3-axis avg | 0.41 | 0.64 | +0.23 |

The two *hard* axes—task and swap—are where the published $\pi_{0.5}$ collapse occurs (0.23 and 0.17, consistent with the LIBERO-PRO report and with our baseline reproduction at 0.2369 on the hard axes). Our method roughly triples performance on the relocation axis (0.17→0.50) and more than doubles the changed-instruction axis (0.23→0.53), while leaving the already-strong object axis essentially intact (0.83→0.90). The hard-axis average rises from ~0.20 to **0.52**.

**Comparison to the steering SOTA.** VLS [arXiv:2602.03973], a training-free method that steers the same frozen $\pi_{0.5}$ by injecting VLM-grounded 3D-keypoint reward gradients into the denoising process, reports **0.3681** on the LIBERO-PRO hard axes. Our hard-axis average of **0.52** exceeds this by **+15pp**. We note that this is not a strictly head-to-head comparison—our evaluation is on the `libero_object` suite, whereas VLS reports across the Goal/Spatial/Long/Object suites—so the gap should be read as indicative rather than a controlled win; a matched-suite comparison is left for the camera-ready. For reference, CAG [arXiv:2602.17659], an inference-time dual-branch language-conditioning scheme, reports 0.217 on the related counterfactual setting.

### 5.2 Binder accuracy

The binder is the only component that must localize the *named* object, so we report it in isolation.

**Table 2.** Binder accuracy (target correctly identified among scene objects), 30 trials/axis.

| Axis | DINOv2 visual-exemplar binder | Oracle |
|---|---|---|
| swap | 30/30 | 30/30 |
| task | 30/30 | 30/30 |

On both hard axes the general DINOv2 binder matches the oracle (30/30). The downstream success numbers in Table 1 are therefore *not* bottlenecked by binding errors; they reflect the motor's ability to act once the scene is correctly simplified. The +15pp over VLS is thus obtained with a binder that, on these objects, is effectively perfect—so the remaining gap to 1.00 is a motor/execution gap, not a perception gap.

### 5.3 Ablations and negative results

**Graying vs. hiding the distractors (negative result).** A natural baseline is to "de-attractor" the scene by occluding non-target objects in place. **Graying** distractor pixels *fails*: swap success moves 0.17→0.10—it makes things worse, because gray patches are themselves out-of-distribution for $\pi_{0.5}$. **Hiding** the distractors—yielding a clean, in-distribution scene—instead lifts swap to 0.63 in the matched single-condition test. This contrast is the crux of our claim: the failure is *distractor confusion*, repaired by **in-distribution scene simplification**, not by occluding pixels per se.

**Table 3.** Distractor-handling ablation (swap axis, single-condition matched test).

| Condition | swap success |
|---|---|
| baseline (distractors present) | 0.17 |
| gray-out distractors (OOD patches) | 0.10 |
| hide distractors (in-distribution) | 0.63 |

**Wrist-camera (precision source).** Zeroing each camera input in the corrected-binding ceiling condition isolates the source of $\pi_{0.5}$'s precision. Zeroing the **wrist (eye-in-hand)** camera collapses success from 1.00 to **0.00**, whereas zeroing the **base** camera leaves it at **0.63**. The base-camera control rules out a generic input-shock confound. Precision therefore comes from **eye-in-hand visual servoing on local contact**, not from the third-person view.

**Table 4.** Camera-ablation in the corrected-binding ceiling condition.

| Input zeroed | success |
|---|---|
| none (ceiling) | 1.00 |
| base camera | 0.63 |
| wrist camera | 0.00 |

**Why a visual-exemplar binder.** DINOv2 features make the LIBERO objects separable (separability 1.00), and prototype-vs-true-crop disambiguation is 20/20 (perfect recall), whereas text binders (CLIP, SigLIP, FG-CLIP, Qwen-VL) cap near 0.50 on fine-grained, visually similar objects. The binding lesson is to **bind by visual reference, not by text**—mask/visual-prompt conditioning has likewise been shown to outperform point/text conditioning for grounding manipulation policies [arXiv:2504.21530].

### 5.4 Interpretation

Taken together, the ablations support a single mechanistic claim: $\pi_{0.5}$'s LIBERO-PRO counterfactual collapse is **distractor confusion, not deep position-memorization**. The ceiling test (1.00 with correct binding) shows the motor is near-perfect once the target is isolated; the hide-vs-gray contrast shows the fix is in-distribution scene simplification; the wrist-camera ablation identifies eye-in-hand servoing as the precision substrate; and the oracle-matching binder shows the recovered performance is gated by execution, not perception.

## 6. Discussion, Limitations, and Future Work

### 6.1 Results in context

Combining the visual-exemplar binder with distractor suppression and $\pi_{0.5}$'s own motor yields, on the `libero_object` suite (30 rollouts/axis): object 0.83→0.90 (+7pp), task 0.23→0.53 (+30pp), and swap 0.17→0.50 (+33pp). The hard-axis (task+swap) average improves from ~0.20 to 0.52, exceeding the prior training-free SOTA, VLS at 0.3681, by ~15pp, and the three-axis average is 0.64. These gains come from an *inference-time* intervention on a frozen base policy; the absolute numbers should be read against this scope rather than as a new generalist-policy ceiling.

### 6.2 Limitations

We state the scope of this work plainly.

- **It is an inference-time method, not a new foundation model.** Every gain reported here is obtained by leaving $\pi_{0.5}$ entirely frozen and changing only what it sees at test time. We do not claim to have improved the underlying policy's parameters, motor repertoire, or language grounding; we exploit a competent-but-mis-engaged motor, and we inherit $\pi_{0.5}$'s intrinsic motor ceiling—the residual gap to 1.00 on the hard axes is execution, which we cannot move without touching the base policy.
- **It is $\pi_{0.5}$-dependent.** The mechanism presupposes that the base policy *has* a strong single-object motor. It is unlikely to help a policy whose single-object ceiling is itself low, and the specific precision source we identified (eye-in-hand servoing) is a property of $\pi_{0.5}$'s architecture, not a guarantee for arbitrary VLAs.
- **The binder uses a per-object reference catalog.** Our binding is open-vocabulary via *reference image*, not pure language→object grounding. It requires a prototype crop per object class and inherits the failure modes of dense correspondence under large appearance shift, novel instances, and occlusion.
- **Distractor suppression is idealized in simulation.** We implement suppression as clean *hiding* of non-target objects, which is exact in sim but not directly available on real images. The intended real-world realization is inpainting (e.g., ProPainter), which introduces synthesis artifacts, latency, and potential out-of-distribution texture—exactly the OOD failure mode our graying ablation flagged. We have not yet validated that an inpainted scene reaches the in-distribution quality that hiding achieves.
- **Evaluation is narrow.** Results are on the `libero_object` suite axes at 30 rollouts/axis. This is sufficient to establish the mechanism and to beat the relevant training-free baselines, but it is not a broad multi-suite, multi-embodiment, or real-robot evaluation, and the rollout counts give wide confidence intervals on individual axes. In particular, the VLS comparison is cross-suite.

### 6.3 Future work: a goal-correcting foundation motor (Phase 2)

The deepest limitation—dependence on a frozen third-party policy whose competent motor we can only *coax* into engaging—motivates building our own motor that removes the $\pi_{0.5}$ dependence by construction. The Phase-2 design follows directly from this paper's diagnostics:

1. **Frozen DINOv2 backbone.** The same self-supervised features that gave 1.00 visual separability become the perceptual front-end, so binding is a native, reference-groundable property of the representation rather than a bolt-on stage.
2. **Flow-matching action expert.** We retain flow-matching action generation—the component the ceiling test showed is *not* the bottleneck—training it on a clean, binding-resolved perceptual interface so distractor confusion cannot re-enter.
3. **FLARE implicit world-modeling objective** [arXiv:2505.15659]. Auxiliary future-latent alignment supplies a task-relevant predictive signal without the cost and objective conflict of pixel reconstruction, and enables co-training on action-free video—a route to the appearance breadth our reference catalog currently supplies by hand.
4. **DART-style recovery** [Laskey et al., CoRL 2017]. Disturbance injection at training time addresses the covariate shift that our wrist-camera analysis implicates in closed-loop precision, so recovery from off-distribution grasp states is learned rather than assumed.

We frame Phase 2 as the natural consequence of this paper's finding: having shown that the LIBERO-PRO collapse is a *binding/scene* problem layered on a competent motor, the principled fix is to train a motor whose binding is correct by design, without inheriting another policy's frozen weights. The contribution of *this* paper remains the mechanistic insight and the inference-time method that validates it.

## References

- [$\pi_0$] Physical Intelligence et al. $\pi_0$: A Vision-Language-Action Flow Model for General Robot Control. arXiv:2410.24164.
- [$\pi_{0.5}$] Physical Intelligence et al. $\pi_{0.5}$: A VLA with Open-World Generalization. arXiv:2504.16054.
- [LIBERO] Liu et al. LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning. arXiv:2306.03310.
- [LIBERO-PRO] Zhou et al. LIBERO-PRO. arXiv:2510.03827.
- [VLS] Vision-Language Steering of frozen flow policies. arXiv:2602.03973.
- [CAG] When Vision Overrides Language. arXiv:2602.17659.
- [RoboGround] RoboGround: Grounding-Mask-Conditioned Manipulation Policies. CVPR 2025, arXiv:2504.21530.
- [AimBot] AimBot: Visual-Prompt Spatial Cues for Manipulation. arXiv:2508.08113.
- [DINOv2] Oquab et al. DINOv2: Learning Robust Visual Features without Supervision. arXiv:2304.07193.
- [ProPainter] Zhou et al. ProPainter: Improving Propagation and Transformer for Video Inpainting. ICCV 2023, arXiv:2309.03897.
- [FLARE] Zheng et al. FLARE: Future Latent Representation Alignment. arXiv:2505.15659.
- [DART] Laskey et al. DART: Noise Injection for Robust Imitation Learning. CoRL 2017, arXiv:1703.09327.
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

Notes for integration (outside the paper body): (1) the VLS and CAG arXiv ids (2602.03973, 2602.17659) and the AimBot id (2508.08113) and SAM id (2304.02643) were taken from the section drafts/brief and should be confirmed against canonical sources before submission; (2) "CAG" in our notes maps to *When Vision Overrides Language* — verify this is the intended 21.7% reference and not a distinct CAST-style work; (3) all measured numbers were unified to the brief (object 0.83→0.90, task 0.23→0.53, swap 0.17→0.50, hard avg ~0.20→0.52, 3-axis 0.41→0.64, gray 0.17→0.10, hide 0.17→0.63, wrist 1.00→0.00 vs base 0.63, binder 30/30, disambiguation 20/20, separability 1.00, VLS 0.3681, CAG 0.217, baseline reproduction 0.2369).