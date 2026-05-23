---
name: YC W26 robotics cohort intel — what gets funded, what doesn't
description: One Robot has the "WM for VLA eval" position locked. W26 funds picks-and-shovels with deployable artifacts, not architectural novelty. Our pitch must complement, not compete with, One Robot.
type: project
originSessionId: 53e17618-aeb4-4007-9d29-d56bfc1900c3
---
**Date:** 2026-04-25.

## One Robot (YC W26) — our closest neighbor

- Build: task-specific action-conditioned video world model, fine-tuned per-customer on customer's teleop/policy data. NOT generic Cosmos/Genie/Dreamer competitor.
- Wedge: "VLA eval cycle time." Sell sim/eval-as-a-service to VLA labs (Figure, PI, 1X type buyers).
- Founders: Hemanth Sarabu (Industrial Next, Google, NASA JPL), Elton Shon (Tesla 5 yrs, Industrial Next). Operators with pedigree — NOT lab spinout.
- Differentiator narrative: "physics matters most" + "your robot's experience" (data-recipe moat, not algorithm moat).
- They do NOT claim architectural novelty. Pitch is failure-mode-named + data-flywheel.
- **Critical: they have "world model for VLA eval" branded. We cannot enter that frame and win.** Our position must be complementary.

## W26 robotics cohort patterns (8 companies analyzed)

- **6 of 8 are picks-and-shovels** (data, sim, eval, hands). Not vertically integrated robots.
- **ZERO of 8 claim architectural novelty in their YC blurb.** All claim a named failure mode + a deployable artifact.
- Funded shapes (in frequency order): data engine > eval/sim > vertical operator > hardware-with-data-moat.
- TechCrunch frame for batch: *"Not just prototypes, but deployed products generating revenue."*

## How research-heavy bets handle "no LOI"

- **Replace LOIs with shipped artifacts.** Origami sells hands to Amazon. Servo7 has a working demo. Luel hit $2M ARR. **The demo IS the LOI.**
- Substitute "frontier labs are buying this category" for company-specific traction.
- Show ROI hypothetically (RoboDock: "$900k/depot/year").

## How to handle "the architecture isn't novel"

- They simply don't claim it is. Pitch is *recipe applied to a specific failure mode*, not algorithm-as-product.
- One Robot says "task-specific WMs on customer data for contact-rich tasks." Servo7 says "lightweight task-specific 55M model." The novelty is in *what they refuse to do generically* — deliberate scoping.

## Founder pattern

Operator-with-pedigree dominates. Industrial Next / Tesla / Zipline / Amazon Lab126 / Plus / Oxford CS PhD. Pure-PhD-only is rare. PhD + meaningful industry shipping is the sweet spot.

## Sentence pattern for moats (Skild / π / One Robot)

**One numbered category claim + one countable moat noun.**
- Skild: "ChatGPT moment for robotics" + "1,000× more data points"
- π: "first generalist policy" + open-source community moat
- One Robot: "physics matters most" + "your robot's experience" (recipe data moat)

## Tactical changes for our pitch (from the audit)

1. **Position as "the inference-time reliability layer that sits ON TOP of any VLA + WM stack."** Make One Robot a *partner shape*, not competitor. Their loop ends at "find failure modes faster"; ours starts at "patch them at inference time without retraining."
2. **Lead with one named failure mode, not the algorithm.** Tree search is the tool, not the headline.
3. **Borrow Skild's "layer not product" framing.** Not a robot, not a VLA — the reasoning layer that makes any VLA reliable enough to deploy.
4. **Ship a demo video of public VLA failing → our wrapper succeeding.** The demo is the LOI. Origami / Servo7 / One Robot all did this.
5. **Two countable moat nouns:** category claim ("first inference-time reasoning layer for VLAs") + countable moat (success-rate delta on LIBERO-PRO, OR compute envelope, OR failure-rollout dataset).

## What NOT to do

- Don't pitch architectural novelty (VLAPS / VLA-Reasoner / V-VLAPS preceded us)
- Don't enter "world model for VLA eval" frame (One Robot owns it)
- Don't pitch "foundation VLA" (zero W26 examples; PI/Skild/Figure/Cosmos already have it)
- Don't apply without a deployable artifact / demo video

## How to apply

When framing the YC application or any external pitch: lead with the failure mode, name a named buyer category, claim the inference-time-reasoning-layer position. Bury MCTS + world models in the "How it works" section, not the headline. The demo (video of public VLA failing → our wrapper succeeding on the same hardware) is the load-bearing artifact.
