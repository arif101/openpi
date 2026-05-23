---
name: PI public roadmap intel — they named our wedge
description: PI has publicly identified inference-time deliberation as missing in 3+ blog posts. Most recently π0.7 (Apr 16 2026): "think through possible ways… reflect on outcomes… revise the task plan." We are building exactly this. Position as the inference-time deliberation layer for openpi.
type: project
originSessionId: 53e17618-aeb4-4007-9d29-d56bfc1900c3
---
**Date:** 2026-04-25.

## The smoking gun — π0.7 blog (Apr 16 2026, 9 days before this memo)

URL: https://www.pi.website/blog/pi07

Direct quote, future-work paragraph:
> "Powerful and steerable models like π0.7 might make it possible in the future to solve even more complex unseen tasks by having the model 'think through' possible ways to perform them, leverage its ability to follow diverse prompts to ground these thoughts in actions, and then reflect on the outcomes to revise the task plan."

That sentence is the loop our system implements: search through candidate plans, ground in actions, evaluate outcomes via WM, revise. PI has framed it as future work, not a shipped capability. **They have publicly named our wedge in their own language.**

## Other PI roadmap quotes naming the same gap

- **Real-Time Chunking** (Jun 2025): "future robot systems will need to make complex inferences at multiple levels of abstraction and multiple time scales, plan out complex and quick dynamic movements, and pause to 'think harder' as needed."
- **Knowledge Insulation** (May 2025): future work envisions "more sophisticated sequential reasoning, planning, and the ability to carry out complex tasks in a deliberate and goal-directed manner."
- **π0 launch** (Oct 2024): "The frontiers of robot foundation model research include long-horizon reasoning and planning, autonomous self-improvement, robustness, and safety." 3 of those 4 are inside our wedge.
- **π0.5** (Apr 2025): "The policies are reactive, and can handle… perturbations" — note "reactive," not "deliberative."
- **Memory** (Mar 2026): "Keeping a full history of the robot's observations in context over minutes or hours is infeasible." Causal confusion warning.
- **Robot Olympics** (Dec 2025): 52% success rate on flagship demos. Unpublished issue: their own demos fail half the time.

## Their existing System-2 (Hi Robot) is hierarchical prompting, NOT search

Hi Robot is a high-level VLM emitting language sub-commands to π0. **No rollout, no scoring, no backtracking, no world-model rollouts.** It's System-2-as-language, not System-2-as-search. There is no closed-loop deliberation between observation and action in any PI release.

## The opening

Three layers in PI's stack today:
- (a) VLA weights (π0/0.5/0.7)
- (b) Real-Time Chunking execution wrapper
- (c) Hi Robot hierarchical conductor + memory module

**No closed-loop deliberation between observation and action.** Our system sits between (a) and (b): treats π0.5/0.7 as stochastic action proposer + latent dynamics oracle, runs MCTS over latent rollouts, hands corrected action chunks to RTC for execution. Drop-in companion, not competitor.

## Pitch sentence (load-bearing)

> "π0.7 generalizes; it does not yet deliberate. We're the inference-time deliberation layer for openpi — a tree search over a latent world model that lets any open VLA pause, simulate, and recover when the world doesn't match what it was trained on. Same model weights, dramatically better OOD robustness, no retraining."

Borrows three pieces of PI's own language: *generalize* (π0.5), *think through / reflect / revise* (π0.7), *pause to think harder* (RTC). A Quan-Vuong-type listener will recognize all three as their own quotes.

## What Quan Vuong specifically would respond to

His public thesis: cross-embodiment scale + emergent capability from data+model. He won't admit a robustness gap publicly, but LIBERO-PRO + 52% Olympics + air fryer failures shout it. The argument that lands: **"You don't have to slow the data flywheel. We give every checkpoint you ship a 2-5× robustness multiplier at inference, without you doing anything. Your scaling story stays intact; ours plugs in on top."**

## Naming — chain off their version stack

π0 → π0.5 → π*0.6 → π0.7 is their pattern. Mirror it: "Search-0.5", "Search-0.7" or a Greek-letter parallel. Makes visible that we ship every time they ship.

## Iconic-demo pattern to copy

PI uses one canonical demo per release (laundry-folding for π0, kitchen cleanup for π0.5, espresso for π0.7). For our YC launch we need ONE demo: **pick a task PI has shown failing (air fryer is perfect — published 9 days ago) and show our layer recovering on top of unmodified π0.7 weights**. Side-by-side video, no commentary. The demo IS the pitch.

## What NOT to do

- Don't position as a competitor to PI — they're our distribution channel and citation source
- Don't claim we built a foundation VLA — we ride on theirs
- Don't pitch architectural novelty — we're an inference-time wrapper, not a new architecture

## How to apply

When framing the YC application, the demo plan, or any external pitch, lead with the π0.7 quote (it's PI's own roadmap), then introduce our system as the named-but-unshipped piece they keep listing as future work. Position as **complementary to PI**, alongside their existing partners (Weave/Ultra are vertical operators; we're the horizontal reasoning layer). Use their language ("Cambrian explosion," "think harder," "Physical Intelligence Layer"). Borrow their iconic-demo discipline.
