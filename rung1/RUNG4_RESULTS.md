# Rung 4 — surprise-gated STRUCTURAL memory: corrections that generalize across episodes (PASSED)

Rung 3 corrected WM errors online but keyed on EXACT global state -> no cross-episode transfer.
Rung 4 keys corrections on the LOCAL, translation/permutation-invariant interaction config around the
pushed object (relative offsets of nearby objects within a window + wall-proximity in push dir). A
correction learned once applies to ANY future arrangement with the same local config.

TEST = build memory on TRAIN episodes, FREEZE, evaluate on HELD-OUT episodes (hardest config:
blocker + 2 clutter; Rung2 WM-only=0%).

| memory mode | held-out success | mem entries |
|---|---|---|
| WM-only | 0% | 0 |
| exact-key (Rung3 style) | 0% | 158 |
| struct-key (Rung4) | 40% | 104 |

PRIMARY: struct-key memory TRANSFERS zero-shot to held-out arrangements (40%); exact-key gives ZERO
transfer (0%) despite 158 entries (global keys never recur). Surprise-gated STRUCTURAL memory =
corrections that generalize. MERGES the WM program with the hippocampus/structural-key (Exp 2a) thread
-- surprise-gated structural retrieval, the unoccupied wedge, now on the world model.

CAVEAT: 40% (partial). Frozen memory covers only TRAIN local-configs; held-out episodes have some novel
local configs. Natural strong version = ONLINE structural learning (each correction generalizes ->
coverage accumulates fast). Frozen-transfer (40 vs 0) is the clean proof structural keys generalize.

## Completes the toy arc (Rungs 1-4)
1: structured WM generalizes (predict).  2: enables reasoning (plan 90 vs 0).  3: surprise+memory make
imperfect WM usable (0->100 online).  4: structural memory makes corrections GENERALIZE (40 vs 0 exact).
=> a coherent, diagnostic-gated demonstration of grounded-reasoning-without-exploitation in miniature.
NEXT (graduation off toys): NeRD (NVIDIA learned robot dynamics) as the realistic structured WM +
our surprise/memory/verification layer on top. + perception. Files: rung1/rung4_structural_memory.py.
