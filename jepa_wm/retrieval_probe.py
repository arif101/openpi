"""Exp 2a — Is the Pi0.5 prefix a STRUCTURAL retrieval key or a SURFACE one?

Pivotal cheap diagnostic for the hippocampus/retrieval architecture. We DON'T need the grounding
LoRA here: it modified the action expert, not the PaliGemma encoder that makes the prefix (the
natural retrieval key). So the first question is whether the BASELINE prefix is already structural.

Key = mean-pooled contextualized prefix hidden state (captured via forward hook on the language model).
Retrieve top-k by cosine similarity (leave-one-out). Metrics:
  - P@k(same instruction)  : STRUCTURAL — does the key encode what the instruction asks?
  - P@k(same scene/task)   : SURFACE    — does the key just encode the scene?

Clean test on libero_goal: 10 tasks SHARE ONE SCENE, different instructions.
  surface key -> same-scene episodes indistinguishable -> P@k(instruction) ~ chance (1/10)
  structural key -> clusters by instruction -> P@k(instruction) high
Also runs libero_object / libero_spatial (target object / relation varies).
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from lerobot.policies.pi05.modeling_pi05 import PI05Policy, make_att_2d_masks
from lerobot.policies.factory import make_pre_post_processors
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK

CKPT = "/workspace/ckpts/pi05_libero_finetuned"
DATASET = "HuggingFaceVLA/libero"


def pooled_prefix_key(policy, batch, dev):
    """Directly run the prefix-only encode (branch 1 of the inner forward) and mean-pool the
    contextualized prefix hidden state — the natural retrieval key (obs+instruction understanding)."""
    m = policy.model
    images, img_masks = policy._preprocess_images(batch)
    tokens = batch[OBS_LANGUAGE_TOKENS]
    masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
    with torch.no_grad():
        prefix_embs, prefix_pad, prefix_att = m.embed_prefix(images, img_masks, tokens, masks)
        # match the model's compute dtype (bf16) — the language model runs in bf16
        mdl_dtype = m.paligemma_with_expert.paligemma.model.language_model.layers[0].self_attn.q_proj.weight.dtype
        prefix_embs = prefix_embs.to(dtype=mdl_dtype)
        att_2d = make_att_2d_masks(prefix_pad, prefix_att)
        pos = torch.cumsum(prefix_pad, dim=1) - 1
        att_4d = m._prepare_attention_masks_4d(att_2d).to(dtype=mdl_dtype)
        result, _ = m.paligemma_with_expert.forward(
            attention_mask=att_4d, position_ids=pos, inputs_embeds=[prefix_embs, None]
        )
        prefix_hidden = result[0].float()                      # (B, prefix_len, D)
        w = prefix_pad.float().unsqueeze(-1)                   # mask padding from the mean
        return (prefix_hidden * w).sum(1) / w.sum(1).clamp(min=1)


def precision_at_k(keys, labels, k=5):
    """Leave-one-out: for each item, fraction of its top-k neighbours sharing its label."""
    keys = torch.nn.functional.normalize(keys, dim=-1)
    sim = keys @ keys.T
    sim.fill_diagonal_(-1e9)
    topk = sim.topk(k, dim=-1).indices.cpu().numpy()
    labels = np.asarray(labels)
    return np.mean([(labels[topk[i]] == labels[i]).mean() for i in range(len(labels))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--suites", nargs="+", default=["libero_goal", "libero_object", "libero_spatial"])
    ap.add_argument("--per-task", type=int, default=12, help="episodes sampled per task")
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()
    dev = "cuda"

    policy = PI05Policy.from_pretrained(args.ckpt).to(dev).eval()
    pre, _ = make_pre_post_processors(policy.config, pretrained_path=args.ckpt)
    meta = LeRobotDatasetMetadata(DATASET)
    chunk = policy.config.chunk_size
    delta = {ACTION: [i / meta.fps for i in range(chunk)]}
    ds = LeRobotDataset(DATASET, delta_timestamps=delta)

    # map suite -> its task strings, to filter episodes
    from cf_dataset import build_task_lookup
    lut = build_task_lookup()
    suite_tasks = {s: [t for t, m in lut.items() if m["suite"] == s] for s in args.suites}

    # build episode frame index by task (first frame of each episode)
    eps = ds.meta.episodes
    n_eps = len(eps)
    frm, to = eps["dataset_from_index"], eps["dataset_to_index"]
    ep_task = eps["tasks"]  # list per episode

    rng = np.random.default_rng(0)
    for suite in args.suites:
        tset = set(suite_tasks[suite])
        # collect episode indices whose task is in this suite, group by task string
        by_task = {}
        for e in range(n_eps):
            t = ep_task[e][0] if isinstance(ep_task[e], list) else ep_task[e]
            if t in tset:
                by_task.setdefault(t, []).append(e)
        # sample episodes, take a mid frame of each
        keys, instr_labels, scene_labels = [], [], []
        for t, elist in by_task.items():
            pick = rng.choice(elist, size=min(args.per_task, len(elist)), replace=False)
            for e in pick:
                mid = (int(frm[e]) + int(to[e])) // 2
                raw = ds[mid]
                raw = {kk: (vv.unsqueeze(0) if torch.is_tensor(vv) else [vv]) for kk, vv in raw.items()}
                batch = pre(raw)
                batch = {kk: (vv.to(dev) if torch.is_tensor(vv) else vv) for kk, vv in batch.items()}
                key = pooled_prefix_key(policy, batch, dev)
                keys.append(key[0].cpu())
                instr_labels.append(t)
                scene_labels.append(suite)  # within-suite scenes are near-identical for goal
        K = torch.stack(keys)
        # instruction label = task string; scene label = a coarse scene id.
        # For libero_goal scenes are shared -> use a constant; structural signal = instruction.
        p_instr = precision_at_k(K, instr_labels, k=args.k)
        n_instr = len(set(instr_labels))
        chance = 1.0 / n_instr
        verdict = "STRUCTURAL" if p_instr > 2 * chance else "surface"
        print(f"[{suite}] n={len(K)}  {n_instr} instructions  "
              f"P@{args.k}(same instruction)={p_instr:.3f}  chance={chance:.3f}  -> {verdict}")


if __name__ == "__main__":
    main()
