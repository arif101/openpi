"""Exp 2 (v2, fast + rigorous): Is the Pi0.5 prefix a STRUCTURAL retrieval key?

Two analyses, both on the BASELINE prefix (our grounding LoRA didn't touch the encoder):

A) COUNTERFACTUAL KEY-SENSITIVITY (works on ALL suites — the clean structural test):
   For each frame, compute key(image, L) and key(image, L_cf) where L_cf swaps the referent
   (same image, different instruction). cosine_drop = 1 - cos(key_L, key_Lcf).
     structural key  -> large drop (instruction moves the key)
     surface  key    -> ~0 drop  (image dominates; instruction ignored)
   Report per suite. Hypothesis: drop is large on libero_goal (grounded), SMALL on
   libero_object/spatial (visual shortcut) — i.e. the key is structural only where language
   is load-bearing, surface where the policy shortcuts.

B) RETRIEVAL P@k(same instruction) on libero_goal (scene-controlled): 10 tasks share one
   scene, so same-instruction != same-scene. P@k well above chance => key encodes the task.

Speed: dataset loaded WITHOUT delta_timestamps (single-frame access, no 50-step action chunk),
encode batched on GPU.
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from lerobot.policies.pi05.modeling_pi05 import PI05Policy, make_att_2d_masks
from lerobot.policies.factory import make_pre_post_processors
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK

from cf_dataset import build_task_lookup, make_counterfactual

CKPT = "/workspace/ckpts/pi05_libero_finetuned"
DATASET = "HuggingFaceVLA/libero"


@torch.no_grad()
def encode_keys(policy, pre, raw_frames, task_strings, dev, bs=16):
    """Batch-encode pooled contextualized-prefix keys for a list of raw frames with given tasks."""
    m = policy.model
    mdl_dtype = m.paligemma_with_expert.paligemma.model.language_model.layers[0].self_attn.q_proj.weight.dtype
    keys = []
    for i in range(0, len(raw_frames), bs):
        chunk = raw_frames[i : i + bs]
        tnk = task_strings[i : i + bs]
        # stack a batch
        batch = {}
        for k in chunk[0]:
            if torch.is_tensor(chunk[0][k]):
                batch[k] = torch.stack([f[k] for f in chunk])
        batch["task"] = list(tnk)
        batch = pre(batch)
        batch = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items()}

        images, img_masks = policy._preprocess_images(batch)
        tokens = batch[OBS_LANGUAGE_TOKENS]
        masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
        prefix_embs, prefix_pad, prefix_att = m.embed_prefix(images, img_masks, tokens, masks)
        prefix_embs = prefix_embs.to(dtype=mdl_dtype)
        att_2d = make_att_2d_masks(prefix_pad, prefix_att)
        pos = torch.cumsum(prefix_pad, dim=1) - 1
        att_4d = m._prepare_attention_masks_4d(att_2d).to(dtype=mdl_dtype)
        result, _ = m.paligemma_with_expert.forward(
            attention_mask=att_4d, position_ids=pos, inputs_embeds=[prefix_embs, None]
        )
        h = result[0].float()
        w = prefix_pad.float().unsqueeze(-1)
        keys.append(((h * w).sum(1) / w.sum(1).clamp(min=1)).cpu())
    return torch.cat(keys)


def precision_at_k(keys, labels, k=5):
    keys = torch.nn.functional.normalize(keys, dim=-1)
    sim = keys @ keys.T
    sim.fill_diagonal_(-1e9)
    topk = sim.topk(k, dim=-1).indices.numpy()
    labels = np.asarray(labels)
    return np.mean([(labels[topk[i]] == labels[i]).mean() for i in range(len(labels))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--suites", nargs="+", default=["libero_goal", "libero_object", "libero_spatial"])
    ap.add_argument("--per-task", type=int, default=8)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.per_task = 2
    dev = "cuda"

    policy = PI05Policy.from_pretrained(args.ckpt).to(dev).eval()
    pre, _ = make_pre_post_processors(policy.config, pretrained_path=args.ckpt)
    ds = LeRobotDataset(DATASET)  # NO delta_timestamps -> fast single-frame access
    lut = build_task_lookup()
    suite_tasks = {s: set(t for t, m in lut.items() if m["suite"] == s) for s in args.suites}

    eps = ds.meta.episodes
    n_eps = len(eps)
    frm, to = eps["dataset_from_index"], eps["dataset_to_index"]
    ep_task = eps["tasks"]
    rng = np.random.default_rng(0)

    print(f"=== {args.ckpt.split('/')[-1]} ===", flush=True)
    for suite in args.suites:
        tset = suite_tasks[suite]
        by_task = {}
        for e in range(n_eps):
            t = ep_task[e][0] if isinstance(ep_task[e], list) else ep_task[e]
            if t in tset:
                by_task.setdefault(t, []).append(e)

        raw_frames, tasks_L, tasks_cf = [], [], []
        for t, elist in by_task.items():
            pick = rng.choice(elist, size=min(args.per_task, len(elist)), replace=False)
            cf = make_counterfactual(t, lut[t], lut, rng)[0]
            for e in pick:
                mid = (int(frm[e]) + int(to[e])) // 2
                raw_frames.append(ds[mid])
                tasks_L.append(t)
                tasks_cf.append(cf)

        keys_L = encode_keys(policy, pre, raw_frames, tasks_L, dev)
        keys_cf = encode_keys(policy, pre, raw_frames, tasks_cf, dev)

        # A) counterfactual key-sensitivity
        kL = torch.nn.functional.normalize(keys_L, dim=-1)
        kC = torch.nn.functional.normalize(keys_cf, dim=-1)
        cos = (kL * kC).sum(-1)
        drop = (1 - cos).mean().item()
        # B) retrieval P@k(same instruction)
        p_instr = precision_at_k(keys_L, tasks_L, k=args.k)
        n_instr = len(set(tasks_L))
        chance = 1.0 / n_instr
        verdict = "STRUCTURAL" if drop > 0.10 else "surface"
        print(f"[{suite}] n={len(keys_L)} {n_instr} instr | "
              f"cf-sensitivity drop={drop:.3f} -> {verdict} | "
              f"retrieval P@{args.k}(instr)={p_instr:.3f} (chance {chance:.3f})", flush=True)


if __name__ == "__main__":
    main()
