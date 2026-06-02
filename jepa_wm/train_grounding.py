"""Phase-1 grounding fine-tune: counterfactual-consistency objective on pi05_libero_finetuned.

Contrastive flow-matching (the only change vs standard BC):
  run the SAME (images, actions, noise, time) through the policy with the CORRECT instruction
  and with a COUNTERFACTUAL instruction (in-scene object/relation swap), then:
    loss = FM(A | img, L)              # standard BC (keep competence)
         + lambda * relu(margin - (FM(A|img,L_cf) - FM(A|img,L)))   # push them apart
If the policy ignores language, FM_cf == FM_L -> margin term is large -> gradient forces
language to change the action prediction. LoRA on the action expert only.

Smoke: --smoke runs 3 steps to validate the loss path.
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.utils.constants import OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK, ACTION

from cf_dataset import build_task_lookup, make_counterfactual

CKPT = "/workspace/ckpts/pi05_libero_finetuned"
DATASET = "HuggingFaceVLA/libero"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--margin", type=float, default=1.0)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--val-frac", type=float, default=0.15, help="fraction of EPISODES held out for val")
    ap.add_argument("--val-every", type=int, default=100)
    ap.add_argument("--val-batches", type=int, default=8)
    ap.add_argument("--out", default="/workspace/ckpts/pi05_grounding_lora")
    args = ap.parse_args()
    dev = "cuda"

    # ---- policy + LoRA on the action expert ----
    policy = PI05Policy.from_pretrained(CKPT)
    policy.wrap_with_peft(peft_cli_overrides={"r": 16, "lora_alpha": 32, "lora_dropout": 0.0})
    policy = policy.to(dev)
    policy.train()
    trainable = [p for p in policy.parameters() if p.requires_grad]
    print(f"trainable params: {sum(p.numel() for p in trainable)/1e6:.2f}M")

    # ---- preprocessor (normalize + tokenize) ----
    pre, _ = make_pre_post_processors(policy.config, pretrained_path=CKPT)

    # ---- counterfactual tooling ----
    lut = build_task_lookup()
    rng = np.random.default_rng(0)

    # ---- data (with ACTION CHUNK via delta_timestamps — pi05 predicts a chunk, not 1 step) ----
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from torch.utils.data import Subset
    meta = LeRobotDatasetMetadata(DATASET)
    chunk = policy.config.chunk_size
    delta = {ACTION: [i / meta.fps for i in range(chunk)]}
    ds = LeRobotDataset(DATASET, delta_timestamps=delta)

    # ---- held-out split BY EPISODE (frame ranges from metadata; O(eps), no column scan) ----
    eps_meta = ds.meta.episodes  # arrow Dataset: per-ep dataset_from_index / dataset_to_index
    n_eps = len(eps_meta)
    frm = eps_meta["dataset_from_index"]; to = eps_meta["dataset_to_index"]
    val_rng = np.random.default_rng(42)
    val_set = set(val_rng.choice(n_eps, size=max(1, int(args.val_frac * n_eps)), replace=False).tolist())
    train_fi, val_fi = [], []
    for e in range(n_eps):
        rng_ = range(int(frm[e]), int(to[e]))
        (val_fi if e in val_set else train_fi).extend(rng_)
    train_ds, val_ds = Subset(ds, train_fi), Subset(ds, val_fi)
    print(f"dataset: {len(ds)} frames | chunk={chunk} @ {meta.fps}fps | "
          f"{n_eps} eps -> train {len(train_fi)} frames / {n_eps-len(val_set)} eps, "
          f"VAL {len(val_fi)} frames / {len(val_set)} eps", flush=True)

    dl = DataLoader(train_ds, batch_size=args.bs, shuffle=True, num_workers=4, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=args.bs, shuffle=True, num_workers=2, drop_last=True)

    def to_dev(b):
        return {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}

    def cf_for(tasks):
        out = []
        for t in tasks:
            m = lut.get(t)
            out.append(make_counterfactual(t, m, lut, rng)[0] if m else t)
        return out

    @torch.no_grad()
    def evaluate():
        policy.eval()
        bcs, gaps = [], []
        it = iter(val_dl)
        for _ in range(args.val_batches):
            try:
                vr = next(it)
            except StopIteration:
                break
            vt = list(vr["task"])
            vr_cf = {k: (v.clone() if torch.is_tensor(v) else list(v)) for k, v in vr.items()}
            vr_cf["task"] = cf_for(vt)
            lL, _ = policy.forward(to_dev(pre(vr)), reduction="none")
            lcf, _ = policy.forward(to_dev(pre(vr_cf)), reduction="none")
            bcs.append(lL.mean().item()); gaps.append((lcf - lL).mean().item())
        policy.train()
        return float(np.mean(bcs)), float(np.mean(gaps))

    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-5)
    best_val = float("inf")
    step = 0
    for raw in dl:
        tasks = list(raw["task"])
        raw_cf = {k: (v.clone() if torch.is_tensor(v) else list(v)) for k, v in raw.items()}
        raw_cf["task"] = cf_for(tasks)

        loss_L, _ = policy.forward(to_dev(pre(raw)), reduction="none")     # (B,) standard BC forward
        loss_cf, _ = policy.forward(to_dev(pre(raw_cf)), reduction="none") # (B,) counterfactual forward

        bc = loss_L.mean()
        margin = torch.relu(args.margin - (loss_cf - loss_L)).mean()
        loss = bc + args.lam * margin

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 20 == 0 or args.smoke:
            print(f"step {step:5d}  train_loss {loss.item():.4f}  train_bc {bc.item():.4f}  "
                  f"margin {margin.item():.4f}  train_gap {(loss_cf-loss_L).mean().item():+.4f}", flush=True)

        if not args.smoke and step % args.val_every == 0:
            v_bc, v_gap = evaluate()
            tag = ""
            if v_bc < best_val:
                best_val = v_bc; policy.save_pretrained(args.out + "_best"); tag = " *BEST(val_bc) saved*"
            print(f"  [VAL step {step:5d}]  val_bc {v_bc:.4f}  val_gap {v_gap:+.4f}  "
                  f"(train_bc {bc.item():.4f})  overfit_gap {v_bc-bc.item():+.4f}{tag}", flush=True)

        step += 1
        if args.smoke and step >= 3:
            print("SMOKE OK"); return
        if step >= args.steps:
            break

    policy.save_pretrained(args.out)
    print(f"saved final {args.out} (best-val at {args.out}_best)")


if __name__ == "__main__":
    main()
