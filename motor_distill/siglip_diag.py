"""Does SigLIP open-vocab patch-matching LOCALIZE the named object (and discriminate T vs M)?

For each captured scene: agentview -> SigLIP vision patches; text "a {T}" / "a {M}" -> SigLIP text emb;
per-patch cosine -> heatmap. Report peak grid-coords for T and M, their separation, and peakiness.
If T-peak and M-peak are well-separated and peaky, SigLIP grounds open-vocab here -> build the binding.
"""
from __future__ import annotations

import argparse, glob, pathlib, re
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bddl-dir", default="data/libero_pro/bddl_files/libero_object_task")
    ap.add_argument("--model", default="google/siglip-so400m-patch14-384")
    ap.add_argument("--n", type=int, default=8)
    args = ap.parse_args()
    import torch
    from transformers import SiglipModel, AutoProcessor
    from libero.libero.envs import OffScreenRenderEnv
    from cf_harness import parse_bddl
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = SiglipModel.from_pretrained(args.model).to(dev).eval()
    proc = AutoProcessor.from_pretrained(args.model)

    @torch.no_grad()
    def patches(img):
        px = proc(images=img, return_tensors="pt").to(dev)
        out = model.vision_model(**px).last_hidden_state[0]          # [P, d]
        return torch.nn.functional.normalize(out, dim=-1)

    @torch.no_grad()
    def textemb(t):
        tk = proc(text=[t], padding="max_length", return_tensors="pt").to(dev)
        e = model.text_model(**tk).pooler_output[0]
        return torch.nn.functional.normalize(e, dim=0)

    bddls = sorted(glob.glob(str(pathlib.Path(args.bddl_dir) / "*.bddl")))[: args.n]
    seps, peaks = [], []
    for bf in bddls:
        _, objs, targets, distractors = parse_bddl(bf)
        if not targets or not distractors:
            continue
        T, M = targets[0], distractors[0]
        tn = re.sub(r"_\d+$", "", T).replace("_", " "); mn = re.sub(r"_\d+$", "", M).replace("_", " ")
        env = OffScreenRenderEnv(bddl_file_name=bf, camera_heights=256, camera_widths=256)
        env.seed(7); env.reset(); obs = env.reset(); env.close()
        img = np.asarray(obs["agentview_image"])[::-1]               # un-flip for natural view
        P = patches(img); g = int(np.sqrt(P.shape[0]))
        sT = (P @ textemb(f"a {tn}")).reshape(g, g).float().cpu().numpy()
        sM = (P @ textemb(f"a {mn}")).reshape(g, g).float().cpu().numpy()
        pT = np.unravel_index(sT.argmax(), sT.shape); pM = np.unravel_index(sM.argmax(), sM.shape)
        sep = float(np.linalg.norm(np.array(pT) - np.array(pM)))
        pk = float(sT.max() / (sT.mean() + 1e-6))
        seps.append(sep); peaks.append(pk)
        print(f"  {pathlib.Path(bf).stem[:24]:26s} T={tn:13s} M={mn:13s} | "
              f"peakT={pT} peakM={pM} sep={sep:.1f}patches peakiness={pk:.2f}", flush=True)
    print(f"\n=== SigLIP localization (N={len(seps)}, grid {g}x{g}) ===", flush=True)
    print(f"  mean T-vs-M peak separation = {np.mean(seps):.1f} patches  (>2 = discriminates objects)", flush=True)
    print(f"  mean peakiness (max/mean)   = {np.mean(peaks):.2f}  (>1.3 = spatially selective)", flush=True)
    print("SIGLIP_DIAG_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
