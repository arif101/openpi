"""CATALOG-FREE fine-grained grounding via a strong VLM (Qwen2.5-VL-7B, cached). Tests whether language understanding
can disambiguate look-alike groceries ('ketchup' vs 'tomato sauce') where GroundingDINO failed (10-20%, confident box on
WRONG object). Prompts Qwen for the target's bbox, parses coords, compares box-center to the true projection. If identity
is high, this REPLACES the DINOv2 catalog entirely (truly open-vocab, language->region, no reference catalog).

Run: PYTHONPATH=/root/LIBERO-PRO:motor_distill MUJOCO_GL=egl HF_HUB_OFFLINE=1 .venv/bin/python motor_distill/diag_qwen.py \
       --bddl-dir <swap> --init-dir <swap> --n 10 --init-start 20 --render 768
"""
from __future__ import annotations
import argparse, glob, pathlib, re, json
import numpy as np, torch
from cf_harness import parse_bddl, resolve_bodies, body_pos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bddl-dir", required=True); ap.add_argument("--init-dir", default="")
    ap.add_argument("--n", type=int, default=10); ap.add_argument("--init-start", type=int, default=20)
    ap.add_argument("--res", type=int, default=256); ap.add_argument("--render", type=int, default=768)  # feed Qwen a hi-res render
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct"); ap.add_argument("--container", default="basket")
    args = ap.parse_args()
    from libero.libero.envs import OffScreenRenderEnv
    import robosuite.utils.camera_utils as cu
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from PIL import Image
    dev = "cuda"; R = args.res; G = args.render
    proc = AutoProcessor.from_pretrained(args.model)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map=dev).eval()

    def ground(img_up, phrase):
        """Ask Qwen for the bbox of `phrase`; return box-center in matrix-native 256 coords, or None."""
        pil = Image.fromarray(img_up)  # G x G upright
        prompt = (f"Locate the {phrase} in this image. Output ONLY its bounding box as JSON: "
                  f'[{{"bbox_2d": [x1, y1, x2, y2]}}]. Coordinates in pixels of this image.')
        msgs = [{"role": "user", "content": [{"type": "image", "image": pil}, {"type": "text", "text": prompt}]}]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = proc(text=[text], images=[pil], return_tensors="pt").to(dev)
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=128, do_sample=False)
        gen = proc.batch_decode(out[:, inp.input_ids.shape[1]:], skip_special_tokens=True)[0]
        nums = re.findall(r"-?\d+\.?\d*", gen)
        if len(nums) < 4: return None, gen
        x1, y1, x2, y2 = [float(v) for v in nums[:4]]
        sc = R / G; cx = (x1 + x2) / 2 * sc; cy_up = (y1 + y2) / 2 * sc
        return (R - 1 - cy_up, cx), gen

    tgt_id = []; tgt_px = []
    for bf in sorted(glob.glob(str(pathlib.Path(args.bddl_dir) / "*.bddl")))[: args.n]:
        instr, objs, targets, distractors = parse_bddl(bf)
        if not targets: continue
        T = targets[0]
        fi = pathlib.Path(args.init_dir) / f"{pathlib.Path(bf).stem}.pruned_init"
        inits = np.asarray(torch.load(fi, weights_only=False)) if fi.exists() else None
        ti = args.init_start
        env = OffScreenRenderEnv(bddl_file_name=bf, camera_heights=R, camera_widths=R, camera_depths=False)
        env.seed(ti); env.reset(); sim = env.env.sim
        obs = env.set_init_state(inits[ti]) if inits is not None else env.reset()
        rb = resolve_bodies(sim, [T])
        img_up = np.asarray(sim.render(width=G, height=G, camera_name="agentview"))[::-1].copy()
        w2p = cu.get_camera_transform_matrix(sim, "agentview", R, R)
        phrase = T.rsplit("_", 1)[0].replace("_", " ")
        gt, raw = ground(img_up, phrase)
        if rb.get(T) is not None and gt is not None:
            pxt = cu.project_points_from_world_to_camera(body_pos(sim, rb[T])[None], w2p, R, R)[0]
            err = float(np.hypot(gt[0] - pxt[0], gt[1] - pxt[1])); tgt_px.append(err); tgt_id.append(int(err < 22))
            print(f"  {pathlib.Path(bf).stem[:22]:24s} '{phrase:14s}' px_err={err:5.1f}  hit={err<22}", flush=True)
        else:
            tgt_id.append(0); print(f"  {pathlib.Path(bf).stem[:22]:24s} '{phrase:14s}' NO PARSE  raw={raw[:50]!r}", flush=True)
        env.close()
    print(f"\n=== Qwen2.5-VL open-vocab (catalog-FREE) on {pathlib.Path(args.bddl_dir).name} N={len(tgt_id)} render={G} ===", flush=True)
    print(f"  TARGET identity(hit<22px)={np.mean(tgt_id):.2f}  px_err={np.mean(tgt_px) if tgt_px else 0:.1f} (n={len(tgt_px)})", flush=True)
    print("QWEN_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
