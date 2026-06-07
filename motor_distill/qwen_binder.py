"""GENERAL binder candidate: scene-level referring grounding with Qwen2.5-VL (open-weight). Instead of
propose->crop->classify (which caps ~50% on similar objects), ask the VLM to localize the NAMED object in the
WHOLE scene via reasoning -> bbox -> depth unproject -> 3D. --validate measures 3D error + which-object acc vs GT.

Run: PYTHONPATH=/root/LIBERO-PRO:motor_distill MUJOCO_GL=egl .venv/bin/python motor_distill/qwen_binder.py \
       --bddl-dir /root/LIBERO-PRO/libero/libero/bddl_files/libero_object \
       --init-dir /root/LIBERO-PRO/libero/libero/init_files/libero_object --res 1024 --n 10 --trials 2
"""
from __future__ import annotations
import argparse, glob, pathlib, re, json
import numpy as np, torch
from cf_harness import parse_bddl, resolve_bodies, body_pos

_M = {}


def nm(o):
    return re.sub(r"_\d+$", "", o).replace("_", " ")


def qwen_box(rgb, phrase, device, mid="Qwen/Qwen2.5-VL-7B-Instruct"):
    """Return [x0,y0,x1,y1] in rgb-pixel coords for `phrase`, or None."""
    from PIL import Image
    from qwen_vl_utils import process_vision_info
    if "q" not in _M:
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        _M["q"] = Qwen2_5_VLForConditionalGeneration.from_pretrained(mid, torch_dtype=torch.bfloat16,
                                                                      device_map=device).eval()
        _M["qp"] = AutoProcessor.from_pretrained(mid)
    model, proc = _M["q"], _M["qp"]
    pil = Image.fromarray(rgb)
    msgs = [{"role": "user", "content": [
        {"type": "image", "image": pil},
        {"type": "text", "text": f"Locate the {phrase} in the image. Output ONLY its bounding box as JSON "
                                 f'[{{"bbox_2d":[x1,y1,x2,y2],"label":"{phrase}"}}].'}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    imgs, vids = process_vision_info(msgs)
    inp = proc(text=[text], images=imgs, videos=vids, padding=True, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=128, do_sample=False)
    gen = out[:, inp.input_ids.shape[1]:]
    txt = proc.batch_decode(gen, skip_special_tokens=True)[0]
    # qwen2.5-vl returns boxes in the (possibly smart-resized) input coord space -> rescale to rgb
    ih, iw = inp["image_grid_thw"][0][1].item() * 14, inp["image_grid_thw"][0][2].item() * 14
    m = re.search(r"\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]", txt)
    if not m:
        return None, txt[:80]
    bx = np.array([int(m.group(i)) for i in range(1, 5)], float)
    H, W = rgb.shape[:2]
    bx[[0, 2]] *= W / iw; bx[[1, 3]] *= H / ih
    return bx.astype(int), txt[:80]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bddl-dir", required=True); p.add_argument("--init-dir", default="")
    p.add_argument("--cam", default="agentview"); p.add_argument("--res", type=int, default=1024)
    p.add_argument("--n", type=int, default=10); p.add_argument("--trials", type=int, default=2)
    p.add_argument("--init-start", type=int, default=20); p.add_argument("--seed", type=int, default=20)
    args = p.parse_args()
    import robosuite.utils.camera_utils as cu
    from libero.libero.envs import OffScreenRenderEnv
    dev = "cuda" if torch.cuda.is_available() else "cpu"; R = args.res
    bddls = sorted(glob.glob(str(pathlib.Path(args.bddl_dir) / "*.bddl")))[: args.n]
    berr = []; ndet = 0; ntot = 0
    for bf in bddls:
        instr, objs, targets, distractors = parse_bddl(bf)
        if not targets: continue
        T = targets[0]; stem = pathlib.Path(bf).stem
        inits = np.asarray(torch.load(pathlib.Path(args.init_dir) / f"{stem}.pruned_init", weights_only=False)) if args.init_dir else None
        s0 = args.init_start; nt = min(args.trials, (len(inits) - s0)) if inits is not None else args.trials
        for t in range(nt):
            ti = s0 + t
            env = OffScreenRenderEnv(bddl_file_name=bf, camera_heights=R, camera_widths=R, camera_depths=True)
            env.seed(args.seed + ti); env.reset(); obs = env.set_init_state(inits[ti]) if inits is not None else env.reset()
            sim = env.env.sim; rb = resolve_bodies(sim, [T])
            rgb = np.asarray(obs[args.cam + "_image"])[::-1].copy()              # upright for the VLM
            box, raw = qwen_box(rgb, nm(T), dev); ntot += 1
            if box is None:
                print(f"  {stem[:24]:26s} NO BOX raw={raw!r}", flush=True); env.close(); continue
            cy = (box[1] + box[3]) / 2; cx = (box[0] + box[2]) / 2               # upright row,col
            r_nat = R - 1 - cy
            real = cu.get_real_depth_map(sim, np.asarray(obs[args.cam + "_depth"]))
            w2p = cu.get_camera_transform_matrix(sim, args.cam, R, R)
            pts = cu.transform_from_pixels_to_world(np.array([[r_nat, cx]]), real[None], np.linalg.inv(w2p))
            true = body_pos(sim, rb[T]); e = float(np.linalg.norm(pts[0] - true)); berr.append(e); ndet += 1
            print(f"  {stem[:24]:26s} want='{nm(T)}' err={e*100:.1f}cm {'OK' if e<0.06 else 'x'}", flush=True)
            env.close()
    med = np.median(berr) if berr else float("nan")
    print(f"\n=== QWEN2.5-VL BINDER (res={R}): det {ndet}/{ntot}, within-6cm {sum(e<0.06 for e in berr)}/{ntot}, "
          f"median 3D err {med*100:.1f}cm ===  [crop-CLIP wall: ~5/10, 13cm]", flush=True)
    print("QWEN_VALIDATE_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
