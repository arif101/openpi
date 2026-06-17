"""LocateAnything-3B binding-accuracy probe (runs in /root/laenv, transformers 4.57.1). Reads pre-rendered PNGs +
GT pixels (render_probe_pngs.py), points at each target noun, scores localization within 18px (in the 256 frame,
matching eval bind_ok). NO sim here -> fully isolated from the tfm5 eval env.

Run: /root/laenv/bin/python motor_distill/la_probe.py --pngs data/la_pngs
"""
import argparse, json, pathlib, re
import numpy as np, torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer, AutoProcessor


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pngs", required=True); p.add_argument("--thresh", type=float, default=18.0)
    p.add_argument("--prompt", default="Point to: {}.")
    args = p.parse_args()
    mid = "nvidia/LocateAnything-3B"
    tok = AutoTokenizer.from_pretrained(mid, trust_remote_code=True)
    proc = AutoProcessor.from_pretrained(mid, trust_remote_code=True)
    model = AutoModel.from_pretrained(mid, torch_dtype=torch.bfloat16, trust_remote_code=True).cuda().eval()
    d = pathlib.Path(args.pngs); labels = json.load(open(d / "labels.json"))
    hit = 0; tot = 0
    for L in labels:
        img = Image.open(d / L["png"]).convert("RGB"); res = L["res"]
        msgs = [{"role": "user", "content": [{"type": "image", "image": img},
                                             {"type": "text", "text": args.prompt.format(L["noun"])}]}]
        text = proc.py_apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        images, videos = proc.process_vision_info(msgs)
        inp = proc(text=[text], images=images, videos=videos, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(pixel_values=inp["pixel_values"].to(torch.bfloat16), input_ids=inp["input_ids"],
                                 attention_mask=inp["attention_mask"], tokenizer=tok, max_new_tokens=512, generation_mode="hybrid")
        ans = tok.decode(out[0], skip_special_tokens=False)
        # parse first point or box-center; coords are 0..1000 (x=col, y=row)
        pts = re.findall(r"<box><(\d+)><(\d+)>(?:<(\d+)><(\d+)>)?</box>", ans)
        if pts:
            g = pts[0]
            if g[2]:
                xc = (int(g[0]) + int(g[2])) / 2.0; yc = (int(g[1]) + int(g[3])) / 2.0
            else:
                xc, yc = float(g[0]), float(g[1])
            pr_c = xc / 1000.0 * res; pr_r = yc / 1000.0 * res
            sc = res / 256.0
            dist = float(np.hypot((pr_r - L["gt_r"]) / sc, (pr_c - L["gt_c"]) / sc))
            hit += int(dist < args.thresh)
        tot += 1
        if tot % 10 == 0: print(f"  {hit}/{tot}", flush=True)
    print(f"\n=== LocateAnything-3B localization: {hit}/{tot} = {hit/max(tot,1):.3f} within {args.thresh}px (vs learned head 27/30=0.90, SAM+DINO 21/30) ===", flush=True)
    print("LAPROBE_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
