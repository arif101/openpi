"""Resolution diagnostic: is the named-object binding failure caused by the objects being unresolvable at
224px (the VLM input), or are LIBERO's meshes genuinely ambiguous even up close? Render agentview at HIGH
res + a zoomed crop of the object cluster + the 224px version the VLM actually sees, side by side, so we can
LOOK and decide tractable(resolution) vs fundamental(ambiguous meshes).

Run: PYTHONPATH=third_party/libero MUJOCO_GL=egl .venv/bin/python motor_distill/render_highres.py --scene 0
"""
from __future__ import annotations
import argparse, glob, pathlib, re
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bddl-dir", default="data/libero_pro/bddl_files/libero_object_task")
    ap.add_argument("--scene", type=int, default=0); ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--hi", type=int, default=1024); ap.add_argument("--out", default="logs")
    args = ap.parse_args()
    from libero.libero.envs import OffScreenRenderEnv
    import robosuite.utils.camera_utils as CU
    from cf_harness import parse_bddl, resolve_bodies, body_pos
    from PIL import Image
    import jax, jax.numpy as jnp

    bf = sorted(glob.glob(str(pathlib.Path(args.bddl_dir) / "*.bddl")))[args.scene]
    instr, objs, targets, distractors = parse_bddl(bf)
    names = list(dict.fromkeys([targets[0]] + list(distractors))) if targets else list(distractors)
    nm = lambda o: re.sub(r"_\d+$", "", o).replace("_", " ")
    print(f"scene {args.scene}: {pathlib.Path(bf).stem}", flush=True)
    print(f"objects: {[nm(x) for x in names]}", flush=True)

    H = args.hi
    env = OffScreenRenderEnv(bddl_file_name=bf, camera_heights=H, camera_widths=H, camera_depths=False)
    env.seed(args.seed); env.reset(); obs = env.reset(); sim = env.env.sim
    img = np.asarray(obs["agentview_image"])[::-1].copy()                 # high-res, upright
    Image.fromarray(img).save(f"{args.out}/hires_full_{args.scene}.png")

    # bounding box of the object cluster (project all object xyz -> hi-res pixels), then crop+zoom
    w2c = CU.get_camera_transform_matrix(sim, "agentview", H, H)
    rb = resolve_bodies(sim, names); pts = []
    for n in names:
        tp = CU.project_points_from_world_to_camera(body_pos(sim, rb[n])[None].astype(np.float32), w2c, H, H)[0]
        pts.append((H - 1 - tp[0], tp[1]))                                # vflip row, col
    pts = np.array(pts); r0, c0 = pts.min(0); r1, c1 = pts.max(0)
    pad = int(0.18 * H)
    r0, c0 = max(0, int(r0 - pad)), max(0, int(c0 - pad)); r1, c1 = min(H, int(r1 + pad)), min(H, int(c1 + pad))
    crop = img[r0:r1, c0:c1]
    Image.fromarray(crop).save(f"{args.out}/hires_crop_{args.scene}.png")

    # the 224px the VLM actually sees (downsample full -> 224)
    v = jax.image.resize(img.astype(np.float32)[None], (1, 224, 224, 3), "bilinear")
    Image.fromarray(np.asarray(jnp.clip(v, 0, 255).astype(jnp.uint8))[0]).save(f"{args.out}/vlm224_{args.scene}.png")
    # and what the cluster looks like at 224 (the crop, downsampled the way the VLM grid sees it ~ each obj 1-2 patches)
    env.close()
    print(f"saved hires_full_{args.scene}.png ({H}px), hires_crop_{args.scene}.png (zoom on cluster), "
          f"vlm224_{args.scene}.png (what VLM sees)", flush=True)
    print("RENDER_HIRES_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
