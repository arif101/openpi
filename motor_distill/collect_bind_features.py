"""B1 (Method A v2): collect LANGUAGE-CONDITIONED pi0.5 spatial features + named-object pixel labels.

For each scene x each object present, prompt pi0.5 with "pick up the {NAME}", extract the un-pooled
agentview patch-token grid (language-conditioned -- the thing Exp 2 omitted), and record the named object's
true pixel in the VLM-input frame. Same scene + different named object -> different label = disambiguation
by construction.

--verify: process 2 scenes and SAVE the VLM-input image with every object's true pixel marked (green) so the
flip/resize convention is confirmed BEFORE a full collection. -> logs/bindfeat_verify_*.png

Run (verify): PYTHONPATH=third_party/libero MUJOCO_GL=egl .venv/bin/python motor_distill/collect_bind_features.py --verify
Run (collect): ... motor_distill/collect_bind_features.py --out data/keystone/bind_feats.npz
"""
from __future__ import annotations
import argparse, glob, pathlib, re
import numpy as np
import jax, jax.numpy as jnp

GRID = 14; N_BASE = GRID * GRID


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bddl-dir", default="data/libero_pro/bddl_files/libero_object_task")
    ap.add_argument("--ckpt", default="gs://openpi-assets/checkpoints/pi05_libero/params")
    ap.add_argument("--seed", type=int, default=7); ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--out", default="data/keystone/bind_feats.npz")
    ap.add_argument("--cam", default="agentview"); ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()

    from openpi.models import model as _model, pi0_config
    from openpi.models.tokenizer import PaligemmaTokenizer
    from openpi.shared import download
    from libero.libero.envs import OffScreenRenderEnv
    import robosuite.utils.camera_utils as CU
    from cf_harness import parse_bddl, resolve_bodies, body_pos

    cfg = pi0_config.Pi0Config(pi05=True, action_horizon=10,
                               paligemma_variant="gemma_2b", action_expert_variant="gemma_300m")
    model = cfg.load(_model.restore_params(download.maybe_download(args.ckpt), dtype=jnp.bfloat16)); model.eval()
    tok = PaligemmaTokenizer(max_len=48)
    H = W = 256; nm = lambda o: re.sub(r"_\d+$", "", o).replace("_", " ")

    def vlm_image(img256):
        """vflip only (the proven binder_viz convention) then resize to 224 (the frame the VLM sees)."""
        im = img256[::-1, :, :]
        out = jax.image.resize(im.astype(np.float32)[None], (1, 224, 224, 3), "bilinear")
        return np.asarray(jnp.clip(out, 0, 255).astype(jnp.uint8))[0]

    def true_pixel_224(xyz, w2c):
        """Project world xyz -> pixel in the 224 VLM frame (vflip + resize, matching vlm_image)."""
        tp = CU.project_points_from_world_to_camera(xyz[None].astype(np.float32), w2c, H, W)[0]
        r256, c256 = tp[0], tp[1]                      # row-from-top, col in unflipped 256 frame
        r = (H - 1 - r256); c = c256                   # vflip only
        return r * 224.0 / H, c * 224.0 / W            # -> 224 frame (row, col)

    @jax.jit
    def feats(obs):
        out, _ = model.extract_vlm_spatial_features(obs)
        return out[:, :N_BASE]                          # agentview patch grid [1,196,D]

    bddls = sorted(glob.glob(str(pathlib.Path(args.bddl_dir) / "*.bddl")))[: args.n]
    X, PX, NAMES, SCENE = [], [], [], []
    for si, bf in enumerate(bddls):
        instr, objs, targets, distractors = parse_bddl(bf)
        names = list(dict.fromkeys([targets[0]] + list(distractors))) if targets else list(distractors)
        if not names: continue
        env = OffScreenRenderEnv(bddl_file_name=bf, camera_heights=H, camera_widths=W, camera_depths=True)
        env.seed(args.seed); env.reset(); obs = env.reset(); sim = env.env.sim
        w2c = CU.get_camera_transform_matrix(sim, args.cam, H, W)
        img256 = np.asarray(obs["agentview_image"])
        vimg = vlm_image(img256)
        rb = resolve_bodies(sim, names)
        if args.verify:
            from PIL import Image, ImageDraw
            im = Image.fromarray(vimg); dr = ImageDraw.Draw(im)
            for nme in names:
                xyz = body_pos(sim, rb[nme]).astype(np.float32)
                r, c = true_pixel_224(xyz, w2c)
                dr.ellipse([c-4, r-4, c+4, r+4], fill=(0, 255, 0)); dr.text((c+5, r-5), nm(nme), fill=(0,255,0))
            im.save(f"logs/bindfeat_verify_{si}.png")
            print(f"  verify scene {si}: saved logs/bindfeat_verify_{si}.png ({len(names)} objs: {[nm(x) for x in names]})", flush=True)
            env.close()
            if si >= 1: break
            continue
        for nme in names:
            xyz = body_pos(sim, rb[nme]).astype(np.float32)
            r, c = true_pixel_224(xyz, w2c)
            prompt = f"pick up the {nm(nme)}"
            tk, tkm = tok.tokenize(prompt, np.zeros(32, np.float32))
            data = {
                "image": {"base_0_rgb": vimg[None], "left_wrist_0_rgb": np.zeros((1,224,224,3), np.uint8),
                          "right_wrist_0_rgb": np.zeros((1,224,224,3), np.uint8)},
                "image_mask": {k: np.array([k == "base_0_rgb"]) for k in ("base_0_rgb","left_wrist_0_rgb","right_wrist_0_rgb")},
                "state": np.zeros((1,32), np.float32),
                "tokenized_prompt": tk[None].astype(np.int32), "tokenized_prompt_mask": tkm[None],
            }
            g = np.asarray(feats(_model.Observation.from_dict(data)), np.float32)[0]   # [196, D]
            X.append(g); PX.append([r/224.0, c/224.0]); NAMES.append(nm(nme)); SCENE.append(si)
        env.close()
        print(f"  scene {si} ({pathlib.Path(bf).stem[:20]}): {len(names)} named targets extracted", flush=True)
    if args.verify:
        print("VERIFY done — inspect logs/bindfeat_verify_*.png (green dots should land ON each named object)", flush=True)
        return
    X = np.stack(X); PX = np.array(PX, np.float32)
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, X=X, px=PX, names=np.array(NAMES), scene=np.array(SCENE))
    print(f"\nsaved {X.shape} feats + {PX.shape} pixel labels ({len(set(NAMES))} object types, {len(set(SCENE))} scenes) -> {args.out}", flush=True)
    print("COLLECT_BIND_FEATURES_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
