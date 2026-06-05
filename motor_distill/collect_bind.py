"""Stage 2 data: counterfactual binding supervision (forces LANGUAGE use).

For each scene, for each present object o, extract frozen VLM prefix features with instruction
"pick up the {o}" and label = o's 3D position. SAME image, DIFFERENT instruction -> DIFFERENT target
label -> the only signal that predicts the label is LANGUAGE (Yin mutual-exclusivity). The binding head
trained on this MUST use language, cannot take the appearance shortcut. Open-vocab generalization comes
from the frozen VLM/SigLIP features (they already ground nouns).

Saves per (scene,object): prefix_out (fp16), n_img, target_pos. -> data/bind/*.npz
"""
from __future__ import annotations

import argparse
import glob
import pathlib
import re

import jax
import jax.numpy as jnp
import numpy as np

from cf_harness import parse_bddl, resolve_bodies, body_pos, build_obs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config-name", default="pi05_libero")
    p.add_argument("--checkpoint", default="gs://openpi-assets/checkpoints/pi05_libero")
    p.add_argument("--bddl-dir", default="data/libero_pro/bddl_files/libero_object_task")
    p.add_argument("--out", default="data/bind")
    p.add_argument("--n-scenes", type=int, default=10)
    p.add_argument("--k-init", type=int, default=3)
    args = p.parse_args()
    from openpi.training import config as _config
    from openpi.policies import policy_config as _policy_config
    from openpi.shared import download
    from openpi.models import model as _model
    from libero.libero.envs import OffScreenRenderEnv
    download.maybe_download(args.checkpoint + "/assets"); download.maybe_download(args.checkpoint + "/params")
    policy = _policy_config.create_trained_policy(_config.get_config(args.config_name), args.checkpoint)
    out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)

    def feats(obs_dict):
        inp = policy._input_transform(jax.tree.map(lambda x: x, obs_dict))
        inp = jax.tree.map(lambda x: jnp.asarray(x)[None], inp)
        o = _model.Observation.from_dict(inp)
        pf, _ = policy._model.extract_vlm_spatial_features(o)
        n_img = int(pf.shape[1] - o.tokenized_prompt.shape[1])
        return np.asarray(pf[0], np.float16), n_img

    bddls = sorted(glob.glob(str(pathlib.Path(args.bddl_dir) / "*.bddl")))[: args.n_scenes]
    saved = 0
    for bf in bddls:
        _, objs, targets, distractors = parse_bddl(bf)
        present = list(dict.fromkeys((targets or []) + (distractors or [])))
        if not present:
            continue
        for k in range(args.k_init):
            env = OffScreenRenderEnv(bddl_file_name=bf, camera_heights=256, camera_widths=256)
            env.seed(100 + k); env.reset(); obs = env.reset()
            sim = env.env.sim; rb = resolve_bodies(sim, present)
            img, wr = np.asarray(obs["agentview_image"]), np.asarray(obs["robot0_eye_in_hand_image"])
            ee, eq, gq = obs["robot0_eef_pos"], obs["robot0_eef_quat"], obs["robot0_gripper_qpos"]
            for o in present:
                oname = re.sub(r"_\d+$", "", o).replace("_", " ")
                pf, n_img = feats(build_obs(img, wr, ee, eq, gq, f"pick up the {oname}"))
                np.savez(out / f"{pathlib.Path(bf).stem[:22]}_{o}_{k}.npz",
                         prefix=pf, n_img=n_img, target_pos=body_pos(sim, rb[o]).astype(np.float32),
                         scene=pathlib.Path(bf).stem, obj=o)
                saved += 1
            env.close()
        print(f"  {pathlib.Path(bf).stem[:30]:32s} objs={[re.sub(r'_d+$','',x) for x in present]}", flush=True)
    print(f"\nSAVED {saved} binding samples -> {out}", flush=True)
    print("COLLECT_BIND_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
