"""Localize WHERE language->object binding breaks in frozen pi0.5 (no training, causal).

For each genuinely-captured LIBERO-PRO TASK scene (baseline grabs the MEMORIZED object M,
instruction names a different target T), run pi0.5 TWICE on the SAME pixels+state, swapping
only the named object in the instruction (name M vs name T), and measure where the swap stops
propagating:

  S_lang  = relative change in LANGUAGE token features      (sanity: should be LARGE)
  S_img   = relative change in IMAGE token features         (does language reach vision?)
  S_act   = relative change in the SAMPLED ACTION           (does the policy act on it?)
  dir     = does the action point toward the NAMED object or the memorized one?

Verdict logic:
  S_img ~ 0                -> language never binds to vision: break is IN THE VLM (need object
                             structure / stronger language->vision grounding).
  S_img > 0 but S_act ~ 0  -> VLM re-binds but the ACTION EXPERT ignores it: break is ROUTING
                             (need goal-conditioning into the motor; NO new representation).

No slots assumed — the architecture is chosen by this result.
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


def name_of(body):
    return re.sub(r"_\d+$", "", body).replace("_", " ")


def feats_and_action(policy, obs_dict, noise):
    from openpi.models import model as _model
    inp = policy._input_transform(jax.tree.map(lambda x: x, obs_dict))
    inp = jax.tree.map(lambda x: jnp.asarray(x)[None], inp)
    o = _model.Observation.from_dict(inp)
    pf, _ = policy._model.extract_vlm_spatial_features(o)
    a, _ = policy._model.sample_actions(jax.random.key(0), o, num_steps=10, noise=noise)
    return np.asarray(pf[0]), np.asarray(a[0]), int(o.tokenized_prompt.shape[1])


def rel(a, b):
    return float(np.linalg.norm(a - b) / (np.linalg.norm(a) + np.linalg.norm(b) + 1e-8) * 2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config-name", default="pi05_libero")
    p.add_argument("--checkpoint", default="gs://openpi-assets/checkpoints/pi05_libero")
    p.add_argument("--bddl-dir", default="data/libero_pro/bddl_files/libero_object_task")
    p.add_argument("--n", type=int, default=12)
    args = p.parse_args()
    import torch
    from openpi.training import config as _config
    from openpi.policies import policy_config as _policy_config
    from openpi.shared import download
    from libero.libero.envs import OffScreenRenderEnv
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    download.maybe_download(args.checkpoint + "/assets"); download.maybe_download(args.checkpoint + "/params")
    policy = _policy_config.create_trained_policy(_config.get_config(args.config_name), args.checkpoint)
    noise = jax.random.normal(jax.random.key(0), (1, 10, 32))

    bddls = sorted(glob.glob(str(pathlib.Path(args.bddl_dir) / "*.bddl")))[: args.n]
    rows = []
    for bf in bddls:
        instr_T, objs, targets, distractors = parse_bddl(bf)   # instr_T names T (the counterfactual target)
        if not targets or not distractors:
            continue
        T, M = targets[0], distractors[0]
        tname, mname = name_of(T), name_of(M)
        if tname not in instr_T:
            continue
        instr_M = instr_T.replace(tname, mname)                # same template, name the MEMORIZED object instead
        env = OffScreenRenderEnv(bddl_file_name=bf, camera_heights=256, camera_widths=256)
        env.seed(7); env.reset(); obs = env.reset()
        sim = env.env.sim
        rb = resolve_bodies(sim, [T, M])
        ee = np.asarray(obs["robot0_eef_pos"], np.float64)
        Tp, Mp = body_pos(sim, rb[T]), body_pos(sim, rb[M])
        uT = (Tp - ee) / (np.linalg.norm(Tp - ee) + 1e-8)
        uM = (Mp - ee) / (np.linalg.norm(Mp - ee) + 1e-8)
        common = dict(wr=np.asarray(obs["robot0_eye_in_hand_image"]), ee=obs["robot0_eef_pos"],
                      eq=obs["robot0_eef_quat"], gq=obs["robot0_gripper_qpos"])
        oT = build_obs(np.asarray(obs["agentview_image"]), common["wr"], common["ee"], common["eq"], common["gq"], instr_T)
        oM = build_obs(np.asarray(obs["agentview_image"]), common["wr"], common["ee"], common["eq"], common["gq"], instr_M)
        env.close()
        pfT, aT, L = feats_and_action(policy, oT, noise)
        pfM, aM, _ = feats_and_action(policy, oM, noise)
        n_img = pfT.shape[0] - L
        S_img = rel(pfT[:n_img], pfM[:n_img])
        S_lang = rel(pfT[n_img:], pfM[n_img:])
        S_act = rel(aT, aM)
        dT = aT[:, 0:3].sum(0); dT = dT / (np.linalg.norm(dT) + 1e-8)   # action dir when told T
        dM = aM[:, 0:3].sum(0); dM = dM / (np.linalg.norm(dM) + 1e-8)   # action dir when told M
        cosT_toT, cosT_toM = float(dT @ uT), float(dT @ uM)
        # CAUSAL contrast: does NAMING T (vs M) shift the action toward T and away from M?
        shift_toT = float((dT - dM) @ uT)        # >0 => naming T causally pulls action toward T
        shift_toM = float((dT - dM) @ uM)        # <0 => naming T causally pushes action off M
        rows.append({"scene": pathlib.Path(bf).stem[:30], "T": tname, "M": mname,
                     "S_lang": S_lang, "S_img": S_img, "S_act": S_act,
                     "dirT->T": cosT_toT, "dirT->M": cosT_toM,
                     "shift->T": shift_toT, "shift_off_M": shift_toM})
        print(f"  {rows[-1]['scene']:30s} T={tname:13s} M={mname:13s} | "
              f"S_img={S_img:.3f} S_act={S_act:.3f} | told-T dir->T={cosT_toT:+.2f} dir->M={cosT_toM:+.2f} | "
              f"CAUSAL shift->T={shift_toT:+.2f} off_M={shift_toM:+.2f}", flush=True)
    if rows:
        f = lambda k: float(np.mean([r[k] for r in rows]))
        print(f"\n=== INSTRUCTION-SENSITIVITY (N={len(rows)}) ===", flush=True)
        print(f"  S_lang(language tokens) = {f('S_lang'):.3f}   (sanity: words changed)", flush=True)
        print(f"  S_img (image tokens)    = {f('S_img'):.4f}   <- does language reach vision?", flush=True)
        print(f"  S_act (sampled action)  = {f('S_act'):.4f}   <- does the policy act on it?", flush=True)
        print(f"  told-T action: dir->T={f('dirT->T'):+.2f}  dir->M={f('dirT->M'):+.2f}  "
              f"(memorization if dir->M >> dir->T)", flush=True)
        print(f"  CAUSAL (naming T vs M): shift->T={f('shift->T'):+.2f}  shift_off_M={f('shift_off_M'):+.2f}  "
              f"(>0 / <0 => language causally moves the reach toward T)", flush=True)
        v = ("BREAK IN VLM (language doesn't reach vision -> need object structure)"
             if f("S_img") < 0.02 else
             "VLM re-binds but ACTION ignores it -> ROUTING fix (goal-condition the motor; no slots)")
        print(f"  VERDICT: {v}", flush=True)
    print("CF_BIND_DIAG_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
