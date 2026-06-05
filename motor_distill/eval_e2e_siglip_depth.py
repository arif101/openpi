"""END-TO-END: SigLIP clean 2D peak + agentview depth -> [u,v,depth] -> learned 3D goal -> reach head.
baseline 20%, pi0.5-feat 50%, SigLIP-planar 30%, oracle 90%.
"""
from __future__ import annotations

import argparse, glob, pathlib, pickle, re
import jax, jax.numpy as jnp, numpy as np
from cf_harness import parse_bddl, resolve_bodies, body_pos
from distill_reach import head_apply
from distill_box import apply as box_apply


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bddl-dir", default="data/libero_pro/bddl_files/libero_object_task")
    ap.add_argument("--model", default="google/siglip-so400m-patch14-384")
    ap.add_argument("--reach-head", default="runs/reach_head.pkl")
    ap.add_argument("--box-head", default="runs/siglipd_head.pkl")
    ap.add_argument("--n", type=int, default=12); ap.add_argument("--horizon", type=int, default=140)
    ap.add_argument("--rebind", type=int, default=20)
    args = ap.parse_args()
    import torch
    from transformers import SiglipModel, AutoProcessor
    from libero.libero.envs import OffScreenRenderEnv
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = SiglipModel.from_pretrained(args.model).to(dev).eval()
    proc = AutoProcessor.from_pretrained(args.model)
    rp = jax.tree.map(jnp.asarray, pickle.load(open(args.reach_head, "rb")))
    bp = jax.tree.map(jnp.asarray, pickle.load(open(args.box_head, "rb")))

    @torch.no_grad()
    def peak(img, text):
        px = proc(images=img, return_tensors="pt").to(dev)
        P = torch.nn.functional.normalize(model.vision_model(**px).last_hidden_state[0], dim=-1)
        tk = proc(text=[text], padding="max_length", return_tensors="pt").to(dev)
        e = torch.nn.functional.normalize(model.text_model(**tk).pooler_output[0], dim=0)
        g = int(np.sqrt(P.shape[0])); s = (P @ e).reshape(g, g).float().cpu().numpy()
        r, c = np.unravel_index(s.argmax(), s.shape)
        return r / (g - 1), c / (g - 1)

    bddls = sorted(glob.glob(str(pathlib.Path(args.bddl_dir) / "*.bddl")))[: args.n]
    rows = []
    for bf in bddls:
        instr, objs, targets, distractors = parse_bddl(bf)
        if not targets or not distractors:
            continue
        T, M = targets[0], distractors[0]; tn = re.sub(r"_\d+$", "", T).replace("_", " ")
        env = OffScreenRenderEnv(bddl_file_name=bf, camera_heights=256, camera_widths=256, camera_depths=True)
        env.seed(7); env.reset(); obs = env.reset()
        sim = env.env.sim; rb = resolve_bodies(sim, [T, M])
        goal = None; rT, rM = 1e9, 1e9; gerr = None
        for step in range(args.horizon):
            if goal is None or step % args.rebind == 0:
                img = np.asarray(obs["agentview_image"]); H, W = img.shape[:2]
                depth = np.asarray(obs["agentview_depth"]).reshape(H, W)
                v, u = peak(img, f"a {tn}")
                d = float(depth[min(int(v * (H - 1)), H - 1), min(int(u * (W - 1)), W - 1)])
                goal = np.asarray(box_apply(bp, jnp.asarray([u, v, d], jnp.float32)))
                if gerr is None:
                    gerr = float(np.linalg.norm(goal - body_pos(sim, rb[T])))
            ee = np.asarray(obs["robot0_eef_pos"], np.float32)
            a = np.asarray(head_apply(rp, jnp.asarray(ee - goal),
                                      jnp.asarray(np.asarray(obs["robot0_eef_quat"], np.float32)),
                                      jnp.asarray(np.asarray(obs["robot0_gripper_qpos"], np.float32))))
            obs, _, done, info = env.step(a.tolist())
            E = np.asarray(obs["robot0_eef_pos"], np.float32)
            rT = min(rT, float(np.linalg.norm(E - body_pos(sim, rb[T]))))
            rM = min(rM, float(np.linalg.norm(E - body_pos(sim, rb[M]))))
            if done:
                break
        env.close()
        rows.append(rT < rM)
        print(f"  {pathlib.Path(bf).stem[:24]:26s} T={T:16s} | bind_err={gerr*100:.1f}cm "
              f"reach_T={rT*100:.1f}cm reach_M={rM*100:.1f}cm reached_T={rT < rM}", flush=True)
    if rows:
        print(f"\n=== END-TO-END SigLIP+DEPTH (no oracle, N={len(rows)}) ===", flush=True)
        print(f"  reaches named target: {sum(rows)}/{len(rows)} = {sum(rows)/len(rows)*100:.0f}%"
              f"   (baseline 20%, pi0.5-feat 50%, SigLIP-planar 30%, oracle 90%)", flush=True)
    print("EVAL_E2E_SIGLIPD_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
