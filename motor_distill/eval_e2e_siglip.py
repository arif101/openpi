"""END-TO-END with SigLIP open-vocab binding: SigLIP(target name) -> heatmap centroid -> learned
planar 2D->3D goal -> distilled reach head closed-loop. No oracle. baseline 20%, pi0.5-feat 50%,
GDINO 20%, oracle 90%.
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
    ap.add_argument("--box-head", default="runs/siglip_head.pkl")
    ap.add_argument("--n", type=int, default=12); ap.add_argument("--horizon", type=int, default=140)
    ap.add_argument("--rebind", type=int, default=20); ap.add_argument("--tau", type=float, default=0.02)
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
    def heat(img, text):
        px = proc(images=img, return_tensors="pt").to(dev)
        P = torch.nn.functional.normalize(model.vision_model(**px).last_hidden_state[0], dim=-1)
        tk = proc(text=[text], padding="max_length", return_tensors="pt").to(dev)
        e = torch.nn.functional.normalize(model.text_model(**tk).pooler_output[0], dim=0)
        g = int(np.sqrt(P.shape[0])); s = (P @ e).reshape(g, g).float().cpu().numpy()
        ij = np.indices((g, g)).reshape(2, -1).T.astype(np.float32)
        w = np.exp((s.ravel() - s.max()) / args.tau); w /= w.sum() + 1e-8
        c = (w[:, None] * ij).sum(0) / (g - 1)
        pk = np.array(np.unravel_index(s.argmax(), s.shape), np.float32) / (g - 1)
        return np.concatenate([c, pk]).astype(np.float32)

    bddls = sorted(glob.glob(str(pathlib.Path(args.bddl_dir) / "*.bddl")))[: args.n]
    rows = []
    for bf in bddls:
        instr, objs, targets, distractors = parse_bddl(bf)
        if not targets or not distractors:
            continue
        T, M = targets[0], distractors[0]; tn = re.sub(r"_\d+$", "", T).replace("_", " ")
        env = OffScreenRenderEnv(bddl_file_name=bf, camera_heights=256, camera_widths=256)
        env.seed(7); env.reset(); obs = env.reset()
        sim = env.env.sim; rb = resolve_bodies(sim, [T, M])
        goal = None; rT, rM = 1e9, 1e9; gerr = None
        for step in range(args.horizon):
            if goal is None or step % args.rebind == 0:
                feat = heat(np.asarray(obs["agentview_image"])[::-1], f"a {tn}")
                goal = np.asarray(box_apply(bp, jnp.asarray(feat)))
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
        print(f"\n=== END-TO-END SigLIP (no oracle, N={len(rows)}) ===", flush=True)
        print(f"  reaches named target: {sum(rows)}/{len(rows)} = {sum(rows)/len(rows)*100:.0f}%"
              f"   (baseline 20%, pi0.5-feat 50%, GDINO 20%, oracle 90%)", flush=True)
    print("EVAL_E2E_SIGLIP_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
