"""END-TO-END: OWLv2 proposes candidate boxes (region proposal); SigLIP VERIFIES which box matches the
target name (open-vocab classification) -> fixes OWLv2 wrong-object picks. Then median real-depth +
geometric unprojection -> 3D goal -> distilled reach head. No oracle, no learned 2D->3D, no per-object train.
OWLv2 localizes (good boxes) + SigLIP discriminates (which is the target) = complementary strengths.
"""
from __future__ import annotations

import argparse, glob, pathlib, pickle, re
import jax, jax.numpy as jnp, numpy as np
from cf_harness import parse_bddl, resolve_bodies, body_pos
from distill_reach import head_apply


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bddl-dir", default="data/libero_pro/bddl_files/libero_object_task")
    ap.add_argument("--reach-head", default="runs/reach_head.pkl")
    ap.add_argument("--n", type=int, default=12); ap.add_argument("--horizon", type=int, default=140)
    ap.add_argument("--rebind", type=int, default=25); ap.add_argument("--cam", default="agentview")
    ap.add_argument("--thr", type=float, default=0.02); ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    import torch
    from transformers import Owlv2Processor, Owlv2ForObjectDetection, SiglipModel, AutoProcessor
    from libero.libero.envs import OffScreenRenderEnv
    import robosuite.utils.camera_utils as CU
    from PIL import Image
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    owlp = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    owlm = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").to(dev).eval()
    sgm = SiglipModel.from_pretrained("google/siglip-so400m-patch14-384").to(dev).eval()
    sgp = AutoProcessor.from_pretrained("google/siglip-so400m-patch14-384")
    rp = jax.tree.map(jnp.asarray, pickle.load(open(args.reach_head, "rb")))
    H = W = 256
    nm = lambda o: re.sub(r"_\d+$", "", o).replace("_", " ")

    @torch.no_grad()
    def candidate_boxes(img, names):
        boxes = []
        for q in names:
            for pr in (q, f"a {q}", f"a photo of a {q}"):
                inp = owlp(text=[[pr]], images=Image.fromarray(img), return_tensors="pt").to(dev)
                r = owlp.post_process_grounded_object_detection(
                    owlm(**inp), threshold=args.thr, target_sizes=torch.tensor([[H, W]]).to(dev))[0]
                for b in r["boxes"].cpu().numpy():
                    boxes.append([int(b[0]), int(b[1]), int(b[2]), int(b[3])])
        # dedup near-identical boxes
        uniq = []
        for b in boxes:
            if not any(abs(b[0] - u[0]) < 8 and abs(b[1] - u[1]) < 8 for u in uniq):
                uniq.append(b)
        return uniq

    @torch.no_grad()
    def siglip_pick(img, boxes, tgt):
        te = torch.nn.functional.normalize(
            sgm.get_text_features(**sgp(text=[f"a {tgt}"], padding="max_length", return_tensors="pt").to(dev))[0], dim=0)
        best = None
        for b in boxes:
            x0, y0, x1, y1 = b
            crop = img[max(0, y0):max(y0 + 1, y1), max(0, x0):max(x0 + 1, x1)]
            if crop.size < 9:
                continue
            ie = torch.nn.functional.normalize(
                sgm.get_image_features(**sgp(images=Image.fromarray(crop), return_tensors="pt").to(dev))[0], dim=0)
            sim = float(te @ ie)
            if best is None or sim > best[0]:
                best = (sim, (y0 + y1) // 2, (x0 + x1) // 2)
        return None if best is None else (best[1], best[2])

    bddls = sorted(glob.glob(str(pathlib.Path(args.bddl_dir) / "*.bddl")))[: args.n]
    rows, berrs = [], []
    for bf in bddls:
        instr, objs, targets, distractors = parse_bddl(bf)
        if not targets or not distractors:
            continue
        T, M = targets[0], distractors[0]
        present = list(dict.fromkeys((targets or []) + (distractors or [])))
        names = [nm(o) for o in present]
        env = OffScreenRenderEnv(bddl_file_name=bf, camera_heights=H, camera_widths=W, camera_depths=True)
        env.seed(args.seed); env.reset(); obs = env.reset()
        sim = env.env.sim; rb = resolve_bodies(sim, [T, M])
        w2c = CU.get_camera_transform_matrix(sim, args.cam, H, W); c2w = np.linalg.inv(w2c)
        goal = None; rT, rM = 1e9, 1e9; gerr = None
        for step in range(args.horizon):
            if goal is None or step % args.rebind == 0:
                img = np.asarray(obs["agentview_image"])[::-1].copy()
                rd = CU.get_real_depth_map(sim, np.asarray(obs["agentview_depth"]).reshape(H, W, 1))[::-1].copy()
                boxes = candidate_boxes(img, names)
                rc = siglip_pick(img, boxes, nm(T)) if boxes else None
                if rc is not None:
                    cy, cx = rc
                    reg = rd[max(0, cy - 6):cy + 6, max(0, cx - 6):cx + 6, 0]
                    dmed = float(np.median(reg)) if reg.size else float(rd[cy, cx, 0])
                    goal = np.asarray(CU.transform_from_pixels_to_world(
                        np.array([cy, cx], float), np.full((H, W, 1), dmed), c2w)[:3], np.float32)
                    if gerr is None:
                        gerr = float(np.linalg.norm(goal - body_pos(sim, rb[T])))
            if goal is None:
                break
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
        rows.append(rT < rM if goal is not None else False)
        if gerr is not None:
            berrs.append(gerr)
        print(f"  {pathlib.Path(bf).stem[:22]:24s} T={nm(T):13s} | bind={'%.1fcm'%(gerr*100) if gerr else 'NOLOC':>7} "
              f"reach_T={rT*100:.0f} reach_M={rM*100:.0f} reached={rows[-1]}", flush=True)
    n = len(rows)
    print(f"\n=== END-TO-END OWLv2+SigLIP-verify (N={n}, seed={args.seed}) ===", flush=True)
    print(f"  reaches named target: {sum(rows)}/{n} = {sum(rows)/n*100:.0f}%   (baseline 20%, OWLv2-only 67%, oracle 90%)", flush=True)
    print(f"  bind_err mean {np.mean(berrs)*100:.1f}cm over {len(berrs)}", flush=True)
    print("EVAL_E2E_OWLSG_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
