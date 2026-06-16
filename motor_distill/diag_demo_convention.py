"""Pin the demo-vs-live convention BEFORE retraining: is eye_in_hand_rgb stored flipped vs the live
render, and does demo ee_ori match eval's _quat2axisangle(eef_quat)? Replays a demo state into the env."""
from __future__ import annotations
import glob, h5py, numpy as np
from cf_harness import _quat2axisangle

DEMO = sorted(glob.glob("/root/LIBERO-PRO/libero/datasets/libero_object/*.hdf5"))[0]
BDDL = "/root/LIBERO-PRO/libero/libero/bddl_files/libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket.bddl"


def main():
    from libero.libero.envs import OffScreenRenderEnv
    f = h5py.File(DEMO, "r"); g = f["data"]["demo_0"]
    t = 60  # mid-trajectory (object in wrist view)
    demo_wrist = np.asarray(g["obs"]["eye_in_hand_rgb"][t])
    demo_eeori = np.asarray(g["obs"]["ee_ori"][t])
    demo_grip = np.asarray(g["obs"]["gripper_states"][t])
    state = np.asarray(g["states"][t])
    env = OffScreenRenderEnv(bddl_file_name=BDDL, camera_heights=128, camera_widths=128)
    env.seed(0); env.reset(); sim = env.env.sim
    sim.set_state_from_flattened(state); sim.forward()
    obs = env.env._get_observations() if hasattr(env.env, "_get_observations") else env._get_obs()
    live_wrist = np.asarray(obs["robot0_eye_in_hand_image"])
    live_quat = np.asarray(obs["robot0_eef_quat"]); live_aa = _quat2axisangle(live_quat)
    live_grip = np.asarray(obs["robot0_gripper_qpos"])

    def mae(a, b):
        a = a.astype(np.float32); b = b.astype(np.float32)
        return float(np.abs(a - b).mean())
    print("=== WRIST CONVENTION ===")
    print("demo vs live (no flip)   :", round(mae(demo_wrist, live_wrist), 2))
    print("demo vs live[::-1]       :", round(mae(demo_wrist, live_wrist[::-1]), 2))
    print("demo vs live[:, ::-1]    :", round(mae(demo_wrist, live_wrist[:, ::-1]), 2))
    print("demo vs live[::-1,::-1]  :", round(mae(demo_wrist, live_wrist[::-1, ::-1]), 2))
    print("=== PROPRIO CONVENTION ===")
    print("demo ee_ori        :", np.round(demo_eeori, 3))
    print("live _quat2axisangle:", np.round(live_aa, 3))
    print("demo gripper       :", np.round(demo_grip, 4), " live gripper_qpos:", np.round(live_grip, 4))
    env.close()


if __name__ == "__main__":
    main()
