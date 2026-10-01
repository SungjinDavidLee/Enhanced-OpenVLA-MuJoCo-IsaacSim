#!/usr/bin/env python3
"""OFT proprio 용 프레임 보정: Lula right_gripper ↔ robosuite robot0_eef (고정 회전 · 오프셋).

OFT 는 robosuite 의 robot0_eef_pos / robot0_eef_quat / robot0_gripper_qpos 로 학습됐다.  Isaac 에서
같은 값을 만들려면 두 프레임 사이의 고정 변환이 필요하다.  MuJoCo 관절값을 Lula FK 에 넣어 직접 잰다.

1 단계 (conda openvla_libero5090): MuJoCo 에서 궤적을 돌리며 관절 · obs(eef_pos, eef_quat, gripper_qpos) 저장
    python3 oft_proprio_calib.py mujoco --out ~/oft_calib_mj.npz
2 단계 (Isaac python.sh): Lula FK 와 비교해 R_off · p_local 계산 → JSON
    /home/user/isaacsim/python.sh oft_proprio_calib.py lula --mj ~/oft_calib_mj.npz \\
        --json ~/isaac_libero/bridge/oft_proprio_calib.json

관계식   R_rg = R_obs · R_off        p_obs = p_rg + R_rg · p_local
판정     R_off 흩어짐 1° 이내, p_local 흩어짐 2 mm 이내, 재구성한 proprio 가 obs 와 1e-2 이내면 사용 가능
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

BASE_POS = np.array([-0.66, 0.0, 0.912])
CFG = "/home/user/isaacsim/extsDeprecated/isaacsim.robot_motion.motion_generation/motion_policy_configs/franka"


def q_xyzw_to_mat(q):
    x, y, z, w = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def quat2axisangle(q):
    q = np.asarray(q, float).copy(); q[3] = np.clip(q[3], -1, 1)
    den = np.sqrt(1 - q[3] ** 2)
    return np.zeros(3) if den < 1e-9 else q[:3] * 2 * np.arccos(q[3]) / den


def mat_to_quat_xyzw(R):
    R = np.asarray(R, float); tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(tr + 1) * 2; q = [(R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s, 0.25 * s]
    else:
        i = int(np.argmax(np.diag(R))); j, k = (i + 1) % 3, (i + 2) % 3
        s = np.sqrt(1 + R[i, i] - R[j, j] - R[k, k]) * 2; q = [0.0, 0.0, 0.0, (R[k, j] - R[j, k]) / s]
        q[i] = 0.25 * s; q[j] = (R[j, i] + R[i, j]) / s; q[k] = (R[k, i] + R[i, k]) / s
    q = np.asarray(q); return q / np.linalg.norm(q)


def stage_mujoco(a):
    sys.path.insert(0, "/home/user/LIBERO")
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv
    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    rows = []
    for t, ep in ((0, 0), (6, 4), (8, 0)):
        import glob
        src = sorted(glob.glob(f"/home/user/isaac_libero/runs/replay_src_demo/task_{t:02d}/*episode_{ep:03d}.npz") +
                     glob.glob(f"/home/user/isaac_libero/runs/replay_src/task_{t:02d}/*episode_{ep:03d}.npz"))
        acts = np.load(src[0], allow_pickle=True)["libero_actions"]
        env = OffScreenRenderEnv(bddl_file_name=suite.get_task_bddl_file_path(t), camera_heights=64, camera_widths=64)
        env.reset(); obs = env.set_init_state(suite.get_task_init_states(t)[ep])
        for _ in range(10):
            obs, *_ = env.step([0, 0, 0, 0, 0, 0, -1])
        sim = env.env.sim
        qadr = [sim.model.get_joint_qpos_addr("robot0_joint%d" % i) for i in range(1, 8)]
        for k in range(len(acts)):
            rows.append(np.concatenate([[t], [float(sim.data.qpos[i]) for i in qadr], obs["robot0_eef_pos"],
                                        obs["robot0_eef_quat"], obs["robot0_gripper_qpos"]]))
            obs, *_ = env.step(acts[k])
        env.close()
        print(f"task {t} e{ep}: {len(acts)} 스텝 ({src[0].split('/')[-1]})")
    rows = np.array(rows)
    np.savez(a.out, task=rows[:, 0], joints=rows[:, 1:8], eef_pos=rows[:, 8:11], eef_quat_xyzw=rows[:, 11:15], gripper_qpos=rows[:, 15:17])
    q = rows[:, 11:15]
    print("저장 %s  (%d 샘플)   첫 obs quat(xyzw) %s   gripper_qpos 범위 %s ~ %s" % (
        a.out, len(rows), np.round(q[0], 4), np.round(rows[:, 15:17].min(0), 4), np.round(rows[:, 15:17].max(0), 4)))


def stage_lula(a):
    d = np.load(a.mj)
    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True})
    from isaacsim.robot_motion.motion_generation import LulaKinematicsSolver
    lula = LulaKinematicsSolver(robot_description_path=CFG + "/rmpflow/robot_descriptor.yaml", urdf_path=CFG + "/lula_franka_gen.urdf")
    lula.set_robot_base_pose(BASE_POS, np.array([1.0, 0.0, 0.0, 0.0]))
    Roffs, dls, Rrg, prg = [], [], [], []
    for j, p_obs, q_obs in zip(d["joints"], d["eef_pos"], d["eef_quat_xyzw"]):
        p, R = lula.compute_forward_kinematics("right_gripper", j)
        R = np.asarray(R, float).reshape(3, 3); p = np.asarray(p, float).reshape(3)
        Ro = q_xyzw_to_mat(q_obs)
        Roffs.append(Ro.T @ R); dls.append(R.T @ (p_obs - p)); Rrg.append(R); prg.append(p)
    u, _, vt = np.linalg.svd(np.sum(Roffs, axis=0)); R_off = u @ vt
    if np.linalg.det(R_off) < 0:
        u[:, -1] *= -1; R_off = u @ vt
    spread = max(np.degrees(np.arccos(np.clip((np.trace(R_off.T @ Rk) - 1) / 2, -1, 1))) for Rk in Roffs)
    p_local = np.mean(dls, axis=0); pspread = max(np.linalg.norm(x - p_local) for x in dls) * 1000
    # 재구성 검사: Lula 자세 → proprio 6 을 obs 에서 만든 값과 비교 (부호 연속성 포함)
    prev = d["eef_quat_xyzw"][0]; err_p, err_a = [], []
    for R, p, p_obs, q_obs in zip(Rrg, prg, d["eef_pos"], d["eef_quat_xyzw"]):
        q = mat_to_quat_xyzw(R @ R_off.T)
        if np.dot(q, prev) < 0:
            q = -q
        prev = q
        err_p.append(np.linalg.norm(p + R @ p_local - p_obs)); err_a.append(np.abs(quat2axisangle(q) - quat2axisangle(q_obs)).max())
    qd = np.sum(d["eef_quat_xyzw"][1:] * d["eef_quat_xyzw"][:-1], axis=1)
    out = {"R_off": R_off.tolist(), "p_local_m": p_local.tolist(), "q_ref_xyzw": d["eef_quat_xyzw"][0].tolist(),
           "R_off_spread_deg": float(spread), "p_local_spread_mm": float(pspread),
           "recon_pos_err_mm_max": float(np.max(err_p) * 1000), "recon_axisangle_err_max": float(np.max(err_a)),
           "obs_quat_min_consecutive_dot": float(qd.min()), "samples": int(len(Roffs))}
    json.dump(out, open(a.json, "w"), indent=2)
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items() if k not in ("R_off",)}, ensure_ascii=False))
    ang = np.degrees(np.arccos(np.clip((np.trace(R_off) - 1) / 2, -1, 1)))
    print("R_off 회전각 %.1f°   저장 %s" % (ang, a.json))
    ok = spread < 1.0 and pspread < 2.0 and np.max(err_a) < 1e-2 and np.max(err_p) < 2e-3
    print("판정:", "사용 가능" if ok else "확인 필요 (위 수치를 보여 주세요)")
    app.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["mujoco", "lula"])
    ap.add_argument("--out", default="/home/user/oft_calib_mj.npz")
    ap.add_argument("--mj", default="/home/user/oft_calib_mj.npz")
    ap.add_argument("--json", default="/home/user/isaac_libero/bridge/oft_proprio_calib.json")
    a = ap.parse_args()
    stage_mujoco(a) if a.stage == "mujoco" else stage_lula(a)
