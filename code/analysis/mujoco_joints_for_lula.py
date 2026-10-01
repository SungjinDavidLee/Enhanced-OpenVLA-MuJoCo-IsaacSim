#!/usr/bin/env python3
"""MuJoCo 관절값으로 Lula 를 검증한다 (두 단계).

1 단계 (conda openvla_libero5090):  MuJoCo 에서 같은 궤적을 재생해 스텝별 관절값·손끝 자세를 저장
    python3 mujoco_joints_for_lula.py mujoco --task 6 --episode 0 --out ~/mj_joints_t6e0.npz

2 단계 (Isaac python.sh):  그 관절값으로 Lula FK 를 계산해 기록 자세와 비교하고, 시드로 IK 를 푼다
    /home/user/isaacsim/python.sh mujoco_joints_for_lula.py lula --joints ~/mj_joints_t6e0.npz \\
        --trace ~/LIBERO/experiments/t6_graded/e000

판정
  FK 차이가 위치 5 mm · 방향 3° 안  →  기구학 일치.  IK 가 MuJoCo 시드로 풀리면 「시드 문제」(A)
  FK 방향 차이가 큼                  →  변환 문제 (B).  어느 스텝부터 벌어지는지 출력
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

BASE_POS = np.array([-0.66, 0.0, 0.912])


def quat_wxyz_to_mat(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def rot_err_deg(Ra, Rb):
    return float(np.degrees(np.arccos(np.clip((np.trace(np.asarray(Ra).T @ np.asarray(Rb)) - 1) / 2, -1, 1))))


def stage_mujoco(a):
    sys.path.insert(0, "/home/user/LIBERO")
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv
    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    d = np.load("/home/user/isaac_libero/runs/replay_src/task_%02d/alltheTask_episode_%03d.npz" % (a.task, a.episode), allow_pickle=True)
    acts = d["libero_actions"]
    env = OffScreenRenderEnv(bddl_file_name=suite.get_task_bddl_file_path(a.task), camera_heights=128, camera_widths=128)
    env.reset(); env.set_init_state(suite.get_task_init_states(a.task)[a.episode])
    for _ in range(10):
        env.step([0, 0, 0, 0, 0, 0, -1])
    sim = env.env.sim
    jn = ["robot0_joint%d" % i for i in range(1, 8)]
    qadr = [sim.model.get_joint_qpos_addr(n) for n in jn]
    joints, eef_pos, eef_quat = [], [], []
    for k in range(len(acts)):
        joints.append([float(sim.data.qpos[i]) for i in qadr])
        obs = env.env._get_observations() if hasattr(env.env, "_get_observations") else None
        eef_pos.append(np.array(env.env._eef_xpos)); eef_quat.append(np.array(env.env._eef_xquat))
        env.step(acts[k])
    joints = np.array(joints); eef_pos = np.array(eef_pos); eef_quat = np.array(eef_quat)
    dev = np.linalg.norm(eef_pos - d["eef_pos"][: len(eef_pos)], axis=1) * 1000
    print("MuJoCo 재생 %d 스텝.  기록된 eef_pos 와의 차이: 중앙 %.1f mm  최대 %.1f mm  (결정적이면 0 근처)" % (len(acts), np.median(dev), dev.max()))
    print("관절 범위 (rad): " + "  ".join("j%d %+.2f~%+.2f" % (i + 1, joints[:, i].min(), joints[:, i].max()) for i in range(7)))
    np.savez(a.out, joints=joints, eef_pos=eef_pos, eef_quat_xyzw=eef_quat, actions=acts)
    print("저장:", a.out)


def stage_lula(a):
    import glob, json
    J = np.load(a.joints); joints = J["joints"]
    r = json.load(open(glob.glob(a.trace + "/**/episode_result.json", recursive=True)[0]))["result"]
    tr = [s for s in r.get("stage_trace") or [] if s.get("target_pose", {}).get("position_m") and s["target_pose"].get("rotation") is not None]
    targets = [(np.array(s["target_pose"]["position_m"]), np.array(s["target_pose"]["rotation"]).reshape(3, 3), s.get("policy_step", i)) for i, s in enumerate(tr)]
    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True})
    from isaacsim.robot_motion.motion_generation import LulaKinematicsSolver
    cfg = "/home/user/isaacsim/extsDeprecated/isaacsim.robot_motion.motion_generation/motion_policy_configs/franka"
    lula = LulaKinematicsSolver(robot_description_path=cfg + "/rmpflow/robot_descriptor.yaml", urdf_path=cfg + "/lula_franka_gen.urdf")
    lula.set_robot_base_pose(BASE_POS, np.array([1.0, 0.0, 0.0, 0.0]))
    print("=== MuJoCo 관절 → Lula FK  vs  브리지 목표(기록 자세 변환) ===")
    print("step   위치차 mm   방향차 °   (33 스텝 이후가 문제 구간)")
    rows = []
    for pos, R, k in targets:
        if k >= len(joints):
            break
        fp, fo = lula.compute_forward_kinematics("right_gripper", joints[k])
        fo = np.asarray(fo, dtype=float)
        Rf = fo.reshape(3, 3) if fo.size == 9 else quat_wxyz_to_mat(fo)      # Lula 는 3x3 행렬을 돌려줌
        pe = np.linalg.norm(np.asarray(fp) - pos) * 1000; oe = rot_err_deg(Rf, R)
        rows.append((k, pe, oe))
    for k, pe, oe in rows[:3] + rows[30:36] + rows[-3:]:
        print("  %3d    %6.1f     %6.1f" % (k, pe, oe))
    pe = np.array([x[1] for x in rows]); oe = np.array([x[2] for x in rows])
    print("전체: 위치차 중앙 %.1f / 최대 %.1f mm   방향차 중앙 %.1f / 최대 %.1f °" % (np.median(pe), pe.max(), np.median(oe), oe.max()))
    print("\n=== 실패 구간(34 스텝~)을 MuJoCo 관절을 시드로 IK ===")
    ok = tot = 0
    for pos, R, k in targets[33:]:
        if k >= len(joints):
            break
        qw = np.array([1, 0, 0, 0.0]); 
        # 회전행렬 → 쿼터니언
        t = np.trace(R)
        if t > 0:
            s = np.sqrt(t + 1) * 2; qw = np.array([0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
        else:
            i = int(np.argmax(np.diag(R))); j, l = (i + 1) % 3, (i + 2) % 3
            s = np.sqrt(1 + R[i, i] - R[j, j] - R[l, l]) * 2; q = np.zeros(4)
            q[0] = (R[l, j] - R[j, l]) / s; q[1 + i] = 0.25 * s; q[1 + j] = (R[j, i] + R[i, j]) / s; q[1 + l] = (R[l, i] + R[i, l]) / s; qw = q
        _, succ = lula.compute_inverse_kinematics("right_gripper", pos, qw, warm_start=joints[k], position_tolerance=0.005, orientation_tolerance=np.radians(3))
        tot += 1; ok += bool(succ)
    print("MuJoCo 관절 시드 + 3° 허용:  %d / %d 풀림" % (ok, tot))
    app.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["mujoco", "lula"])
    ap.add_argument("--task", type=int, default=6); ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--out", default="/home/user/mj_joints_t6e0.npz")
    ap.add_argument("--joints", default="/home/user/mj_joints_t6e0.npz")
    ap.add_argument("--trace", default="/home/user/LIBERO/experiments/t6_graded/e000")
    a = ap.parse_args()
    stage_mujoco(a) if a.stage == "mujoco" else stage_lula(a)
