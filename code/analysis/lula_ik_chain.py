#!/usr/bin/env python3
"""기록된 목표 자세(위치+방향)를 스텝마다 이어 풀어, 드리프트 없이 따라갈 수 있는지 본다.

재생 결과의 stage_trace 에는 브리지가 변환한 목표(target_pose.position_m, rotation 3x3)가 있다.
그걸 Lula 로 순서대로 풀되 warm start 를 「직전 스텝의 해」로 준다 (브리지는 드리프트한 현재
관절을 줌).  실패하면 시드 여러 개와 허용 오차 10°/20° 로 재시도한다.

    /home/user/isaacsim/python.sh lula_ik_chain.py ~/LIBERO/experiments/t6_graded/e000

결과
  strict / 10° / 20° / 실패  스텝 수,  실패 스텝의 위치,  해의 관절이 한계에 얼마나 붙는지
"""

from __future__ import annotations

import glob
import json
import sys

import numpy as np

D = sys.argv[1] if len(sys.argv) > 1 else "/home/user/LIBERO/experiments/t6_graded/e000"
CFG = "/home/user/isaacsim/extsDeprecated/isaacsim.robot_motion.motion_generation/motion_policy_configs/franka"
BASE_POS = np.array([-0.66, 0.0, 0.912]); BASE_QUAT = np.array([1.0, 0.0, 0.0, 0.0])
FRAME = "right_gripper"
LIM = np.array([[-2.897, 2.897], [-1.763, 1.763], [-2.897, 2.897], [-3.072, -0.070], [-2.897, 2.897], [-0.018, 3.752], [-2.897, 2.897]])


def mat_to_quat_wxyz(R):
    R = np.asarray(R, float).reshape(3, 3); t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1) * 2; w = 0.25 * s; x = (R[2, 1] - R[1, 2]) / s; y = (R[0, 2] - R[2, 0]) / s; z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2]) * 2; w = (R[2, 1] - R[1, 2]) / s; x = 0.25 * s; y = (R[0, 1] + R[1, 0]) / s; z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1 + R[1, 1] - R[0, 0] - R[2, 2]) * 2; w = (R[0, 2] - R[2, 0]) / s; x = (R[0, 1] + R[1, 0]) / s; y = 0.25 * s; z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1 + R[2, 2] - R[0, 0] - R[1, 1]) * 2; w = (R[1, 0] - R[0, 1]) / s; x = (R[0, 2] + R[2, 0]) / s; y = (R[1, 2] + R[2, 1]) / s; z = 0.25 * s
    return np.array([w, x, y, z])


r = json.load(open(glob.glob(D + "/**/episode_result.json", recursive=True)[0]))["result"]
steps = [(np.array(s["target_pose"]["position_m"]), np.array(s["target_pose"]["rotation"]).reshape(3, 3))
         for s in r.get("stage_trace") or [] if s.get("target_pose", {}).get("position_m") and s["target_pose"].get("rotation") is not None]
print("목표 자세 %d 스텝" % len(steps))

from isaacsim import SimulationApp  # noqa: E402
app = SimulationApp({"headless": True})
from isaacsim.robot_motion.motion_generation import LulaKinematicsSolver  # noqa: E402
lula = LulaKinematicsSolver(robot_description_path=CFG + "/rmpflow/robot_descriptor.yaml", urdf_path=CFG + "/lula_franka_gen.urdf")
lula.set_robot_base_pose(BASE_POS, BASE_QUAT)

SEEDS = [np.array([0.0, 0.8, 0.0, -1.2, 0.0, 2.0, 0.78]), np.array([0.0, 1.2, 0.0, -0.6, 0.0, 1.8, 0.78]),
         np.array([0.0, 0.4, 0.0, -1.8, 0.0, 2.2, 0.78]), np.array([0.0, -0.3, 0.0, -2.2, 0.0, 2.0, 0.78])]


def solve(pos, quat, warm, tol_deg):
    q, ok = lula.compute_inverse_kinematics(FRAME, pos, quat, warm_start=warm, position_tolerance=0.005, orientation_tolerance=np.radians(tol_deg))
    return (np.asarray(q, float) if ok else None)


prev = None; counts = {"strict": 0, "10deg": 0, "20deg": 0, "seed": 0, "fail": 0}; fails = []; margins = []
for i, (pos, R) in enumerate(steps):
    quat = mat_to_quat_wxyz(R); q = None; how = None
    for tol, name in ((2.0, "strict"), (10.0, "10deg"), (20.0, "20deg")):
        q = solve(pos, quat, prev, tol)
        if q is not None:
            how = name; break
    if q is None:
        for s in SEEDS:
            q = solve(pos, quat, s, 10.0)
            if q is not None:
                how = "seed"; break
    if q is None:
        counts["fail"] += 1; fails.append((i, pos)); continue
    counts[how] += 1; prev = q
    margins.append(np.min(np.minimum(q - LIM[:, 0], LIM[:, 1] - q)))

print("결과: strict %(strict)d  10° %(10deg)d  20° %(20deg)d  시드 %(seed)d  실패 %(fail)d" % counts)
if fails:
    xs = [p[0] for _, p in fails]; print("실패 스텝 %s  x 범위 %.3f~%.3f (베이스에서 %.2f~%.2f m)" % ([i for i, _ in fails][:12], min(xs), max(xs), min(xs) + 0.66, max(xs) + 0.66))
if margins:
    print("해의 관절 한계 여유: 최소 %.3f rad  (0.05 미만이면 한계에 붙어 있음)" % min(margins))
app.close()
