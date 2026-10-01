#!/usr/bin/env python3
"""장면 없이 Lula IK 만 띄워, 재생 로그의 [IK Failure] 목표를 직접 풀어 본다.

브리지와 같은 기술 파일(rmpflow/robot_descriptor.yaml + lula_franka_gen.urdf)과 베이스 자세를 쓴다.
목표 위치와 그때의 관절값(warm start)은 로그에서 읽는다.

    /home/user/isaacsim/python.sh lula_ik_probe.py ~/LIBERO/experiments/replay_all/task_06_e000.log

시험
  A  위치만, warm start = 로그의 현재 관절       ← 브리지의 fallback=current 와 가장 비슷
  B  위치만, warm start 없음 (기본 시드)
  C  위치만, 허용 오차 0.005 → 0.02 m
  D  위치 + 아래 향한 방향, warm start
  E  A 에서 실패한 목표에 「팔 편 시드」 여러 개로 재시도
  F  도달 스윕: z=1.00 에서 x 를 늘려 가며 아래 향한 방향으로 어디까지 풀리는지
"""

from __future__ import annotations

import re
import sys

import numpy as np

LOG = sys.argv[1] if len(sys.argv) > 1 else "/home/user/LIBERO/experiments/replay_all/task_06_e000.log"
CFG = "/home/user/isaacsim/extsDeprecated/isaacsim.robot_motion.motion_generation/motion_policy_configs/franka"
BASE_POS = np.array([-0.66, 0.0, 0.912])
BASE_QUAT = np.array([1.0, 0.0, 0.0, 0.0])
FRAME = "right_gripper"
DOWN_QUAT = np.array([1.0, 0.0, 0.0, 0.0])          # 초기 정렬 로그의 목표 방향이 [0.999, -0.038, ...] 이었음 → 거의 단위

pat = re.compile(r"IK Failure\] target_position=\[([^\]]+)\] current_right_gripper=\[([^\]]+)\] current_joints=\[([^\]]+)\]")
rows = []
for line in open(LOG, errors="ignore"):
    m = pat.search(line)
    if m:
        t = np.array([float(v) for v in m.group(1).split(",")])
        q = np.array([float(v) for v in m.group(3).split(",")])[:7]
        rows.append((t, q))
# 같은 목표가 반복 기록되므로 6 mm 격자로 중복 제거
seen, uniq = set(), []
for t, q in rows:
    k = tuple(np.round(t / 0.006).astype(int))
    if k not in seen:
        seen.add(k); uniq.append((t, q))
print("IK 실패 로그 %d 줄, 서로 다른 목표 %d 개" % (len(rows), len(uniq)))
if not uniq:
    sys.exit(0)

from isaacsim import SimulationApp  # noqa: E402
app = SimulationApp({"headless": True})
from isaacsim.robot_motion.motion_generation import LulaKinematicsSolver  # noqa: E402

lula = LulaKinematicsSolver(robot_description_path=CFG + "/rmpflow/robot_descriptor.yaml", urdf_path=CFG + "/lula_franka_gen.urdf")
lula.set_robot_base_pose(BASE_POS, BASE_QUAT)
print("Lula 설정 가능 항목:", [a for a in dir(lula) if any(k in a.lower() for k in ("seed", "iter", "toler", "descent", "bfgs", "ccd"))])


def ik(pos, quat=None, warm=None, tol=0.005):
    q, ok = lula.compute_inverse_kinematics(FRAME, np.asarray(pos, dtype=np.float64), None if quat is None else np.asarray(quat, dtype=np.float64),
                                            warm_start=None if warm is None else np.asarray(warm, dtype=np.float64),
                                            position_tolerance=tol, orientation_tolerance=0.1)
    return ok


def run(label, fn):
    ok = sum(1 for t, q in uniq if fn(t, q))
    print("  %-40s %3d / %d  (%.0f%%)" % (label, ok, len(uniq), 100 * ok / len(uniq)))
    return ok


print("\n=== 실패 목표 재시도 ===")
run("A  위치만, warm=현재 관절", lambda t, q: ik(t, None, q))
run("B  위치만, 기본 시드", lambda t, q: ik(t, None, None))
run("C  위치만, warm, 허용 0.02 m", lambda t, q: ik(t, None, q, 0.02))
run("D  위치+아래 방향, warm", lambda t, q: ik(t, DOWN_QUAT, q))

seeds = [
    np.array([0.0, 0.8, 0.0, -1.2, 0.0, 2.0, 0.78]),      # 앞으로 뻗은 자세
    np.array([0.0, 1.2, 0.0, -0.6, 0.0, 1.8, 0.78]),      # 더 편 자세
    np.array([0.0, 0.4, 0.0, -1.8, 0.0, 2.2, 0.78]),      # 접은 자세
    np.array([0.0, -0.3, 0.0, -2.2, 0.0, 2.0, 0.78]),     # Isaac 기본
]
def multi_seed(t, q):
    if ik(t, None, q):
        return True
    return any(ik(t, None, s) for s in seeds)
run("E  A 실패분에 팔 편 시드 4 개 재시도", multi_seed)

print("\n=== 도달 스윕 (아래 방향, z=1.00, y=0) ===")
last_ok = None
for x in np.arange(-0.10, 0.36, 0.02):
    ok = ik([x, 0.0, 1.00], DOWN_QUAT, seeds[0]) or ik([x, 0.0, 1.00], DOWN_QUAT, seeds[1])
    if ok:
        last_ok = x
print("  아래 향한 자세로 풀리는 최대 x = %s  (베이스에서 %.2f m)   재생 실패 목표 x 범위 0.08~0.14 (0.74~0.80 m)"
      % ("%.2f" % last_ok if last_ok is not None else "없음", (last_ok + 0.66) if last_ok is not None else 0))
last_ok = None
for x in np.arange(-0.10, 0.36, 0.02):
    ok = ik([x, 0.0, 1.00], None, seeds[0]) or ik([x, 0.0, 1.00], None, seeds[1])
    if ok:
        last_ok = x
print("  방향 무시로 풀리는 최대 x     = %s  (베이스에서 %.2f m)" % ("%.2f" % last_ok if last_ok is not None else "없음", (last_ok + 0.66) if last_ok is not None else 0))
app.close()
