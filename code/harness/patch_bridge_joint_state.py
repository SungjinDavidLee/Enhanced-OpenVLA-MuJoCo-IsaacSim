#!/usr/bin/env python3
"""스텝별 observations 에 관절값을 추가한다 (GR00T · π0.5 관절 임바디먼트 대비).

  state["joint_positions"]          팔 관절 7 개 (rad)
  state["gripper_joint_positions"]  손가락 관절 2 개 (m)
observations 로 그대로 복사되어 episode_result.json 의 stage_trace 에 남는다.

    python3 patch_bridge_joint_state.py ~/isaac_libero/bridge/run_openvla_isaac_bridge.py
"""

import shutil
import sys
from pathlib import Path

INIT_OLD = '''            "gripper_finger_mean_m": None,
            "gripper_open_ok": None,'''
INIT_NEW = '''            "gripper_finger_mean_m": None,
            "joint_positions": None,
            "gripper_joint_positions": None,
            "gripper_open_ok": None,'''

FILL_OLD = '''                    finger_mean = float(np.mean(joints[finger_indices]))
                    state["gripper_finger_mean_m"] = finger_mean'''
FILL_NEW = '''                    finger_mean = float(np.mean(joints[finger_indices]))
                    state["gripper_finger_mean_m"] = finger_mean
                    state["joint_positions"] = [float(v) for v in joints[:7]]
                    state["gripper_joint_positions"] = [float(v) for v in joints[finger_indices]]'''

OBS_OLD = '''            "gripper_finger_mean_m": state.get("gripper_finger_mean_m"),'''
OBS_NEW = '''            "gripper_finger_mean_m": state.get("gripper_finger_mean_m"),
            "joint_positions": state.get("joint_positions"),
            "gripper_joint_positions": state.get("gripper_joint_positions"),'''


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__); return 2
    path = Path(sys.argv[1]).expanduser(); text = path.read_text()
    if '"joint_positions": state.get("joint_positions")' in text:
        print("이미 적용됨:", path); return 0
    for old in (INIT_OLD, FILL_OLD, OBS_OLD):
        if text.count(old) != 1:
            print("기준 문자열이 1곳이 아님 (%d):\n%s" % (text.count(old), old)); return 1
    shutil.copy2(path, path.with_suffix(path.suffix + ".bak_joints"))
    text = text.replace(INIT_OLD, INIT_NEW, 1).replace(FILL_OLD, FILL_NEW, 1).replace(OBS_OLD, OBS_NEW, 1)
    path.write_text(text); print("적용 완료:", path); return 0


if __name__ == "__main__":
    raise SystemExit(main())
