#!/usr/bin/env python3
"""브리지의 집게 명령을 속도 제한한다.

기존: 닫기 명령이 오면 손가락 목표 폭이 0.04 → 0.0 으로 바로 바뀌고, 물리가 최대 속도로
닫는다 (21 물리스텝 ≈ 0.35 s). MuJoCo(robosuite Panda) 는 같은 4 cm 를 14 제어스텝 ≈ 0.7 s
에 닫는다. 정책은 「닫기 명령 뒤 0.7 s 동안 팔이 계속 내려가며 잡힌다」를 학습했으므로,
Isaac 이 즉시 닫으면 테두리에 닿기 전에 허공에서 닫힌다 (재생 e6: 명령 시점 z 985, 테두리 950).

패치: 손가락 목표 폭을 정책 스텝마다 최대 speed × dt 만큼만 움직인다.
    --gripper-speed-mps 0.057      4 cm / 0.7 s.  MuJoCo 와 같은 닫힘 시간
    (인자 없음)                     기존과 동일 (즉시)

팔 명령 적용과 hold_current_pose 의 width 계산을 램프 헬퍼로 바꾸고, startup 용
command_gripper_state 는 즉시 유지하며 램프 상태만 동기화한다. argparse 에 인자를 추가한다.

    python3 patch_bridge_gripper_speed.py ~/isaac_libero/bridge/run_openvla_isaac_bridge.py
"""

import shutil
import sys
from pathlib import Path

WIDTH_BLOCK = '''        width = (
            float(self.args.open_finger_width_m)
            if self.last_gripper > 0.0
            else float(self.args.close_finger_width_m)
        )'''
WIDTH_NEW = '''        width = self._finger_width_target()'''

HELPER_ANCHOR = '''    def set_gripper(self, gripper: float) -> None:
        """Store gripper state; arm+gripper are applied together below."""'''
HELPER = '''    def _finger_width_target(self) -> float:
        """Commanded finger width, rate-limited so closing takes MuJoCo's ~0.7 s.

        Without --gripper-speed-mps the goal width is returned at once (old
        behaviour). With it, the command moves toward the goal by at most
        speed * (sim_steps_per_action / physics_hz) per policy step, so the
        arm keeps descending while the fingers close, as in robosuite.
        """
        goal = (
            float(self.args.open_finger_width_m)
            if self.last_gripper > 0.0
            else float(self.args.close_finger_width_m)
        )
        speed = getattr(self.args, "gripper_speed_mps", None)
        if speed is None or float(speed) <= 0.0:
            self._finger_cmd = goal
            return goal
        physics_hz = float(getattr(self, "physics_hz", 0.0) or 60.0)
        dt = float(getattr(self.args, "sim_steps_per_action", 10)) / physics_hz
        step = float(speed) * dt
        cur = getattr(self, "_finger_cmd", None)
        if cur is None:
            cur = goal
        cur = cur + max(-step, min(step, goal - cur))
        self._finger_cmd = float(cur)
        return float(cur)

'''

ARG_ANCHOR = '''    parser.add_argument("--open-finger-width-m", type=float, default=0.04)'''
ARG_NEW = '''    parser.add_argument(
        "--gripper-speed-mps",
        type=float,
        default=None,
        help=(
            "rate-limit the commanded finger width (m/s). 0.057 closes 4 cm in "
            "0.7 s like robosuite's Panda gripper; omit for the old instant close"
        ),
    )
    parser.add_argument("--open-finger-width-m", type=float, default=0.04)'''


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    path = Path(sys.argv[1]).expanduser()
    text = path.read_text()
    if "_finger_width_target" in text:
        print("이미 적용됨:", path)
        return 0
    n = text.count(WIDTH_BLOCK)
    if n != 3:
        print(f"width 계산 블록이 {n}곳 (3곳이어야 함: 팔 명령, hold, command_gripper_state)")
        return 1
    for a in (HELPER_ANCHOR, ARG_ANCHOR):
        if text.count(a) != 1:
            print("기준 문자열이 1곳이 아님:\n" + a)
            return 1
    backup = path.with_suffix(path.suffix + ".bak_gripper_speed")
    shutil.copy2(path, backup)
    # 앞의 두 곳(정책 스텝마다 호출)은 램프, 세 번째(startup/settle 용 command_gripper_state)는
    # 즉시 유지하되 램프 상태를 동기화한다.
    text = text.replace(WIDTH_BLOCK, WIDTH_NEW, 2)
    text = text.replace(WIDTH_BLOCK, WIDTH_BLOCK + "\n        self._finger_cmd = float(width)", 1)
    text = text.replace(HELPER_ANCHOR, HELPER + HELPER_ANCHOR, 1)
    text = text.replace(ARG_ANCHOR, ARG_NEW, 1)
    path.write_text(text)
    print("적용 완료:", path)
    print("백업:", backup)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
