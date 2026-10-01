#!/usr/bin/env python3
"""solve_ik 에 「위치만」 fallback 을 추가한다.

지금은 목표 방향 → 현재 방향 → 마지막 성공 방향 순으로 시도하고 그래도 실패하면 IKFailure 다.
Lula 프로브 결과: 재생에서 실패한 목표 138 개가 방향을 붙이면 0 개, 위치만 풀면 138 개 전부 풀림.
아래 향한 방향으로는 베이스에서 0.62 m 까지, 방향 없이는 0.90 m 까지 풀린다.  MuJoCo 의 OSC 는
먼 곳에서 손을 기울여서라도 위치를 맞추므로, 같은 타협을 마지막 단계로 넣는다.

    python3 patch_bridge_ik_position_fallback.py ~/isaac_libero/bridge/run_openvla_isaac_bridge.py

인자 --no-ik-position-fallback 으로 끌 수 있다 (기본 켜짐).
"""

import shutil
import sys
from pathlib import Path

OLD = '''                if bool(fallback_success):
                    LOGGER.warning(
                        "[IK Recovery] target orientation rejected; solved same "
                        "position with %s right_gripper orientation",
                        label,
                    )
                    action, success = fallback_action, fallback_success
                    used_rotation = fallback_rotation.copy()
                    break
        if not bool(success):'''
NEW = '''                if bool(fallback_success):
                    LOGGER.warning(
                        "[IK Recovery] target orientation rejected; solved same "
                        "position with %s right_gripper orientation",
                        label,
                    )
                    action, success = fallback_action, fallback_success
                    used_rotation = fallback_rotation.copy()
                    break
            if not bool(success) and not getattr(self.args, "no_ik_position_fallback", False):
                # 마지막 단계: 방향을 풀고 위치만 맞춘다.  팔을 거의 다 편 자세에서는 어떤
                # 고정 방향으로도 해가 없지만 위치만은 풀린다 (Lula 프로브 138/138).  MuJoCo
                # OSC 가 먼 목표에서 손을 기울이는 것과 같은 타협이다.
                free_action, free_success = self.kinematics.compute_inverse_kinematics(
                    target_position,
                    None,
                )
                if bool(free_success):
                    LOGGER.warning(
                        "[IK Recovery] all orientations rejected; solved position only "
                        "(orientation released) at target=%s",
                        np.round(target_position, 4).tolist(),
                    )
                    action, success = free_action, free_success
                    used_rotation = fallback_rotations[0][1].copy()
        if not bool(success):'''

ARG_ANCHOR = '''    parser.add_argument(
        "--ik-backtrack-factors",'''
ARG_NEW = '''    parser.add_argument(
        "--no-ik-position-fallback",
        action="store_true",
        help="disable the last IK fallback that releases the orientation and solves position only",
    )
    parser.add_argument(
        "--ik-backtrack-factors",'''


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    path = Path(sys.argv[1]).expanduser()
    text = path.read_text()
    if "orientation released" in text:
        print("이미 적용됨:", path)
        return 0
    if text.count(OLD) != 1 or text.count(ARG_ANCHOR) != 1:
        print("기준 문자열이 1곳이 아님: solve_ik %d, argparse %d" % (text.count(OLD), text.count(ARG_ANCHOR)))
        return 1
    backup = path.with_suffix(path.suffix + ".bak_ik_posfb")
    shutil.copy2(path, backup)
    text = text.replace(OLD, NEW, 1).replace(ARG_ANCHOR, ARG_NEW, 1)
    path.write_text(text)
    print("적용 완료:", path)
    print("백업:", backup)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
