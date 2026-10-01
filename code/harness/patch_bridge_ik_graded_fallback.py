#!/usr/bin/env python3
"""IK fallback 을 단계형으로 바꾼다 (patch_bridge_ik_position_fallback.py 적용 뒤에 실행).

「위치만」 fallback 은 IK 실패를 0 으로 만들었지만 손 방향이 55~82° 틀어져 파지가 안 됐다.
대신 기록된 목표 방향을 두고 허용 오차를 10° → 20° → 30° → 45° 로 넓혀 가며 풀고,
그래도 안 되면 마지막에만 방향을 푼다.  MuJoCo OSC 가 먼 목표에서 손을 「조금」 기울이는
것과 같은 타협이다.

    python3 patch_bridge_ik_graded_fallback.py ~/isaac_libero/bridge/run_openvla_isaac_bridge.py

    --ik-orientation-steps-deg 10 20 30 45   단계 (기본).  빈 목록이면 단계 없이 곧바로 풀기
"""

import shutil
import sys
from pathlib import Path

OLD = '''            if not bool(success) and not getattr(self.args, "no_ik_position_fallback", False):
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
                    used_rotation = fallback_rotations[0][1].copy()'''
NEW = '''            if not bool(success) and not getattr(self.args, "no_ik_position_fallback", False):
                # 단계적 완화: 기록된 목표 방향을 두고 허용 오차를 넓혀 가며 푼다.  팔을 거의
                # 다 편 자세에서는 정확한 방향으로는 해가 없지만 10~30° 안에서는 대개 풀린다
                # (MuJoCo OSC 도 먼 목표에서 손을 조금 기울인다).  전부 실패하면 마지막에만
                # 방향을 푼다 — 완전히 풀면 55~82° 틀어져 파지가 안 되므로 최후 수단이다.
                steps_deg = getattr(self.args, "ik_orientation_steps_deg", None)
                if steps_deg is None:
                    steps_deg = [10.0, 20.0, 30.0, 45.0]
                for tol_deg in steps_deg:
                    tol_action, tol_success = self.kinematics.compute_inverse_kinematics(
                        target_position,
                        target_quaternion,
                        orientation_tolerance=float(np.radians(tol_deg)),
                    )
                    if bool(tol_success):
                        LOGGER.warning(
                            "[IK Recovery] solved with orientation tolerance %.0f deg at target=%s",
                            tol_deg, np.round(target_position, 4).tolist(),
                        )
                        action, success = tol_action, tol_success
                        used_rotation = target_rotation.copy()
                        break
            if not bool(success) and not getattr(self.args, "no_ik_position_fallback", False):
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
                    used_rotation = fallback_rotations[0][1].copy()'''

ARG_ANCHOR = '''    parser.add_argument(
        "--no-ik-position-fallback",'''
ARG_NEW = '''    parser.add_argument(
        "--ik-orientation-steps-deg",
        type=float,
        nargs="*",
        default=None,
        help="graded orientation tolerances (deg) tried before releasing orientation; default 10 20 30 45",
    )
    parser.add_argument(
        "--no-ik-position-fallback",'''


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    path = Path(sys.argv[1]).expanduser()
    text = path.read_text()
    if "orientation tolerance %.0f deg" in text:
        print("이미 적용됨:", path)
        return 0
    if text.count(OLD) != 1 or text.count(ARG_ANCHOR) != 1:
        print("기준 문자열이 1곳이 아님 (patch_bridge_ik_position_fallback.py 를 먼저 적용했는지 확인): %d / %d"
              % (text.count(OLD), text.count(ARG_ANCHOR)))
        return 1
    backup = path.with_suffix(path.suffix + ".bak_ik_graded")
    shutil.copy2(path, backup)
    path.write_text(text.replace(OLD, NEW, 1).replace(ARG_ANCHOR, ARG_NEW, 1))
    print("적용 완료:", path)
    print("백업:", backup)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
