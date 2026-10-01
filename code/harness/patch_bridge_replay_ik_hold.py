#!/usr/bin/env python3
"""EE-pose 재생 중 서브스텝 보간에서 난 IK 실패를 잡아 그 서브스텝만 건너뛴다.

지금은 물리 서브스텝 루프 안의 bridge.execute_pose_target() 이 IKFailure 를 던지면
메인 루프의 except (5121행) 바깥이라 브리지가 통째로 종료된다 (task 4·9 재생이 첫 IK 실패에서
죽은 원인). --replay-hold-on-ik-failure 가 켜져 있으면 실패한 서브스텝은 현재 자세를 유지하고
계속 간다.  꺼져 있으면 기존처럼 전파한다.

    python3 patch_bridge_replay_ik_hold.py ~/isaac_libero/bridge/run_openvla_isaac_bridge.py
"""

import shutil
import sys
from pathlib import Path

OLD = '''                    bridge.execute_pose_target(
                        interpolated_pose,
                        1.0 if close_transition else action.gripper,
                    )
                scene.world.step('''
NEW = '''                    try:
                        bridge.execute_pose_target(
                            interpolated_pose,
                            1.0 if close_transition else action.gripper,
                        )
                    except IKFailure as ik_error:
                        if not getattr(args, "replay_hold_on_ik_failure", False):
                            raise
                        total_ik_failures += 1
                        LOGGER.warning(
                            "[Replay IK Hold] substep %d/%d failed; holding current pose: %s",
                            sim_step + 1, max(1, int(args.sim_steps_per_action)), ik_error,
                        )
                        bridge.hold_current_pose(1.0 if close_transition else action.gripper)
                scene.world.step('''


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    path = Path(sys.argv[1]).expanduser()
    text = path.read_text()
    if "[Replay IK Hold]" in text:
        print("이미 적용됨:", path)
        return 0
    if text.count(OLD) != 1:
        print("기준 문자열이 1곳이 아님 (%d곳)" % text.count(OLD))
        return 1
    backup = path.with_suffix(path.suffix + ".bak_replay_ik")
    shutil.copy2(path, backup)
    path.write_text(text.replace(OLD, NEW, 1))
    print("적용 완료:", path)
    print("백업:", backup)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
