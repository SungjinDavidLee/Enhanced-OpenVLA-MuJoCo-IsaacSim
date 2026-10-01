#!/usr/bin/env python3
"""정책 스텝당 물리 서브스텝 중 마지막에서만 렌더한다 (RTX 렌더 횟수 1/N).

카메라는 서브스텝이 끝난 뒤 읽으므로 캡처 화면은 동일하다. 재생(--sim-steps-per-action 20)
에서 렌더가 20 배 줄어, Isaac 인스턴스 여러 개를 동시에 돌려도 굶지 않는다.

    python3 patch_bridge_render_last_substep.py ~/isaac_libero/bridge/run_openvla_isaac_bridge.py
"""

import shutil
import sys
from pathlib import Path

HOLD_OLD = '''                        for _ in range(max(1, int(args.sim_steps_per_action))):
                            if not env_module.simulation_app.is_running():
                                stop_requested = True
                                break
                            scene.world.step(render=True)'''
HOLD_NEW = '''                        _n_sub = max(1, int(args.sim_steps_per_action))
                        for _sub in range(_n_sub):
                            if not env_module.simulation_app.is_running():
                                stop_requested = True
                                break
                            scene.world.step(render=(_sub == _n_sub - 1))'''

MAIN_OLD = '''                scene.world.step(render=True)
                action_physics_steps += 1'''
MAIN_NEW = '''                scene.world.step(render=(sim_step == max(1, int(args.sim_steps_per_action)) - 1))
                action_physics_steps += 1'''


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    path = Path(sys.argv[1]).expanduser()
    text = path.read_text()
    if "render=(sim_step ==" in text:
        print("이미 적용됨:", path)
        return 0
    for a in (HOLD_OLD, MAIN_OLD):
        if text.count(a) != 1:
            print("기준 문자열이 1곳이 아님:\n" + a)
            return 1
    backup = path.with_suffix(path.suffix + ".bak_render")
    shutil.copy2(path, backup)
    text = text.replace(HOLD_OLD, HOLD_NEW, 1).replace(MAIN_OLD, MAIN_NEW, 1)
    path.write_text(text)
    print("적용 완료:", path)
    print("백업:", backup)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
