#!/usr/bin/env python3
"""손목 프레임 뒤집기 옵션 (patch_wrist_camera.py 적용 뒤).

yaml wrist_camera 에
    flip_horizontal: true     좌우 반전 (GR00T 팀의 WFLIP=h 와 같은 조치)
    flip_vertical: false      상하 반전
캡처 시점에 적용되므로 wrist_rgb.mp4 · npz · 추론 입력이 모두 같은 화면이 된다.

    python3 patch_wrist_flip.py ~/isaac_libero/bridge/libero_spatial_env_isaac601.py
"""

import shutil
import sys
from pathlib import Path

OLD = '''        if rgb.ndim != 3 or rgb.shape[-1] < 3:
            return None
        return np.ascontiguousarray(rgb[..., :3])
'''
NEW = '''        if rgb.ndim != 3 or rgb.shape[-1] < 3:
            return None
        rgb = rgb[..., :3]
        cfg = self.config.get("wrist_camera") or {}
        if bool(cfg.get("flip_horizontal", False)):
            rgb = rgb[:, ::-1]
        if bool(cfg.get("flip_vertical", False)):
            rgb = rgb[::-1, :]
        return np.ascontiguousarray(rgb)
'''


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__); return 2
    p = Path(sys.argv[1]).expanduser(); t = p.read_text()
    if "flip_horizontal" in t:
        print("이미 적용됨:", p); return 0
    if t.count(OLD) != 1:
        print("기준 문자열이 1곳이 아님 (%d) — patch_wrist_camera.py 가 먼저 적용됐는지 확인" % t.count(OLD)); return 1
    shutil.copy2(p, p.with_suffix(p.suffix + ".bak_wflip")); p.write_text(t.replace(OLD, NEW, 1)); print("적용 완료:", p); return 0


if __name__ == "__main__":
    raise SystemExit(main())
