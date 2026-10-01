#!/usr/bin/env python3
"""평가용 OFT yaml 두 개를 만든다 (본 yaml 과 task 5 용 _t5).  손목 카메라만 LIBERO 규격으로 바꾼다.

  LIBERO eye_in_hand: 손 프레임 (0.05, 0, 0), 손끝 방향 그대로(기울임 0), fovy 75°, 정사각 256²
  roll 180 · flip_horizontal 은 D405 조정 때 맞춘 화면 방향 규약이라 그대로 둔다.

    python3 make_oft_yaml.py ~/isaac_libero/bridge
"""
import re, sys
from pathlib import Path
root = Path(sys.argv[1]).expanduser()
for src_name in ("libero_spatial_environment.yaml", "libero_spatial_environment_t5.yaml"):
    src = root / src_name
    if not src.exists():
        print("없음:", src); continue
    t = src.read_text()
    if "wrist_camera:" not in t:
        print("  %s 에 wrist_camera 블록이 없음 → 본 yaml 에서 다시 만드세요 (안내 참고)" % src.name); continue
    i = t.index("wrist_camera:")
    head, blk = t[:i], t[i:]
    rep = {"enabled": "true", "position_xyz": "[0.050, 0.0, 0.0]", "tilt_deg": "0.0", "resolution": "[256, 256]",
           "horizontal_fov_deg": "75.0", "vertical_fov_deg": "75.0"}
    for k, v in rep.items():
        blk, n = re.subn(r"(?m)^(  %s:)\s*.*$" % k, r"\1 %s" % v, blk, count=1)
        if n == 0:
            print("  키 없음:", k)
    dst = root / src_name.replace("libero_spatial_environment", "libero_spatial_environment_oft")
    dst.write_text(head + blk)
    print("만듦:", dst)
    print("   " + "\n   ".join(l for l in blk.splitlines()[:16] if any(k in l for k in list(rep) + ["roll_deg", "flip_"])))
