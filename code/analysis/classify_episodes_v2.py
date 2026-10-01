#!/usr/bin/env python3
"""에피소드를 네 부류로 나누고, 부류마다 무엇이 다른지 지표로 비교한다.

  A  성공                      task_success
  B  물었는데 못 들어 올림        실패, 그릇 최대 상승 < 40 mm, 손가락 폭 2.5–12 mm 에 3 샘플 이상 머묾 (테두리를 실제로 묾)
  C  허공에서 닫음               실패, 위 흔적 없음 (닫지 않았거나 빈손으로 닫힘)
  D  들고 가다 실패              실패, 그릇 최대 상승 ≥ 40 mm, 마지막 그릇이 접시에서 100 mm 이상
  E  접시 근처에서 실패           실패, 그릇 최대 상승 ≥ 40 mm, 마지막 그릇이 접시에서 100 mm 안 (놓기 · 안착 실패)

    python3 classify_episodes.py ~/LIBERO/experiments/zs_s00_S10 ~/LIBERO/experiments/zs_oft_s00

지표 (부류별 중앙값)
  첫닫힘 수평      첫 닫힘 때 손끝 − 그릇 중심 수평 거리 (mm)
  첫닫힘 z         첫 닫힘 때 손끝 − 그릇 윗면 (mm, 음수 = 테두리 아래로 내려감)
  닫힌 폭          첫 닫힘 뒤 손가락 평균 폭 (mm). 0 에 가까우면 허공, 3–8 이면 테두리를 문 것
  닫기 횟수        손가락이 열림(>30 mm) → 닫힘(<12 mm) 으로 바뀐 횟수
  첫닫힘 스텝      첫 닫힘까지 걸린 정책 스텝
  최대 상승        그릇 바닥의 최대 상승 (mm)
  끝 거리          마지막 그릇 중심 − 접시 중심 수평 거리 (mm)
  끝 높이          마지막 그릇 바닥 − 접시 윗면 (mm)
"""

from __future__ import annotations

import glob
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np


def features(r: dict):
    tr = [s for s in r.get("stage_trace") or [] if s.get("observations")]
    if not tr:
        return None
    o0 = tr[0]["observations"]
    z0 = o0["target_bowl_aabb_min_m"][2]
    w = np.array([s["observations"].get("gripper_finger_mean_m") or 0.04 for s in tr])
    bz = np.array([s["observations"]["target_bowl_aabb_min_m"][2] for s in tr])
    held = any(s["observations"].get("gripper_holding_ok") for s in tr)
    closes = 0; opened = True
    for x in w:
        if opened and x < 0.012:
            closes += 1; opened = False
        elif not opened and x > 0.030:
            opened = True
    ci = next((i for i, x in enumerate(w) if x < 0.012), None)
    f = {"closes": closes, "lift": (bz.max() - z0) * 1000, "held": held}
    if ci is not None:
        o = tr[ci]["observations"]; ie = np.array(o["eef_position_m"]); b = np.array(o["target_bowl_aabb_center_m"])
        f["close_h"] = np.linalg.norm((ie - b)[:2]) * 1000
        f["close_z"] = (ie[2] - o["target_bowl_aabb_max_m"][2]) * 1000
        after = w[ci:ci + 6]
        f["close_w"] = float(np.median(after)) * 1000
        f["close_step"] = tr[ci].get("policy_step", ci)
        # 실제로 무언가를 물었는가: 손가락 폭이 빈손 문턱(2.5 mm) 과 닫힘 문턱(12 mm) 사이에 3 샘플 이상 연속으로 머묾.
        # gripper_holding_ok 는 빈손에서도 켜지는 경우가 있어 쓰지 않는다.
        run = best = 0
        for x in w:
            run = run + 1 if 0.0025 < x < 0.012 else 0
            best = max(best, run)
        f["grip_evidence"] = best >= 3
    else:
        f["grip_evidence"] = False
    last = tr[-1]["observations"]
    if last.get("plate_aabb_center_m"):
        f["end_d"] = np.linalg.norm((np.array(last["target_bowl_aabb_center_m"]) - np.array(last["plate_aabb_center_m"]))[:2]) * 1000
        f["end_h"] = (last["target_bowl_aabb_min_m"][2] - last["plate_aabb_max_m"][2]) * 1000
    return f


def category(r, f):
    if r.get("task_success"):
        return "A"
    if f["lift"] >= 40:
        return "E" if f.get("end_d", 1e9) < 100 else "D"
    return "B" if f["grip_evidence"] else "C"


NAMES = {"A": "A 성공", "B": "B 물었는데 못 듦", "C": "C 허공에서 닫음", "D": "D 들고 가다 실패", "E": "E 접시 근처서 실패"}
KEYS = (("close_h", "첫닫힘 수평"), ("close_z", "첫닫힘 z"), ("close_w", "닫힌 폭"), ("closes", "닫기 횟수"),
        ("close_step", "첫닫힘 스텝"), ("lift", "최대 상승"), ("end_d", "끝 거리"), ("end_h", "끝 높이"))


def main() -> int:
    roots = [os.path.expanduser(p) for p in sys.argv[1:]] or ["/home/user/LIBERO/experiments/zs_s00_S10"]
    for root in roots:
        per_task = defaultdict(Counter); feats = defaultdict(lambda: defaultdict(list)); stages = defaultdict(Counter); bad = 0
        for fp in sorted(glob.glob(root + "/task_*_e*/episode_result.json")):
            r = json.load(open(fp)).get("result", {}); t = int(fp.split("/")[-2].split("_")[1])
            f = features(r)
            if f is None:
                bad += 1; continue
            c = category(r, f); per_task[t][c] += 1; stages[c]["S%s" % r.get("stage_level")] += 1
            for k, _ in KEYS:
                if k in f:
                    feats[c][k].append(f[k])
        print("\n==== %s  (기록 없는 판 %d 제외)" % (os.path.basename(root), bad))
        print("task   " + "   ".join("%-16s" % NAMES[c] for c in "ABCDE"))
        tot = Counter()
        for t in sorted(per_task):
            print("T%-4d  " % t + "   ".join("%-16d" % per_task[t][c] for c in "ABCDE")); tot.update(per_task[t])
        print("합계   " + "   ".join("%-16d" % tot[c] for c in "ABCDE"))
        print("\n부류별 중앙값        " + "  ".join("%-12s" % NAMES[c].split(" ")[0] for c in "ABCDE"))
        for k, name in KEYS:
            vals = []
            for c in "ABCDE":
                v = feats[c][k]
                vals.append(("%.0f" % np.median(v)) if v else "-")
            print("  %-16s   " % name + "  ".join("%-12s" % x for x in vals))
        print("\n도달 단계 분포")
        for c in "ABCDE":
            print("  %-18s %s" % (NAMES[c], dict(sorted(stages[c].items()))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
