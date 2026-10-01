#!/usr/bin/env python3
"""Isaac 평가 결과 루트를 읽어 태스크별·전체 성공률과 실패 단계 분포를 집계한다.

    python3 summarize_isaac_eval.py ~/LIBERO/experiments/zs_faithful_50ep [--ref 422/500]

결과 폴더 이름은 task_XX_eYYY (eval_zs_all_isaac.sh 가 만드는 형식).  요약 텍스트가 아니라
episode_result.json 을 직접 읽으므로 재시작 · 병렬 실행과 무관하게 정확하다.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
from collections import Counter


def z_two_prop(s1: int, n1: int, s2: int, n2: int) -> float:
    p1, p2 = s1 / n1, s2 / n2
    p = (s1 + s2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    return (p1 - p2) / se if se > 0 else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root")
    ap.add_argument("--ref", default=None, help="비교 기준 성공/전체, 예: 422/500 (MuJoCo 제로샷)")
    a = ap.parse_args()

    rows = []
    for d in sorted(glob.glob(os.path.join(a.root, "task_*_e*"))):
        if not os.path.isdir(d):
            continue
        fs = glob.glob(d + "/**/episode_result.json", recursive=True)
        if not fs:
            continue
        r = json.load(open(fs[0]))["result"]
        name = os.path.basename(d)
        t = int(name.split("_")[1]); e = int(name.split("_e")[-1])
        rows.append((t, e, bool(r.get("task_success")), str(r.get("stage_level", "?")),
                     int(r.get("processed_steps", 0)), str(r.get("termination_reason", ""))))
    if not rows:
        print("결과 없음:", a.root); return 1

    print("task   성공/판   성공률    실패 단계 분포")
    tot_s = tot_n = 0
    for t in range(10):
        tr = [x for x in rows if x[0] == t]
        if not tr:
            print("task %d   (없음)" % t); continue
        s = sum(1 for x in tr if x[2]); n = len(tr); tot_s += s; tot_n += n
        fails = Counter("S" + x[3] for x in tr if not x[2])
        print("task %d   %3d/%-3d   %5.1f%%    %s" % (t, s, n, 100 * s / n,
              "  ".join("%s:%d" % kv for kv in sorted(fails.items()))))
    print("전체     %3d/%-3d   %5.1f%%" % (tot_s, tot_n, 100 * tot_s / tot_n))

    term = Counter(x[5] for x in rows)
    print("\n종료 사유:", "  ".join("%s:%d" % kv for kv in term.most_common()))
    early = sum(v for k, v in term.items() if "stopped_before" in k)
    print("조기 종료(stopped_before_*):", early, "판")

    if a.ref:
        rs, rn = (int(v) for v in a.ref.split("/"))
        z = z_two_prop(tot_s, tot_n, rs, rn)
        print("\n기준 %d/%d (%.1f%%) 대비  z = %+.2f" % (rs, rn, 100 * rs / rn, z))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
