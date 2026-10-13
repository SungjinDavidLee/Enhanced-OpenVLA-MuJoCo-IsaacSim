#!/usr/bin/env python3
"""Table 1 의 Metric 을 결과 폴더에서 같은 정의로 계산한다 (Isaac Sim · MuJoCo 모두). 읽기만 한다.
판정 규칙은 vla_signals.py (같은 폴더) 하나에 모여 있다.

  gripper close / grasp   집게를 한 번이라도 닫은 판 / 그릇 벽 두께 범위에서 멈춘 채 물고 있던 판
  placement               성공 판정을 받은 판 (Isaac Sim: 우리 판정, MuJoCo: LIBERO 자체 판정)
  undisturbed (≤5 mm)     목표가 아닌 물체가 모두 5 mm 안에서만 움직인 판 (MuJoCo 만 기록 있음)
  median placement error  끝났을 때 그릇 중심과 접시 중심 사이 수평 거리의 중앙값
  radius 13/20/30/50 mm   그 거리가 각 반경 안인 판

    python3 table1_metrics.py OpZ·M·배치50 OpZ·B·배치50 "MuJoCo 기본·배치50" "MuJoCo-B·배치50"
    python3 table1_metrics.py zs_ovla_cal_50ep --keys
"""
import argparse, glob, json, os, statistics, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.expanduser("~/tsk/VLA-repo_all/scripts/analysis"))
import vla_signals as vs
_sys_path = __import__("os").path.expanduser("~/tsk/VLA-repo_all/scripts/analysis")
__import__("sys").path.insert(0, _sys_path)
import episodes as epi   # 판 폴더 찾기 — 새 구조(task_XX/eYYY) · 옛 구조(task_XX_eYYY) 모두

EXP = os.path.expanduser("~/LIBERO/experiments")
NAMES = {"OpZ·B·배치고정": "zs_s00_S10", "OpZ·M·배치고정": "zs_s00_S3_full", "OpZ·B·배치50": "OpZ_B_layout50",
         "OpZ·M·배치50": "zs_ovla_cal_50ep", "OpF4·B·배치고정": "OpF4_B_layout1", "OpF4·M·배치고정": "OpF4_M_layout1",
         "OpF4·B·배치50": "OpF4_B_layout50", "OpF4·M·배치50": "ft_v4b_4000_full",
         "MuJoCo 기본·배치50": "MJ_OpZ_default_layout50", "MuJoCo-B·배치50": "MJ_OpZ_B_layout50"}

ap = argparse.ArgumentParser()
ap.add_argument("runs", nargs="+")
ap.add_argument("--keys", action="store_true", help="판 기록의 항목 이름만 보여 줌")
a = ap.parse_args()

for run in a.runs:
    folder = NAMES.get(run, run)
    fs = [p for _t, _e, p in epi.ep_results(f"{EXP}/{folder}")]
    if not fs:
        print(f"\n{run}: 결과 없음 ({folder})"); continue
    if a.keys:
        r = json.load(open(fs[0])).get("result", {}); obs = vs.observations(r)
        print(f"\n{run} ({folder}) — 판 기록 항목:", sorted(obs[0].keys()) if obs else "없음")
        print("   결과 항목:", sorted(k for k in r if k != "stage_trace")); continue
    n = closed = held = ok = 0; err = []; undist = []
    for f in fs:
        r = json.load(open(f)).get("result", {}); n += 1; ok += bool(r.get("task_success"))
        obs = vs.observations(r)
        if not obs:
            continue
        closed += vs.closed(r, obs); held += vs.held(r, obs)
        e = vs.final_xy_error_mm(obs)
        if e is not None:
            err.append(e)
        m = vs.others_moved_m(r, obs)
        if m is not None:
            undist.append(m <= 0.005)
    print(f"\n{run} ({folder}, {n}판, {vs.sim(r)})")
    print("  gripper close / grasp    %.1f%% / %.1f%%" % (100 * closed / n, 100 * held / n))
    print("  placement (성공 판정)     %.1f%% (%d)" % (100 * ok / n, ok))
    print("  undisturbed (≤5 mm)      " + ("%.1f%%" % (100 * sum(undist) / len(undist)) if undist else "판 기록에 다른 물체 위치가 없음"))
    if err:
        print("  median placement error   %.1f mm" % statistics.median(err))
        print("  radius 13/20/30/50 mm    " + "/".join("%.1f" % (100 * sum(x <= r_ for x in err) / len(err)) for r_ in (13, 20, 30, 50)) + "%")
