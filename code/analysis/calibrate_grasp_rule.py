#!/usr/bin/env python3
"""「그릇을 물음」 기준을 정답을 아는 판으로 채점해서 고른다 (Isaac Sim · MuJoCo 따로). 기본은 읽기만 한다.

정답 묶음 (손가락 폭과 상관없는 그릇의 움직임으로 정함)
  물었던 판   성공한 판 / 그릇 중심이 5 mm 이상 올라간 판 / 기울기만 한 판(그릇을 둘러싼 상자의 높이가 5 mm 이상 커짐, Isaac Sim)
  안 문 판    그릇 중심이 2 mm 미만 올라가고, 5 mm 미만 밀리고, 기울지 않고(상자 높이 변화 3 mm 미만), 회전하지 않은(0.2 rad/s 미만) 실패 판
  그 밖       어느 쪽인지 알 수 없어 채점에서 뺌
채점 = (물었던 판을 「물음」으로 잡은 비율) − (안 문 판을 「물음」으로 잡은 비율). 가장 높은 후보를 고른다.
  같은 점수면 멈춤 기준이 없는(더 단순한) 쪽, 그다음 범위 아래쪽이 낮은 쪽.

    python3 calibrate_grasp_rule.py            # 채점 결과와 고른 기준, 고른 기준으로 다시 센 실패 종류(이전 규칙과 나란히)
    python3 calibrate_grasp_rule.py --write    # 고른 기준을 grasp_rule.json 에 써서 모든 분석 도구가 쓰게 함
"""
import argparse, glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vla_signals as vs

EXP = os.path.expanduser("~/LIBERO/experiments")
CELLS = [("OpZ·B·배치고정", "zs_s00_S10"), ("OpZ·M·배치고정", "zs_s00_S3_full"), ("OpZ·B·배치50", "OpZ_B_layout50"),
         ("OpZ·M·배치50", "zs_ovla_cal_50ep"), ("OpF4·B·배치고정", "OpF4_B_layout1"), ("OpF4·M·배치고정", "OpF4_M_layout1"),
         ("OpF4·B·배치50", "OpF4_B_layout50"), ("OpF4·M·배치50", "ft_v4b_4000_full"),
         ("MuJoCo 기본·배치50", "MJ_OpZ_default_layout50"), ("MuJoCo-B·배치50", "MJ_OpZ_B_layout50")]
OLD = {"isaac": ([0.0025, 0.012], None), "mujoco": ([0.0010, 0.006], None)}
BANDS = {"isaac": [[0.0025, 0.012], [0.0015, 0.012], [0.0025, 0.020]], "mujoco": [[0.0010, 0.018], [0.0015, 0.018], [0.0020, 0.018]]}
STALLS = [None, 0.0003, 0.0005, 0.001, 0.002]
ap = argparse.ArgumentParser(); ap.add_argument("--write", action="store_true"); a = ap.parse_args()


def truth(r, obs):
    s0 = vs.start(r, obs); o = obs[s0:]
    if r.get("task_success"):
        return "pos"
    cz = [x["target_bowl_aabb_center_m"][2] for x in o]; xy = [np.asarray(x["target_bowl_aabb_center_m"][:2]) for x in o]
    rise = max(cz) - cz[0]; shift = max(float(np.linalg.norm(q - xy[0])) for q in xy)
    if "target_bowl_aabb_max_m" in o[0]:
        h = [x["target_bowl_aabb_max_m"][2] - x["target_bowl_aabb_min_m"][2] for x in o]; tilt = max(h) - h[0]
    else:
        tilt = 0.0
    ang = max((x.get("bowl_angular_speed_radps") or 0.0) for x in o)
    if rise >= 0.005 or tilt >= 0.005:
        return "pos"
    if rise < 0.002 and shift < 0.005 and tilt < 0.003 and ang < 0.2:
        return "neg"
    return None


data = {}            # 결과 폴더 → [(r, 손가락 폭, 정답)]
for name, f in CELLS:
    rows = []
    for p in glob.glob(f"{EXP}/{f}/**/episode_result.json", recursive=True):
        r = json.load(open(p)).get("result", {}); obs = vs.observations(r)
        if obs:
            rows.append((r, vs.finger(obs)[vs.start(r, obs):], truth(r, obs)))
    data[f] = rows

chosen = {}
for simname in ("isaac", "mujoco"):
    rows = [x for _, f in CELLS for x in data[f] if vs.sim(x[0]) == simname]
    pos = [w for r, w, t in rows if t == "pos"]; neg = [w for r, w, t in rows if t == "neg"]
    if not pos or not neg:
        print(f"\n{simname}: 정답 묶음이 비어 채점할 수 없음 (물었던 판 {len(pos)}, 안 문 판 {len(neg)})"); continue
    print(f"\n=== {'Isaac Sim' if simname == 'isaac' else 'MuJoCo'} — 물었던 판 {len(pos)}개, 안 문 판 {len(neg)}개")
    print("  %-16s %-10s %14s %16s %8s" % ("범위 (mm)", "멈춤", "물었던 판 잡음", "안 문 판 잘못 잡음", "점수"))
    cands = [(OLD[simname][0], OLD[simname][1], "이전 규칙")] + [(b, s, "") for b in BANDS[simname] for s in STALLS]
    best = None
    for band, stall, tag in cands:
        tp = np.mean([vs.held_w(w, band, stall) for w in pos]); fp = np.mean([vs.held_w(w, band, stall) for w in neg]); sc = tp - fp
        key = (round(sc, 3), stall is None, -band[0])
        print("  %-16s %-10s %13.1f%% %15.1f%% %8.3f  %s" % ("%.1f–%.1f" % (1000 * band[0], 1000 * band[1]), "없음" if stall is None else "%.1f mm" % (1000 * stall), 100 * tp, 100 * fp, sc, tag))
        if not tag and (best is None or key > best[0]):
            best = (key, band, stall, tp, fp)
    _, band, stall, tp, fp = best
    chosen[simname] = {"held": band, "stall": stall}
    print("  → 고른 기준: 범위 %.1f–%.1f mm, 멈춤 %s (물었던 판 %.1f%% 잡음, 안 문 판 %.1f%% 잘못 잡음)" % (
        1000 * band[0], 1000 * band[1], "없음" if stall is None else "%.1f mm" % (1000 * stall), 100 * tp, 100 * fp))

print("\n=== 고른 기준으로 다시 센 실패 종류 (위: 이전 규칙, 아래: 고른 기준)")
short = ["성공", "못 듦", "물지 못함", "안 닫음", "옮기다 실패", "접시 근처"]
print("  %-20s %-8s " % ("시험", "기준") + " ".join("%-9s" % s for s in short))
for name, f in CELLS:
    if not data[f]:
        continue
    for tag, pick in (("이전", lambda s: OLD[s]), ("고른 것", lambda s: (chosen.get(s, {"held": OLD[s][0]})["held"], chosen.get(s, {"stall": None})["stall"]))):
        c = []
        for r, w, t in data[f]:
            band, stall = pick(vs.sim(r)); c.append(vs.classify(r, stall=stall, band=band))
        print("  %-20s %-8s " % (name if tag == "이전" else "", tag) + " ".join("%-9d" % c.count(k) for k in vs.KINDS))

if a.write and chosen:
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "grasp_rule.json")
    for s in chosen:
        chosen[s]["closed_below"] = vs.RULE[s]["closed_below"]
    json.dump(chosen, open(out, "w"), indent=1, ensure_ascii=False)
    print(f"\n고른 기준을 {out} 에 씀 — 이제 모든 분석 도구가 이 기준을 씀")
elif chosen:
    print("\n(아직 쓰지 않음. 결과를 확인한 뒤 --write 로 쓰면 모든 분석 도구에 반영됨)")
