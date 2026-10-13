#!/usr/bin/env python3
"""엄격한 기준: 「한 번에 집어서 한 번에 놓았는가」를 판마다 판정한다. 기존 성공 판정과 나란히 보여 준다. 읽기만 한다
(판별 결과 표만 ~/LIBERO/experiments/_analysis/strict_<결과 폴더>.csv 로 저장 — 영상 확인용).

판마다 붙는 이름
  정석 성공        기존 기준 성공 + 아래 ①②③ 모두 만족
  회귀 성공        기존 기준 성공이지만, 접시로 옮긴 시도 전에 「그릇에 닿았는데 실패한 시도」가 있었거나, 들었다가 떨어뜨린 적이 있음
  판정 의심 성공    기존 기준 성공이지만 ① 또는 ③ 이 어긋남 (우연히 접시에 닿아 성공 판정이 났을 수 있음 — 영상으로 확인)
  회귀 실패        실패. 그릇에 닿았는데 실패했거나 떨어뜨린 뒤, 다시 집기를 시도함
  그 밖의 실패      실패. 실패 종류는 vla_signals.classify (집게를 닫지 않음 등)

정석 성공의 세 조건
  ① 제대로 들었다      집게가 닫힌 채(35 mm 아래) 그릇 바닥이 처음보다 1 cm 이상 떴고, 그동안 (가) 손도 그릇이 오른 높이의
                       절반 이상 같이 올라갔거나 (나) 그릇이 접시 중심에서 8 cm 밖에 있었음
                       (판 4, 10/8: 그릇이 접시에 붙어 있는 과제 0 · 8 에서는 (나)만으로는 제대로 들어도 걸렸음.
                        밀어서 접시 턱을 타고 오르면 손은 그대로라 (가)에 걸리지 않음)
  ② 한 번에 옮겼다      그 시도 전에 「닿았는데 실패한 시도」가 없고, 들어 올린 뒤 접시에 내려앉을 때까지 떨어뜨림이 없음
  ③ 놓아서 내려놓았다   그릇이 접시에 내려앉을 때 아직 물고 있었거나, 마지막으로 물고 있던 순간의 그릇 바닥이 접시 윗면에서
                       3 cm 이내 (미끄러져 떨어지거나 높은 데서 떨어뜨려 우연히 접시에 걸린 것을 거름)
                       「물고 있음」 = 집게가 닫혀 있고(35 mm 아래) 그릇이 손과 함께 움직임 — 그릇과 손의 높이 차가 가장 높이
                       들었던 순간과 1.5 cm 안으로 같음 (세게 쥐어 손가락이 범위 아래여도 물고 있는 것으로 봄, 미끄러지면 차가 벌어짐)

집기 시도 = 손가락 폭이 35 mm 아래로 내려갔다가 다시 35 mm 위로 올라갈 때까지 (vla_signals.CLOSED_BELOW)
닿음      = 그 시도 동안(끝난 뒤 2스텝 포함) 손가락이 그릇 벽 폭에서 멈췄거나, 그릇이 5 mm 이상 밀렸거나,
            (Isaac Sim) 기울어졌거나(둘러싼 상자 높이 3 mm 이상 변화) 돌기 시작함(0.2 rad/s 이상)
            → 닿지 않은 헛손질은 시도로 치지 않음
떨어뜨림   = 그릇 바닥이 2 cm 넘게 들렸다가, 접시 중심에서 12 cm 넘게 떨어진 곳에서 처음 높이 + 1 cm 안으로 내려왔고,
            그때 손가락이 그릇 벽 폭에 있지 않음 (판 2, 10/7: 접시 가장자리에 내려놓는 것을 떨어뜨림으로 세던 것을 고침)
접시 위    = 그릇 바닥이 접시 윗면 + 1 cm 안이고, 그릇 중심과 접시 중심의 수평 거리 8 cm 안
MuJoCo 는 그릇 · 접시의 중심 위치만 있어서, 높이는 그 결과 폴더의 성공 판들이 끝났을 때의 (그릇 중심 − 접시 중심) 높이
차를 「접시 위에 놓인 높이」로 보고 그 차이로 잰다.

    python3 strict_outcomes.py                       # 있는 결과 폴더 모두
    python3 strict_outcomes.py zs_oft_cal_50ep Pi05Z_M_layout50
"""
import csv, glob, json, os, statistics, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vla_signals as vs
_sys_path = __import__("os").path.expanduser("~/tsk/VLA-repo_all/scripts/analysis")
__import__("sys").path.insert(0, _sys_path)
import episodes as epi   # 판 폴더 찾기 — 새 구조(task_XX/eYYY) · 옛 구조(task_XX_eYYY) 모두

EXP = os.path.expanduser("~/LIBERO/experiments")
OUT = os.path.join(EXP, "_analysis")
LABEL = {"zs_s00_S10": "OpZ·B·배치고정", "zs_s00_S3_full": "OpZ·M·배치고정", "OpZ_B_layout50": "OpZ·B·배치50", "zs_ovla_cal_50ep": "OpZ·M·배치50",
         "OpF4_B_layout1": "OpF4·B·배치고정", "OpF4_M_layout1": "OpF4·M·배치고정", "OpF4_B_layout50": "OpF4·B·배치50", "ft_v4b_4000_full": "OpF4·M·배치50",
         "zs_oft_s00": "OFT·B·배치고정", "zs_oft_cal_50ep": "OFT·M·배치50", "Pi05Z_M_layout50": "Pi05Z·M·배치50", "Pi05Z_B_layout50": "Pi05Z·B·배치50",
         "Pi05Z_B_layout1": "Pi05Z·B·배치고정", "Pi05Z_M_layout1": "Pi05Z·M·배치고정", "MJ_Pi05Z_default_layout50": "MuJoCo 기본·Pi05Z",
         "MJ_Pi05Z_B_layout50": "MuJoCo-B·Pi05Z", "MJ_OFT_default_layout50": "MuJoCo 기본·OFT", "MJ_OFT_B_layout50": "MuJoCo-B·OFT",
         "OFT_M_layout1": "OFT·M·배치고정", "OFT_B_layout50": "OFT·B·배치50",
         "MJ_OpZ_default_layout50": "MuJoCo 기본·OpZ", "MJ_OpZ_B_layout50": "MuJoCo-B·OpZ", "MJ_OpF4_default_layout50": "MuJoCo 기본·OpF4", "MJ_OpF4_B_layout50": "MuJoCo-B·OpF4"}
LIFT, DROP_LIFT, TABLE_BAND, PLATE_XY, ON_PLATE_Z, RELEASE_MAX, DROP_AWAY = 0.04, 0.02, 0.01, 0.08, 0.01, 0.03, 0.12
LIFT_OFF = 0.01          # ① 접시 밖에서 떠 있어야 하는 높이
TOUCH_DISP, TOUCH_TILT, TOUCH_ANG = 0.005, 0.003, 0.2
NAMES = ["정석 성공", "회귀 성공", "판정 의심 성공", "회귀 실패", "그 밖의 실패"]


def load(folder):
    rs = []
    for t, e, p in epi.ep_results(f"{EXP}/{folder}"):
        r = json.load(open(p)).get("result", {}); obs = vs.observations(r)
        if obs:
            rs.append((epi.ep_id(t, e), r, obs[vs.start(r, obs):]))
    return rs


def mujoco_offset(rs):
    d = [o[-1]["target_bowl_aabb_center_m"][2] - o[-1]["plate_aabb_center_m"][2] for _, r, o in rs
         if r.get("task_success") and o[-1].get("plate_aabb_center_m")]
    return statistics.median(d) if d else 0.03


def judge(r, o, offset):
    s = vs.sim(r); band = vs.RULE[s]["held"]; stall = vs.RULE[s]["stall"]
    w = vs.finger(o); n = len(o)
    c = np.asarray([x["target_bowl_aabb_center_m"] for x in o], float)
    if s == "isaac":
        bot = np.asarray([x["target_bowl_aabb_min_m"][2] for x in o], float)
        hgt = np.asarray([x["target_bowl_aabb_max_m"][2] - x["target_bowl_aabb_min_m"][2] for x in o], float) if "target_bowl_aabb_max_m" in o[0] else np.zeros(n)
        ang = np.asarray([x.get("bowl_angular_speed_radps") or 0.0 for x in o], float)
        top = np.asarray([(x.get("plate_aabb_max_m") or [0, 0, -9])[2] for x in o], float)
        above = bot - top
    else:
        bot = c[:, 2]; hgt = np.zeros(n); ang = np.zeros(n)
        pz = np.asarray([x["plate_aabb_center_m"][2] for x in o], float)
        above = (c[:, 2] - pz) - offset
    pxy = np.asarray([(x.get("plate_aabb_center_m") or [9, 9])[:2] for x in o], float)
    dxy = np.linalg.norm(c[:, :2] - pxy, axis=1)
    on_plate = (above <= ON_PLATE_Z) & (np.linalg.norm(c[:, :2] - pxy, axis=1) <= PLATE_XY)
    rise = bot - bot[0]
    inband = np.asarray([band[0] < x < band[1] for x in w])
    eef = np.asarray([(x.get("eef_position_m") or [np.nan] * 3)[2] for x in o], float)
    rel = c[:, 2] - eef                                   # 그릇 중심과 손의 높이 차

    # 집기 시도와 닿음
    att, i = [], 0
    while i < n:
        if w[i] < vs.CLOSED_BELOW:
            j = i
            while j + 1 < n and w[j + 1] < vs.CLOSED_BELOW:
                j += 1
            k = min(n - 1, j + 2)
            hold = vs.held_w(w[i:j + 1], band, stall)
            touched = hold or float(np.max(np.linalg.norm(c[i:k + 1] - c[i], axis=1))) >= TOUCH_DISP \
                or float(np.max(hgt[i:k + 1]) - hgt[i]) >= TOUCH_TILT or float(np.max(ang[i:k + 1])) >= TOUCH_ANG
            seg = np.arange(i, j + 1)
            ground = [t for t in seg if rise[t] < 0.002]                     # 집게가 닫힌 채 그릇이 아직 바닥에 있던 마지막 순간
            t0 = ground[-1] if ground else i
            hand_up = (eef[seg] - eef[t0]) if not np.isnan(eef[t0]) else np.full(len(seg), -9.0)
            with_hand = (rise[seg] >= LIFT_OFF) & (hand_up >= 0.5 * rise[seg])          # (가) 손과 함께 올라감
            off_plate = (rise[seg] >= LIFT_OFF) & (dxy[seg] > PLATE_XY)                  # (나) 접시 밖에서 떠 있음
            lifted = bool(np.any(with_hand | off_plate))
            att.append({"s": i, "e": j, "touched": touched, "lifted": lifted})
            i = j + 1
        else:
            i += 1
    # 떨어뜨림
    drops, up = [], False
    for t in range(n):
        if rise[t] > DROP_LIFT:
            up = True
        elif up and rise[t] <= TABLE_BAND and dxy[t] > DROP_AWAY and not inband[t]:
            drops.append(t); up = False
    touched = [a for a in att if a["touched"]]
    m = {"시도": len(att), "닿은 시도": len(touched), "떨어뜨림": len(drops), "이유": "", "놓은 높이": None}

    if r.get("task_success"):
        carry = next((a for a in reversed(att) if a["lifted"]), None)
        if carry is None:
            return "판정 의심 성공", dict(m, 이유="① 손과 함께 1 cm 넘게 들어 올린 적 없음 (밀어서 올렸을 수 있음)")
        t_land = next((t for t in range(carry["s"], n) if on_plate[t] and rise[t] < LIFT), None)
        if t_land is None:
            t_land = next((t for t in range(carry["s"], n) if on_plate[t]), None)
        if t_land is None:
            return "판정 의심 성공", dict(m, 이유="접시 위에 놓인 순간을 찾지 못함")
        win = range(carry["s"], max(carry["s"] + 1, t_land + 1))
        t_ref = max(win, key=lambda t: rise[t])           # 가장 높이 들었던 순간 (확실히 물고 있음)
        if np.isnan(rel[t_ref]):
            holding = lambda t: bool(inband[t])           # 손 위치 기록이 없으면 손가락 범위로
        else:
            holding = lambda t: w[t] < vs.CLOSED_BELOW and abs(rel[t] - rel[t_ref]) < 0.015
        if holding(t_land):
            h = 0.0
        else:
            last_hold = next((t for t in range(t_land, carry["s"] - 1, -1) if holding(t)), None)
            h = float(above[last_hold]) if last_hold is not None else 9.0
        m["놓은 높이"] = h
        if h > RELEASE_MAX:
            return "판정 의심 성공", dict(m, 이유="③ 접시 위 %.0f mm 에서 놓침 · 미끄러짐" % (1000 * h) if h < 9 else "③ 물고 있던 순간 없음")
        prior = [a for a in touched if a["s"] < carry["s"]]
        dropped = [t for t in drops if t < t_land]
        if prior or dropped:
            return "회귀 성공", dict(m, 이유="닿았는데 실패한 시도 %d · 떨어뜨림 %d" % (len(prior), len(dropped)))
        return "정석 성공", m
    first_fail = next((x for x in touched if not x["lifted"]), None)
    retry = (first_fail is not None and any(a["s"] > first_fail["e"] for a in att)) or (drops and any(a["s"] > drops[0] for a in att))
    return ("회귀 실패" if retry else "그 밖의 실패"), m


def main():
    folders = sys.argv[1:] or [f for f in LABEL if os.path.isdir(os.path.join(EXP, f))]
    os.makedirs(OUT, exist_ok=True)
    print("%-18s %4s %8s | %8s %8s %10s %8s %10s | %6s %8s %8s" % ("시험", "판", "기존 성공", "정석 성공", "회귀 성공", "판정 의심", "회귀 실패", "그 밖 실패", "시도", "닿은 시도", "떨어뜨림판"))
    details = []
    for f in folders:
        rs = load(f)
        if not rs:
            continue
        off = mujoco_offset(rs) if vs.sim(rs[0][1]) == "mujoco" else 0.0
        rows = []
        for ep, r, o in rs:
            name, m = judge(r, o, off)
            rows.append((ep, bool(r.get("task_success")), name, m, vs.classify(r)))
        cnt = {k: sum(1 for x in rows if x[2] == k) for k in NAMES}; n = len(rows); old = sum(x[1] for x in rows)
        print("%-18s %4d %8s | %8s %8d %10d %8d %10d | %6.2f %8.2f %8d" % (
            LABEL.get(f, f)[:18], n, "%d (%.1f%%)" % (old, 100 * old / n), "%d (%.1f%%)" % (cnt[NAMES[0]], 100 * cnt[NAMES[0]] / n),
            cnt[NAMES[1]], cnt[NAMES[2]], cnt[NAMES[3]], cnt[NAMES[4]],
            np.mean([x[3]["시도"] for x in rows]), np.mean([x[3]["닿은 시도"] for x in rows]), sum(1 for x in rows if x[3]["떨어뜨림"])))
        with open(os.path.join(OUT, f"strict_{f}.csv"), "w", newline="", encoding="utf-8") as fh:
            wr = csv.writer(fh); wr.writerow(["판", "기존 성공", "엄격한 판정", "이유", "실패 종류", "집기 시도", "닿은 시도", "떨어뜨림", "놓은 높이 mm"])
            for ep, ok, name, m, kind in rows:
                h = m.get("놓은 높이"); wr.writerow([ep, int(ok), name, m["이유"], kind, m["시도"], m["닿은 시도"], m["떨어뜨림"], "" if h is None or h >= 9 else round(1000 * h)])
        hs = [1000 * x[3]["놓은 높이"] for x in rows if x[3].get("놓은 높이") is not None and x[3]["놓은 높이"] < 9]
        if hs:
            print("%-18s   성공 판의 놓은 높이(접시 윗면 위, mm): 중앙값 %.0f · 90%% %.0f · 최대 %.0f · 30 mm 넘음 %d판" % (
                "", np.median(hs), np.percentile(hs, 90), max(hs), sum(h > 30 for h in hs)))
        details.append((f, rows))
    print("\n판정 의심 성공의 이유, 이름별 예시 판 (영상 확인용 — 전체는 ~/LIBERO/experiments/_analysis/strict_<폴더>.csv)")
    for f, rows in details:
        sus = [x[3]["이유"].split(" ")[0] for x in rows if x[2] == "판정 의심 성공"]
        if sus or any(x[2] in ("회귀 성공", "회귀 실패") for x in rows):
            print(f"\n  {LABEL.get(f, f)}  판정 의심 이유: " + ", ".join(f"{k} {sus.count(k)}" for k in sorted(set(sus))) if sus else f"\n  {LABEL.get(f, f)}")
            for k in ("판정 의심 성공", "회귀 성공", "회귀 실패"):
                ex = [x[0] for x in rows if x[2] == k][:5]
                if ex:
                    print(f"    {k}: " + " ".join(ex))


if __name__ == "__main__":
    main()
