#!/usr/bin/env python3
"""보고서용 표 · 그래프를 결과 폴더 24개(모델 4 × 칸 6)에서 한 번에 만든다. 결과 폴더와 분석 코드는 읽기만 한다.

무엇을 하나
  1단계 (모으기)  결과 폴더 24개의 판 기록(episode_result.json)을 모두 읽어 판마다 한 줄씩 episodes_all.csv 에 적는다.
                  판정은 기존 분석 코드를 그대로 불러 쓴다. 새 규칙을 만들지 않는다.
                  거리 · 높이 값은 반올림하지 않고 그대로 적는다 (경계값 판정이 기존 도구와 똑같도록, 10/13 고침).
                    실패 종류                    vla_signals.classify
                    한번에 옮겼는지 (다섯 가지)   strict_outcomes.judge
                    Table 1 지표                 vla_signals.closed · held · final_xy_error_mm · others_moved_m
  2단계 (만들기)  episodes_all.csv 하나에서 표(CSV), 마크다운 표(한국어 · 영어), 그래프(한국어 · 영어)를 만든다.
                  모든 보고서가 이 파일 하나에서 숫자를 가져가므로 보고서끼리 숫자가 어긋나지 않는다.

바꾸지 않는 것
  ~/LIBERO/experiments 의 결과 폴더, ~/tsk/VLA-repo_all/scripts/analysis 의 분석 코드 — 모두 읽기만 한다.
  새로 쓰는 곳은 --out 폴더 하나뿐이다 (기본 ~/tsk/VLA-repo_all/middle_/assets/<오늘 날짜>).

실행
  python3 make_report_assets_v2.py               # 1단계 + 2단계 (처음 한 번. 판 12,000개를 읽으므로 오래 걸릴 수 있음)
  python3 make_report_assets_v2.py --reuse       # 2단계만 (이미 만든 episodes_all.csv 로 표 · 그래프만 다시)
  python3 make_report_assets_v2.py --jobs 6      # 결과 폴더를 동시에 6개씩 읽음 (기본 4, 메모리가 모자라면 줄일 것)

결과 (--out 아래)
  episodes_all.csv      판마다 한 줄 (모든 표 · 그래프의 원천)
  screen.csv            OpenVLA 재학습 단계 고르기 (과제마다 0~4번 판, 같은 50판)
  collect_log.txt       결과 폴더별 판 수 · 읽기 오류 · MuJoCo 높이 기준값
  tables/               표 CSV (영문 열 이름)
  md/                   보고서에 붙여 넣을 마크다운 표 (overall · openvla · oft · pi05, 한국어 _ko · 영어 _en)
  figures_ko/ figures_en/   그래프 (같은 파일 이름, 언어만 다름)
  README_assets.md      위 파일 목록과 설명
"""
import argparse
import collections
import csv
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

HOME = os.path.expanduser("~")
DEF_EXP = os.path.join(HOME, "LIBERO", "experiments")
DEF_ADIR = os.path.join(HOME, "tsk", "VLA-repo_all", "scripts", "analysis")
DEF_OUT = os.path.join(HOME, "tsk", "VLA-repo_all", "middle_", "assets", time.strftime("%Y-%m-%d"))

# ------------------------------------------------------------------ 시험 목록
MODELS = ["OpZ", "OpF4", "OFT", "Pi05"]
CELLS = ["mj", "mjb", "M1", "M50", "B1", "B50"]
ISAAC = ["M1", "M50", "B1", "B50"]
# MuJoCo 대비 남은 비율은 MuJoCo 데이터로만 학습한 모델에만 뜻이 있다 (재학습 모델 OpF4 는 Isaac Sim 데이터로 학습했으므로 뺀다)
RET_MODELS = ["OpZ", "OFT", "Pi05"]
FOLDER = {
    ("OpZ", "mj"): "MJ_OpZ_default_layout50", ("OpZ", "mjb"): "MJ_OpZ_B_layout50",
    ("OpZ", "M1"): "zs_s00_S3_full", ("OpZ", "M50"): "zs_ovla_cal_50ep",
    ("OpZ", "B1"): "zs_s00_S10", ("OpZ", "B50"): "OpZ_B_layout50",
    ("OpF4", "mj"): "MJ_OpF4_default_layout50", ("OpF4", "mjb"): "MJ_OpF4_B_layout50",
    ("OpF4", "M1"): "OpF4_M_layout1", ("OpF4", "M50"): "ft_v4b_4000_full",
    ("OpF4", "B1"): "OpF4_B_layout1", ("OpF4", "B50"): "OpF4_B_layout50",
    ("OFT", "mj"): "MJ_OFT_default_layout50", ("OFT", "mjb"): "MJ_OFT_B_layout50",
    ("OFT", "M1"): "OFT_M_layout1", ("OFT", "M50"): "zs_oft_cal_50ep",
    ("OFT", "B1"): "zs_oft_s00", ("OFT", "B50"): "OFT_B_layout50",
    ("Pi05", "mj"): "MJ_Pi05Z_default_layout50", ("Pi05", "mjb"): "MJ_Pi05Z_B_layout50",
    ("Pi05", "M1"): "Pi05Z_M_layout1", ("Pi05", "M50"): "Pi05Z_M_layout50",
    ("Pi05", "B1"): "Pi05Z_B_layout1", ("Pi05", "B50"): "Pi05Z_B_layout50",
}
# OpenVLA 재학습 단계 고르기: 과제마다 0~4번 판(같은 50판). 원본은 OpZ · M · 배치 50 에서 같은 판만 뽑는다.
SCREEN = [("0", "zs_ovla_cal_50ep"), ("2000", "ft_v4b_2000_screen"), ("4000", "ft_v4b_4000_screen"),
          ("6000", "ft_v4b_6000_screen"), ("8000", "ft_v4b_8000_screen")]
SCREEN_EPS = range(5)
GROUPS = {"overall": MODELS, "openvla": ["OpZ", "OpF4"], "oft": ["OFT"], "pi05": ["Pi05"]}
SLUG = {"OpZ": "openvla_zs", "OpF4": "openvla_ft", "OFT": "oft", "Pi05": "pi05"}

# strict_outcomes.py 의 이름 (내부용) — 보고서에는 아래 CLS_KO / CLS_EN 으로 쓴다
CLS = ["정석 성공", "회귀 성공", "판정 의심 성공", "회귀 실패", "그 밖의 실패"]
# vla_signals.KINDS 와 같은 순서
KINDS = ["성공", "물었지만 들지 못함", "물지 못하고 닫음", "집게를 닫지 않음", "들었지만 접시까지 못 감", "접시 근처에서 실패"]

COLS = ["model", "cell", "folder", "sim", "task", "ep", "ep_id", "success", "trace", "steps", "outcome", "reason_code",
        "reason", "attempts", "touched", "drops", "release_mm", "kind", "closed", "held", "final_err_mm",
        "others_moved_m", "note"]

# ------------------------------------------------------------------ 이름표 (한국어 · 영어)
L = {
    "ko": {
        "model": {"OpZ": "OpenVLA 원본", "OpF4": "OpenVLA 재학습", "OFT": "OFT", "Pi05": "π0.5"},
        "model_long": {"OpZ": "OpenVLA 원본 (제로샷)", "OpF4": "OpenVLA 재학습 (Isaac Sim 데이터, 4000단계)",
                       "OFT": "OpenVLA-OFT (제로샷)", "Pi05": "π0.5 (제로샷)"},
        "cell": {"mj": "MuJoCo 기본", "mjb": "MuJoCo-B", "M1": "Isaac M · 배치 고정", "M50": "Isaac M · 배치 50",
                 "B1": "Isaac B · 배치 고정", "B50": "Isaac B · 배치 50"},
        "cell_br": {"mj": "MuJoCo\n기본", "mjb": "MuJoCo-B", "M1": "Isaac M\n배치 고정", "M50": "Isaac M\n배치 50",
                    "B1": "Isaac B\n배치 고정", "B50": "Isaac B\n배치 50"},
        "cls": ["놓치지 않고 한번에 옮김", "놓친 뒤 다시 집어 성공", "판정 의심 (영상 확인 필요)", "놓친 뒤 끝내 실패", "그 밖의 실패"],
        "kind": ["성공", "물었지만 들지 못함", "물지 못하고 닫음", "집게를 닫지 않음", "들었지만 접시까지 못 감", "접시 근처에서 실패"],
        "reason": {1: "1 들어 올린 적 없음", 2: "2 접시 위 순간 없음", 3: "3 놓을 때 쥐고 있지 않음"},
        "task": ["접시와 라메킨 사이", "라메킨 옆", "테이블 중앙", "쿠키 상자 위", "캐비닛 윗 서랍 안",
                 "라메킨 위", "쿠키 상자 옆", "스토브 위", "접시 옆", "나무 캐비닛 위"],
        "succ": "기존 판정 성공", "one": "놓치지 않고 한번에 옮김", "pct_ep": "판 비율 (%)", "rate": "성공률 (%)",
        "dark": "진한 부분: 한번에 옮김", "light": "연한 부분: 다시 집어 성공 · 판정 의심",
        "ret": "MuJoCo 기본 대비 남은 비율 (%)", "dist": "끝났을 때 그릇 중심 – 접시 중심 수평 거리 (mm)",
        "cum": "그 거리 안에 놓인 판 (%)", "height": "마지막으로 쥐고 있던 순간 그릇 바닥의 접시 윗면 위 높이 (mm, 로그 눈금)",
        "crit": "기준 3 cm", "diff": "차이 (%p)", "steps": "학습 단계", "succ50": "같은 50판 중 성공",
        "base": "원본",
    },
    "en": {
        "model": {"OpZ": "OpenVLA ZS", "OpF4": "OpenVLA FT", "OFT": "OFT ZS", "Pi05": "π0.5 ZS"},
        "model_long": {"OpZ": "OpenVLA (zero-shot)", "OpF4": "OpenVLA (fine-tuned on Isaac Sim data, 4000 steps)",
                       "OFT": "OpenVLA-OFT (zero-shot)", "Pi05": "π0.5 (zero-shot)"},
        "cell": {"mj": "MuJoCo default", "mjb": "MuJoCo-B", "M1": "Isaac M · fixed layout", "M50": "Isaac M · 50 layouts",
                 "B1": "Isaac B · fixed layout", "B50": "Isaac B · 50 layouts"},
        "cell_br": {"mj": "MuJoCo\ndefault", "mjb": "MuJoCo-B", "M1": "Isaac M\nfixed", "M50": "Isaac M\n50 layouts",
                    "B1": "Isaac B\nfixed", "B50": "Isaac B\n50 layouts"},
        "cls": ["One-pass success", "Success after regrasp", "Unverified success (video check)", "Failure after a lost grasp",
                "Other failure"],
        "kind": ["Success", "Grasped but not lifted", "Closed without grasping", "Never closed", "Lifted, not carried to plate",
                 "Failed near plate"],
        "reason": {1: "1 never lifted with the hand", 2: "2 no on-plate moment", 3: "3 not held at release"},
        "task": ["between plate and ramekin", "next to ramekin", "table center", "on cookie box", "in cabinet top drawer",
                 "on ramekin", "next to cookie box", "on stove", "next to plate", "on wooden cabinet"],
        "succ": "Success (original criterion)", "one": "One-pass success", "pct_ep": "% of episodes", "rate": "Success rate (%)",
        "dark": "dark: one-pass", "light": "light: after regrasp / unverified",
        "ret": "Retained vs MuJoCo default (%)", "dist": "Final bowl-to-plate center distance (mm)",
        "cum": "Episodes within that distance (%)", "height": "Bowl bottom above plate top at last hold (mm, log scale)",
        "crit": "3 cm criterion", "diff": "Difference (%p)", "steps": "Training steps", "succ50": "Successes on the same 50 episodes",
        "base": "base",
    },
}
COMPS = [("M1", "B1"), ("M50", "B50"), ("mj", "mjb"), ("M1", "M50"), ("B1", "B50"), ("mj", "M50"), ("mj", "B50")]
MC = {"OpZ": "#4C72B0", "OpF4": "#55A868", "OFT": "#C44E52", "Pi05": "#8172B2"}
CLC = ["#2E7D32", "#9CCC65", "#FBC02D", "#E57373", "#BDBDBD"]
KC = ["#2E7D32", "#FFB74D", "#E57373", "#9E9E9E", "#64B5F6", "#BA68C8"]
CC = {"mj": ("#000000", "-"), "mjb": ("#000000", "--"), "M1": ("#1F77B4", "--"), "M50": ("#1F77B4", "-"),
      "B1": ("#D62728", "--"), "B50": ("#D62728", "-")}


# ------------------------------------------------------------------ 통계
def wilson(k, n, z=1.96):
    if not n:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


def ztest(k1, n1, k2, n2):
    """두 비율의 차이 (2 − 1). 반환: 차이 %p, z, 양쪽 p, 차이의 95 % 구간."""
    if not n1 or not n2:
        return (float("nan"),) * 5
    p1, p2 = k1 / n1, k2 / n2
    p = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    z = (p2 - p1) / se if se > 0 else 0.0
    se_u = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    d = p2 - p1
    return 100 * d, z, math.erfc(abs(z) / math.sqrt(2)), 100 * (d - 1.96 * se_u), 100 * (d + 1.96 * se_u)


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")


def median(xs):
    xs = sorted(xs)
    if not xs:
        return float("nan")
    k = len(xs) // 2
    return xs[k] if len(xs) % 2 else (xs[k - 1] + xs[k]) / 2


def reason_code(s):
    s = s or ""
    if s.startswith("①"):
        return 1
    if s.startswith("접시 위에 놓인"):
        return 2
    if s.startswith("③"):
        return 3
    return 0


# ------------------------------------------------------------------ 1단계: 모으기
def collect_cell(job):
    model, cell, folder, exp, adir = job
    if adir not in sys.path:
        sys.path.insert(0, adir)
    import vla_signals as vs
    import episodes as epi
    import strict_outcomes as so
    base = os.path.join(exp, folder)
    if not os.path.isdir(base):
        return model, cell, folder, None, 0.0, []
    loaded = []
    for t, e, p in epi.ep_results(base):
        try:
            with open(p) as fh:
                r = json.load(fh).get("result", {}) or {}
            loaded.append((t, e, r, vs.observations(r), ""))
        except Exception as ex:  # noqa: BLE001
            loaded.append((t, e, None, [], "read %s: %s" % (type(ex).__name__, ex)))
    rs = [(epi.ep_id(t, e), r, obs[vs.start(r, obs):]) for t, e, r, obs, _ in loaded if r is not None and obs]
    off = so.mujoco_offset(rs) if rs and vs.sim(rs[0][1]) == "mujoco" else 0.0
    rows = []
    for t, e, r, obs, err in loaded:
        row = {k: "" for k in COLS}
        row.update(model=model, cell=cell, folder=folder, task=t, ep=e, ep_id=epi.ep_id(t, e), note=err)
        if r is None:
            rows.append(row)
            continue
        row.update(sim=vs.sim(r), success=int(bool(r.get("task_success"))), trace=int(bool(obs)), steps=len(obs))
        row["kind"] = vs.classify(r) or ""
        if obs:
            try:
                name, m = so.judge(r, obs[vs.start(r, obs):], off)
                h = m.get("놓은 높이")
                row.update(outcome=name, reason=m.get("이유", ""), reason_code=reason_code(m.get("이유", "")),
                           attempts=m.get("시도", ""), touched=m.get("닿은 시도", ""), drops=m.get("떨어뜨림", ""),
                           release_mm="" if h is None or h >= 9 else 1000 * h)
            except Exception as ex:  # noqa: BLE001
                row["note"] = "judge %s: %s" % (type(ex).__name__, ex)
            row["closed"] = int(bool(vs.closed(r, obs)))
            row["held"] = int(bool(vs.held(r, obs)))
            fe = vs.final_xy_error_mm(obs)
            row["final_err_mm"] = "" if fe is None else fe          # 반올림하지 않음 (13/20/30/50 mm 경계 판정이 바뀌지 않게)
            mv = vs.others_moved_m(r, obs)
            row["others_moved_m"] = "" if mv is None else mv          # m 그대로 (table1_metrics 와 같은 0.005 m 비교)
        rows.append(row)
    return model, cell, folder, len(loaded), off, rows


def collect_screen(exp, adir):
    if adir not in sys.path:
        sys.path.insert(0, adir)
    import episodes as epi
    out = []
    for label, folder in SCREEN:
        base = os.path.join(exp, folder)
        if not os.path.isdir(base):
            continue
        for t, e, p in epi.ep_results(base):
            if e in SCREEN_EPS:
                with open(p) as fh:
                    r = json.load(fh).get("result", {}) or {}
                out.append({"step": label, "folder": folder, "task": t, "ep": e, "success": int(bool(r.get("task_success")))})
    return out


def write_csv(path, rows, cols):
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def stage_collect(a):
    jobs = [(m, c, FOLDER[(m, c)], a.exp, a.adir) for m in MODELS for c in CELLS]
    rows, log = [], []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        futs = [ex.submit(collect_cell, j) for j in jobs]
        for f in as_completed(futs):
            m, c, folder, n, off, rr = f.result()
            rows += rr
            bad = sum(1 for r in rr if r["note"])
            msg = "%-5s %-4s %-28s 판 %s  읽기/판정 오류 %d  MuJoCo 높이 기준 %s  (%.0f초)" % (
                m, c, folder, "없음(폴더 없음)" if n is None else n, bad, "%.4f m" % off if off else "-", time.time() - t0)
            print(msg, flush=True)
            log.append(msg)
    key = {(m, c): i for i, (m, c) in enumerate((m, c) for m in MODELS for c in CELLS)}
    rows.sort(key=lambda r: (key[(r["model"], r["cell"])], int(r["task"]), int(r["ep"])))
    write_csv(os.path.join(a.out, "episodes_all.csv"), rows, COLS)
    sc = collect_screen(a.exp, a.adir)
    write_csv(os.path.join(a.out, "screen.csv"), sc, ["step", "folder", "task", "ep", "success"])
    with open(os.path.join(a.out, "collect_log.txt"), "w", encoding="utf-8") as fh:
        fh.write("결과 폴더: %s\n분석 코드: %s\n만든 시각: %s\n\n" % (a.exp, a.adir, time.strftime("%Y-%m-%d %H:%M")))
        fh.write("\n".join(sorted(log)) + "\n")
        fh.write("\n재학습 단계 고르기: %d판\n" % len(sc))


# ------------------------------------------------------------------ 2단계: 모아 세기
def load_rows(path):
    with open(path, encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        if "others_moved_m" not in r and r.get("others_moved_mm") not in (None, ""):   # 10/13 첫 판 CSV 와도 맞게
            r["others_moved_m"] = str(float(r["others_moved_mm"]) / 1000)
        for k in ("task", "ep", "success", "trace", "steps", "reason_code", "attempts", "touched", "drops", "closed", "held"):
            v = r.get(k)
            r[k] = int(float(v)) if v not in ("", None) else None
        for k in ("release_mm", "final_err_mm", "others_moved_m"):
            v = r.get(k)
            r[k] = float(v) if v not in ("", None) else None
    return rows


def aggregate(rows):
    by = collections.defaultdict(list)
    for r in rows:
        by[(r["model"], r["cell"])].append(r)
    A = {}
    for m in MODELS:
        for c in CELLS:
            rr = by.get((m, c), [])
            n = len(rr)
            d = {"n": n, "folder": FOLDER[(m, c)], "sim": rr[0]["sim"] if rr else ""}
            d["succ"] = sum(r["success"] or 0 for r in rr)
            oc = collections.Counter(r["outcome"] for r in rr)
            d["cls"] = [oc.get(k, 0) for k in CLS]
            d["one"] = d["cls"][0]
            d["unjudged"] = n - sum(d["cls"])
            d["att"] = mean([r["attempts"] for r in rr])
            d["touched"] = mean([r["touched"] for r in rr])
            d["drop_eps"] = sum(1 for r in rr if (r["drops"] or 0) > 0)
            sus = [r for r in rr if r["outcome"] == CLS[2]]
            d["reasons"] = [sum(1 for r in sus if r["reason_code"] == k) for k in (1, 2, 3)]
            hs = sorted(r["release_mm"] for r in sus if r["reason_code"] == 3 and r["release_mm"] is not None)
            d["heights"] = hs
            d["hb"] = [sum(h < 50 for h in hs), sum(50 <= h < 100 for h in hs), sum(h >= 100 for h in hs)]
            d["r3_nohold"] = d["reasons"][2] - len(hs)
            d["r1_tasks"] = collections.Counter(r["task"] for r in sus if r["reason_code"] == 1)
            kc = collections.Counter(r["kind"] for r in rr)
            d["kinds"] = [kc.get(k, 0) for k in KINDS]
            d["unkind"] = n - sum(d["kinds"])
            d["closed"] = sum(r["closed"] or 0 for r in rr)
            d["held"] = sum(r["held"] or 0 for r in rr)
            d["err"] = sorted(r["final_err_mm"] for r in rr if r["final_err_mm"] is not None)
            mv = [r["others_moved_m"] for r in rr if r["others_moved_m"] is not None]
            d["undist"] = (sum(x <= 0.005 for x in mv), len(mv))
            d["task_n"] = [sum(1 for r in rr if r["task"] == t) for t in range(10)]
            d["task_succ"] = [sum(1 for r in rr if r["task"] == t and r["success"]) for t in range(10)]
            d["task_one"] = [sum(1 for r in rr if r["task"] == t and r["outcome"] == CLS[0]) for t in range(10)]
            d["notes"] = sum(1 for r in rr if r.get("note"))
            A[(m, c)] = d
    return A


def pct(k, n):
    return 100.0 * k / n if n else float("nan")


def f1(x, sign=False):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return ("%+.1f" if sign else "%.1f") % x


def f2(x, sign=True):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return ("%+.2f" if sign else "%.2f") % x


def kp(k, n):
    return "—" if not n else "%d (%.1f%%)" % (k, pct(k, n))


def comp_row(A, ma, ca, mb, cb):
    """같은 측정으로 두 칸을 비교. 한번에 옮김은 판정 의심이 풀릴 때의 범위도 함께."""
    a, b = A[(ma, ca)], A[(mb, cb)]
    out = {}
    out["succ"] = ztest(a["succ"], a["n"], b["succ"], b["n"])
    out["one"] = ztest(a["one"], a["n"], b["one"], b["n"])
    sa, sb = a["cls"][2], b["cls"][2]
    lo = ztest(a["one"] + sa, a["n"], b["one"], b["n"])
    hi = ztest(a["one"], a["n"], b["one"] + sb, b["n"])
    out["range"] = (lo[0], hi[0], lo[1], hi[1])
    return out


# ------------------------------------------------------------------ 표 (CSV)
def write_tables(A, screen, out):
    td = os.path.join(out, "tables")
    os.makedirs(td, exist_ok=True)
    t1, t2, t3, t4, t5, t6 = [], [], [], [], [], []
    for m in MODELS:
        mj = A[(m, "mj")]
        for c in CELLS:
            d = A[(m, c)]
            n = d["n"]
            lo, hi = wilson(d["succ"], n)
            t1.append({"model": m, "cell": c, "folder": d["folder"], "n": n, "success": d["succ"],
                       "success_pct": f1(pct(d["succ"], n)), "ci95_low": f1(lo), "ci95_high": f1(hi),
                       "one_pass": d["one"], "one_pass_pct": f1(pct(d["one"], n)),
                       "one_pass_max_if_unverified_ok": d["one"] + d["cls"][2],
                       "retained_vs_mujoco_pct": f1(100 * pct(d["succ"], n) / pct(mj["succ"], mj["n"]) if mj["succ"] and n else float("nan")),
                       "one_pass_retained_vs_mujoco_pct": f1(100 * pct(d["one"], n) / pct(mj["one"], mj["n"]) if mj["one"] and n else float("nan"))})
            t2.append({"model": m, "cell": c, "n": n, "success": d["succ"], "one_pass": d["cls"][0], "regrasp_success": d["cls"][1],
                       "unverified_success": d["cls"][2], "failure_after_lost_grasp": d["cls"][3], "other_failure": d["cls"][4],
                       "not_judged": d["unjudged"], "mean_closings": f2(d["att"], False), "mean_touching_attempts": f2(d["touched"], False),
                       "episodes_with_drop": d["drop_eps"]})
            t3.append({"model": m, "cell": c, "unverified": d["cls"][2], "r1_never_lifted": d["reasons"][0],
                       "r2_no_on_plate_moment": d["reasons"][1], "r3_not_held_at_release": d["reasons"][2],
                       "r3_below_50mm": d["hb"][0], "r3_50_to_99mm": d["hb"][1], "r3_100mm_or_more": d["hb"][2],
                       "r3_never_held": d["r3_nohold"],
                       "r1_tasks": " ".join("%d:%d" % kv for kv in sorted(d["r1_tasks"].items())),
                       "r3_heights_mm": " ".join("%.0f" % h for h in d["heights"])})
            row = {"model": m, "cell": c, "n": n}
            for k, v in zip(["success", "grasped_not_lifted", "closed_without_grasping", "never_closed",
                             "lifted_not_carried", "failed_near_plate"], d["kinds"]):
                row[k] = v
            row["not_classified"] = d["unkind"]
            t4.append(row)
            e = d["err"]
            t5.append({"model": m, "cell": c, "n": n, "gripper_close_pct": f1(pct(d["closed"], n)), "grasp_pct": f1(pct(d["held"], n)),
                       "placement_pct": f1(pct(d["succ"], n)),
                       "undisturbed_pct": f1(pct(*d["undist"])) if d["undist"][1] else "", "undisturbed_n": d["undist"][1],
                       "median_err_mm": f1(median(e)), "within_13mm_pct": f1(pct(sum(x <= 13 for x in e), len(e))),
                       "within_20mm_pct": f1(pct(sum(x <= 20 for x in e), len(e))), "within_30mm_pct": f1(pct(sum(x <= 30 for x in e), len(e))),
                       "within_50mm_pct": f1(pct(sum(x <= 50 for x in e), len(e))), "err_n": len(e)})
            for t in range(10):
                t6.append({"model": m, "cell": c, "task": t, "task_name": L["en"]["task"][t], "n": d["task_n"][t],
                           "success": d["task_succ"][t], "one_pass": d["task_one"][t]})
    write_csv(os.path.join(td, "t1_success.csv"), t1, list(t1[0]))
    write_csv(os.path.join(td, "t2_outcomes.csv"), t2, list(t2[0]))
    write_csv(os.path.join(td, "t3_unverified.csv"), t3, list(t3[0]))
    write_csv(os.path.join(td, "t4_failure_kinds.csv"), t4, list(t4[0]))
    write_csv(os.path.join(td, "t5_metrics.csv"), t5, list(t5[0]))
    write_csv(os.path.join(td, "t6_per_task.csv"), t6, list(t6[0]))

    def comp_dict(kind, label, ma, ca, mb, cb):
        o = comp_row(A, ma, ca, mb, cb)
        res = []
        for meas in ("succ", "one"):
            d, z, p, lo, hi = o[meas]
            a, b = A[(ma, ca)], A[(mb, cb)]
            k1 = a["succ"] if meas == "succ" else a["one"]
            k2 = b["succ"] if meas == "succ" else b["one"]
            row = {"kind": kind, "comparison": label, "from_model": ma, "from_cell": ca, "to_model": mb, "to_cell": cb,
                   "measure": "success" if meas == "succ" else "one_pass", "from_k": k1, "from_n": a["n"], "to_k": k2, "to_n": b["n"],
                   "diff_pp": f1(d, True), "z": f2(z), "p": "%.4f" % p if not math.isnan(p) else "—", "ci95_low_pp": f1(lo, True),
                   "ci95_high_pp": f1(hi, True), "range_min_pp": "", "range_max_pp": "", "z_at_min": "", "z_at_max": ""}
            if meas == "one":
                rlo, rhi, zlo, zhi = o["range"]
                row.update(range_min_pp=f1(rlo, True), range_max_pp=f1(rhi, True), z_at_min=f2(zlo), z_at_max=f2(zhi))
            res.append(row)
        return res

    t7 = []
    for m in MODELS:
        for ca, cb in COMPS:
            t7 += comp_dict("within_model", "%s: %s -> %s" % (m, ca, cb), m, ca, m, cb)
    for c in CELLS:
        t7 += comp_dict("finetune", "OpZ -> OpF4: %s" % c, "OpZ", c, "OpF4", c)
    write_csv(os.path.join(td, "t7_within_model.csv"), t7, list(t7[0]))
    t8 = []
    for c in CELLS:
        for i, ma in enumerate(MODELS):
            for mb in MODELS[i + 1:]:
                t8 += comp_dict("between_models", "%s: %s -> %s" % (c, ma, mb), ma, c, mb, c)
    write_csv(os.path.join(td, "t8_between_models.csv"), t8, list(t8[0]))
    if screen:
        t9 = []
        for label, folder in SCREEN:
            rr = [r for r in screen if r["step"] == label]
            t9.append({"step": label, "folder": folder, "n": len(rr), "success": sum(int(r["success"]) for r in rr)})
        write_csv(os.path.join(td, "t9_finetune_step_screen.csv"), t9, list(t9[0]))


# ------------------------------------------------------------------ 마크다운 표
def md_table(head, rows):
    s = "| " + " | ".join(head) + " |\n|" + "|".join("---" for _ in head) + "|\n"
    for r in rows:
        s += "| " + " | ".join(str(x) for x in r) + " |\n"
    return s + "\n"


H = {
    "ko": {
        "cell": "칸", "n": "판", "succ": "기존 판정 성공", "ci": "95 % 구간", "one": "한번에 옮김", "onemax": "판정 의심이 모두 한번에일 때",
        "ret": "MuJoCo 기본 대비 (기존 / 한번에)", "att": "평균 닫은 횟수", "tou": "평균 닿은 시도", "susp": "판정 의심",
        "r3b": ["3 중 5 cm 미만", "5~10 cm", "10 cm 이상", "쥔 순간 없음"], "r1t": "1번이 나온 과제 (과제:판)",
        "close": "닫음 시도 %", "grasp": "집음 %", "place": "놓기 성공 %", "undist": "다른 물체 안 건드림 %",
        "med": "접시 중심까지 중앙값 (mm)", "rad": ["13 mm 안 %", "20 mm 안 %", "30 mm 안 %", "50 mm 안 %"],
        "task": "과제", "where": "그릇 위치", "sum": "합계", "comp": "비교", "now": "지금 차이 (z)",
        "range": "영상 확인 후 가능한 범위 (z)", "mj": "MuJoCo 기본",
        "s_succ": "성공률", "s_cls": "그릇을 놓치지 않고 한번에 옮긴 경우", "s_out": "모든 판을 다섯 가지로 나눔 (그릇을 놓치지 않고 한번에 옮겼는가)", "s_susp": "판정 의심의 이유와 놓은 높이",
        "s_kind": "실패 종류", "s_met": "자세한 지표 (Table 1 지표)", "s_task_s": "과제별 성공 (기존 판정, 과제당 판 수 대비)",
        "s_task_o": "과제별 한번에 옮긴 판", "s_comp": "한 가지만 바꿨을 때", "s_ft": "재학습 효과 (OpenVLA 원본 → 재학습, 같은 칸)",
        "s_screen": "재학습 단계 고르기 (OpZ · M · 배치 50 설정, 과제마다 0~4번 판, 같은 50판)",
        "overall_a": "기존 판정 성공 (모델 × 칸)", "overall_b": "놓치지 않고 한번에 옮긴 판 (모델 × 칸)",
        "overall_c": "MuJoCo 기본 대비 Isaac Sim 에서 남은 비율 (기존 판정 / 한번에 옮김, MuJoCo 데이터로만 학습한 모델)",
        "note_undist": "다른 물체 안 건드림은 다른 물체 위치 기록이 있는 판만 셈 (지금은 MuJoCo 만 기록이 있음).",
        "note_rad": "거리 지표는 실패 판도 포함한다.",
    },
    "en": {
        "cell": "Cell", "n": "Episodes", "succ": "Success (original)", "ci": "95% CI", "one": "One-pass", "onemax": "If all unverified are one-pass",
        "ret": "Retained vs MuJoCo default (original / one-pass)", "att": "Mean closings", "tou": "Mean touching attempts",
        "susp": "Unverified", "r3b": ["3: <5 cm", "5–10 cm", "≥10 cm", "never held"], "r1t": "Tasks with reason 1 (task:count)",
        "close": "Gripper close %", "grasp": "Grasp %", "place": "Placement %", "undist": "Undisturbed %",
        "med": "Median error (mm)", "rad": ["≤13 mm %", "≤20 mm %", "≤30 mm %", "≤50 mm %"],
        "task": "Task", "where": "Bowl location", "sum": "Total", "comp": "Comparison", "now": "Difference now (z)",
        "range": "Possible range after video check (z)", "mj": "MuJoCo default",
        "s_succ": "Success rate", "s_cls": "One-pass success (bowl carried without being lost)",
        "s_out": "All episodes in five groups (was the bowl carried in one pass?)",
        "s_susp": "Unverified successes: reasons and release height", "s_kind": "Failure kinds",
        "s_met": "Detailed metrics (Table 1 metrics)", "s_task_s": "Per-task success (original criterion)",
        "s_task_o": "Per-task one-pass successes", "s_comp": "One factor changed at a time",
        "s_ft": "Effect of fine-tuning (OpenVLA zero-shot → fine-tuned, same cell)",
        "s_screen": "Choosing the fine-tuning step (OpZ · M · 50 layouts setting, episodes 0–4 of every task, same 50 episodes)",
        "overall_a": "Success, original criterion (model × cell)", "overall_b": "One-pass successes (model × cell)",
        "overall_c": "Retained in Isaac Sim vs MuJoCo default (original / one-pass; models trained on MuJoCo data only)",
        "note_undist": "Undisturbed counts only episodes that log other objects' positions (currently MuJoCo only).",
        "note_rad": "Distance metrics include failed episodes.",
    },
}


def md_model(A, m, lang):
    h, lb = H[lang], L[lang]
    cells = [c for c in CELLS if A[(m, c)]["n"]]
    mj = A[(m, "mj")]
    s = "## %s\n\n" % lb["model_long"][m]
    rows = []
    for c in cells:
        d = A[(m, c)]
        lo, hi = wilson(d["succ"], d["n"])
        ret = "—"
        if m in RET_MODELS and c != "mj" and mj["n"] and mj["succ"] and mj["one"]:
            ret = "%.0f %% / %.0f %%" % (100 * pct(d["succ"], d["n"]) / pct(mj["succ"], mj["n"]),
                                         100 * pct(d["one"], d["n"]) / pct(mj["one"], mj["n"]))
        rows.append([lb["cell"][c], d["n"], kp(d["succ"], d["n"]), "%.1f–%.1f" % (lo, hi), kp(d["one"], d["n"]),
                     kp(d["one"] + d["cls"][2], d["n"]), ret])
    s += "### %s\n\n" % h["s_succ"] + md_table([h["cell"], h["n"], h["succ"], h["ci"], h["one"], h["onemax"], h["ret"]], rows)
    rows = []
    for c in cells:
        d = A[(m, c)]
        rows.append([lb["cell"][c], d["succ"], kp(d["cls"][0], d["n"]), d["cls"][1], d["cls"][2], d["cls"][3], d["cls"][4],
                     f2(d["att"], False), f2(d["touched"], False)])
    s += "### %s\n\n" % h["s_cls"] + md_table([h["cell"], h["succ"]] + lb["cls"] + [h["att"], h["tou"]], rows)
    rows = []
    for c in cells:
        d = A[(m, c)]
        rows.append([lb["cell"][c], d["cls"][2]] + d["reasons"] + d["hb"] + [d["r3_nohold"],
                    " ".join("%d:%d" % kv for kv in sorted(d["r1_tasks"].items())) or "—"])
    s += "### %s\n\n" % h["s_susp"] + md_table([h["cell"], h["susp"]] + [lb["reason"][k] for k in (1, 2, 3)] + h["r3b"] + [h["r1t"]], rows)
    rows = []
    for c in cells:
        d = A[(m, c)]
        rows.append([lb["cell"][c]] + d["kinds"])
    s += "### %s\n\n" % h["s_kind"] + md_table([h["cell"]] + lb["kind"], rows)
    rows = []
    for c in cells:
        d = A[(m, c)]
        e = d["err"]
        rows.append([lb["cell"][c], f1(pct(d["closed"], d["n"])), f1(pct(d["held"], d["n"])), f1(pct(d["succ"], d["n"])),
                     f1(pct(*d["undist"])) if d["undist"][1] else "—", f1(median(e))] +
                    [f1(pct(sum(x <= r for x in e), len(e))) for r in (13, 20, 30, 50)])
    s += "### %s\n\n" % h["s_met"] + md_table([h["cell"], h["close"], h["grasp"], h["place"], h["undist"], h["med"]] + h["rad"], rows)
    s += "%s %s\n\n" % (h["note_undist"], h["note_rad"])
    for key, title in (("task_succ", h["s_task_s"]), ("task_one", h["s_task_o"])):
        rows = []
        for t in range(10):
            rows.append([t, lb["task"][t]] + [("%d" % A[(m, c)][key][t]) if A[(m, c)]["task_n"][t] == 50 else
                                              "%d/%d" % (A[(m, c)][key][t], A[(m, c)]["task_n"][t]) for c in cells])
        rows.append([h["sum"], ""] + [sum(A[(m, c)][key]) for c in cells])
        s += "### %s\n\n" % title + md_table([h["task"], h["where"]] + [lb["cell"][c] for c in cells], rows)
    rows = []
    for ca, cb in COMPS:
        if not (A[(m, ca)]["n"] and A[(m, cb)]["n"]):
            continue
        o = comp_row(A, m, ca, m, cb)
        d, z = o["succ"][0], o["succ"][1]
        d1, z1 = o["one"][0], o["one"][1]
        rlo, rhi, zlo, zhi = o["range"]
        rows.append(["%s → %s" % (lb["cell"][ca], lb["cell"][cb]), "%s (%s)" % (f1(d, True), f2(z)),
                     "%s (%s)" % (f1(d1, True), f2(z1)), "%s ~ %s (%s ~ %s)" % (f1(rlo, True), f1(rhi, True), f2(zlo), f2(zhi))])
    s += "### %s\n\n" % h["s_comp"] + md_table([h["comp"], h["succ"] + " " + h["now"], h["one"] + " " + h["now"], h["one"] + " " + h["range"]], rows)
    return s


def md_finetune(A, screen, lang):
    h, lb = H[lang], L[lang]
    rows = []
    for c in CELLS:
        if not (A[("OpZ", c)]["n"] and A[("OpF4", c)]["n"]):
            continue
        o = comp_row(A, "OpZ", c, "OpF4", c)
        rows.append([lb["cell"][c], "%s → %s" % (f1(pct(A[("OpZ", c)]["succ"], A[("OpZ", c)]["n"])), f1(pct(A[("OpF4", c)]["succ"], A[("OpF4", c)]["n"]))),
                     "%s (%s)" % (f1(o["succ"][0], True), f2(o["succ"][1])),
                     "%s → %s" % (f1(pct(A[("OpZ", c)]["one"], A[("OpZ", c)]["n"])), f1(pct(A[("OpF4", c)]["one"], A[("OpF4", c)]["n"]))),
                     "%s (%s)" % (f1(o["one"][0], True), f2(o["one"][1]))])
    s = "## %s\n\n" % h["s_ft"] + md_table([h["cell"], h["succ"] + " %", h["now"], h["one"] + " %", h["now"]], rows)
    if screen:
        rows = []
        for label, folder in SCREEN:
            rr = [r for r in screen if r["step"] == label]
            if rr:
                rows.append([lb["base"] if label == "0" else label, folder, "%d / %d" % (sum(int(r["success"]) for r in rr), len(rr))])
        s += "## %s\n\n" % h["s_screen"] + md_table([L[lang]["steps"], "결과 폴더" if lang == "ko" else "result folder", L[lang]["succ50"]], rows)
    return s


def md_overall(A, lang):
    h, lb = H[lang], L[lang]
    s = ""
    for key, title in (("succ", h["overall_a"]), ("one", h["overall_b"])):
        rows = [[lb["model"][m]] + [kp(A[(m, c)][key], A[(m, c)]["n"]) for c in CELLS] for m in MODELS]
        s += "## %s\n\n" % title + md_table([""] + [lb["cell"][c] for c in CELLS], rows)
    rows = []
    for m in RET_MODELS:
        mj = A[(m, "mj")]
        r = [lb["model"][m]]
        for c in ISAAC:
            d = A[(m, c)]
            if mj["succ"] and mj["one"] and d["n"]:
                r.append("%.0f %% / %.0f %%" % (100 * pct(d["succ"], d["n"]) / pct(mj["succ"], mj["n"]),
                                                100 * pct(d["one"], d["n"]) / pct(mj["one"], mj["n"])))
            else:
                r.append("—")
        rows.append(r)
    s += "## %s\n\n" % h["overall_c"] + md_table([""] + [lb["cell"][c] for c in ISAAC], rows)
    return s


def write_md(A, screen, out):
    md = os.path.join(out, "md")
    os.makedirs(md, exist_ok=True)
    for lang in ("ko", "en"):
        with open(os.path.join(md, "overall_%s.md" % lang), "w", encoding="utf-8") as fh:
            fh.write(md_overall(A, lang))
        for g, ms in GROUPS.items():
            if g == "overall":
                continue
            s = "".join(md_model(A, m, lang) for m in ms)
            if g == "openvla":
                s += md_finetune(A, screen, lang)
            with open(os.path.join(md, "%s_%s.md" % (g, lang)), "w", encoding="utf-8") as fh:
                fh.write(s)


# ------------------------------------------------------------------ 그래프
def setup_mpl(lang):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager as fm
    fam = ["DejaVu Sans"]
    if lang == "ko":
        names = {f.name for f in fm.fontManager.ttflist}
        for want in ("NanumGothic", "NanumBarunGothic", "Noto Sans CJK KR", "Noto Sans KR", "Noto Sans CJK JP", "UnDotum", "Baekmuk Gulim"):
            if want in names:
                fam = [want, "DejaVu Sans"]
                break
        else:
            for p in ("/usr/share/fonts/truetype/nanum/NanumGothic.ttf", "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"):
                if os.path.exists(p):
                    fm.fontManager.addfont(p)
                    fam = [fm.FontProperties(fname=p).get_name(), "DejaVu Sans"]
                    break
            else:
                print("  !! 한국어 글꼴을 찾지 못함 — 한국어 그래프의 글자가 네모로 나올 수 있음. 설치: sudo apt install fonts-nanum", flush=True)
    plt.rcParams.update({"font.family": fam, "axes.unicode_minus": False, "figure.dpi": 100, "savefig.dpi": 200,
                         "axes.spines.top": False, "axes.spines.right": False, "font.size": 10})
    return plt


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)


def fig_success_all(plt, A, lang, path):
    import numpy as np
    lb = L[lang]
    fig, ax = plt.subplots(figsize=(12, 5.2))
    x = np.arange(len(CELLS))
    w = 0.2
    for i, m in enumerate(MODELS):
        xs = x + (i - 1.5) * w
        tot, one, lo, hi = [], [], [], []
        for c in CELLS:
            d = A[(m, c)]
            n = d["n"]
            tot.append(pct(d["succ"], n))
            one.append(pct(d["one"], n))
            a, b = wilson(d["succ"], n)
            lo.append(a)
            hi.append(b)
        tot, one, lo, hi = map(np.array, (tot, one, lo, hi))
        ax.bar(xs, one, w * 0.92, color=MC[m], label=lb["model"][m])
        ax.bar(xs, tot - one, w * 0.92, bottom=one, color=MC[m], alpha=0.35)
        ax.errorbar(xs, tot, yerr=[tot - lo, hi - tot], fmt="none", ecolor="#333333", lw=0.8, capsize=2)
        for xx, t, hh in zip(xs, tot, hi):
            if not math.isnan(t):
                ax.text(xx, hh + 1.2, "%.0f" % t, ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([lb["cell_br"][c] for c in CELLS])
    ax.set_ylim(0, 112)
    ax.set_ylabel(lb["rate"])
    ax.axvline(1.5, color="#999999", lw=0.8, ls=":")
    from matplotlib.patches import Patch
    hs, ls_ = ax.get_legend_handles_labels()
    hs += [Patch(color="#555555"), Patch(color="#555555", alpha=0.35)]
    ls_ += [lb["dark"], lb["light"]]
    ax.legend(hs, ls_, ncol=3, fontsize=8, loc="upper right", frameon=False)
    ax.set_title(lb["succ"] + " · " + lb["one"], fontsize=11)
    save(fig, path)


def fig_retention(plt, A, lang, path):
    import numpy as np
    lb = L[lang]
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.6), sharey=True)
    top = 0.0
    for ax, key, title in ((axs[0], "succ", lb["succ"]), (axs[1], "one", lb["one"])):
        x = np.arange(len(RET_MODELS))
        w = 0.2
        for j, c in enumerate(ISAAC):
            vals = []
            for m in RET_MODELS:
                mj, d = A[(m, "mj")], A[(m, c)]
                vals.append(100 * pct(d[key], d["n"]) / pct(mj[key], mj["n"]) if mj[key] and d["n"] else float("nan"))
            col, ls = CC[c]
            ax.bar(x + (j - 1.5) * w, vals, w * 0.92, color=col, alpha=0.45 if ls == "--" else 0.85,
                   hatch="//" if ls == "--" else None, edgecolor="white", label=lb["cell"][c])
            for xx, v in zip(x + (j - 1.5) * w, vals):
                if not math.isnan(v):
                    ax.text(xx, v + 1, "%.0f" % v, ha="center", fontsize=7)
                    top = max(top, v)
        ax.set_xticks(x)
        ax.set_xticklabels([lb["model"][m] for m in RET_MODELS])
        ax.set_title(title, fontsize=10)
        ax.axhline(100, color="#999999", lw=0.7, ls=":")
    axs[0].set_ylim(0, max(110, top * 1.1))
    axs[0].set_ylabel(lb["ret"])
    hs, ls_ = axs[0].get_legend_handles_labels()
    fig.legend(hs, ls_, ncol=4, fontsize=8, frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.04))
    fig.subplots_adjust(bottom=0.18)
    save(fig, path)


def fig_model_success(plt, A, m, lang, path):
    import numpy as np
    lb = L[lang]
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    x = np.arange(len(CELLS))
    bottom = np.zeros(len(CELLS))
    for k in range(3):
        v = np.array([pct(A[(m, c)]["cls"][k], A[(m, c)]["n"]) for c in CELLS])
        ax.bar(x, v, 0.6, bottom=bottom, color=CLC[k], label=lb["cls"][k])
        if k == 0:
            for xx, vv in zip(x, v):
                if vv >= 6:
                    ax.text(xx, vv / 2, "%.1f" % vv, ha="center", va="center", fontsize=8, color="white")
        bottom = bottom + np.nan_to_num(v)
    tot = np.array([pct(A[(m, c)]["succ"], A[(m, c)]["n"]) for c in CELLS])
    ci = [wilson(A[(m, c)]["succ"], A[(m, c)]["n"]) for c in CELLS]
    lo = np.array([a for a, _ in ci])
    hi = np.array([b for _, b in ci])
    ax.errorbar(x, tot, yerr=[tot - lo, hi - tot], fmt="none", ecolor="#333333", lw=0.8, capsize=3)
    for xx, t, hh in zip(x, tot, hi):
        if not math.isnan(t):
            ax.text(xx, hh + 1.5, "%.1f" % t, ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([lb["cell_br"][c] for c in CELLS])
    ax.set_ylim(0, 112)
    ax.set_ylabel(lb["rate"])
    ax.axvline(1.5, color="#999999", lw=0.8, ls=":")
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    ax.set_title(lb["model_long"][m], fontsize=11)
    save(fig, path)


def fig_stacked_h(plt, A, m, lang, path, key, names, colors, title):
    import numpy as np
    fig, ax = plt.subplots(figsize=(10, 3.8))
    lb = L[lang]
    y = np.arange(len(CELLS))
    left = np.zeros(len(CELLS))
    for k, (nm, col) in enumerate(zip(names, colors)):
        v = np.array([pct(A[(m, c)][key][k], A[(m, c)]["n"]) for c in CELLS])
        cnt = [A[(m, c)][key][k] for c in CELLS]
        ax.barh(y, v, left=left, color=col, label=nm, edgecolor="white", lw=0.5)
        for yy, l0, vv, cc in zip(y, left, v, cnt):
            if vv >= 5:
                ax.text(l0 + vv / 2, yy, "%d" % cc, ha="center", va="center", fontsize=7,
                        color="white" if col in (CLC[0], KC[0], CLC[3], KC[2]) else "black")
        left = left + np.nan_to_num(v)
    ax.set_yticks(y)
    ax.set_yticklabels([lb["cell"][c] for c in CELLS])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel(lb["pct_ep"])
    ax.legend(ncol=3, fontsize=7.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    ax.set_title(title, fontsize=10.5)
    save(fig, path)


def fig_per_task(plt, A, m, lang, path):
    import numpy as np
    lb = L[lang]
    fig, axs = plt.subplots(1, 2, figsize=(12, 5.2))
    for ax, key, title in ((axs[0], "task_succ", lb["succ"]), (axs[1], "task_one", lb["one"])):
        mat = np.array([[A[(m, c)][key][t] if A[(m, c)]["task_n"][t] else np.nan for c in CELLS] for t in range(10)], float)
        ax.imshow(mat, vmin=0, vmax=50, cmap="YlGn", aspect="auto")
        for t in range(10):
            for j, c in enumerate(CELLS):
                n = A[(m, c)]["task_n"][t]
                if n:
                    v = A[(m, c)][key][t]
                    ax.text(j, t, ("%d" % v) if n == 50 else "%d/%d" % (v, n), ha="center", va="center", fontsize=8,
                            color="white" if v > 33 else "black")
        ax.set_xticks(range(len(CELLS)))
        ax.set_xticklabels([lb["cell_br"][c] for c in CELLS], fontsize=8)
        ax.set_yticks(range(10))
        ax.set_yticklabels(["%d %s" % (t, lb["task"][t]) for t in range(10)], fontsize=8)
        ax.set_title(title + " (/50)", fontsize=10)
        for sp in ax.spines.values():
            sp.set_visible(False)
    fig.suptitle(lb["model_long"][m], fontsize=11)
    save(fig, path)


def fig_cdf(plt, A, m, lang, path):
    import numpy as np
    lb = L[lang]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    xs = np.linspace(0, 250, 501)
    for c in CELLS:
        e = np.array(A[(m, c)]["err"])
        if not len(e):
            continue
        col, ls = CC[c]
        ax.plot(xs, [100 * np.mean(e <= x) for x in xs], color=col, ls=ls, lw=1.6, label=lb["cell"][c])
    for r in (13, 20, 30, 50):
        ax.axvline(r, color="#BBBBBB", lw=0.7, ls=":")
        ax.text(r, 101, "%d" % r, ha="center", fontsize=7, color="#777777")
    ax.set_xlim(0, 250)
    ax.set_ylim(0, 104)
    ax.set_xlabel(lb["dist"])
    ax.set_ylabel(lb["cum"])
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    ax.set_title(lb["model_long"][m], fontsize=10.5)
    save(fig, path)


def fig_release(plt, A, m, lang, path):
    import numpy as np
    lb = L[lang]
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    rng = np.random.default_rng(0)
    for i, c in enumerate(CELLS):
        hs = np.array(A[(m, c)]["heights"])
        col, _ = CC[c]
        if len(hs):
            ax.scatter(hs, i + rng.uniform(-0.18, 0.18, len(hs)), s=16, color=col, alpha=0.75, edgecolor="none")
        hb = A[(m, c)]["hb"]
        ax.text(1.02, i, "n=%d  (<5 cm %d · 5–10 cm %d · ≥10 cm %d)" % (len(hs), *hb), transform=ax.get_yaxis_transform(),
                va="center", fontsize=7.5)
    for v, lab in ((30, lb["crit"]), (50, "5 cm"), (100, "10 cm")):
        ax.axvline(v, color="#D62728" if v == 30 else "#AAAAAA", lw=0.9, ls="-" if v == 30 else ":")
        ax.text(v * 1.03, -0.7, lab, ha="left", va="center", fontsize=7, color="#D62728" if v == 30 else "#777777")
    ax.set_xscale("log")
    ax.set_xlim(20, 600)
    from matplotlib.ticker import FixedLocator, NullLocator, FixedFormatter
    ticks = [20, 30, 50, 100, 200, 400]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FixedFormatter([str(t) for t in ticks]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_ylim(-0.9, len(CELLS) - 0.4)
    ax.set_yticks(range(len(CELLS)))
    ax.set_yticklabels([lb["cell"][c] for c in CELLS], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel(lb["height"], fontsize=8.5)
    ax.set_title(lb["model_long"][m] + " · " + lb["reason"][3], fontsize=10)
    save(fig, path)


def fig_comparisons(plt, A, m, lang, path, pairs=None, title=None):
    lb = L[lang]
    pairs = pairs or [(m, ca, m, cb) for ca, cb in COMPS]
    pairs = [p for p in pairs if A[(p[0], p[1])]["n"] and A[(p[2], p[3])]["n"]]
    fig, ax = plt.subplots(figsize=(9.5, 0.55 * len(pairs) + 1.4))
    for i, (ma, ca, mb, cb) in enumerate(pairs):
        o = comp_row(A, ma, ca, mb, cb)
        d, z, _, lo, hi = o["succ"]
        ax.errorbar([d], [i - 0.15], xerr=[[d - lo], [hi - d]], fmt="o", color=MC[mb], ms=5, capsize=2, lw=1)
        d1, z1, _, lo1, hi1 = o["one"]
        rlo, rhi, _, _ = o["range"]
        ax.plot([rlo, rhi], [i + 0.15, i + 0.15], color=CLC[2], lw=6, alpha=0.6, solid_capstyle="butt")
        ax.errorbar([d1], [i + 0.15], xerr=[[d1 - lo1], [hi1 - d1]], fmt="s", color=CLC[0], ms=5, capsize=2, lw=1)
        ax.text(1.01, i, "%s %s (z %s)  /  %s %s (z %s)" % ("기존" if lang == "ko" else "orig.", f1(d, True), f2(z),
                                                         "한번에" if lang == "ko" else "one-pass", f1(d1, True), f2(z1)),
                transform=ax.get_yaxis_transform(), va="center", fontsize=7.5)
    ax.axvline(0, color="#555555", lw=0.8)
    ax.set_yticks(range(len(pairs)))
    if pairs and pairs[0][0] == pairs[0][2]:
        labs = ["%s → %s" % (lb["cell"][ca], lb["cell"][cb]) for _, ca, _, cb in pairs]
    else:
        labs = ["%s: %s → %s" % (lb["cell"][ca], lb["model"][ma], lb["model"][mb]) for ma, ca, mb, cb in pairs]
    ax.set_yticklabels(labs, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel(lb["diff"])
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], marker="o", color=MC[pairs[0][2]] if pairs else "#555555", ls="none"),
          Line2D([], [], marker="s", color=CLC[0], ls="none"),
          Line2D([], [], color=CLC[2], lw=6, alpha=0.6)]
    ax.legend(hs, [lb["succ"] + " (95% CI)", lb["one"] + " (95% CI)",
                   "판정 의심이 풀릴 때 가능한 범위" if lang == "ko" else "range once unverified are resolved"],
              fontsize=7.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12 if len(pairs) > 4 else -0.25), ncol=3)
    ax.set_title(title or lb["model_long"][m], fontsize=10.5)
    save(fig, path)


def fig_screen(plt, screen, lang, path):
    lb = L[lang]
    xs, ys, ns = [], [], []
    for label, _ in SCREEN:
        rr = [r for r in screen if r["step"] == label]
        if rr:
            xs.append(int(label))
            ys.append(sum(int(r["success"]) for r in rr))
            ns.append(len(rr))
    if not xs:
        return False
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.plot(xs, ys, "o-", color=MC["OpF4"])
    for x, y, n in zip(xs, ys, ns):
        ax.text(x, y + 0.6, "%d/%d" % (y, n), ha="center", fontsize=8)
    ax.set_xticks(xs)
    ax.set_xticklabels([lb["base"] if x == 0 else str(x) for x in xs])
    ax.set_xlabel(lb["steps"])
    ax.set_ylabel(lb["succ50"])
    ax.set_ylim(0, max(ys) + 4)
    save(fig, path)
    return True


def write_figures(A, screen, out):
    made = []
    for lang in ("ko", "en"):
        plt = setup_mpl(lang)
        fd = os.path.join(out, "figures_%s" % lang)
        os.makedirs(fd, exist_ok=True)
        lb = L[lang]
        jobs = [("fig01_success_all.png", lambda p: fig_success_all(plt, A, lang, p)),
                ("fig02_retention.png", lambda p: fig_retention(plt, A, lang, p))]
        for m in MODELS:
            s = SLUG[m]
            jobs += [
                ("%s_1_success.png" % s, lambda p, m=m: fig_model_success(plt, A, m, lang, p)),
                ("%s_2_outcomes.png" % s, lambda p, m=m: fig_stacked_h(plt, A, m, lang, p, "cls", lb["cls"], CLC,
                                                                        lb["model_long"][m] + " · " + H[lang]["s_out"])),
                ("%s_3_failure_kinds.png" % s, lambda p, m=m: fig_stacked_h(plt, A, m, lang, p, "kinds", lb["kind"], KC,
                                                                             lb["model_long"][m] + " · " + H[lang]["s_kind"])),
                ("%s_4_per_task.png" % s, lambda p, m=m: fig_per_task(plt, A, m, lang, p)),
                ("%s_5_placement_cdf.png" % s, lambda p, m=m: fig_cdf(plt, A, m, lang, p)),
                ("%s_6_release_height.png" % s, lambda p, m=m: fig_release(plt, A, m, lang, p)),
                ("%s_7_comparisons.png" % s, lambda p, m=m: fig_comparisons(plt, A, m, lang, p)),
            ]
        jobs.append(("openvla_8_finetune_effect.png",
                     lambda p: fig_comparisons(plt, A, "OpF4", lang, p, pairs=[("OpZ", c, "OpF4", c) for c in CELLS],
                                               title=H[lang]["s_ft"])))
        for name, fn in jobs:
            p = os.path.join(fd, name)
            try:
                fn(p)
                made.append(os.path.relpath(p, out))
            except Exception as ex:  # noqa: BLE001
                print("  !! 그래프 실패 %s: %s: %s" % (name, type(ex).__name__, ex), flush=True)
        p = os.path.join(fd, "openvla_9_step_screen.png")
        if screen and fig_screen(plt, screen, lang, p):
            made.append(os.path.relpath(p, out))
    return made


# ------------------------------------------------------------------ 확인 · 목록
def check(A):
    print("\n칸별 확인 (판 · 기존 성공 · 한번에 옮김 · 판정 의심 · 판정 못 한 판 · 메모 있는 판)")
    ok = True
    for m in MODELS:
        for c in CELLS:
            d = A[(m, c)]
            flag = ""
            if d["n"] != 500:
                flag += "  ← 판 수가 500이 아님"
                ok = False
            if d["unjudged"] or d["notes"]:
                flag += "  ← 판정 못 한 판/메모 있음 (episodes_all.csv 의 note 열 확인)"
            if d["cls"][0] + d["cls"][1] + d["cls"][2] != d["succ"] and not d["unjudged"]:
                flag += "  ← 한번에+다시 집어+판정 의심 ≠ 기존 성공"
                ok = False
            print("  %-5s %-4s %-28s %4d  %4d  %4d  %3d  %3d  %3d%s" % (m, c, d["folder"], d["n"], d["succ"], d["one"], d["cls"][2],
                                                                     d["unjudged"], d["notes"], flag))
    print("  → 모두 정상" if ok else "  → 위 표시된 칸을 확인할 것")


def write_readme(out, figs):
    s = """# 보고서용 표 · 그래프 (make_report_assets_v2.py 로 만듦, %s)

모든 숫자는 `episodes_all.csv` 하나에서 나온다. 판정 규칙은 기존 분석 코드(vla_signals.py · strict_outcomes.py)를 그대로 썼다.

## 칸 이름
| 이름 | 뜻 |
|---|---|
| mj | MuJoCo 기본 (원래 LIBERO 평가, 동작 하나 0.05초) |
| mjb | MuJoCo-B (동작 하나 0.10초) |
| M1 / M50 | Isaac Sim M (동작 하나 0.05초, 움직이는 거리를 MuJoCo 에 맞춤) · 배치 고정 / 배치 50 |
| B1 / B50 | Isaac Sim B (동작 하나 0.17초, 맞추지 않음) · 배치 고정 / 배치 50 |

모델: OpZ = OpenVLA 원본(제로샷), OpF4 = OpenVLA 재학습(Isaac Sim 데이터 4000단계), OFT = OpenVLA-OFT(제로샷), Pi05 = π0.5(제로샷)

## 파일
| 파일 | 내용 |
|---|---|
| episodes_all.csv | 판마다 한 줄: 기존 판정 성공, 한번에 옮겼는지(outcome · 이유 · 놓은 높이), 집게를 닫은 횟수(attempts), 그릇에 닿은 시도(touched), 실패 종류, 지표용 값 |
| screen.csv | OpenVLA 재학습 단계 고르기 (과제마다 0~4번 판) |
| collect_log.txt | 결과 폴더별 판 수 · 오류 |
| tables/t1_success.csv | 칸별 성공률, 95 %% 구간, 한번에 옮김, MuJoCo 기본 대비 남은 비율 |
| tables/t2_outcomes.csv | 다섯 가지 나누기, 평균 닫은 횟수, 평균 닿은 시도, 떨어뜨린 판 |
| tables/t3_unverified.csv | 판정 의심의 이유(1 · 2 · 3), 3번의 놓은 높이 구간, 1번이 나온 과제 |
| tables/t4_failure_kinds.csv | 실패 종류 여섯 가지 |
| tables/t5_metrics.csv | Table 1 지표 (닫음 · 집음 · 놓기 · 안 건드림 · 거리) |
| tables/t6_per_task.csv | 과제별 성공 · 한번에 옮김 |
| tables/t7_within_model.csv | 한 모델 안에서 한 가지만 바꾼 비교 + OpenVLA 재학습 효과 (차이, z, p, 95 %% 구간, 판정 의심 범위) |
| tables/t8_between_models.csv | 같은 칸에서 모델끼리 비교 |
| tables/t9_finetune_step_screen.csv | 재학습 단계별 성공 |
| md/overall_ko.md · _en.md | README(총 정리)용 표 |
| md/openvla_ko.md · _en.md | 보고서 2 (OpenVLA 제로샷 · 재학습)용 표 |
| md/oft_ko.md · _en.md | 보고서 3 (OFT 제로샷)용 표 |
| md/pi05_ko.md · _en.md | 보고서 4 (π0.5 제로샷)용 표 |

## 그래프 (figures_ko/ · figures_en/ 에 같은 이름)
| 파일 | 내용 |
|---|---|
| fig01_success_all | 모델 4 × 칸 6 성공률. 진한 부분 = 한번에 옮김, 막대 위 = 95 %% 구간 |
| fig02_retention | MuJoCo 기본 대비 Isaac Sim 에서 남은 비율 |
| <모델>_1_success | 성공을 한번에 / 다시 집어 / 판정 의심으로 나눈 막대 |
| <모델>_2_outcomes | 다섯 가지 나누기 (판 비율) |
| <모델>_3_failure_kinds | 실패 종류 (판 비율) |
| <모델>_4_per_task | 과제 × 칸 성공 수 (왼쪽 기존, 오른쪽 한번에) |
| <모델>_5_placement_cdf | 끝났을 때 그릇–접시 중심 거리의 누적 분포 |
| <모델>_6_release_height | 판정 의심 3번 판의 놓은 높이 |
| <모델>_7_comparisons | 한 가지만 바꾼 비교 (차이와 95 %% 구간, 판정 의심 범위) |
| openvla_8_finetune_effect | OpenVLA 원본 → 재학습 (같은 칸) |
| openvla_9_step_screen | 재학습 단계 고르기 |

<모델> = openvla_zs · openvla_ft · oft · pi05

만든 그래프 %d개.
""" % (time.strftime("%Y-%m-%d %H:%M"), len(figs))
    with open(os.path.join(out, "README_assets.md"), "w", encoding="utf-8") as fh:
        fh.write(s)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", default=DEF_EXP, help="결과 폴더들이 있는 곳 (기본 ~/LIBERO/experiments)")
    ap.add_argument("--adir", default=DEF_ADIR, help="분석 코드 폴더 (기본 ~/tsk/VLA-repo_all/scripts/analysis)")
    ap.add_argument("--out", default=DEF_OUT, help="결과를 쓸 폴더 (기본 ~/tsk/VLA-repo_all/middle_/assets/<오늘>)")
    ap.add_argument("--jobs", type=int, default=4, help="동시에 읽을 결과 폴더 수")
    ap.add_argument("--reuse", action="store_true", help="episodes_all.csv 를 다시 만들지 않고 표 · 그래프만")
    a = ap.parse_args()
    a.exp, a.adir, a.out = (os.path.expanduser(x) for x in (a.exp, a.adir, a.out))
    for f in ("vla_signals.py", "strict_outcomes.py", "episodes.py"):
        if not os.path.exists(os.path.join(a.adir, f)):
            sys.exit("분석 코드가 없음: %s" % os.path.join(a.adir, f))
    os.makedirs(a.out, exist_ok=True)
    ep_csv = os.path.join(a.out, "episodes_all.csv")
    if a.reuse and os.path.exists(ep_csv):
        print("episodes_all.csv 를 다시 씀:", ep_csv)
    else:
        print("1단계: 결과 폴더 %d개를 읽는 중 (동시에 %d개)" % (len(FOLDER), a.jobs), flush=True)
        stage_collect(a)
    print("2단계: 표 · 그래프 만드는 중", flush=True)
    A = aggregate(load_rows(ep_csv))
    sc_path = os.path.join(a.out, "screen.csv")
    screen = list(csv.DictReader(open(sc_path, encoding="utf-8-sig"))) if os.path.exists(sc_path) else []
    check(A)
    write_tables(A, screen, a.out)
    write_md(A, screen, a.out)
    figs = write_figures(A, screen, a.out)
    write_readme(a.out, figs)
    print("\n끝: %s  (표 tables/ · 마크다운 md/ · 그래프 %d개 figures_ko/ · figures_en/)" % (a.out, len(figs)))


if __name__ == "__main__":
    main()
