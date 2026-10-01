#!/usr/bin/env python3
"""보고서 그림 생성: figure_data.json → fig*.png (+ 아키텍처 · 단계 사다리는 graphviz svg/png).

    python3 make_report_figures.py            # 같은 폴더의 figure_data.json 사용
"""

from __future__ import annotations

import json
import math
import os
import subprocess

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, FancyBboxPatch, Polygon, Rectangle  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(HERE, "figure_data.json")))
from matplotlib import font_manager as _fm  # noqa: E402
for _f in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"):
    if os.path.exists(_f):
        _fm.fontManager.addfont(_f)
_FAMILY = "Noto Sans CJK JP" if any("Noto Sans CJK JP" == f.name for f in _fm.fontManager.ttflist) else "NanumGothic"
plt.rcParams.update({"font.family": _FAMILY, "axes.unicode_minus": False, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 170,
                     "savefig.bbox": "tight", "axes.titleweight": "bold"})
C = {"old": "#9aa0a6", "s3": "#e8a33d", "s10": "#2f6fb3", "groot": "#8a5cb8", "ok": "#3a9d5d", "bad": "#c9483c",
     "demo": "#1f9e89", "light": "#dfe6ee"}
T = D["tasks"]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


def save(fig, name):
    path = os.path.join(HERE, name)
    fig.savefig(path)
    plt.close(fig)
    print("저장", name)


# ---------------------------------------------------------------- fig1 제로샷 개요
def fig1():
    z = D["zeroshot"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.5, 4.4), gridspec_kw={"width_ratios": [1.1, 1.3]})
    old = z["paper_old_harness_s00"]; s10 = z["fixed_S10_s00"]
    k10 = sum(v[0] for v in s10["per_task"].values()); n10 = sum(v[1] for v in s10["per_task"].values())
    g_old = old["success"] - old["graspless"]
    xs = [0, 1]
    a.bar(0, g_old / old["n"] * 100, color=C["old"], width=0.55, label="파지 동반 성공")
    a.bar(0, old["graspless"] / old["n"] * 100, bottom=g_old / old["n"] * 100, color="white", edgecolor=C["old"],
          hatch="///", width=0.55, label="파지 없이 밀어 올린 성공 (판정 결함)")
    a.bar(1, k10 / n10 * 100, color=C["s10"], width=0.55)
    for x, (k, n) in zip(xs, [(g_old, old["n"]), (k10, n10)]):
        lo, hi = wilson(k, n)
        a.errorbar(x, k / n * 100, yerr=[[k / n * 100 - lo * 100], [hi * 100 - k / n * 100]], color="black", capsize=4, lw=1)
    a.text(0, old["success"] / old["n"] * 100 + 1.2, "보고값 %.1f%%\n(파지 동반 %.1f%%)" % (old["success"] / old["n"] * 100, g_old / old["n"] * 100),
           ha="center", va="bottom", fontsize=9)
    gl = s10.get("graspless")
    a.text(1, k10 / n10 * 100 + 3.2, "%.1f%%\n%d/%d · %s" % (k10 / n10 * 100, k10, n10, "파지 없는 성공 %d" % gl if gl is not None else "파지 동반 감사 중"),
           ha="center", va="bottom", fontsize=9)
    a.set_xticks(xs, ["논문 수치\n(옛 하네스)", "수정 하네스\n(S=10)"])
    a.set_ylabel("성공률 (%)"); a.set_ylim(0, 35)
    a.set_title("OpenVLA 제로샷 — 같은 배치(s00) · 같은 정책")
    a.legend(loc="upper left", fontsize=8, frameon=False)

    labels, vals, cols, notes = [], [], [], []
    s3 = z["fixed_S3_50layouts"]["per_task"]; k3 = sum(v[0] for v in s3.values()); n3 = sum(v[1] for v in s3.values())
    ga = z["groot_reference"]["anchor_layout"]; gr = z["groot_reference"]["random_layouts"]
    rows = [("OpenVLA · 50 배치\n수정 하네스 S=3", k3, n3, C["s3"], "S=10 재측정 예정"),
            ("GR00T N1.6 · 앵커 배치 1 개\n방해 그릇 없음 · 50 mm 판정", ga[0], ga[1], C["groot"], "손목 + state 입력"),
            ("GR00T N1.6 · 배치 30 세트", gr[0], gr[1], C["groot"], "")]
    for i, (lab, k, n, col, note) in enumerate(rows):
        hatch = None if i == 0 else ".."
        b.barh(i, k / n * 100, color=col if i == 0 else "white", edgecolor=col, hatch=hatch, height=0.55)
        lo, hi = wilson(k, n)
        b.errorbar(k / n * 100, i, xerr=[[k / n * 100 - lo * 100], [hi * 100 - k / n * 100]], color="black", capsize=3, lw=1)
        b.text(hi * 100 + 1.5, i, "%.1f%% (%d/%d)  %s" % (k / n * 100, k, n, note), va="center", fontsize=8.5)
    b.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=8.5); b.invert_yaxis()
    b.set_xlim(0, 105); b.set_xlabel("성공률 (%)")
    b.set_title("참고 — 조건이 다른 수치 (직접 비교 불가)")
    fig.suptitle("그림 1. Isaac Sim LIBERO-Spatial 제로샷 성공률", x=0.01, ha="left", fontsize=12, fontweight="bold")
    save(fig, "fig1_zeroshot_overview.png")


# ---------------------------------------------------------------- fig2 태스크별 + 실패 단계
def fig2():
    z = D["zeroshot"]; s10 = z["fixed_S10_s00"]; s3p = z["fixed_S3_s00_partial"]["per_task"]
    fig, (a, b) = plt.subplots(2, 1, figsize=(11, 7.2), gridspec_kw={"height_ratios": [1, 1.05]}, sharex=True)
    w = 0.38
    for t in T:
        k, n = s10["per_task"][str(t)]
        a.bar(t + w / 2, k / n * 100, width=w, color=C["s10"], label="S=10 (최종)" if t == 0 else None)
        lo, hi = wilson(k, n); a.errorbar(t + w / 2, k / n * 100, yerr=[[k / n * 100 - lo * 100], [hi * 100 - k / n * 100]], color="black", capsize=2, lw=0.8)
        a.text(t + w / 2, hi * 100 + 1, "%d/%d" % (k, n), ha="center", fontsize=7.5)
        if str(t) in s3p:
            k3, n3 = s3p[str(t)]
            a.bar(t - w / 2, k3 / n3 * 100, width=w, color=C["s3"], label="S=3 · gain 0.65" if t == 0 else None)
            a.text(t - w / 2, k3 / n3 * 100 + 1, "%d/%d" % (k3, n3), ha="center", fontsize=7.5)
    a.set_ylabel("성공률 (%)"); a.set_ylim(0, 100)
    a.set_title("(a) 태스크별 성공률 — 배치 s00 고정, 수정 하네스.  S=3 은 중단 전 4 개 태스크만 측정")
    a.legend(frameon=False, fontsize=8.5, loc="upper left")
    a.annotate("잡은 뒤 들지 못함 → 들어 올림", xy=(6 + w / 2, 80), xytext=(7.3, 88), fontsize=8.5,
               arrowprops=dict(arrowstyle="->", lw=0.8))

    order = ["성공", "S7+", "S5–S6", "S4", "S3", "S0"]
    colmap = {"성공": C["ok"], "S7+": "#7fb4d9", "S5–S6": "#b9d4ea", "S4": "#f0b37e", "S3": C["bad"], "S0": "#555555"}
    bottoms = [0] * 10
    for key in order:
        vals = []
        for t in T:
            fs = s10["fail_stage"][str(t)]
            if key == "성공":
                v = s10["per_task"][str(t)][0]
            elif key == "S7+":
                v = sum(fs.get(s, 0) for s in ("S7", "S8", "S9"))
            elif key == "S5–S6":
                v = fs.get("S5", 0) + fs.get("S6", 0)
            else:
                v = fs.get(key, 0)
            vals.append(v)
        b.bar(T, vals, bottom=bottoms, color=colmap[key], width=0.7, label=key)
        bottoms = [x + y for x, y in zip(bottoms, vals)]
    fails = {k: sum(s10["fail_stage"][str(t)].get(k, 0) for t in T) for k in ("S3", "S4")}
    nf = sum(sum(v.values()) for v in s10["fail_stage"].values())
    b.set_ylabel("에피소드 수 (각 50)"); b.set_ylim(0, 50)
    b.set_xticks(T, ["T%d" % t for t in T])
    b.set_title("(b) 도달 단계 분포 (stage_level = 연속 통과한 마지막 단계).  실패 %d 판 중 S3–S4 에서 멈춤 %d 판 (%.0f%%)"
                % (nf, fails["S3"] + fails["S4"], 100 * (fails["S3"] + fails["S4"]) / nf))
    b.legend(ncol=6, frameon=False, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    fig.suptitle("그림 2. 태스크별 제로샷 결과와 실패 지점", x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    save(fig, "fig2_per_task_and_stages.png")


# ---------------------------------------------------------------- fig3 실행 속도
def fig3():
    dz = D["dynamics_task6"]; sp = D["speed"]; ab = D["setting_ab"]
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(15, 4.4), gridspec_kw={"width_ratios": [1.05, 1.25, 0.9]})
    colk = {"mujoco": C["old"], "fast": C["s10"], "slow": C["s3"]}
    for i, (lab, v, k) in enumerate(sp["rows"]):
        a.barh(i, v, color=colk[k], height=0.6); a.text(v + 0.6, i, "%.1f" % v, va="center", fontsize=8.5)
    a.set_yticks(range(len(sp["rows"])), [r[0] for r in sp["rows"]], fontsize=8.5); a.invert_yaxis()
    a.set_xlabel("손끝 이동 (mm / 정책 스텝)"); a.set_xlim(0, 42)
    a.set_title("(a) 명령 하나당 이동량")
    a.text(40, 4.35, "정책 롤아웃 기준.  같은 시연 명령의 개루프 비교는 그림 15", ha="right", fontsize=8, color=C["s10"])
    w = 0.38; x = range(4)
    for dx, key, col, lab in ((-w / 2, "task6", C["s10"], "T6"), (w / 2, "task0", C["demo"], "T0")):
        for i, (name, k, n) in enumerate(ab[key]):
            b.bar(i + dx, 100 * k / n, width=w, color=col, alpha=1.0 if "gain 1.0" in name else 0.45, label=lab if i == 1 else None)
            b.text(i + dx, 100 * k / n + 1.5, "%d/%d" % (k, n), ha="center", fontsize=7.5)
    b.set_xticks(list(x), [r[0] for r in ab["task6"]], fontsize=8); b.set_ylim(0, 100); b.set_ylabel("성공률 (%)")
    b.set_title("(b) 같은 배치 s00, 실행 설정만 다름 (5–50 판)")
    b.legend(frameon=False, fontsize=8.5, loc="upper right")
    import random
    random.seed(1)
    y3 = dz["S3_fail_bowl_max_z_mm"]; y10 = dz["S10_ab_bowl_max_z_mm"]; ok10 = dz["S10_ab_success"]
    c.scatter([0 + random.uniform(-0.12, 0.12) for _ in y3], y3, color=C["s3"], s=20, alpha=0.85, label="S=3: 잡은 뒤 실패 24 판")
    for y, ok in zip(y10, ok10):
        c.scatter(1 + random.uniform(-0.12, 0.12), y, color=C["s10"] if ok else "white", edgecolor=C["s10"], s=28)
    c.scatter([], [], color=C["s10"], label="S=10: 성공"); c.scatter([], [], color="white", edgecolor=C["s10"], label="S=10: 실패")
    c.axhline(dz["table_bowl_bottom_mm"], color="#666", ls="--", lw=0.8)
    c.text(1.45, dz["table_bowl_bottom_mm"] + 3, "테이블 위 그릇 (898 mm)", fontsize=7.5, color="#555", ha="right")
    c.set_xticks([0, 1], ["S=3·gain 0.65", "S=10·gain 1.0"]); c.set_xlim(-0.5, 1.5)
    c.set_ylabel("그릇 최고 높이 (mm)"); c.set_title("(c) OpenVLA T6: 그릇을 들었는가")
    c.legend(frameon=False, fontsize=7.5, loc="upper left")
    fig.suptitle("그림 3. 실행 설정과 성공률 (고정 배치 s00, 9/27–28 측정) — 50 배치 · 500 판 결과는 그림 13",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    save(fig, "fig3_execution_speed.png")


# ---------------------------------------------------------------- fig4 쿼터니언 순서
def fig4():
    q = D["quaternion_bug"]
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(13, 4.0), gridspec_kw={"width_ratios": [1.2, 1, 0.9]})
    a.plot(q["steps"], q["target_vs_mujoco_deg"], "o-", color=C["bad"], label="수정 전 (eef_quat 을 wxyz 로 읽음)")
    a.axhline(0.5, color=C["ok"], lw=2, label="수정 후 추종 오차 0.5–0.6°")
    a.set_xlabel("재생 스텝"); a.set_ylabel("재생 목표 방향 − MuJoCo 실제 방향 (°)"); a.set_ylim(0, 60)
    a.set_title("(a) T6 e0: 목표 손 방향의 오차가 누적"); a.legend(frameon=False, fontsize=8)
    x = range(len(q["interp_steps"])); w = 0.27
    b.bar([i - w for i in x], q["wxyz_rel_deg"], width=w, color=C["bad"], label="wxyz 해석 (std %.1f°)" % q["wxyz_std"])
    b.bar(list(x), q["bridge_target_rel_deg"], width=w, color=C["s3"], label="브리지 목표 (std %.1f°)" % q["bridge_std"])
    b.bar([i + w for i in x], q["xyzw_rel_deg"], width=w, color=C["ok"], label="xyzw 해석 (std %.1f°)" % q["xyzw_std"])
    b.set_xticks(list(x), ["step %d" % s for s in q["interp_steps"]]); b.set_ylabel("기록 → MuJoCo 실제 상대 회전 (°)")
    b.set_title("(b) 올바른 해석은 상대 회전이 일정 (90°)"); b.legend(frameon=False, fontsize=7.5, loc="upper right"); b.set_ylim(0, 200)
    bef = q["task6_ik_failures_before"]; aft = q["task6_ik_failures_after"]
    c.bar([0, 1], bef, color=C["bad"], width=0.35, label="수정 전 (실패)")
    c.bar([0.4, 1.4], [max(v, 8) for v in aft], color=C["ok"], width=0.35, label="수정 후 (성공)")
    for xx, v in zip([0, 1], bef):
        c.text(xx, v + 30, str(v), ha="center", fontsize=8.5)
    for xx in (0.4, 1.4):
        c.text(xx, 40, "0", ha="center", fontsize=8.5)
    c.set_xticks([0.2, 1.2], ["T6 e0", "T6 e5"]); c.set_ylabel("에피소드당 IK 실패 횟수")
    c.set_title("(c) 먼 태스크의 IK 실패 원인"); c.legend(frameon=False, fontsize=8)
    fig.suptitle("그림 4. 재생 데이터의 쿼터니언 순서 오류 (LIBERO eef_quat 는 xyzw, 브리지 기본값 wxyz)",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    save(fig, "fig4_quaternion_order.png")


# ---------------------------------------------------------------- fig5 재생 성공 vs 거리
def fig5():
    r = D["replay_vs_distance"]; dist = r["bowl_base_distance_m"]
    fig, a = plt.subplots(figsize=(8.6, 4.6))
    for key, col, mk, lab in (("rollout_before_fix", C["bad"], "o", "수정 전 · 롤아웃 궤적"),
                              ("rollout_after_fix", C["s10"], "s", "수정 후 · 롤아웃 궤적 (xyzw, z −6 mm)"),
                              ("demo_after_fix", C["demo"], "D", "수정 후 · 사람 시연 (xyzw)")):
        pts = sorted((dist[t], k / n * 100, t) for t, (k, n) in r[key].items())
        a.plot([p[0] for p in pts], [p[1] for p in pts], mk, color=col, ms=7, label=lab, alpha=0.9)
        if key == "rollout_before_fix":
            for x, y, t in pts:
                a.annotate("T%s" % t, (x, y), textcoords="offset points", xytext=(5, -3), fontsize=7.5, color=col)
    a.set_xlabel("로봇 베이스에서 목표 그릇까지 수평 거리 (m)"); a.set_ylabel("재생 성공률 (%)"); a.set_ylim(-3, 105)
    a.axvspan(0.72, 0.81, color=C["light"], zorder=0); a.text(0.765, 50, "먼 태스크\nT3·T4·T6·T8·T9", ha="center", fontsize=8, color="#555")
    a.legend(frameon=False, fontsize=8.5, loc="lower left")
    a.set_title("그림 5. 재생 성공률과 도달 거리 — 거리 의존성은 손 방향 오류에서 왔음", loc="left", fontsize=12)
    a.text(0.40, 12, "T5 (0.49 m) 은 라메킨 장면 문제", fontsize=8, color="#555")
    save(fig, "fig5_replay_vs_distance.png")


# ---------------------------------------------------------------- fig6 배치 (위에서 본 그림)
def fig6():
    xy = D["bowl_xy_m"]; bx, by = D["robot_base_xy_m"]; s10 = D["zeroshot"]["fixed_S10_s00"]["per_task"]
    sh = D["demo_layout_shift_mm"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 5.0), gridspec_kw={"width_ratios": [1.25, 1]})
    for r in (0.5, 0.6, 0.7, 0.8):
        a.add_patch(Circle((bx, by), r, fill=False, ls=":", lw=0.8, color="#888"))
        a.text(bx + r * math.cos(math.radians(25)), by + r * math.sin(math.radians(25)), "%.1f m" % r, fontsize=7.5, color="#777")
    a.add_patch(Rectangle((bx - 0.06, by - 0.06), 0.12, 0.12, color="#444")); a.text(bx, by - 0.1, "로봇 베이스", ha="center", fontsize=8)
    cmap = plt.get_cmap("RdYlGn")
    for t, (x, y) in xy.items():
        k, n = s10[t]; p = k / n
        a.scatter(x, y, s=260, color=cmap(p), edgecolor="black", zorder=3)
        a.text(x, y, "T" + t, ha="center", va="center", fontsize=8, fontweight="bold", zorder=4)
        a.text(x, y - 0.045, "%d%%" % round(100 * p), ha="center", fontsize=7.5, zorder=4)
    a.set_aspect("equal"); a.set_xlim(-0.75, 0.25); a.set_ylim(-0.42, 0.45)
    a.set_xlabel("x (m, LIBERO 월드)"); a.set_ylabel("y (m)")
    a.set_title("(a) 배치 s00 의 목표 그릇 위치와 제로샷 성공률 (S=10)")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 100)); fig.colorbar(sm, ax=a, fraction=0.035, label="성공률 (%)")
    b.bar(T, sh["max"], color=C["light"], label="최대")
    b.bar(T, sh["median"], color=C["demo"], label="중앙값")
    b.set_xticks(T, ["T%d" % t for t in T]); b.set_ylabel("그릇 위치 차이 (mm)")
    b.set_title("(b) 사람 시연 배치 vs 평가 배치(s00) — 학습·평가 배치가 겹치지 않음")
    b.legend(frameon=False, fontsize=8.5)
    fig.suptitle("그림 6. 장면 배치", x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    save(fig, "fig6_layout.png")


# ---------------------------------------------------------------- fig7 데이터 수집
def fig7():
    cs = D["collection_snapshot"]; r = D["replay_vs_distance"]; ft = D["replay_failure_types_rollout"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(12.5, 4.4), gridspec_kw={"width_ratios": [1.35, 1]})
    for t in T:
        if str(t) in cs["demo"]:
            ok, tried = cs["demo"][str(t)]
            a.bar(t, tried, color=C["light"], width=0.7, label="재생 시도 (시드 7·8·9)" if t == 0 else None)
            a.bar(t, ok, color=C["demo"], width=0.7, label="성공 궤적 (서로 다름)" if t == 0 else None)
            a.text(t, ok + 1, str(ok), ha="center", fontsize=8)
        else:
            a.text(t, 3, "대기", ha="center", fontsize=8, color="#777")
    a.axhline(cs["target_per_task"], color="black", ls="--", lw=0.8); a.text(9.4, cs["target_per_task"] + 1, "목표 50", ha="right", fontsize=8)
    a.set_xticks(T, ["T%d" % t for t in T]); a.set_ylabel("판 수"); a.set_ylim(0, 72)
    a.set_title("(a) 사람 시연 재생 수집 현황 (%s 기준, 진행 중)" % cs["time"]); a.legend(frameon=False, fontsize=8, loc="upper right")
    cats = ["미끄러짐", "윗면 위 닫힘", "수평 어긋남", "놓기 실패"]
    colc = [C["s3"], C["bad"], "#999", "#7fb4d9"]
    left = [0, 0]
    for cat, col in zip(cats, colc):
        vals = [ft["0"].get(cat, 0), ft["1"].get(cat, 0)]
        b.barh([0, 1], vals, left=left, color=col, label=cat, height=0.5)
        left = [x + y for x, y in zip(left, vals)]
    b.set_yticks([0, 1], ["T0 롤아웃\n(실패 42)", "T1 롤아웃\n(실패 18)"]); b.invert_yaxis(); b.set_xlabel("실패 판 수")
    rec = ft["zoff6_recovered"]
    b.set_title("(b) 수정 후 재생 실패 유형 — 접촉 물리 차이.\n    목표 z −6 mm 로 T0 실패 %d 판 중 %d 판 회복" % (rec[1], rec[0]))
    b.legend(frameon=False, fontsize=8, ncol=2, loc="lower right")
    fig.suptitle("그림 7. 학습 데이터 수집 (Isaac 재렌더링)", x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    save(fig, "fig7_data_collection.png")


# ---------------------------------------------------------------- fig8 손목 카메라 배치
def fig8():
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, title, cam, tilt, fov, res in ((a, "(a) 수집용: D405 광학, LIBERO 자리 근처", (0.050, -0.010), 8, 56, "640×360 · 가로 87° (세로는 해상도 비로 약 56°)"),
                                            (b, "(b) OFT 제로샷 평가용: LIBERO eye_in_hand 규격", (0.050, 0.0), 0, 75, "256×256 · 75°×75°")):
        ax.add_patch(Rectangle((-0.045, -0.03), 0.09, 0.06, color="#bbb"))                        # 손 몸체
        ax.add_patch(Rectangle((-0.01, 0.03), 0.02, 0.075, color="#888"))
        ax.text(-0.075, 0.08, "손가락\n(y 방향으로\n벌어짐)", fontsize=7.5, color="#666")
        ax.plot([0, 0], [-0.05, 0.16], ":", color="#999"); ax.text(0.003, 0.155, "손끝 방향 (+z)", fontsize=8, color="#666")
        cx, cz = cam
        ax.add_patch(Rectangle((cx - 0.008, cz - 0.008), 0.016, 0.016, color=C["s10"]))
        th = math.radians(tilt)
        for sgn in (-1, 1):
            ang = th + sgn * math.radians(fov / 2)
            ax.plot([cx, cx - 0.2 * math.sin(ang)], [cz, cz + 0.2 * math.cos(ang)], color=C["s10"], lw=1)
        ax.text(cx + 0.012, cz - 0.004, "카메라\n(x %.3f, z %+.3f)\n기울임 %d°" % (cx, cz, tilt), fontsize=8, color=C["s10"])
        ax.set_xlim(-0.12, 0.13); ax.set_ylim(-0.06, 0.2); ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(title + "\n" + res + " · roll 180° + 좌우 반전 (화면 규약)", fontsize=9.5)
    fig.suptitle("그림 8. 손목 카메라 배치 (panda_hand 프레임, 옆에서 본 모식도)", x=0.01, ha="left", fontsize=12, fontweight="bold")
    save(fig, "fig8_wrist_camera.png")


# ---------------------------------------------------------------- graphviz 다이어그램
def dot(name: str, src: str):
    p = os.path.join(HERE, name)
    open(p + ".dot", "w").write(src)
    for fmt in ("png", "svg"):
        subprocess.run(["dot", "-T" + fmt, "-Gdpi=150" if fmt == "png" else "-Gdpi=72", p + ".dot", "-o", p + "." + fmt], check=True)
    print("저장", name + ".png / .svg")


FONT = 'fontname="Noto Sans CJK KR"'


def fig9_architecture():
    dot("fig9a_eval_pipeline", f'''digraph G {{
  graph [rankdir=TB, {FONT}, fontsize=12, nodesep=0.5, ranksep=0.35, label="그림 9a. 평가 파이프라인 (제로샷 · 파인튜닝 평가 공통)", labelloc=t, labeljust=l];
  node [shape=box, style="rounded,filled", {FONT}, fontsize=10, fillcolor="#f4f6f9", color="#8a94a3"];
  edge [{FONT}, fontsize=9, color="#555555"];
  scene [label="Isaac 장면\\nLIBERO 배치 JSON\\n(s00–s49 / 시연별)"];
  cams [label="카메라\\nagentview D405\\n손목 D405 (신규)"];
  cli [label="정책 클라이언트\\nOpenVLAClient\\nOFTClient (신규: 청크 큐 · proprio 8)"];
  srv [label="정책 서버\\nOpenVLA :8000  (crop 0.35 → 224)\\nOFT :8001 (신규: agentview+손목+state)", fillcolor="#e3eefa"];
  ac [label="액션 규약 · 어댑터\\n관문 해제 · 집게 PD 자연 닫힘\\nz 오프셋 0"];
  ik [label="Lula IK (right_gripper)\\n실패 시 현재 자세 유지"];
  phys [label="PhysX\\n명령당 S=3 서브스텝 · gain 0.65\\n(사람 시연 명령 개루프 재생으로\\nMuJoCo 와 이동비 1.02 확인)", fillcolor="#fdf1dd"];
  st [label="StageTelemetry S0–S10\\n성공 = S9", fillcolor="#d9efe0"];
  cap [label="캡처\\nraw_rgb · openvla_rgb\\nwrist_rgb (신규) · 관절 (신규)"];
  wd [label="watchdog.sh (신규)\\n15 분 무응답 판 재시도", fillcolor="#eeeeee"];
  scene -> cams -> cli; cli -> srv [dir=both, label="HTTP"];
  cli -> ac -> ik -> phys; phys -> cams [constraint=false, style=dashed, label="다음 관측"];
  phys -> st; phys -> cap; wd -> phys [style=dotted, arrowhead=none];
  {{ rank=same; cli; srv; }}  {{ rank=same; phys; wd; }}  {{ rank=same; st; cap; }}
}}''')
    dot("fig9b_data_pipeline", f'''digraph D {{
  graph [rankdir=TB, {FONT}, fontsize=12, nodesep=0.5, ranksep=0.3, label="그림 9b. 학습 데이터 수집 파이프라인 (신규)", labelloc=t, labeljust=l];
  node [shape=box, style="rounded,filled", {FONT}, fontsize=10, fillcolor="#f4f6f9", color="#8a94a3"];
  edge [{FONT}, fontsize=9, color="#555555"];
  h5 [label="LIBERO-Spatial\\n사람 시연 HDF5\\n10 태스크 × 50"];
  npz [label="libero_demo_to_replay_npz.py\\nno-op 제거 · eef_quat(xyzw)\\n관절 · 집게 상태"];
  pj [label="make_demo_pose_jsons.py\\n시연 첫 상태의 물체 배치"];
  col [label="collect_success_per_task.py\\n태스크당 서로 다른 성공 50\\n시드 7 → 8 → 9 · 재시도 자동", fillcolor="#e0f3ee"];
  rp [label="브리지 재생\\nEEF 자세 → right_gripper\\nxyzw · S=20 · IK hold"];
  sel [label="선별\\nS9 성공 · 파지 동반\\n궤적 중복 없음"];
  bld [label="데이터셋 변환기\\nimage · wrist · state 8\\njoints 7 · action 7"];
  rl [label="RLDS\\nOpenVLA · OFT", fillcolor="#e3eefa"];
  lr [label="LeRobot v2\\nπ0.5 · GR00T", fillcolor="#e3eefa"];
  h5 -> npz -> col; h5 -> pj -> col; col -> rp -> sel -> bld; bld -> rl; bld -> lr;
  {{ rank=same; npz; pj; }}  {{ rank=same; rl; lr; }}
}}''')


def fig10_stages():
    dot("fig10_stage_ladder", f'''digraph S {{
  graph [rankdir=TB, {FONT}, fontsize=12, nodesep=0.2, ranksep=0.35, label="그림 10. 평가 단계 사다리 (StageTelemetry) — 성공 = S9 도달", labelloc=t, labeljust=l];
  node [shape=box, style="rounded,filled", {FONT}, fontsize=10, fillcolor="#f4f6f9", color="#8a94a3", width=1.3];
  S0 [label="S0\\n초기 상태·센서\\n(정책 실패와 분리)", fillcolor="#eeeeee"];
  S1 [label="S1\\n첫 닫힘이 목표\\n그릇 근처"]; S2 [label="S2\\n접근\\n(pregrasp)"];
  S3 [label="S3\\n정렬\\n(닫힐 때 위치)", fillcolor="#f7d9d5"]; S4 [label="S4\\n접촉·파지", fillcolor="#fbe3cc"];
  S5 [label="S5\\n안정 파지\\n(손가락 폭)"]; S6 [label="S6\\n들기"]; S7 [label="S7\\n안전 이탈"];
  S8 [label="S8\\n이송\\n(접시 위 xy·높이)"];
  S9 [label="S9  성공\\nfootprint · 높이\\n· 정지 · 위에서 하강", fillcolor="#d9efe0", penwidth=2];
  S10 [label="S10 (참고)\\n해제 + 2 초 안정"];
  {{ rank=same; S0; S1; S2; S3; }}  {{ rank=same; S4; S5; S6; S7; }}  {{ rank=same; S8; S9; S10; }}
  S0 -> S1 -> S2 -> S3; S4 -> S5 -> S6 -> S7; S8 -> S9 -> S10;
  S0 -> S4 -> S8 [style=invis];  S3 -> S4 [constraint=false];  S7 -> S8 [constraint=false];
}}''')


if __name__ == "__main__":
    for f in (fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8, fig9_architecture, fig10_stages):
        f()
