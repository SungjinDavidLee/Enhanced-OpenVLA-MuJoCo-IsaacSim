#!/usr/bin/env python3
"""v2 보고서의 새 그림 (그림 13–19).  수치는 figure_data_v2.json 에서 읽는다.

    python3 make_report_figures_v2.py
"""

from __future__ import annotations

import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager as _fm  # noqa: E402

for _f in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"):
    if os.path.exists(_f):
        _fm.fontManager.addfont(_f)
_FAMILY = "Noto Sans CJK JP" if any("Noto Sans CJK JP" == f.name for f in _fm.fontManager.ttflist) else "NanumGothic"
plt.rcParams.update({"font.family": _FAMILY, "axes.unicode_minus": False, "font.size": 10, "savefig.dpi": 160,
                     "savefig.bbox": "tight", "axes.titleweight": "bold", "axes.spines.top": False, "axes.spines.right": False})
HERE = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(HERE, "figure_data_v2.json")))
C = {"ovla": "#3b6ea8", "oft": "#d9822b", "mj": "#2a9d60", "bad": "#c0504d", "grey": "#9aa3ad",
     "A": "#2a9d60", "B": "#e8a13a", "C": "#c0504d", "D": "#8e6cc0", "E": "#aab2bd"}


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d


def save(fig, name):
    fig.savefig(os.path.join(HERE, name)); plt.close(fig); print("저장", name)


def fig13():
    z = D["zeroshot"]
    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    groups = [("빠른 실행 (S=10 · gain 1.0)\n고정 배치 s00", "fast_s00"), ("보정 실행 (S=3 · gain 0.65)\nLIBERO 초기 상태 50 개", "cal_50")]
    w = 0.34
    for gi, (glab, key) in enumerate(groups):
        for mi, (m, col) in enumerate((("OpenVLA", C["ovla"]), ("OFT", C["oft"]))):
            k, n, note = z[key][m]
            x = gi + (mi - 0.5) * w
            lo, hi = wilson(k, n)
            ax.bar(x, 100 * k / n, w * 0.92, color=col, label=m if gi == 0 else None)
            ax.errorbar(x, 100 * k / n, yerr=[[100 * (k / n - lo)], [100 * (hi - k / n)]], color="#333", capsize=3, lw=1)
            ax.text(x, 100 * hi + 1.5, "%.1f %%\n%d/%d%s" % (100 * k / n, k, n, ("\n" + note) if note else ""), ha="center", fontsize=8.5)
    o = z["old_harness"]
    ax.axhline(100 * o["success"] / o["n"], color=C["grey"], ls="--", lw=1)
    ax.text(-0.45, 100 * o["success"] / o["n"] + 0.8, "옛 하네스 16.8 % (쥐지 않은 성공 21 판 포함, 실제 12.6 %)", ha="left", fontsize=8, color="#555")
    ax.set_xticks([0, 1], [g[0] for g in groups]); ax.set_ylim(0, 72); ax.set_ylabel("성공률 (%) · Wilson 95 %")
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("그림 13. 고친 하네스의 제로샷 성공률 — 모델 × 실행 설정 (220 스텝, D405, 쥐지 않은 성공 0)", loc="left", fontsize=11.5)
    save(fig, "fig13_zeroshot_v2.png")


def fig14():
    b = D["breakdown"]; cats = ["A", "B", "C", "D", "E"]
    lab = {"A": "A 성공", "B": "B 물었는데 못 듦", "C": "C 허공에서 닫음", "D": "D 들고 가다 실패", "E": "E 접시 근처 실패"}
    names = list(b.keys())
    fig, ax = plt.subplots(figsize=(11, 3.4))
    for i, nm in enumerate(names):
        left = 0; tot = sum(b[nm].values())
        for c in cats:
            v = b[nm][c]; pct = 100 * v / tot
            ax.barh(i, pct, left=left, color=C[c], label=lab[c] if i == 0 else None)
            if pct >= 4:
                ax.text(left + pct / 2, i, "%d" % v, ha="center", va="center", fontsize=8.5, color="white")
            left += pct
    ax.set_yticks(range(len(names)), names); ax.invert_yaxis(); ax.set_xlim(0, 100); ax.set_xlabel("판 비율 (%) — 막대 안 숫자는 판 수")
    ax.legend(frameon=False, ncol=5, fontsize=8.5, loc="lower center", bbox_to_anchor=(0.5, -0.5))
    ax.set_title("그림 14. 결과 5 부류 (물리 신호 기준) — 보정 실행에서 「허공에서 닫음」 이 사라지고 「들기」 실패만 남음", loc="left", fontsize=11.5)
    save(fig, "fig14_outcome_classes.png")


def fig15():
    v = D["openloop"]; names = list(v.keys())
    fig, (a, b) = plt.subplots(1, 2, figsize=(12.5, 4.0), gridspec_kw={"width_ratios": [1.15, 1]})
    win = ["0–8 스텝", "0–24 스텝", "첫 닫기까지"]; w = 0.26
    cols = {"now": C["bad"], "cal": C["mj"], "ss": C["grey"]}
    for i, nm in enumerate(names):
        vals = v[nm]["ratio"]
        a.bar([j + (i - 1) * w for j in range(3)], vals, w * 0.95, color=cols[nm], label=v[nm]["label"])
        for j, y in enumerate(vals):
            a.text(j + (i - 1) * w, y + 0.05, "%.2f" % y, ha="center", fontsize=8)
    a.axhline(1.0, color="#333", lw=0.8, ls="--"); a.set_xticks(range(3), win); a.set_ylabel("Isaac 이동 ÷ MuJoCo 이동")
    a.set_title("(a) 같은 사람 시연 명령의 이동비 (시연 4 개)"); a.legend(frameon=False, fontsize=8.5)
    ax_lab = ["x", "y", "z"]
    for i, nm in enumerate(names):
        vals = v[nm]["axis"]
        b.bar([j + (i - 1) * w for j in range(3)], vals, w * 0.95, color=cols[nm])
        for j, y in enumerate(vals):
            b.text(j + (i - 1) * w, y + 0.04, "%.2f" % y, ha="center", fontsize=8)
    b.axhspan(0.9, 1.1, color=C["mj"], alpha=0.08); b.axhline(1.0, color="#333", lw=0.8, ls="--")
    b.set_xticks(range(3), ax_lab); b.set_title("(b) 축별 이동비 (0–24 스텝), 초록 띠 = 합격 ±0.10")
    for i, nm in enumerate(names):
        b.text(2.55, 1.85 - 0.16 * i, "%s: 파지 시점 %d mm · %d°" % (v[nm]["short"], v[nm]["pos_mm"], v[nm]["rot_deg"]), fontsize=8, color=cols[nm], ha="right")
    b.set_ylim(0, 2.0)
    fig.suptitle("그림 15. 어댑터 실행 배율 개루프 검증 — 정책 없이 사람 시연 명령을 재생해 MuJoCo 손끝 궤적과 대조", x=0.01, ha="left", fontsize=11.5, fontweight="bold")
    fig.tight_layout(); save(fig, "fig15_openloop_verification.png")


def fig16():
    p = D["probe"]; c = D["closure"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.5, 3.6))
    names = ["크기비", "방향 코사인", "목표 방향 진행비"]; vals = [p["magnitude_ratio"], p["cosine"], p["progress_ratio"]]
    a.bar(names, vals, color=C["oft"], width=0.55)
    for i, y in enumerate(vals):
        a.text(i, y + 0.02, "%.2f" % y, ha="center", fontsize=9)
    a.axhline(1.0, color="#333", lw=0.8, ls="--"); a.set_ylim(0, 1.2)
    a.set_title("(a) 같은 로봇 자세에서 Isaac 화면 ÷ MuJoCo 화면\n    OFT 명령 비교 (43 샘플 중앙값) → 화면 차이는 판단을 거의 안 바꿈", fontsize=10)
    labs = ["빠른 실행\n(고정 배치 500 판)", "보정 실행\n(50 배치 499 판)"]
    b.bar(labs, [c["fast"], c["cal"]], color=[C["bad"], C["mj"]], width=0.5)
    for i, y in enumerate([c["fast"], c["cal"]]):
        b.text(i, y + 2, "%d 판" % y, ha="center", fontsize=10)
    b.set_ylim(0, 115); b.set_ylabel("허공에서 닫은 판 수")
    b.set_title("(b) OFT 의 「허공에서 닫음」 — 실행 배율을 맞추자 사라짐", fontsize=10)
    fig.suptitle("그림 16. 실패 원인 가설 검증 — 인식이 아니라 실행 동역학", x=0.01, ha="left", fontsize=11.5, fontweight="bold")
    fig.tight_layout(); save(fig, "fig16_hypothesis_tests.png")


def fig17():
    g = D["grasp_bench"]; depths = ["-10", "-20", "-30"]
    order = list(g["isaac"].keys())
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 4.3))
    w = 0.8 / (len(order) + 1)
    for j, d in enumerate(depths):
        a.bar(j - 0.4 + w / 2, g["mujoco"][d]["rise"], w * 0.95, color=C["mj"], label="MuJoCo" if j == 0 else None)
        for i, k in enumerate(order):
            col = C["ovla"] if k in ("기본 85 g",) else ("#7aa6d6" if "6 g" in k else C["grey"])
            a.bar(j - 0.4 + w * (i + 1.5), g["isaac"][k][d][0], w * 0.95, color=col, label=k if j == 0 else None)
    a.axhline(80, color="#333", lw=0.8, ls="--"); a.text(2.45, 81.5, "손이 들어 올린 높이 80 mm", fontsize=8, ha="right")
    a.set_xticks(range(3), ["테두리 −10 mm", "−20 mm", "−30 mm"]); a.set_ylabel("그릇 바닥 상승 (mm)"); a.set_ylim(0, 92)
    a.legend(frameon=False, fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    a.set_title("(a) 같은 동작으로 쥐고 8 cm 들기 — 그릇 바닥이 따라 올라온 높이")
    for i, k in enumerate(order):
        b.plot(range(3), [g["isaac"][k][d][1] for d in depths], marker="o", lw=1.2,
               color=C["ovla"] if k == "기본 85 g" else ("#7aa6d6" if "6 g" in k else C["grey"]), label=k)
    b.axhline(53, color=C["mj"], lw=1.2, ls="--"); b.text(2.05, 54.5, "수평 (53 mm)", fontsize=8, color=C["mj"])
    b.set_xticks(range(3), ["−10 mm", "−20 mm", "−30 mm"]); b.set_ylabel("들고 난 뒤 그릇 세로 크기 (mm)"); b.set_ylim(45, 110)
    b.set_title("(b) Isaac: 그릇 기울기 (세로 크기가 클수록 많이 기움).  MuJoCo 기울기 2 · 8 · 15°")
    fig.suptitle("그림 17. 정책 없는 파지 시험 (task 0) — Isaac 은 한쪽 테두리 파지 시 그릇이 쥔 곳을 축으로 기움.  그릇 쪽 PhysX 설정은 효과 없음, 질량만 일부",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout(); save(fig, "fig17_grasp_bench.png")


def fig18():
    f = D["camera"]
    fig, ax = plt.subplots(figsize=(9.5, 4.0))
    xs = [i / 100 for i in range(10, 101)]
    t = math.tan(math.radians(f["d405_vfov_square_deg"] / 2))
    ax.plot(xs, [2 * math.degrees(math.atan(t * math.sqrt(x))) for x in xs], color=C["ovla"], lw=1.6, label="D405 정사각 → 면적 a 중앙 크롭")
    ax.axhline(f["libero_native_deg"], color=C["mj"], ls="--", lw=1); ax.text(0.11, f["libero_native_deg"] + 0.8, "LIBERO 학습 화면 45°", color=C["mj"], fontsize=8.5)
    ax.axhline(f["libero_after_openvla_crop_deg"], color=C["mj"], ls=":", lw=1); ax.text(0.11, f["libero_after_openvla_crop_deg"] - 2.4, "LIBERO + OpenVLA 0.9 크롭 42.9°", color=C["mj"], fontsize=8.5)
    for a_, lab, res in f["points"]:
        y = 2 * math.degrees(math.atan(t * math.sqrt(a_)))
        ax.scatter([a_], [y], color=C["bad"] if "0/5" in res else C["oft"], zorder=3)
        off = {0.9: (-150, -8), 1.0: (-150, 10)}.get(a_, (6, 6))
        ax.annotate("%s\n%.1f° · %s" % (lab, y, res), (a_, y), textcoords="offset points", xytext=off, fontsize=8)
    ax.set_xlabel("크롭 면적 a (정사각 기준)"); ax.set_ylabel("모델이 보는 세로 화각 (°)"); ax.set_ylim(20, 78)
    ax.legend(frameon=False, loc="upper left", fontsize=8.5)
    ax.set_title("그림 18. D405 크롭과 실효 화각 (세로 70.9°, 정사각 픽셀) — LIBERO 화각 근처에서는 같고, 크게 넓어지면 무너짐", loc="left", fontsize=11)
    save(fig, "fig18_camera_fov.png")


def fig19():
    v = D["alignment"]; lags = [-2, -1, 0, 1, 2]
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 3.8))
    for key, col, lab in (("v4", C["bad"], "v4 (화면 k)"), ("v4b", C["ovla"], "v4b (화면 k+1)"), ("mujoco", C["mj"], "MuJoCo 원본")):
        a.plot(lags, v["v5b"][key], marker="o", color=col, label=lab)
        b.plot(lags, v["v5"][key], marker="o", color=col, label=lab)
    a.set_title("(a) 화면 변화 ↔ 자기 손끝 속도 (지연별 상관)"); b.set_title("(b) 화면 변화 ↔ 액션 크기")
    for ax in (a, b):
        ax.set_xticks(lags); ax.set_xlabel("지연 L (스텝)"); ax.legend(frameon=False, fontsize=8.5)
    a.set_ylabel("상관")
    a.text(1.9, min(v["v5b"]["mujoco"]) + 0.01, "봉우리: v4 %+.2f · v4b %+.2f · 원본 %+.2f" % tuple(v["vertex_v5b"]), ha="right", fontsize=8)
    b.text(1.9, min(v["v5"]["mujoco"]) + 0.01, "봉우리: v4 %+.2f · v4b %+.2f · 원본 %+.2f\n(남은 0.6 = 재생 추종 지연 0.44)" % tuple(v["vertex_v5"]), ha="right", fontsize=8)
    fig.suptitle("그림 19. 학습 데이터 시간 정렬 — 재생 기록의 화면을 한 칸 옮기자(v4 → v4b) 곡선이 원본 LIBERO 와 겹침", x=0.01, ha="left", fontsize=11.5, fontweight="bold")
    fig.tight_layout(); save(fig, "fig19_dataset_alignment.png")


if __name__ == "__main__":
    for fn in (fig13, fig14, fig15, fig16, fig17, fig18, fig19):
        fn()
