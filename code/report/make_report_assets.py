#!/usr/bin/env python3
"""보고서 · 포트폴리오용 그림을 실제 Isaac Sim 실행 기록에서 한 번에 뽑는다.

    conda activate openvla_libero5090
    python3 make_report_assets.py                  # 기본 경로에서 전부 (10–20 분, CPU 만)
    python3 make_report_assets.py --only charts gifs

출력 ~/report_assets/
  charts/      overall_success.png · success_by_task.png · failure_breakdown.png   (실행 기록에서 직접 계산)
  moments/     <조건>/task_XX.png · overview.png   (초기 · 파지 · 놓기, make_task_moments_v2.py 사용)
  filmstrips/  <조건>_task_XX.png   (성공 판 하나를 시작 → 성공까지 6 장)
  gifs/        <조건>_task_XX_success.gif · fail_*.gif   (가로 360 px, 약 8 fps)
  failures/    fail_overshoot.png · fail_tilt.png · fail_task5.png · bench_tilt.png
  camera/      d405_cropbox.png · model_view.png · wrist_view.png
  manifest.md  파일별 한국어 · 영어 설명 (문서에서 이 이름을 그대로 참조)
같은 폴더에 make_task_moments_v2.py 가 있어야 moments 를 만든다.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HOME = os.path.expanduser("~")
EXP = f"{HOME}/LIBERO/experiments"
COND = {   # 이름: (폴더, 표시 이름)
    "ovla_fast": (f"{EXP}/zs_s00_S10", "OpenVLA · fast exec (S10, g1.0) · fixed layout"),
    "oft_fast": (f"{EXP}/zs_oft_s00", "OFT · fast exec (S10, g1.0) · fixed layout"),
    "oft_cal": (f"{EXP}/zs_oft_cal_50ep", "OFT · calibrated exec (S3, g0.65) · 50 LIBERO layouts"),
    "ovla_cal": (f"{EXP}/zs_ovla_cal_50ep", "OpenVLA · calibrated exec (S3, g0.65) · 50 LIBERO layouts"),
}
OUT = f"{HOME}/report_assets"
FONTS = ["/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf", "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
         "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]
MANIFEST = []


def font(size):
    for p in FONTS:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except OSError:
                pass
    return ImageFont.load_default()


def note(path, ko, en):
    MANIFEST.append((os.path.relpath(path, OUT), ko, en))


# ---------------------------------------------------------------- 기록 읽기
def episodes(root):
    out = []
    for d in sorted(glob.glob(f"{root}/task_*_e*")):
        if not os.path.isdir(d):
            continue
        fs = glob.glob(d + "/**/episode_result.json", recursive=True)
        if not fs:
            continue
        try:
            r = json.load(open(fs[0])).get("result", {})
        except Exception:  # noqa: BLE001
            continue
        by = {}
        for s in r.get("stage_trace") or []:
            ps = s.get("policy_step")
            if s.get("observations") and isinstance(ps, int) and ps >= 0 and ps not in by:
                by[ps] = s["observations"]
        t = int(os.path.basename(d).split("_")[1]); e = int(d.rsplit("_e", 1)[-1])
        out.append({"dir": d, "task": t, "ep": e, "success": bool(r.get("task_success")), "obs": [by[k] for k in sorted(by)],
                    "video": (glob.glob(d + "/**/raw_rgb.mp4", recursive=True) or [None])[0],
                    "wrist": (glob.glob(d + "/**/wrist_rgb.mp4", recursive=True) or [None])[0]})
    return out


def classify(ep):
    """A 성공 · B 물었는데 못 듦 · C 허공에서 닫음 · D 들고 가다 실패 · E 접시 근처 실패 (classify_episodes_v2 규칙)."""
    if ep["success"]:
        return "A"
    o = ep["obs"]
    if not o:
        return None
    w = [x.get("gripper_finger_mean_m") or 0.04 for x in o]
    bz = [x["target_bowl_aabb_min_m"][2] for x in o]
    lift = max(bz) - bz[0]
    held = any(all(0.0025 < w[j] < 0.012 for j in range(i, i + 3)) for i in range(len(w) - 2))
    if lift < 0.04:
        return "B" if held else "C"
    b = np.asarray(o[-1]["target_bowl_aabb_center_m"][:2]); p = np.asarray((o[-1].get("plate_aabb_center_m") or [9, 9])[:2])
    return "E" if np.linalg.norm(b - p) < 0.10 else "D"


def frames(path, idx=None):
    cap = cv2.VideoCapture(path); fr = []
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    want = set(range(n)) if idx is None else {max(0, min(i, n - 1)) for i in idx}
    for i in range(n):
        ok, f = cap.read()
        if not ok:
            break
        if i in want:
            fr.append((i, cv2.cvtColor(f, cv2.COLOR_BGR2RGB)))
    cap.release()
    return fr


def label(img, text, size=20):
    im = Image.fromarray(img) if isinstance(img, np.ndarray) else img
    dr = ImageDraw.Draw(im); f = font(size)
    tw = dr.textlength(text, font=f)
    dr.rectangle([0, 0, tw + 16, size + 12], fill=(0, 0, 0))
    dr.text((8, 5), text, fill=(255, 255, 255), font=f)
    return im


# ---------------------------------------------------------------- charts
def make_charts(data):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(f"{OUT}/charts", exist_ok=True)
    names = [k for k in COND if data.get(k)]
    # 1 전체 성공률
    fig, ax = plt.subplots(figsize=(8, 3.6))
    vals = [100 * np.mean([e["success"] for e in data[k]]) for k in names]
    ns = [len(data[k]) for k in names]
    bars = ax.barh([COND[k][1] for k in names], vals, color=["#8aa", "#aab", "#2a7", "#27a"][:len(names)])
    for b, v, n in zip(bars, vals, ns):
        ax.text(v + 0.8, b.get_y() + b.get_height() / 2, "%.1f %%  (n=%d)" % (v, n), va="center", fontsize=10)
    ax.set_xlim(0, 100); ax.set_xlabel("task success (%)"); ax.invert_yaxis(); ax.set_title("Zero-shot success in Isaac Sim (LIBERO-Spatial, 220 steps)")
    fig.tight_layout(); p = f"{OUT}/charts/overall_success.png"; fig.savefig(p, dpi=160); plt.close(fig)
    note(p, "조건별 제로샷 성공률 (고친 하네스, 220 스텝)", "Zero-shot success by condition (fixed harness, 220 steps)")
    # 2 태스크별
    fig, ax = plt.subplots(figsize=(10, 4)); w = 0.8 / len(names)
    for j, k in enumerate(names):
        rate = [100 * np.mean([e["success"] for e in data[k] if e["task"] == t] or [0]) for t in range(10)]
        ax.bar(np.arange(10) + (j - (len(names) - 1) / 2) * w, rate, w, label=COND[k][1])
    ax.set_xticks(range(10)); ax.set_xticklabels(["T%d" % t for t in range(10)]); ax.set_ylabel("success (%)"); ax.set_ylim(0, 105)
    ax.legend(fontsize=8, loc="upper left"); ax.set_title("Success by task"); fig.tight_layout()
    p = f"{OUT}/charts/success_by_task.png"; fig.savefig(p, dpi=160); plt.close(fig)
    note(p, "태스크별 성공률", "Success rate by task")
    # 3 실패 분포
    fig, ax = plt.subplots(figsize=(9, 3.8)); cats = ["A", "B", "C", "D", "E"]
    colors = {"A": "#2a7", "B": "#e94", "C": "#c55", "D": "#96c", "E": "#aaa"}
    desc = {"A": "success", "B": "grasped, not lifted", "C": "closed on air", "D": "lifted, lost on the way", "E": "failed near plate"}
    left = np.zeros(len(names))
    for c in cats:
        v = np.array([100 * np.mean([classify(e) == c for e in data[k]]) for k in names])
        ax.barh([COND[k][1] for k in names], v, left=left, color=colors[c], label="%s %s" % (c, desc[c]))
        for i, x in enumerate(v):
            if x >= 5:
                ax.text(left[i] + x / 2, i, "%.0f" % x, ha="center", va="center", fontsize=9, color="white")
        left += v
    ax.set_xlim(0, 100); ax.invert_yaxis(); ax.set_xlabel("% of episodes"); ax.legend(fontsize=8, ncol=5, loc="lower center", bbox_to_anchor=(0.5, -0.42))
    ax.set_title("Outcome breakdown (physical signals)"); fig.tight_layout()
    p = f"{OUT}/charts/failure_breakdown.png"; fig.savefig(p, dpi=160, bbox_inches="tight"); plt.close(fig)
    note(p, "결과 분포 (A 성공 · B 물었는데 못 듦 · C 허공에서 닫음 · D 들고 가다 실패 · E 접시 근처 실패)", "Outcome breakdown by failure mode")


# ---------------------------------------------------------------- moments
def make_moments():
    tool = os.path.join(os.path.dirname(os.path.abspath(__file__)), "make_task_moments_v2.py")
    if not os.path.exists(tool):
        print("  make_task_moments_v2.py 가 같은 폴더에 없어 moments 를 건너뜀"); return
    for k in ("ovla_fast", "oft_cal", "ovla_cal"):
        root, name = COND[k]
        if not glob.glob(root + "/task_*"):
            continue
        out = f"{OUT}/moments/{k}"
        subprocess.run([sys.executable, tool, "--root", root, "--model", k, "--out", out], check=False)
        note(out + "/overview.png", "태스크별 초기 · 파지 · 놓기 장면 — " + name, "Initial / grasp / place per task — " + name)
    root = COND["ovla_fast"][0]
    if glob.glob(root + "/task_*"):
        out = f"{OUT}/moments/ovla_fast_modelview"
        subprocess.run([sys.executable, tool, "--root", root, "--model", "ovla_fast", "--out", out, "--crop-area", "0.35"], check=False)
        note(out + "/overview.png", "같은 장면을 모델 입력(정사각 → 면적 0.35 → 224²)으로", "Same scenes as model input (square → area 0.35 → 224²)")


# ---------------------------------------------------------------- filmstrips · gifs
def pick_success(eps, t):
    s = [e for e in eps if e["task"] == t and e["success"] and e["video"]]
    return s[0] if s else None


def make_filmstrips(data):
    os.makedirs(f"{OUT}/filmstrips", exist_ok=True)
    for k in ("oft_cal", "ovla_fast"):
        for t in range(10):
            ep = pick_success(data.get(k) or [], t)
            if not ep:
                continue
            n = len(ep["obs"]); idx = [int(round(i * (n - 1) / 5)) for i in range(6)]
            fr = frames(ep["video"], idx)
            if not fr:
                continue
            tiles = [label(cv2.resize(f, (320, 240)), "step %d" % i, 16) for i, f in fr]
            strip = Image.new("RGB", (320 * len(tiles) + 4 * (len(tiles) - 1), 240 + 40), (255, 255, 255))
            ImageDraw.Draw(strip).text((6, 8), "Task %d · %s · ep %03d (success)" % (t, COND[k][1], ep["ep"]), fill=(0, 0, 0), font=font(18))
            for i, tile in enumerate(tiles):
                strip.paste(tile, (i * 324, 40))
            p = f"{OUT}/filmstrips/{k}_task_{t:02d}.png"; strip.save(p)
            note(p, "task %d 성공 판을 시작부터 성공까지 6 장 (%s)" % (t, COND[k][1]), "Task %d success, start to finish in 6 frames" % t)


def to_gif(path_video, out, title, stride=2, width=360, max_frames=160):
    fr = frames(path_video)
    if not fr:
        return False
    sel = [f for i, f in fr][::stride][:max_frames]
    h = int(sel[0].shape[0] * width / sel[0].shape[1])
    ims = [label(cv2.resize(f, (width, h)), title, 14).convert("P", palette=Image.ADAPTIVE) for f in sel]
    ims[0].save(out, save_all=True, append_images=ims[1:], duration=120, loop=0, optimize=True)
    return True


def make_gifs(data):
    os.makedirs(f"{OUT}/gifs", exist_ok=True)
    for t in range(10):
        ep = pick_success(data.get("oft_cal") or [], t) or pick_success(data.get("ovla_fast") or [], t)
        if ep:
            k = "oft_cal" if ep in (data.get("oft_cal") or []) else "ovla_fast"
            p = f"{OUT}/gifs/{k}_task_{t:02d}_success.gif"
            if to_gif(ep["video"], p, "T%d success (%s)" % (t, "OFT cal" if k == "oft_cal" else "OpenVLA fast")):
                note(p, "task %d 성공 장면 (움직이는 그림)" % t, "Task %d success (animated)" % t)
    # 실패 장면
    of = [e for e in (data.get("oft_fast") or []) if e["task"] == 0 and not e["success"] and e["video"]]
    if of and to_gif(of[0]["video"], f"{OUT}/gifs/fail_overshoot_oft_fast_t0.gif", "OFT fast: overshoots bowl"):
        note(f"{OUT}/gifs/fail_overshoot_oft_fast_t0.gif", "빠른 설정 OFT 가 첫 청크에 그릇을 지나쳐 접시 근처에서 닫음 (task 0)", "OFT at fast execution overshoots the bowl in its first chunk")
    oc = [e for e in (data.get("oft_cal") or []) if e["task"] == 0 and classify(e) == "B" and e["video"]]
    if oc and to_gif(oc[0]["video"], f"{OUT}/gifs/fail_tilt_oft_cal_t0.gif", "OFT calibrated: grasps, bowl tilts"):
        note(f"{OUT}/gifs/fail_tilt_oft_cal_t0.gif", "보정 설정 OFT 는 테두리를 정확히 물지만 들 때 그릇이 기움 (task 0)", "Calibrated OFT grasps the rim but the bowl tilts on lift")
    o5 = [e for e in (data.get("oft_cal") or []) if e["task"] == 5 and e["video"]]
    if o5 and to_gif(o5[0]["video"], f"{OUT}/gifs/fail_task5_oft_cal.gif", "Task 5: bowl on ramekin"):
        note(f"{OUT}/gifs/fail_task5_oft_cal.gif", "task 5 (라메킨 위 그릇) 실패 장면", "Task 5 (bowl on ramekin) failure")


# ---------------------------------------------------------------- failures · bench · camera
def key_frame(ep, which):
    o = ep["obs"]; w = [x.get("gripper_finger_mean_m") or 0.04 for x in o]
    if which == "first_close":
        return next((i for i in range(len(w)) if w[i] < 0.012), len(w) // 2)
    if which == "after_lift":
        i = next((i for i in range(len(w)) if 0.0025 < w[i] < 0.012), 0)
        return min(len(w) - 1, i + 20)
    return len(w) - 1


def make_failures(data):
    os.makedirs(f"{OUT}/failures", exist_ok=True)
    items = [("oft_fast", 0, "first_close", "fail_overshoot.png", "빠른 설정 OFT: 그릇(왼쪽)을 지나쳐 접시 근처에서 닫는 순간", "Fast-exec OFT closes past the bowl, near the plate"),
             ("oft_cal", 0, "after_lift", "fail_tilt.png", "보정 설정 OFT: 테두리를 문 뒤 들 때 그릇이 기운 모습", "Calibrated OFT: bowl tilts after a rim grasp"),
             ("oft_cal", 5, "after_lift", "fail_task5.png", "task 5: 라메킨 위 그릇을 물었지만 들리지 않음", "Task 5: grasped on the ramekin but not lifted")]
    for k, t, which, name, ko, en in items:
        cand = [e for e in (data.get(k) or []) if e["task"] == t and not e["success"] and e["video"] and (k != "oft_cal" or classify(e) == "B")]
        if not cand:
            continue
        i = key_frame(cand[0], which); fr = frames(cand[0]["video"], [i])
        if fr:
            p = f"{OUT}/failures/{name}"; label(fr[0][1], "%s · T%d · ep %03d · step %d" % (k, t, cand[0]["ep"], i)).save(p); note(p, ko, en)
    # 파지 시험 (정책 없음): Isaac 기울기
    for d in sorted(glob.glob(f"{EXP}/grasp_bench/base2_t0_d-20")):
        v = (glob.glob(d + "/**/raw_rgb.mp4", recursive=True) or [None])[0]
        if v:
            fr = frames(v)
            if fr:
                a = label(cv2.resize(fr[60][1] if len(fr) > 60 else fr[0][1], (480, 360)), "closed on rim (step 60)")
                b = label(cv2.resize(fr[-1][1], (480, 360)), "after 8 cm lift: bowl hangs tilted")
                im = Image.new("RGB", (964, 360), (255, 255, 255)); im.paste(a, (0, 0)); im.paste(b, (484, 0))
                p = f"{OUT}/failures/bench_tilt.png"; im.save(p)
                note(p, "정책 없는 파지 시험 (Isaac): 같은 동작에서 MuJoCo 는 수평으로 들리고(기울기 2–15°) Isaac 은 크게 기움", "Scripted grasp bench in Isaac: bowl tilts heavily, MuJoCo lifts it level")


def make_camera(data):
    os.makedirs(f"{OUT}/camera", exist_ok=True)
    ep = pick_success(data.get("ovla_fast") or [], 6) or pick_success(data.get("oft_cal") or [], 6)
    if ep:
        fr = frames(ep["video"], [0])
        if fr:
            img = fr[0][1].copy(); h, w = img.shape[:2]; s = min(h, w); c = int(round(s * np.sqrt(0.35)))
            x0, y0 = (w - c) // 2, (h - c) // 2
            box = img.copy(); cv2.rectangle(box, (x0, y0), (x0 + c, y0 + c), (255, 60, 60), 3)
            p = f"{OUT}/camera/d405_cropbox.png"; label(box, "D405 %dx%d · red = model input (area 0.35)" % (w, h)).save(p)
            note(p, "D405 원본 화면(640×480)과 모델이 보는 영역(빨간 네모, 세로 약 46°)", "D405 frame with the model-input region (red box, ~46° vertical)")
            mv = cv2.resize(img[y0:y0 + c, x0:x0 + c], (224, 224), interpolation=cv2.INTER_AREA)
            p = f"{OUT}/camera/model_view.png"; Image.fromarray(cv2.resize(mv, (448, 448), interpolation=cv2.INTER_NEAREST)).save(p)
            note(p, "모델이 실제로 받는 224×224 입력 (2 배 확대)", "Actual 224×224 model input (2× upscaled)")
    wr = [e for e in episodes(f"{EXP}/replay_demo/seed7") if e["success"] and e["wrist"]][:1] if os.path.isdir(f"{EXP}/replay_demo/seed7") else []
    if wr:
        n = len(wr[0]["obs"]); fr = frames(wr[0]["wrist"], [0, n // 3, 2 * n // 3])
        if fr:
            tiles = [label(cv2.resize(f, (400, 225)), "wrist D405 · step %d" % i, 14) for i, f in fr]
            im = Image.new("RGB", (400 * len(tiles) + 8, 225), (255, 255, 255))
            for i, tl in enumerate(tiles):
                im.paste(tl, (i * 404, 0))
            p = f"{OUT}/camera/wrist_view.png"; im.save(p)
            note(p, "손목 D405 카메라 화면 (사람 시연 재생 데이터 수집 중)", "Wrist D405 view during human-demo replay collection")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", default=None, choices=["charts", "moments", "filmstrips", "gifs", "failures", "camera"])
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    want = set(a.only or ["charts", "moments", "filmstrips", "gifs", "failures", "camera"])
    data = {}
    for k, (root, name) in COND.items():
        data[k] = episodes(root) if os.path.isdir(root) else []
        print("%-9s %4d 판  (%s)" % (k, len(data[k]), root if data[k] else "없음"))
    for step, fn in (("charts", lambda: make_charts(data)), ("moments", make_moments), ("filmstrips", lambda: make_filmstrips(data)),
                     ("gifs", lambda: make_gifs(data)), ("failures", lambda: make_failures(data)), ("camera", lambda: make_camera(data))):
        if step in want:
            print("==", step); fn()
    with open(f"{OUT}/manifest.md", "w") as f:
        f.write("# report_assets 목록 / Asset manifest\n\n| 파일 / File | 설명 | Caption |\n|---|---|---|\n")
        for rel, ko, en in MANIFEST:
            f.write("| `%s` | %s | %s |\n" % (rel, ko, en))
    print("\n완료:", OUT, " (%d 개, manifest.md 참고)" % len(MANIFEST))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
