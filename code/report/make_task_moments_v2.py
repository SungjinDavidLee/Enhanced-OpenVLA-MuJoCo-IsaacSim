#!/usr/bin/env python3
"""태스크별로 「초기 · 파지 · 놓기」 세 장면을 평가 영상에서 뽑아 사진으로 만든다.

    python3 make_task_moments.py                                   # OpenVLA 500 판 (zs_s00_S10)
    python3 make_task_moments.py --root ~/LIBERO/experiments/zs_oft_cal_50ep --model OFT
    python3 make_task_moments.py --pick 5=12 7=3                   # 태스크별 판 번호 직접 지정
    python3 make_task_moments.py --crop-area 0.35                  # 모델이 실제로 보는 화면 (정사각 → 면적 0.35 → 224²)
    python3 make_task_moments.py --crop-area 0.35 --show-crop-box  # 원본 위에 크롭 영역을 빨간 네모로

출력 (--out, 기본 ~/task_moments_<모델>/)
  task_00.png … task_09.png   태스크별 한 장 (초기 | 파지 | 놓기)
  overview.png                열 개 태스크를 위에서 아래로 이어 붙인 한 장
  picks.txt                   어느 판의 몇 번째 스텝을 썼는지

장면을 고르는 기준
  초기   첫 화면
  파지   손가락 폭이 2.5–12 mm (그릇을 문 상태) 로 3 스텝 이상 이어지고, 이후 30 스텝 안에 그릇이 20 mm 이상 올라간 첫 순간
  놓기   그릇이 가장 높이 올라간 뒤 손가락이 30 mm 넘게 벌어진 첫 순간 (없으면 판의 마지막 화면)
판 고르기   성공한 판 중 번호가 가장 작은 것.  성공 판이 없으면 그릇을 가장 높이 든 실패 판 (그림에 표시)
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

PANEL_W, PANEL_H, BAR = 480, 360, 44
FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf", "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc", "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc", "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
]


def load_font(size):
    for p in FONT_CANDIDATES:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size), True
            except OSError:
                pass
    return ImageFont.load_default(), False


FONT, KOREAN = load_font(22)
FONT_S, _ = load_font(17)
L = {"init": "초기" if KOREAN else "Initial", "grasp": "파지" if KOREAN else "Grasp", "place": "놓기" if KOREAN else "Place",
     "nosucc": "성공 판 없음 — 가장 높이 든 실패 판" if KOREAN else "no success - highest-lift failure",
     "noframe": "해당 장면 없음" if KOREAN else "not reached", "step": "스텝" if KOREAN else "step"}


def episode_data(d):
    fs = glob.glob(d + "/**/episode_result.json", recursive=True)
    vs = glob.glob(d + "/**/raw_rgb.mp4", recursive=True)
    if not fs or not vs:
        return None
    res = json.load(open(fs[0])).get("result", {})
    by = {}
    for s in res.get("stage_trace") or []:
        ps = s.get("policy_step")
        if s.get("observations") and isinstance(ps, int) and ps >= 0 and ps not in by:
            by[ps] = s["observations"]
    if not by:
        return None
    ks = sorted(by)
    return {"dir": d, "success": bool(res.get("task_success")), "steps": ks, "obs": [by[k] for k in ks],
            "video": vs[0], "instruction": res.get("instruction") or res.get("task_instruction") or ""}


def moments(ep):
    obs = ep["obs"]; ks = ep["steps"]
    w = np.array([o.get("gripper_finger_mean_m") or 0.04 for o in obs])
    bz = np.array([o["target_bowl_aabb_min_m"][2] for o in obs])
    lift = bz - bz[0]
    grasp = None
    for i in range(len(w) - 3):
        if all(0.0025 < w[j] < 0.012 for j in range(i, i + 3)) and lift[i:i + 30].max() > 0.02:
            grasp = i; break
    if grasp is None:
        grasp = next((i for i in range(len(w)) if 0.0025 < w[i] < 0.012), None)
    place = None
    if grasp is not None:
        top = grasp + int(np.argmax(lift[grasp:]))
        place = next((i for i in range(top, len(w)) if w[i] > 0.030), None)
        if place is None and ep["success"]:
            place = len(w) - 1
    return {"init": 0, "grasp": grasp, "place": place}, float(lift.max())


def read_frames(path, want):
    cap = cv2.VideoCapture(path); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); out = {}
    for name, idx in want.items():
        if idx is None:
            out[name] = None; continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, min(idx, n - 1)))
        ok, fr = cap.read()
        out[name] = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB) if ok else None
    cap.release()
    return out


def crop_box(img, area):
    """정사각(짧은 변) 중앙 → 면적 area 중앙 크롭 영역 (x0, y0, x1, y1)."""
    h, w = img.shape[:2]; s = min(h, w); cs = int(round(s * np.sqrt(area)))
    x0 = (w - cs) // 2; y0 = (h - cs) // 2
    return x0, y0, x0 + cs, y0 + cs


def model_view(img, area, size):
    if img is None:
        return None
    x0, y0, x1, y1 = crop_box(img, area)
    return cv2.resize(img[y0:y1, x0:x1], (size, size), interpolation=cv2.INTER_AREA)


def with_box(img, area):
    if img is None:
        return None
    out = img.copy(); x0, y0, x1, y1 = crop_box(img, area)
    cv2.rectangle(out, (x0, y0), (x1 - 1, y1 - 1), (255, 60, 60), 3)
    return out


def panel(img, title, sub):
    canvas = Image.new("RGB", (PANEL_W, PANEL_H + BAR), (245, 245, 245))
    dr = ImageDraw.Draw(canvas)
    dr.text((12, 8), title, fill=(20, 20, 20), font=FONT)
    if sub:
        dr.text((PANEL_W - 12 - dr.textlength(sub, font=FONT_S), 12), sub, fill=(90, 90, 90), font=FONT_S)
    if img is None:
        dr.rectangle([0, BAR, PANEL_W, PANEL_H + BAR], fill=(225, 225, 225))
        t = L["noframe"]; dr.text(((PANEL_W - dr.textlength(t, font=FONT)) / 2, BAR + PANEL_H / 2 - 12), t, fill=(120, 120, 120), font=FONT)
    else:
        h, w = img.shape[:2]; s = min(PANEL_W / w, PANEL_H / h)
        im = Image.fromarray(img).resize((int(w * s), int(h * s)), Image.BILINEAR)
        canvas.paste(im, ((PANEL_W - im.width) // 2, BAR + (PANEL_H - im.height) // 2))
    return canvas


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=os.path.expanduser("~/LIBERO/experiments/zs_s00_S10"))
    ap.add_argument("--model", default="OpenVLA")
    ap.add_argument("--out", default=None)
    ap.add_argument("--tasks", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--pick", nargs="*", default=[], help="task=판번호 (예: 5=12)")
    ap.add_argument("--crop-area", type=float, default=None, help="모델 입력 크롭 면적 (OpenVLA 우리 평가 0.35).  주면 모델이 보는 화면으로")
    ap.add_argument("--model-size", type=int, default=224, help="모델 입력 해상도 (OpenVLA 224)")
    ap.add_argument("--show-crop-box", action="store_true", help="잘라 보여 주는 대신 원본 위에 크롭 영역을 빨간 네모로 표시")
    a = ap.parse_args()
    suffix = "" if not a.crop_area else ("_cropbox%03d" if a.show_crop_box else "_crop%03d") % int(round(a.crop_area * 100))
    out = os.path.expanduser(a.out or "~/task_moments_%s%s" % (a.model, suffix))
    os.makedirs(out, exist_ok=True)
    forced = {int(k): int(v) for k, v in (p.split("=") for p in a.pick)}
    rows, log = [], []
    for t in a.tasks:
        dirs = sorted(d for d in glob.glob(os.path.join(a.root, "task_%02d_e*" % t)) if os.path.isdir(d))
        eps = [e for e in (episode_data(d) for d in dirs) if e]
        if not eps:
            print("task %d: 결과 없음" % t); continue
        note = ""
        if t in forced:
            ep = next((e for e in eps if e["dir"].endswith("_e%03d" % forced[t])), None) or eps[0]
        else:
            succ = [e for e in eps if e["success"]]
            if succ:
                ep = succ[0]
            else:
                ep = max(eps, key=lambda e: moments(e)[1]); note = L["nosucc"]
        mom, lift = moments(ep)
        frames = read_frames(ep["video"], {k: (None if v is None else v) for k, v in mom.items()})
        ep_id = ep["dir"].rsplit("_e", 1)[-1]
        head = "Task %d · %s · ep %s%s" % (t, a.model, ep_id, ("" if ep["success"] else (" · " + ("실패" if KOREAN else "fail"))))
        if a.crop_area:
            head += ("  [크롭 영역 표시 %.2f]" if KOREAN else "  [crop box %.2f]") % a.crop_area if a.show_crop_box else \
                    ("  [모델 입력: 정사각 → 면적 %.2f → %d²]" if KOREAN else "  [model input: square -> area %.2f -> %d px]") % (a.crop_area, a.model_size)
        instr = ep["instruction"]
        if a.crop_area:
            frames = {n: (with_box(f, a.crop_area) if a.show_crop_box else model_view(f, a.crop_area, a.model_size)) for n, f in frames.items()}
        panels = [panel(frames[n], L[n], ("%s %d" % (L["step"], ep["steps"][mom[n]])) if mom[n] is not None else "") for n in ("init", "grasp", "place")]
        top_h = 70 if (instr or note) else 40
        img = Image.new("RGB", (PANEL_W * 3 + 16, PANEL_H + BAR + top_h + 8), (255, 255, 255))
        dr = ImageDraw.Draw(img)
        dr.text((10, 8), head, fill=(0, 0, 0), font=FONT)
        sub = " — ".join(x for x in (instr, note) if x)
        if sub:
            dr.text((10, 38), sub, fill=(180, 40, 40) if note else (70, 70, 70), font=FONT_S)
        for i, p in enumerate(panels):
            img.paste(p, (4 + i * (PANEL_W + 4), top_h))
        path = os.path.join(out, "task_%02d.png" % t); img.save(path); rows.append(img)
        line = "task %d  ep %s  성공=%s  초기 step %s · 파지 step %s · 놓기 step %s  최대 상승 %.0f mm  %s" % (
            t, ep_id, ep["success"], *[ep["steps"][mom[n]] if mom[n] is not None else "-" for n in ("init", "grasp", "place")], lift * 1000, note)
        log.append(line); print(line)
    if rows:
        W = max(r.width for r in rows); H = sum(r.height for r in rows) + 6 * (len(rows) - 1)
        ov = Image.new("RGB", (W, H), (200, 200, 200)); y = 0
        for r in rows:
            ov.paste(r, (0, y)); y += r.height + 6
        ov.save(os.path.join(out, "overview.png"))
        open(os.path.join(out, "picks.txt"), "w").write("\n".join(log) + "\n")
        print("저장:", out, "(task_XX.png · overview.png · picks.txt)" + ("" if KOREAN else "  — 한글 글꼴이 없어 영어 라벨 사용"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
