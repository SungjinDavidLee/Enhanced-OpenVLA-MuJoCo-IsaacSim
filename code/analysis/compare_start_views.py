#!/usr/bin/env python3
"""태스크 0~9 시작 화면을 MuJoCo 와 Isaac 의 「정책 입력」 기준으로 나란히 비교한다.

  MuJoCo   LIBERO 초기 상태 (--state, 기본 0) → 무동작 10 스텝 (OpenVLA 평가와 같음) → agentview 256
           → 180° 회전 → 중앙 크롭 0.9 → 224   (OpenVLA 공식 LIBERO 전처리)
  Isaac    s00 평가 캡처의 openvla_rgb.mp4 첫 프레임 = 서버가 crop(0.35) 해서 정책에 넣은 224 화면
           (없으면 raw_rgb_first.png 에 같은 crop 을 적용)
  세 번째 칸은 두 화면을 반씩 겹친 것 — 물체 위치 · 크기가 맞으면 겹쳐 보이고, 다르면 두 겹으로 보인다.

    conda activate openvla_libero5090
    python3 compare_start_views.py --isaac-root ~/LIBERO/experiments/zs_s00_S10 --out ~/start_views.png
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import sys

import cv2
import numpy as np


def center_crop_resize(img: np.ndarray, area: float, size: int = 224) -> np.ndarray:
    h, w = img.shape[:2]
    s = min(h, w)
    img = img[(h - s) // 2:(h - s) // 2 + s, (w - s) // 2:(w - s) // 2 + s]
    f = math.sqrt(area)
    c = int(round(s * f))
    o = (s - c) // 2
    return cv2.resize(img[o:o + c, o:o + c], (size, size), interpolation=cv2.INTER_AREA)


def isaac_view(root: str, task: int, crop_area: float) -> np.ndarray | None:
    d = os.path.join(root, f"task_{task:02d}_e000")
    v = glob.glob(d + "/**/openvla_rgb.mp4", recursive=True)
    if v:
        cap = cv2.VideoCapture(v[0]); ok, fr = cap.read(); cap.release()
        if ok:
            return cv2.resize(fr, (224, 224))
    p = glob.glob(d + "/**/raw_rgb_first.png", recursive=True)
    if p:
        return center_crop_resize(cv2.imread(p[0]), crop_area)
    return None


def label(img: np.ndarray, text: str) -> np.ndarray:
    img = img.copy()
    cv2.rectangle(img, (0, 0), (img.shape[1], 18), (0, 0, 0), -1)
    cv2.putText(img, text, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--isaac-root", default=os.path.expanduser("~/LIBERO/experiments/zs_s00_S10"))
    ap.add_argument("--state", type=int, default=0, help="LIBERO 초기 상태 번호 (s00 평가면 0)")
    ap.add_argument("--crop-area", type=float, default=0.35, help="Isaac raw 화면에 쓸 crop (openvla_rgb.mp4 가 없을 때만)")
    ap.add_argument("--out", default=os.path.expanduser("~/start_views.png"))
    a = ap.parse_args()
    sys.path.insert(0, "/home/user/LIBERO")
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv
    suite = benchmark.get_benchmark_dict()["libero_spatial"]()

    rows = []
    for t in range(10):
        env = OffScreenRenderEnv(bddl_file_name=suite.get_task_bddl_file_path(t), camera_heights=256, camera_widths=256)
        env.reset(); env.set_init_state(suite.get_task_init_states(t)[a.state])
        obs = None
        for _ in range(10):
            obs, *_ = env.step([0, 0, 0, 0, 0, 0, -1])
        mj = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
        mj = cv2.cvtColor(center_crop_resize(mj, 0.9), cv2.COLOR_RGB2BGR)
        env.close()
        isa = isaac_view(a.isaac_root, t, a.crop_area)
        if isa is None:
            isa = np.zeros_like(mj); blend = mj.copy()
        else:
            blend = cv2.addWeighted(mj, 0.5, isa, 0.5, 0)
        name = suite.get_task(t).language[:52]
        row = np.concatenate([label(mj, f"T{t} MuJoCo"), label(isa, f"T{t} Isaac (crop)"), label(blend, "blend 50/50")], axis=1)
        cap = np.zeros((20, row.shape[1], 3), np.uint8)
        cv2.putText(cap, name, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        rows.append(np.concatenate([cap, row], axis=0))
        cv2.imwrite(os.path.splitext(a.out)[0] + f"_t{t}.png", rows[-1])
        print(f"task {t}  {'Isaac 캡처 있음' if isa.any() else 'Isaac 캡처 없음'}  {name}")
    left, right = rows[0::2], rows[1::2]
    grid = np.concatenate([np.concatenate(left, axis=0), np.concatenate(right, axis=0)], axis=1)
    cv2.imwrite(a.out, grid)
    print("저장:", a.out, grid.shape[1], "x", grid.shape[0])
    return 0


if __name__ == "__main__":
    sys.exit(main())
