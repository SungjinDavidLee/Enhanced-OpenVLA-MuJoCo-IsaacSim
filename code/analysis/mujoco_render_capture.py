#!/usr/bin/env python3
"""MuJoCo(LIBERO) 에서 같은 궤적을 재생하며 agentview 와 eye_in_hand(손목) 화면을 찍는다.

Isaac 캡처 폴더와 같은 파일 이름으로 저장하므로 view_wrist_capture.py 로 나란히 볼 수 있다.

    conda activate openvla_libero5090
    python3 mujoco_render_capture.py --task 0 --episode 0 --out ~/LIBERO/experiments/wrist_test/mujoco_t0e0

저장:  raw_rgb.mp4 (agentview) · wrist_rgb.mp4 (eye_in_hand) · *_first.png · *_last.png
LIBERO 손목 카메라: right_hand 기준 pos (0.05, 0, 0), fovy 75°, 원본 128×128 (여기서는 256 으로 렌더)
"""

from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", type=int, default=0)
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=256)
    a = ap.parse_args()
    sys.path.insert(0, "/home/user/LIBERO")
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    d = np.load("/home/user/isaac_libero/runs/replay_src/task_%02d/alltheTask_episode_%03d.npz" % (a.task, a.episode), allow_pickle=True)
    acts = d["libero_actions"]
    env = OffScreenRenderEnv(bddl_file_name=suite.get_task_bddl_file_path(a.task), camera_heights=a.size, camera_widths=a.size,
                             camera_names=["agentview", "robot0_eye_in_hand"])
    env.reset(); env.set_init_state(suite.get_task_init_states(a.task)[a.episode])
    obs = None
    for _ in range(10):
        obs, *_ = env.step([0, 0, 0, 0, 0, 0, -1])
    os.makedirs(a.out, exist_ok=True)
    writers = {}
    frames = {"raw_rgb": [], "wrist_rgb": []}
    for k in range(len(acts) + 1):
        # LIBERO 화면은 위아래가 뒤집혀 나오므로 180° 돌려 저장 (OpenVLA 전처리와 같음)
        av = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]); wr = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
        for name, img in (("raw_rgb", av), ("wrist_rgb", wr)):
            bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            if name not in writers:
                writers[name] = cv2.VideoWriter(os.path.join(a.out, name + ".mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (bgr.shape[1], bgr.shape[0]))
                cv2.imwrite(os.path.join(a.out, name + "_first.png"), bgr)
            writers[name].write(bgr); frames[name] = bgr
        if k < len(acts):
            obs, *_ = env.step(acts[k])
    for name, w in writers.items():
        w.release(); cv2.imwrite(os.path.join(a.out, name + "_last.png"), frames[name])
    env.close()
    print("저장: %s  (%d 프레임, agentview + eye_in_hand %dx%d)" % (a.out, len(acts) + 1, a.size, a.size))
    return 0


if __name__ == "__main__":
    sys.exit(main())
