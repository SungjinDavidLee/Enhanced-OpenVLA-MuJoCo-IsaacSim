#!/usr/bin/env python3
"""OFT 서버 오프라인 검증 — Isaac 없이 MuJoCo 관측을 그대로 넣어 시연 액션을 재현하는지 본다.

OFT(libero_spatial) 는 LIBERO 사람 시연으로 학습됐으므로, 서버의 전처리 · 정규화 · 청크 출력이 옳다면
MuJoCo 관측을 넣었을 때 시연 액션과 거의 같은 값이 나와야 한다.  크게 다르면 서버 쪽 문제다.

1 단계 (conda openvla_libero5090):  MuJoCo 에서 시연을 재생하며 관측 저장
    python3 oft_offline_check.py record --tasks 0 6 --demos 0 1 --out ~/oft_check_mj.npz
2 단계 (아무 환경, OFT 서버 :8001 이 떠 있어야 함):  서버에 보내 비교
    python3 oft_offline_check.py query --npz ~/oft_check_mj.npz

관측 규약은 OFT 공식 LIBERO 평가와 같다: agentview · eye_in_hand 256 을 180° 회전([::-1, ::-1]),
state = eef_pos 3 + quat2axisangle(eef_quat) 3 + gripper_qpos 2.  서버에는 crop_area 1.0 으로 보낸다.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import urllib.request

import numpy as np


def quat2axisangle(q):
    q = np.asarray(q, float).copy(); q[3] = np.clip(q[3], -1, 1)
    den = np.sqrt(1 - q[3] ** 2)
    return np.zeros(3) if den < 1e-9 else q[:3] * 2 * np.arccos(q[3]) / den


def record(a):
    import h5py
    sys.path.insert(0, "/home/user/LIBERO")
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    import os
    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    ds = os.path.join(get_libero_path("datasets"), "libero_spatial")
    out = {"agent": [], "wrist": [], "state": [], "action": [], "task": [], "demo": [], "step": [], "instruction": []}
    for t in a.tasks:
        task = suite.get_task(t)
        with h5py.File(os.path.join(ds, task.name + "_demo.hdf5"), "r") as f:
            for i in a.demos:
                g = f["data"]["demo_%d" % i]
                acts = np.asarray(g["actions"]); states = np.asarray(g["states"])
                env = OffScreenRenderEnv(bddl_file_name=suite.get_task_bddl_file_path(t), camera_heights=256, camera_widths=256,
                                         camera_names=["agentview", "robot0_eye_in_hand"])
                env.reset(); obs = env.set_init_state(states[0])
                for k in range(0, min(len(acts), a.max_steps), a.every):
                    obs = env.regenerate_obs_from_state(states[k])
                    out["agent"].append(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]))
                    out["wrist"].append(np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1]))
                    out["state"].append(np.concatenate([obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"]]))
                    out["action"].append(acts[k:k + 8] if k + 8 <= len(acts) else np.vstack([acts[k:], np.repeat(acts[-1:], k + 8 - len(acts), 0)]))
                    out["task"].append(t); out["demo"].append(i); out["step"].append(k); out["instruction"].append(task.language)
                env.close()
                print("task %d demo %d: %d 샘플" % (t, i, len(range(0, min(len(acts), a.max_steps), a.every))))
    np.savez_compressed(a.out, **{k: np.asarray(v) for k, v in out.items()})
    print("저장", a.out)


def png_b64(arr):
    from PIL import Image
    buf = io.BytesIO(); Image.fromarray(np.asarray(arr, dtype=np.uint8)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def query(a):
    d = np.load(a.npz, allow_pickle=True)
    n = len(d["state"]); rows = []
    for k in range(n):
        body = json.dumps({"instruction": str(d["instruction"][k]), "image_base64": png_b64(d["agent"][k]),
                           "wrist_base64": png_b64(d["wrist"][k]), "state": [float(v) for v in d["state"][k]],
                           "crop_area": 1.0}).encode()
        req = urllib.request.Request(a.server + "/predict_chunk", data=body, headers={"Content-Type": "application/json"}, method="POST")
        res = json.loads(urllib.request.urlopen(req, timeout=60).read().decode())
        pred = np.asarray(res["actions"], float)[:8, :7]
        gt = np.asarray(d["action"][k], float)[:8, :7].copy()
        p_g = np.where(pred[:, 6] > 0.5, -1.0, 1.0)          # 정책 규약(1=열림) → 환경 규약(−1 열림)
        rows.append((int(d["task"][k]), int(d["step"][k]), pred, gt, p_g))
    print("task  step   |오차| xyz (청크 평균)   크기비 |pred|/|gt| (xyz)   방향 코사인   집게 일치")
    allerr = []
    for t, s, pred, gt, pg in rows:
        e = np.abs(pred[:, :3] - gt[:, :3]).mean(); allerr.append(e)
        mp, mg = np.linalg.norm(pred[:, :3], axis=1).mean(), np.linalg.norm(gt[:, :3], axis=1).mean()
        cos = np.mean([np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-9) for x, y in zip(pred[:, :3], gt[:, :3])])
        print("  %d   %4d      %.3f                  %.2f                   %+.2f         %d/8" % (t, s, e, mp / (mg + 1e-9), cos, int((pg == np.sign(gt[:, 6])).sum())))
    print("\n전체 평균 |오차| %.3f   (시연 액션 크기 평균 %.3f)" % (np.mean(allerr), np.mean([np.abs(r[3][:, :3]).mean() for r in rows])))
    print("판정: 크기비 0.8–1.2 · 방향 코사인 0.8 이상 · 집게 대부분 일치면 서버는 정상.  크게 벗어나면 서버 전처리 · 정규화 확인")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["record", "query"])
    ap.add_argument("--tasks", type=int, nargs="+", default=[0, 6])
    ap.add_argument("--demos", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--every", type=int, default=10, help="몇 스텝마다 한 샘플")
    ap.add_argument("--max-steps", type=int, default=120)
    ap.add_argument("--out", default="/home/user/oft_check_mj.npz")
    ap.add_argument("--npz", default="/home/user/oft_check_mj.npz")
    ap.add_argument("--server", default="http://127.0.0.1:8001")
    a = ap.parse_args()
    record(a) if a.stage == "record" else query(a)
