#!/usr/bin/env python3
"""같은 로봇 상태에서 MuJoCo 화면 대 Isaac 화면 — OFT 예측이 어떻게 달라지는지 잰다.

1 단계 replay (openvla_libero5090, Isaac 재생 4 판 ≈ 12 분): 사람 시연을 OFT 평가 설정(LIBERO 규격 손목)으로 Isaac 에서 재생
    python3 isaac_vs_mujoco_probe.py replay --tasks 0 6 --demos 0 1
2 단계 query (OFT 서버 :8001 필요): oft_offline_check.py 의 MuJoCo 샘플마다, 손끝 위치가 가장 가까운 Isaac 스텝을 찾아
    두 화면의 예측을 비교
    python3 isaac_vs_mujoco_probe.py query --mj ~/oft_check_mj.npz

MuJoCo 샘플은 원본 HDF5 스텝 번호이고 Isaac 재생은 no-op 을 뺀 궤적이라 번호가 다르다. 그래서 손끝 위치 (+ 집게 상태) 가
가장 가까운 Isaac 스텝을 짝으로 쓴다.  Isaac 상태는 oft_proprio_calib.json 으로 robosuite 규약으로 바꾼다.
"""

from __future__ import annotations

import argparse
import base64
import glob
import io
import json
import os
import subprocess
import sys
import urllib.request

import numpy as np

BR = "/home/user/isaac_libero/bridge"
ISAAC = "/home/user/isaacsim/python.sh"
OUT = "/home/user/LIBERO/experiments/probe_isaac_oftcam"


def q_mat_to_xyzw(R):
    R = np.asarray(R, float).reshape(3, 3); tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(tr + 1) * 2; q = [(R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s, 0.25 * s]
    else:
        i = int(np.argmax(np.diag(R))); j, k = (i + 1) % 3, (i + 2) % 3
        s = np.sqrt(1 + R[i, i] - R[j, j] - R[k, k]) * 2; q = [0.0, 0.0, 0.0, (R[k, j] - R[j, k]) / s]
        q[i] = 0.25 * s; q[j] = (R[j, i] + R[i, j]) / s; q[k] = (R[k, i] + R[i, k]) / s
    q = np.asarray(q); return q / np.linalg.norm(q)


def quat2axisangle(q):
    q = np.asarray(q, float).copy(); q[3] = np.clip(q[3], -1, 1)
    den = np.sqrt(1 - q[3] ** 2)
    return np.zeros(3) if den < 1e-9 else q[:3] * 2 * np.arccos(q[3]) / den


def replay(a):
    for t in a.tasks:
        for i in a.demos:
            d = f"{OUT}/task_{t:02d}_d{i:03d}"
            if glob.glob(d + "/**/episode_result.json", recursive=True):
                print("있음", d); continue
            os.makedirs(d, exist_ok=True)
            cmd = [ISAAC, f"{BR}/run_openvla_isaac_bridge.py", "--env-script", f"{BR}/libero_spatial_env_isaac601.py",
                   "--config", f"{BR}/libero_spatial_environment_oft.yaml",
                   "--pose-json-path", f"{BR}/poses_demo/task_{t:02d}/demo_episode_{i:03d}.json", "--libero-world-offset-m", "0", "0", "0",
                   "--task-id", str(t), "--episode-id", str(i), "--seed", "7", "--no-cv-display", "--stage-telemetry", "--steps", "220",
                   "--action-replay", f"/home/user/isaac_libero/runs/replay_src_demo/task_{t:02d}/demo_episode_{i:03d}.npz",
                   "--replay-action-field", "libero_actions", "--replay-reference", "libero_eef_pose", "--align-initial-ee",
                   "--libero-eef-quat-order", "xyzw", "--replay-hold-on-ik-failure", "--sim-steps-per-action", "20",
                   "--ik-backtrack-factors", "1.0", "0.5", "--close-approach-settle-steps", "0", "--gripper-settle-steps", "1",
                   "--target-position-tolerance-m", "0.5", "--target-rotation-tolerance-deg", "90", "--gripper-speed-mps", "0.057",
                   "--capture-dir", d]
            with open(d + ".log", "w") as log:
                subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
            ok = glob.glob(d + "/**/episode_result.json", recursive=True)
            print("task %d demo %d 재생 %s" % (t, i, "완료" if ok else "실패 (로그 확인)"))


def frames(path):
    import cv2
    cap = cv2.VideoCapture(path); out = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        out.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    return out


def png_b64(arr):
    from PIL import Image
    buf = io.BytesIO(); Image.fromarray(np.asarray(arr, dtype=np.uint8)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def post(server, instr, agent, wrist, state, crop):
    body = {"instruction": instr, "image_base64": png_b64(agent), "wrist_base64": png_b64(wrist), "state": [float(v) for v in state]}
    if crop is not None:
        body["crop_area"] = crop
    req = urllib.request.Request(server + "/predict_chunk", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}, method="POST")
    return np.asarray(json.loads(urllib.request.urlopen(req, timeout=60).read().decode())["actions"], float)[:8, :7]


def query(a):
    mj = np.load(a.mj, allow_pickle=True)
    cal = json.load(open(f"{BR}/oft_proprio_calib.json"))
    R_off = np.asarray(cal["R_off"]); p_local = np.asarray(cal["p_local_m"]); q_ref = np.asarray(cal["q_ref_xyzw"])
    cache = {}
    rows = []
    for k in range(len(mj["state"])):
        t, i = int(mj["task"][k]), int(mj["demo"][k])
        key = (t, i)
        if key not in cache:
            d = f"{OUT}/task_{t:02d}_d{i:03d}"
            fs = glob.glob(d + "/**/episode_result.json", recursive=True)
            if not fs:
                cache[key] = None; continue
            tr_all = [s for s in json.load(open(fs[0]))["result"]["stage_trace"] if s.get("observations")]
            ag = frames(glob.glob(d + "/**/raw_rgb.mp4", recursive=True)[0]); wr = frames(glob.glob(d + "/**/wrist_rgb.mp4", recursive=True)[0])
            # 기록에는 판 시작 전후 관측이 더 있어 영상보다 길다.  기록의 policy_step 번호로 영상 프레임과 짝짓는다.
            by_step = {}
            for s in tr_all:
                ps = s.get("policy_step")
                if isinstance(ps, int) and 0 <= ps < min(len(ag), len(wr)) and ps not in by_step:
                    by_step[ps] = s
            if by_step:
                keys = sorted(by_step); tr = [by_step[k] for k in keys]; ag = [ag[k] for k in keys]; wr = [wr[k] for k in keys]
            else:
                n0 = min(len(tr_all), len(ag), len(wr)); tr = tr_all[:n0]; ag = ag[:n0]; wr = wr[:n0]
            n = len(tr)
            print("  task %d demo %d: 기록 %d · 영상 %d → 짝 %d (%s)" % (t, i, len(tr_all), len(frames(glob.glob(d + "/**/raw_rgb.mp4", recursive=True)[0])) if False else len(ag), n, "스텝 번호" if by_step else "순서"))
            states = []; prev = q_ref.copy()
            for s in tr[:n]:
                o = s["observations"]; R = np.asarray(o["eef_rotation"], float).reshape(3, 3); p = np.asarray(o["eef_position_m"], float)
                q = q_mat_to_xyzw(R @ R_off.T)
                if np.dot(q, prev) < 0:
                    q = -q
                prev = q
                g = o.get("gripper_joint_positions") or [o.get("gripper_finger_mean_m", 0.04)] * 2
                states.append(np.concatenate([p + R @ p_local, quat2axisangle(q), [g[0], -g[1]]]))
            cache[key] = (np.asarray(states), ag[:n], wr[:n])
        if cache[key] is None:
            continue
        st, ag, wr = cache[key]
        s_mj = mj["state"][k]
        dist = np.linalg.norm(st[:, :3] - s_mj[:3], axis=1) + 0.5 * np.abs(st[:, 6] - s_mj[6])
        j = int(np.argmin(dist))
        instr = str(mj["instruction"][k])
        p_mj = post(a.server, instr, mj["agent"][k], mj["wrist"][k], s_mj, 1.0)
        p_is = post(a.server, instr, ag[j], wr[j], st[j], None)
        gt = np.asarray(mj["action"][k], float)[:8, :3]
        m_mj = np.linalg.norm(p_mj[:, :3], axis=1).mean(); m_is = np.linalg.norm(p_is[:, :3], axis=1).mean()
        cos = np.mean([np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-9) for x, y in zip(p_is[:, :3], p_mj[:, :3])])
        u = gt.sum(0); u = u / (np.linalg.norm(u) + 1e-9)                   # 시연의 이동 방향
        prj_mj = float((p_mj[:, :3] @ u).sum()); prj_is = float((p_is[:, :3] @ u).sum())
        grip = int(((p_is[:, 6] > 0.5) == (p_mj[:, 6] > 0.5)).sum())
        rows.append((t, int(mj["step"][k]), dist[j] * 1000, m_is / (m_mj + 1e-9), cos, prj_is / (prj_mj + 1e-9) if abs(prj_mj) > 1e-3 else np.nan, grip))
    print("task  step   상태 짝 거리(mm)   크기비 Isaac/MuJoCo   방향 코사인   진행 방향 비   집게 일치")
    for r in rows:
        print("  %d   %4d      %6.1f              %.2f              %+.2f         %5.2f        %d/8" % r)
    R = np.array([r[3:6] for r in rows], float)
    print("\n중앙값: 크기비 %.2f   방향 코사인 %.2f   진행 방향 비 %.2f   (n=%d)" % (np.nanmedian(R[:, 0]), np.nanmedian(R[:, 1]), np.nanmedian(R[:, 2]), len(rows)))
    print("진행 방향 비 < 1 이면 Isaac 화면에서 같은 상태인데도 목표 쪽으로 덜 가라고 명령함 (= 못 미침의 원인)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["replay", "query"])
    ap.add_argument("--tasks", type=int, nargs="+", default=[0, 6])
    ap.add_argument("--demos", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--mj", default="/home/user/oft_check_mj.npz")
    ap.add_argument("--server", default="http://127.0.0.1:8001")
    a = ap.parse_args()
    replay(a) if a.stage == "replay" else query(a)
