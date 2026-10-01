#!/usr/bin/env python3
"""어댑터 이동 배율 검증 — 사람 시연의 실제 명령을 정책 없이 어댑터에 넣고, Isaac 손끝 궤적을 MuJoCo 기록과 대조한다.

정책 · 카메라와 무관하다.  「같은 명령에 Isaac 손이 MuJoCo 손과 같은 길을 가는가」만 본다.
브리지의 integrated_action 재생은 정책 평가와 같은 어댑터 경로(gain · IK · 물리 스텝)를 탄다.

1 단계 (Isaac 재생, 판당 1–2 분):
    python3 verify_adapter_scale.py run --tasks 0 6 --demos 0 1
      기본 설정 세 가지:  now = S10·gain1.0 (지금 평가)   cal = S3·gain0.65 (9/25 보정)   ss = S10·gain0.25 (정상 상태 계산값)
2 단계 (대조):
    python3 verify_adapter_scale.py report

출력 (설정별, 시연 4 개 평균)
  구간 이동비     Isaac 이동 ÷ MuJoCo 이동.  구간: 처음 8 스텝 / 처음 24 스텝 / 첫 닫힘까지     (1.00 이 맞음)
  축별 이동비     처음 24 스텝의 x · y · z 각각                                          (축마다 다르면 축별 보정 필요)
  위치 오차       첫 닫힘 순간 Isaac 손끝 − MuJoCo 손끝 (mm)
  회전 오차       첫 닫힘 순간 두 손끝 자세 차이 (°)
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess

import numpy as np

BR = "/home/user/isaac_libero/bridge"
ISAAC = "/home/user/isaacsim/python.sh"
SRC = "/home/user/isaac_libero/runs/replay_src_demo"
OUT = "/home/user/LIBERO/experiments/adapter_verify"
SETTINGS = {"now": (10, 1.0), "cal": (3, 0.65), "ss": (10, 0.25)}


def run(a):
    for name in a.settings:
        S, G = SETTINGS[name]
        for t in a.tasks:
            for i in a.demos:
                d = f"{OUT}/{name}/task_{t:02d}_d{i:03d}"
                if glob.glob(d + "/**/episode_result.json", recursive=True):
                    print("있음", d); continue
                os.makedirs(d, exist_ok=True)
                npz = f"{SRC}/task_{t:02d}/demo_episode_{i:03d}.npz"
                n = len(np.load(npz)["libero_actions"])
                cmd = [ISAAC, f"{BR}/run_openvla_isaac_bridge.py", "--env-script", f"{BR}/libero_spatial_env_isaac601.py",
                       "--config", f"{BR}/libero_spatial_environment.yaml",
                       "--pose-json-path", f"{BR}/poses_demo/task_{t:02d}/demo_episode_{i:03d}.json", "--libero-world-offset-m", "0", "0", "0",
                       "--task-id", str(t), "--episode-id", str(i), "--seed", "7", "--no-cv-display", "--stage-telemetry",
                       "--steps", str(min(n, 220)), "--action-replay", npz, "--replay-action-field", "libero_actions",
                       "--replay-reference", "integrated_action", "--align-initial-ee", "--libero-eef-quat-order", "xyzw",
                       "--replay-hold-on-ik-failure", "--sim-steps-per-action", str(S), "--convergence-gain", str(G),
                       "--ik-backtrack-factors", "1.0", "0.5", "--close-approach-settle-steps", "0", "--gripper-settle-steps", "1",
                       "--target-position-tolerance-m", "0.5", "--target-rotation-tolerance-deg", "90", "--capture-dir", d]
                with open(d + ".log", "w") as log:
                    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    try:
                        proc.wait(timeout=a.timeout_s)
                    except subprocess.TimeoutExpired:
                        # Isaac 이 멈추면 판 전체(Isaac 과 자식 프로세스)를 끊고 다음 판으로
                        import signal
                        os.killpg(proc.pid, signal.SIGKILL); proc.wait()
                        print("%-4s task %d demo %d  시간 초과 %d 초 → 끊고 다음으로" % (name, t, i, a.timeout_s))
                ok = glob.glob(d + "/**/episode_result.json", recursive=True)
                print("%-4s task %d demo %d  %s" % (name, t, i, "완료" if ok else "실패 (로그 확인)"))


def q_to_mat(q):
    x, y, z, w = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def ang(Ra, Rb):
    return float(np.degrees(np.arccos(np.clip((np.trace(Ra.T @ Rb) - 1) / 2, -1, 1))))


def report(a):
    try:
        cal = json.load(open(f"{BR}/oft_proprio_calib.json"))       # right_gripper → robosuite eef 프레임 (없으면 보정 없이)
        R_off = np.asarray(cal["R_off"]); p_local = np.asarray(cal["p_local_m"])
    except OSError:
        R_off = np.eye(3); p_local = np.zeros(3); print("(oft_proprio_calib.json 없음 — 프레임 보정 없이 비교, 회전오차는 참고만)")
    print("설정   이동비(0–8)  이동비(0–24)  이동비(닫힘까지)   축별 0–24 (x / y / z)        닫힘 순간 위치오차   회전오차   (n)")
    for name in SETTINGS:
        rows = []
        for d in sorted(glob.glob(f"{OUT}/{name}/task_*_d*")):
            if not os.path.isdir(d):
                continue
            fs = glob.glob(d + "/**/episode_result.json", recursive=True)
            if not fs:
                continue
            t = int(d.split("task_")[1][:2]); i = int(d.split("_d")[-1])
            mj = np.load(f"{SRC}/task_{t:02d}/demo_episode_{i:03d}.npz")
            mp, mq, ma = mj["eef_pos"], mj["eef_quat"], mj["libero_actions"]
            by = {}
            for s in json.load(open(fs[0]))["result"]["stage_trace"]:
                ps = s.get("policy_step")
                if s.get("observations") and isinstance(ps, int) and ps >= 0 and ps not in by:
                    by[ps] = s["observations"]
            ks = [k for k in sorted(by) if k < len(mp)]
            if len(ks) < 10:
                continue
            ip = np.array([np.asarray(by[k]["eef_position_m"]) + np.asarray(by[k]["eef_rotation"]).reshape(3, 3) @ p_local for k in ks])
            iR = [np.asarray(by[k]["eef_rotation"]).reshape(3, 3) @ R_off.T for k in ks]
            m = mp[ks]
            close = next((j for j, k in enumerate(ks) if ma[k][6] > 0), len(ks) - 1)
            def ratio(j1):
                j1 = min(j1, len(ks) - 1); dm = np.linalg.norm(m[j1] - m[0]); di = np.linalg.norm(ip[j1] - ip[0])
                return di / dm if dm > 1e-4 else np.nan
            j24 = min(24, len(ks) - 1)
            ax = [(ip[j24, c] - ip[0, c]) / (m[j24, c] - m[0, c]) if abs(m[j24, c] - m[0, c]) > 0.01 else np.nan for c in range(3)]
            rows.append((ratio(8), ratio(24), ratio(close), *ax, np.linalg.norm(ip[close] - m[close]) * 1000, ang(iR[close], q_to_mat(mq[ks[close]]))))
        if not rows:
            print("%-5s  (결과 없음)" % name); continue
        R = np.nanmean(np.array(rows, float), axis=0)
        print("%-5s    %.2f         %.2f          %.2f            %5.2f / %5.2f / %5.2f           %5.0f mm        %4.1f°    (%d)" % (name, *R, len(rows)))
    print("\n이동비 1.00 · 축별 1.00 근처 · 위치오차 수 mm · 회전오차 1–2° 이내면 그 설정은 MuJoCo 와 같은 길을 감.")
    print("0–8 구간만 1 보다 크면 출발 구간(과도 응답) 차이, 전 구간이 크면 배율 자체가 큼.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["run", "report"])
    ap.add_argument("--settings", nargs="+", default=list(SETTINGS), choices=list(SETTINGS))
    ap.add_argument("--tasks", type=int, nargs="+", default=[0, 6])
    ap.add_argument("--demos", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--timeout-s", type=int, default=480, help="판 하나의 최대 시간 (초). 넘으면 끊고 다음 판으로")
    a = ap.parse_args()
    run(a) if a.stage == "run" else report(a)
