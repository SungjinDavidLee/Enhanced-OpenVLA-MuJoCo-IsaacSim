#!/usr/bin/env python3
"""정책 없는 파지 시험 — MuJoCo 와 Isaac 에서 같은 동작으로 그릇 테두리를 깊이별로 쥐고 들어 본다.

동작 (두 시뮬레이터 공통)
  1 시작 자세 → 테두리 위 8 cm 로 이동 (집게 열림)
  2 정해진 깊이까지 내림 (손끝 − 테두리 윗면 = --depths mm)
  3 집게 닫고 그 자리에서 대기 (20 스텝)
  4 8 cm 들어 올림 (20 스텝, 스텝당 4 mm) → 20 스텝 대기
  판정: 마지막에 그릇 바닥이 60 mm 이상 올라와 있으면 「들림」
쥐는 위치: 그릇 중심에서 ±y 방향 테두리 (기본 손 방향에서 손가락이 y 로 닫히므로 벽을 직각으로 쥠).
          두 쪽 중 다른 물체에서 더 먼 쪽.  배치는 평가와 같은 s00.

    conda activate openvla_libero5090
    python3 grasp_bench.py mujoco --tasks 0 6 --depths -10 -20 -30          # MuJoCo (몇 분)
    python3 grasp_bench.py isaac  --tasks 0 6 --depths -10 -20 -30          # Isaac 재생 (판당 1–2 분)
    python3 grasp_bench.py report
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys

import numpy as np

BR = "/home/user/isaac_libero/bridge"
ISAAC = "/home/user/isaacsim/python.sh"
OUT = "/home/user/LIBERO/experiments/grasp_bench"
POSE = BR + "/poses_all/libero_spatial_task_poses_s00.json"
BOWL = "akita_black_bowl_1_main"
P_LOCAL = np.zeros(3)
BOWL_H = 0.0527          # 그릇 원점(바닥) → 테두리 윗면
RIM_R = 0.053            # 쥐는 점: 그릇 중심에서 수평 53 mm (벽 위치)
LIFT = 0.08


def plan(center_xy, rim_top_z, side_sign, depth_m, start_pos):
    """스텝별 손끝 목표 (T×3) 와 집게 명령 (T,)."""
    grasp = np.array([center_xy[0], center_xy[1] + side_sign * RIM_R, rim_top_z + depth_m])
    above = grasp + np.array([0, 0, 0.08 - depth_m])
    seg = lambda a, b, n: [a + (b - a) * (k + 1) / n for k in range(n)]
    pts = seg(np.asarray(start_pos, float), above, 40) + seg(above, grasp, 20) + [grasp] * 5
    grip = [-1.0] * len(pts)
    pts += [grasp] * 20; grip += [1.0] * 20
    top = grasp + np.array([0, 0, LIFT])
    pts += seg(grasp, top, 20) + [top] * 20; grip += [1.0] * 40
    return np.array(pts), np.array(grip)


MUST_REACH = {39, 59, 64}     # 테두리 위 도착 · 쥐는 깊이 도착 · 닫기 직전 — MuJoCo 는 여기서 오차 2 mm 까지 기다림


def isaac_bowl_geom(task):
    """평가 로그에 찍힌 「실제로 놓인」 그릇 AABB (배치 JSON 의 z 는 받침 정렬 전 값이라 쓰지 않는다)."""
    import re
    logs = sorted(glob.glob("/home/user/LIBERO/experiments/zs_s00_S10/task_%02d_e000.log" % task)) + sorted(glob.glob("/home/user/LIBERO/experiments/*/task_%02d_e000.log" % task))
    pat = re.compile(r"\[Camera Geometry\] target_bowl \|.*?aabb_center=\[([^\]]+)\].*?aabb_max=\[([^\]]+)\]")
    for lg in logs:
        try:
            for line in open(lg, errors="ignore"):
                m = pat.search(line)
                if m:
                    c = np.array([float(v) for v in m.group(1).split(",")]); mx = np.array([float(v) for v in m.group(2).split(",")])
                    return c[:2], mx[2], lg
        except OSError:
            continue
    return None, None, None


def pick_side(center_xy, others_xy):
    best, far = 1, -1.0
    for s in (1, -1):
        p = np.array([center_xy[0], center_xy[1] + s * RIM_R])
        d = min([np.linalg.norm(p - np.asarray(o)) for o in others_xy] or [9.0])
        if d > far:
            best, far = s, d
    return best


def layout(task):
    poses = json.load(open(POSE))["tasks"][str(task)]["poses"]
    b = np.array(poses[BOWL]["position_xyz"])
    others = [np.array(v["position_xyz"][:2]) for k, v in poses.items() if k != BOWL and "cabinet" not in k]
    return b, others


# ---------------------------------------------------------------- MuJoCo
def run_mujoco(a):
    sys.path.insert(0, "/home/user/LIBERO")
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv
    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    res = []
    for t in a.tasks:
        for dmm in a.depths:
            env = OffScreenRenderEnv(bddl_file_name=suite.get_task_bddl_file_path(t), camera_heights=32, camera_widths=32)
            env.reset(); obs = env.set_init_state(suite.get_task_init_states(t)[0])
            for _ in range(10):
                obs, *_ = env.step([0, 0, 0, 0, 0, 0, -1])
            m, d = env.env.sim.model, env.env.sim.data; bid = m.body_name2id(BOWL)
            vis = [g for g in range(m.ngeom) if m.geom_bodyid[g] == bid and m.geom_group[g] == 1]
            P = np.vstack([d.geom_xpos[v] + m.mesh_vert[m.mesh_vertadr[m.geom_dataid[v]]:m.mesh_vertadr[m.geom_dataid[v]] + m.mesh_vertnum[m.geom_dataid[v]]] @ d.geom_xmat[v].reshape(3, 3).T for v in vis])
            center = (P.min(0) + P.max(0))[:2] / 2; rim_top = P[:, 2].max(); z0 = P[:, 2].min()
            others = [np.array(d.body_xpos[i][:2]) for i in range(m.nbody) if (m.body_id2name(i) or "").endswith("_main") and m.body_id2name(i) != BOWL and "cabinet" not in m.body_id2name(i)]
            side = pick_side(center, others)
            pts, grip = plan(center, rim_top, side, dmm / 1000.0, obs["robot0_eef_pos"])
            bz = []; at_close = None; kc = int(np.argmax(grip > 0))
            for idx, (p, g) in enumerate(zip(pts, grip)):
                if idx == kc and at_close is None:
                    at_close = (np.linalg.norm(p - obs["robot0_eef_pos"]) * 1000, (obs["robot0_eef_pos"][2] - rim_top) * 1000, np.round(obs["robot0_eef_pos"], 3))
                for n_step in range(150 if idx in MUST_REACH else 3):   # 목표마다 최소 1 스텝은 반드시 진행 (집게가 닫힐 시간 보장)
                    err = p - obs["robot0_eef_pos"]
                    act = np.concatenate([np.clip(err / 0.05 * 2.0, -1, 1), [0, 0, 0], [g]])
                    obs, *_ = env.step(act)
                    if np.linalg.norm(p - obs["robot0_eef_pos"]) < (0.002 if idx in MUST_REACH else 0.003):
                        break
                bz.append(d.body_xpos[bid][2]); tilt_mj = float(np.degrees(np.arccos(np.clip(d.body_xmat[bid].reshape(3, 3)[2, 2], -1, 1))))
            env.close()
            rise = (bz[-1] - bz[0]) * 1000
            gq = obs["robot0_gripper_qpos"]
            res.append({"sim": "mujoco", "task": t, "depth_mm": dmm, "side": side, "rise_mm": rise, "max_rise_mm": (max(bz) - bz[0]) * 1000,
                        "close_err_mm": float(at_close[0]), "close_depth_mm": float(at_close[1]), "finger_mm": float(np.mean(np.abs(gq)) * 1000),
                        "tilt_deg": tilt_mj})
            print("MuJoCo task %d  깊이 %+d mm  쪽 %+d  → 그릇 상승 %5.1f mm (%s)   닫기 직전 손끝 오차 %.0f mm · 실제 깊이 %+.0f mm · 손 %s   끝 손가락 %.1f mm   끝 기울기 %.0f°" % (
                t, dmm, side, rise, "들림" if rise >= 60 else "안 들림", at_close[0], at_close[1], at_close[2], np.mean(np.abs(gq)) * 1000, tilt_mj))
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(OUT + "/mujoco.json", "w"), indent=1)


# ---------------------------------------------------------------- Isaac
def run_isaac(a):
    os.makedirs(OUT, exist_ok=True)
    ref = np.load(sorted(glob.glob("/home/user/isaac_libero/runs/replay_src_demo/task_00/demo_episode_000.npz"))[0])
    start_pos, start_quat = ref["eef_pos"][0], ref["eef_quat"][0]
    for t in a.tasks:
        b, others = layout(t)
        c_xy, top_z, src = isaac_bowl_geom(t)
        if top_z is None:
            print("task %d: 평가 로그에서 그릇 위치를 못 찾음 — 이 태스크 건너뜀 (먼저 s00 평가를 한 판 돌려 두세요)" % t); continue
        print("task %d: 그릇 중심 %s · 테두리 윗면 %.4f m  (%s)" % (t, np.round(c_xy, 4), top_z, src.split("experiments/")[-1]))
        side = pick_side(c_xy, others)
        for dmm in a.depths:
            tag = "%s_t%d_d%+03d" % (a.name, t, dmm); d = f"{OUT}/{tag}"
            if glob.glob(d + "/**/episode_result.json", recursive=True):
                print("있음", tag); continue
            os.makedirs(d, exist_ok=True)
            pts, grip = plan(c_xy, top_z, side, dmm / 1000.0, start_pos)
            acts = np.zeros((len(pts), 7), np.float32); acts[:, 6] = grip
            npz = d + "/plan.npz"
            np.savez(npz, eef_pos=pts, eef_quat=np.repeat(start_quat[None], len(pts), 0), libero_actions=acts)
            cmd = [ISAAC, f"{BR}/run_openvla_isaac_bridge.py", "--env-script", f"{BR}/libero_spatial_env_isaac601.py",
                   "--config", a.config, "--pose-json-path", POSE, "--libero-world-offset-m", "0", "0", "0",
                   "--task-id", str(t), "--episode-id", "0", "--seed", "7", "--no-cv-display", "--stage-telemetry",
                   "--steps", str(len(pts)), "--action-replay", npz, "--replay-action-field", "libero_actions",
                   "--replay-reference", "libero_eef_pose", "--align-initial-ee", "--libero-eef-quat-order", "xyzw",
                   "--replay-hold-on-ik-failure", "--sim-steps-per-action", "10", "--ik-backtrack-factors", "1.0", "0.5",
                   "--close-approach-settle-steps", "0", "--gripper-settle-steps", "1",
                   "--target-position-tolerance-m", "0.5", "--target-rotation-tolerance-deg", "90", "--capture-dir", d] + a.extra
            with open(d + ".log", "w") as log:
                p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    p.wait(timeout=480)
                except subprocess.TimeoutExpired:
                    import signal
                    os.killpg(p.pid, signal.SIGKILL); p.wait(); print("시간 초과", tag)
            print("Isaac %-22s 쪽 %+d  %s" % (tag, side, "완료" if glob.glob(d + "/**/episode_result.json", recursive=True) else "실패 (로그 확인)"))


def report(_a):
    global P_LOCAL
    try:
        P_LOCAL = np.asarray(json.load(open(BR + "/oft_proprio_calib.json"))["p_local_m"])
    except OSError:
        P_LOCAL = np.zeros(3)
    rows = {}; diag = []
    if os.path.exists(OUT + "/mujoco.json"):
        for r in json.load(open(OUT + "/mujoco.json")):
            rows[("MuJoCo", r["task"], r["depth_mm"])] = (r["rise_mm"], r["max_rise_mm"], None)
    for d in sorted(glob.glob(OUT + "/*_t*_d*")):
        if not os.path.isdir(d):
            continue
        fs = glob.glob(d + "/**/episode_result.json", recursive=True)
        if not fs:
            continue
        name, t, dm = os.path.basename(d).rsplit("_t", 1)[0], int(d.split("_t")[-1].split("_d")[0]), int(d.split("_d")[-1])
        res = json.load(open(fs[0]))["result"]
        tr = [s["observations"] for s in res["stage_trace"] if s.get("observations")]
        by = {s["policy_step"]: s["observations"] for s in res["stage_trace"] if s.get("observations") and isinstance(s.get("policy_step"), int)}
        bz = [o["target_bowl_aabb_min_m"][2] for o in tr]; w = [o.get("gripper_finger_mean_m") or 0.04 for o in tr]
        close_w = np.median([x for x in w[-45:-25]]) * 1000 if len(w) > 45 else np.nan
        rows[("Isaac:" + name, t, dm)] = ((bz[-1] - bz[0]) * 1000, (max(bz) - bz[0]) * 1000, close_w)
        plan = np.load(d + "/plan.npz"); kc = int(np.argmax(plan["libero_actions"][:, 6] > 0)) - 1
        o = by.get(kc) or tr[min(kc, len(tr) - 1)]
        e = np.asarray(o["eef_position_m"]) + np.asarray(o["eef_rotation"]).reshape(3, 3) @ P_LOCAL
        h0 = (tr[0]["target_bowl_aabb_max_m"][2] - tr[0]["target_bowl_aabb_min_m"][2]) * 1000
        h1 = (tr[-1]["target_bowl_aabb_max_m"][2] - tr[-1]["target_bowl_aabb_min_m"][2]) * 1000
        top_rise = (tr[-1]["target_bowl_aabb_max_m"][2] - tr[0]["target_bowl_aabb_max_m"][2]) * 1000
        diag.append("   Isaac:%s T%d %+d mm  닫기 직전 손끝 오차 %.0f mm  실제 깊이 %+.0f mm  IK 실패 %s   끝 그릇 높이(AABB) %.0f mm (처음 %.0f, 크면 기울어짐)  그릇 윗면 상승 %.0f mm" % (
            name, t, dm, np.linalg.norm(e - plan["eef_pos"][kc]) * 1000, (e[2] - o["target_bowl_aabb_max_m"][2]) * 1000, res.get("ik_failures"), h1, h0, top_rise))
    print("시뮬레이터 · 설정          task  깊이    마지막 그릇 상승   최대 상승   쥔 폭      판정")
    for (sim, t, dm), (rise, mx, cw) in sorted(rows.items(), key=lambda kv: (kv[0][1], kv[0][2], kv[0][0])):
        print("%-24s  T%d   %+4d mm    %6.1f mm        %6.1f mm   %s   %s" % (sim, t, dm, rise, mx, ("%4.1f mm" % cw) if cw else "   -   ", "들림" if rise >= 60 else "안 들림"))
    if diag:
        print("\n닫기 직전 손 위치 점검 (오차가 10 mm 를 넘으면 손이 테두리에 못 간 것 — 그 줄은 파지 시험으로 무효)")
        print("\n".join(diag))
    if os.path.exists(OUT + "/mujoco.json"):
        for r in json.load(open(OUT + "/mujoco.json")):
            if "close_err_mm" in r:
                print("   MuJoCo T%d %+d mm  닫기 직전 손끝 오차 %.0f mm  실제 깊이 %+.0f mm  끝 손가락 %.1f mm  끝 기울기 %.0f°" % (r["task"], r["depth_mm"], r["close_err_mm"], r["close_depth_mm"], r["finger_mm"], r.get("tilt_deg", float("nan"))))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["mujoco", "isaac", "report"])
    ap.add_argument("--tasks", type=int, nargs="+", default=[0, 6])
    ap.add_argument("--depths", type=int, nargs="+", default=[-10, -20, -30])
    ap.add_argument("--name", default="base2", help="Isaac 설정 이름 (물리 설정을 바꿔 시험할 때 구분용)")
    ap.add_argument("--config", default=BR + "/libero_spatial_environment.yaml")
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[], help="브리지에 더 넘길 인자 (맨 뒤에)")
    a = ap.parse_args()
    {"mujoco": run_mujoco, "isaac": run_isaac, "report": report}[a.stage](a)
