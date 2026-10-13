#!/usr/bin/env python3
"""어댑터 이동 배율 측정 (v4) — 사람 시범의 동작 명령을 모델 없이 Isaac Sim 에 넣고, 손이 움직인 거리를 MuJoCo 기록과 비교한다.

v4 에서 더한 것 (10/12)
  - 과제 10개 (기본: 과제마다 시범 0번 하나)
  - 배율 여러 개: M(0.05초 · 0.65), 0.05초 · 1.0 / 1.3 / 1.6, 그리고 B(0.17초 · 1.0)
  - 그릇 있음 · 없음 — 「그릇 없음」은 시범의 물체 배치 파일에서 그릇(이름에 bowl)을 3 m 옆으로 옮긴 장면.
    들어 올릴 때 덜 움직이는 원인이 그릇 무게(85 g, MuJoCo 6 g)인지 팔 자체인지 가른다.
  - 여러 판을 동시에 (--workers, 기본 4)
  - 결과는 과제별 구조: ~/LIBERO/experiments/adapter_verify/v4/<설정>/<bowl|nobowl>/task_XX/d000/

비교하는 값 (Isaac ÷ MuJoCo, 1.00 이면 MuJoCo 와 같음)
  다가가는 동안    처음부터 첫 닫기 명령까지 손이 움직인 거리
  처음 24스텝      처음 24스텝 동안 움직인 거리
  들어 올리는 동안  첫 닫기 뒤 MuJoCo 손이 5 mm 넘게 오르기 시작한 스텝부터 20스텝 동안 올라간 높이
  들기 처음 3스텝   그 구간의 처음 3스텝 (작은 명령의 첫 반응)

    python3 verify_adapter_scale_v4.py inspect          # 그릇을 옮길 수 있는지 배치 파일 확인 (GPU 안 씀)
    python3 verify_adapter_scale_v4.py run              # Isaac Sim 재생 (판당 1–2분, 4개씩 동시)
    python3 verify_adapter_scale_v4.py report           # 표와 「1배가 되는 배율」
정책 · 카메라와 무관하다. 브리지의 integrated_action 재생은 정책 평가와 같은 어댑터 경로(배율 · 관절 계산 · 물리 스텝)를 탄다.
"""
from __future__ import annotations

import argparse, copy, glob, json, os, signal, subprocess
from concurrent.futures import ThreadPoolExecutor

import numpy as np

BR = "/home/user/isaac_libero/bridge"
ISAAC = "/home/user/isaacsim/python.sh"
SRC = "/home/user/isaac_libero/runs/replay_src_demo"
OUT = "/home/user/LIBERO/experiments/adapter_verify/v4"
SETTINGS = {"M": (3, 0.65), "M_g1.0": (3, 1.0), "M_g1.3": (3, 1.3), "M_g1.6": (3, 1.6), "B": (10, 1.0)}
M_FAMILY = ["M", "M_g1.0", "M_g1.3", "M_g1.6"]


# ---------------------------------------------------------------- 그릇 없는 장면
def _shift(v, dx):
    if isinstance(v, dict):
        for pk in ("position_xyz", "position", "pos", "translation", "xyz", "p", "position_m"):
            if pk in v and isinstance(v[pk], list) and len(v[pk]) >= 3 and all(isinstance(x, (int, float)) for x in v[pk][:3]):
                v[pk][0] += dx; return 1
        for pk in ("pose", "world_pose", "transform", "init_pose"):
            if pk in v:
                return _shift(v[pk], dx)
        return 0
    if isinstance(v, list) and len(v) >= 3 and all(isinstance(x, (int, float)) for x in v[:3]):
        v[0] += dx; return 1
    return 0


def move_bowls(obj, dx=3.0, found=None):
    found = [] if found is None else found
    if isinstance(obj, dict):
        for k, v in obj.items():
            if "bowl" in str(k).lower():
                if _shift(v, dx):
                    found.append(str(k))
            else:
                move_bowls(v, dx, found)
    elif isinstance(obj, list):
        for v in obj:
            tag = next((str(v.get(f)) for f in ("name", "prim", "prim_path", "path", "id", "object") if isinstance(v, dict) and "bowl" in str(v.get(f, "")).lower()), None)
            if tag and _shift(v, dx):
                found.append(tag)
            else:
                move_bowls(v, dx, found)
    return found


def pose_path(t, i, bowl):
    src = f"{BR}/poses_demo/task_{t:02d}/demo_episode_{i:03d}.json"
    if bowl:
        return src, []
    dst = f"{OUT}/_poses_nobowl/task_{t:02d}/demo_episode_{i:03d}.json"
    data = copy.deepcopy(json.load(open(src))); found = move_bowls(data)
    if found:
        os.makedirs(os.path.dirname(dst), exist_ok=True); json.dump(data, open(dst, "w"), indent=1)
    return dst, found


def inspect(a):
    t, i = a.tasks[0], a.demos[0]
    src = f"{BR}/poses_demo/task_{t:02d}/demo_episode_{i:03d}.json"
    data = json.load(open(src))
    print("배치 파일:", src)
    print("맨 위 항목:", list(data)[:20] if isinstance(data, dict) else f"목록 {len(data)}개")
    _, found = pose_path(t, i, bowl=False)
    print("옮길 그릇 항목:", found or "찾지 못함 — 「그릇 없음」 재생은 건너뜀 (배치 파일 구조를 알려 주세요)")


# ---------------------------------------------------------------- 재생
def one(job):
    name, bowl, t, i, timeout = job
    S, G = SETTINGS[name]
    d = f"{OUT}/{name}/{'bowl' if bowl else 'nobowl'}/task_{t:02d}/d{i:03d}"
    if glob.glob(d + "/**/episode_result.json", recursive=True):
        return f"있음   {name:7s} {'그릇' if bowl else '없음'} task {t} demo {i}"
    pose, found = pose_path(t, i, bowl)
    if not bowl and not found:
        return f"건너뜀 {name:7s} 없음 task {t} demo {i} (그릇 항목을 못 찾음)"
    os.makedirs(d, exist_ok=True)
    npz = f"{SRC}/task_{t:02d}/demo_episode_{i:03d}.npz"
    n = len(np.load(npz)["libero_actions"])
    cmd = [ISAAC, f"{BR}/run_openvla_isaac_bridge.py", "--env-script", f"{BR}/libero_spatial_env_isaac601.py",
           "--config", f"{BR}/libero_spatial_environment.yaml", "--pose-json-path", pose, "--libero-world-offset-m", "0", "0", "0",
           "--task-id", str(t), "--episode-id", str(i), "--seed", "7", "--no-cv-display", "--stage-telemetry",
           "--steps", str(min(n, 220)), "--action-replay", npz, "--replay-action-field", "libero_actions",
           "--replay-reference", "integrated_action", "--align-initial-ee", "--libero-eef-quat-order", "xyzw",
           "--replay-hold-on-ik-failure", "--sim-steps-per-action", str(S), "--convergence-gain", str(G),
           "--ik-backtrack-factors", "1.0", "0.5", "--close-approach-settle-steps", "0", "--gripper-settle-steps", "1",
           "--target-position-tolerance-m", "0.5", "--target-rotation-tolerance-deg", "90", "--capture-dir", d]
    with open(d + ".log", "w") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL); proc.wait()
            return f"시간 초과 {name:7s} {'그릇' if bowl else '없음'} task {t} demo {i}"
    ok = glob.glob(d + "/**/episode_result.json", recursive=True)
    return f"{'완료' if ok else '실패'}   {name:7s} {'그릇' if bowl else '없음'} task {t} demo {i}"


def run(a):
    jobs = [(nm, b, t, i, a.timeout_s) for b in (True, False) for nm in a.settings for t in a.tasks for i in a.demos]
    print(f"재생 {len(jobs)}판, {a.workers}개씩 동시")
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for msg in ex.map(one, jobs):
            print(msg, flush=True)


# ---------------------------------------------------------------- 비교
def load(d, p_local):
    fs = glob.glob(d + "/**/episode_result.json", recursive=True)
    if not fs:
        return None
    t = int(d.split("task_")[1][:2]); i = int(d.rsplit("/d", 1)[-1])
    mj = np.load(f"{SRC}/task_{t:02d}/demo_episode_{i:03d}.npz"); mp, ma = mj["eef_pos"], mj["libero_actions"]
    by = {}
    for s in json.load(open(fs[0]))["result"]["stage_trace"]:
        ps = s.get("policy_step")
        if s.get("observations") and isinstance(ps, int) and ps >= 0 and ps not in by:
            by[ps] = s["observations"]
    ks = [k for k in sorted(by) if k < len(mp)]
    if len(ks) < 10:
        return None
    ip = np.array([np.asarray(by[k]["eef_position_m"]) + np.asarray(by[k]["eef_rotation"]).reshape(3, 3) @ p_local for k in ks])
    return ks, ip, mp[ks], ma


def measure(r, window):
    ks, ip, m, ma = r
    path = lambda x, j0, j1: float(np.sum(np.linalg.norm(np.diff(x[j0:j1 + 1], axis=0), axis=1)))
    jc = next((j for j, k in enumerate(ks) if ma[k][6] > 0), None)
    out = {}
    if jc and jc > 2:
        pm = path(m, 0, jc); out["approach"] = path(ip, 0, jc) / pm if pm > 1e-3 else np.nan
    j24 = min(24, len(ks) - 1); pm = path(m, 0, j24); out["first24"] = path(ip, 0, j24) / pm if pm > 1e-3 else np.nan
    if jc is not None:
        j1 = next((j for j in range(jc, len(ks)) if m[j, 2] - m[jc, 2] > 0.005), None)
        if j1 and j1 >= 1:
            j1 -= 1; j2 = min(j1 + window, len(ks) - 1); j3 = min(j1 + 3, len(ks) - 1)
            dm, dm3 = m[j2, 2] - m[j1, 2], m[j3, 2] - m[j1, 2]
            out["lift"] = (ip[j2, 2] - ip[j1, 2]) / dm if dm > 0.003 else np.nan
            out["lift3"] = (ip[j3, 2] - ip[j1, 2]) / dm3 if dm3 > 0.001 else np.nan
    return out


def report(a):
    try:
        p_local = np.asarray(json.load(open(f"{BR}/oft_proprio_calib.json"))["p_local_m"])
    except OSError:
        p_local = np.zeros(3)
    cols = ["approach", "first24", "lift", "lift3"]
    table = {}
    print("Isaac ÷ MuJoCo (1.00 이면 같음), 과제별 평균\n")
    print("%-8s %-6s %10s %10s %10s %12s  %s" % ("설정", "그릇", "다가가는 동안", "처음 24스텝", "들어 올리는 동안", "들기 처음 3스텝", "(판 수)"))
    for nm in SETTINGS:
        for b in ("bowl", "nobowl"):
            rows = [measure(r, a.window) for d in sorted(glob.glob(f"{OUT}/{nm}/{b}/task_*/d*")) if os.path.isdir(d)
                    for r in [load(d, p_local)] if r]
            if not rows:
                continue
            R = {c: float(np.nanmean([x.get(c, np.nan) for x in rows])) for c in cols}
            table[(nm, b)] = R
            print("%-8s %-6s %10.2f %10.2f %12.2f %14.2f  (%d)" % (nm, "있음" if b == "bowl" else "없음", R["approach"], R["first24"], R["lift"], R["lift3"], len(rows)))
    print("\n1배가 되는 배율 (0.05초 설정들 사이를 이어서 계산)")
    for b in ("bowl", "nobowl"):
        pts = [(SETTINGS[nm][1], table[(nm, b)]) for nm in M_FAMILY if (nm, b) in table]
        if len(pts) < 2:
            continue
        for c, lab in (("approach", "다가가는 동안"), ("lift", "들어 올리는 동안")):
            g = np.array([p[0] for p in pts]); v = np.array([p[1][c] for p in pts]); ok = ~np.isnan(v)
            if ok.sum() >= 2:
                fit = np.polyfit(v[ok], g[ok], 1); need = float(np.polyval(fit, 1.0))
                print("  그릇 %s · %-8s → 배율 약 %.2f  (측정 범위 %.2f–%.2f배)" % ("있음" if b == "bowl" else "없음", lab, need, np.nanmin(v), np.nanmax(v)))
    if ("M", "bowl") in table and ("M", "nobowl") in table:
        lb, ln = table[("M", "bowl")]["lift"], table[("M", "nobowl")]["lift"]
        print("\nM 에서 들어 올리는 동안: 그릇 있음 %.2f배, 없음 %.2f배 → %s" % (lb, ln,
              "그릇 무게 탓이 큼" if ln - lb > 0.2 else "그릇이 없어도 덜 올라감 → 팔(제어) 자체의 차이"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["inspect", "run", "report"])
    ap.add_argument("--settings", nargs="+", default=list(SETTINGS), choices=list(SETTINGS))
    ap.add_argument("--tasks", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--demos", type=int, nargs="+", default=[0])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--window", type=int, default=20)
    ap.add_argument("--timeout-s", type=int, default=600)
    a = ap.parse_args()
    {"inspect": inspect, "run": run, "report": report}[a.stage](a)
