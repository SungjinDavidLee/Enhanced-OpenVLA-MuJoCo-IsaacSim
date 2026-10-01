#!/usr/bin/env python3
"""태스크마다 「서로 다른 궤적」 의 재생 성공이 N 개(기본 50) 될 때까지 자동으로 재생한다.

  1 순위  시드 7 로 각 소스 궤적 1 회
  2 순위  실패한 궤적만 시드 8 로
  3 순위  그래도 부족하면 시드 9 로
  4 순위  여전히 부족하면 이미 성공한 궤적을 다른 시드로 (다른 화면) — --allow-duplicates 일 때만
N 개가 되면 그 태스크는 멈춘다.  결과가 있는 판은 건너뛰므로 중단 후 다시 실행해도 된다.

    nohup python3 collect_success_per_task.py --tasks 0 1 2 3 4 --out ~/LIBERO/experiments/replay_all > ~/collect_w1.log 2>&1 &
    nohup python3 collect_success_per_task.py --tasks 5 6 7 8 9 --out ~/LIBERO/experiments/replay_all > ~/collect_w2.log 2>&1 &

결과 폴더:  <out>/seed<S>/task_XX_eYYY/    (시드별로 분리)
진행 확인:  python3 collect_success_per_task.py --tasks 0 1 2 3 4 5 6 7 8 9 --out ... --status
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time

BR = "/home/user/isaac_libero/bridge"
SRC = "/home/user/isaac_libero/runs/replay_src"     # --src 로 바꿀 수 있음 (예: replay_src_demo)
POSE_ROOT = None                                   # --pose-root: 시연별 배치 JSON 루트 (없으면 poses_all/sNN 규약)
ISAAC = "/home/user/isaacsim/python.sh"


def source_file(task: int, ep: int) -> str | None:
    fs = glob.glob(f"{SRC}/task_{task:02d}/*episode_{ep:03d}.npz")
    return fs[0] if fs else None


def sources(task: int) -> list[int]:
    eps = []
    for f in glob.glob(f"{SRC}/task_{task:02d}/*episode_*.npz"):
        eps.append(int(os.path.basename(f).split("_")[-1].split(".")[0]))
    return sorted(set(eps))


def result_of(d: str):
    fs = glob.glob(d + "/**/episode_result.json", recursive=True)
    if not fs:
        return None
    try:
        return bool(json.load(open(fs[0])).get("result", {}).get("task_success"))
    except Exception:
        return False


Z_OFFSET_M = -0.006   # 재생 목표 z 오프셋.  MuJoCo 가 테두리를 얕게 문 궤적을 Isaac 에서 살리는 값 (task 0 실패 14 판 중 6 판 회복)


def run_episode(task: int, ep: int, seed: int, out_root: str, steps: int) -> bool | None:
    d = f"{out_root}/seed{seed}/task_{task:02d}_e{ep:03d}"
    r = result_of(d)
    if r is not None:
        return r
    os.makedirs(d, exist_ok=True)
    pose = f"{BR}/poses_all/libero_spatial_task_poses_s{ep:02d}.json"
    cfg = f"{BR}/libero_spatial_environment.yaml"
    if task == 5 and os.path.isdir(f"{BR}/poses_t5"):
        pose = f"{BR}/poses_t5/libero_spatial_task_poses_s{ep:02d}.json"
        cfg = f"{BR}/libero_spatial_environment_t5.yaml"
    if POSE_ROOT:
        # 시연별 배치: <pose_root>/task_XX/*episode_YYY.json,  task 5 는 <pose_root>_t5
        root5 = POSE_ROOT.rstrip("/") + "_t5"
        cand = glob.glob(f"{root5 if task == 5 and os.path.isdir(root5) else POSE_ROOT}/task_{task:02d}/*episode_{ep:03d}.json")
        if not cand:
            print(f"  task {task} e{ep:03d}: 배치 JSON 없음 ({POSE_ROOT})", flush=True); return None
        pose = cand[0]
        cfg = f"{BR}/libero_spatial_environment_t5.yaml" if (task == 5 and os.path.isdir(root5)) else f"{BR}/libero_spatial_environment.yaml"
    cmd = [ISAAC, f"{BR}/run_openvla_isaac_bridge.py",
           "--env-script", f"{BR}/libero_spatial_env_isaac601.py", "--config", cfg,
           "--pose-json-path", pose, "--libero-world-offset-m", "0", "0", str(Z_OFFSET_M),
           "--task-id", str(task), "--episode-id", str(ep), "--seed", str(seed),
           "--no-cv-display", "--stage-telemetry", "--steps", str(steps),
           "--action-replay", source_file(task, ep),
           "--replay-action-field", "libero_actions", "--replay-reference", "libero_eef_pose", "--align-initial-ee",
           "--libero-eef-quat-order", "xyzw", "--replay-hold-on-ik-failure",
           "--sim-steps-per-action", "20", "--ik-backtrack-factors", "1.0", "0.5",
           "--close-approach-settle-steps", "0", "--gripper-settle-steps", "1",
           "--target-position-tolerance-m", "0.5", "--target-rotation-tolerance-deg", "90",
           "--gripper-speed-mps", "0.057", "--capture-dir", d]
    json.dump({"seed": seed, "z_offset_m": Z_OFFSET_M, "steps": steps}, open(d + "/replay_args.json", "w"))
    with open(d + ".log", "w") as log:
        subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    return result_of(d)


def tally(task: int, out_root: str, seeds):
    """궤적별로 어느 시드에서 성공했는지."""
    ok_by_ep: dict[int, list[int]] = {}
    tried: dict[int, list[int]] = {}
    for s in seeds:
        for d in glob.glob(f"{out_root}/seed{s}/task_{task:02d}_e*"):
            if not os.path.isdir(d):
                continue
            ep = int(d.split("_e")[-1]); r = result_of(d)
            if r is None:
                continue
            tried.setdefault(ep, []).append(s)
            if r:
                ok_by_ep.setdefault(ep, []).append(s)
    return ok_by_ep, tried


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--out", default="/home/user/LIBERO/experiments/replay_all")
    ap.add_argument("--target", type=int, default=50, help="태스크당 목표 (서로 다른 궤적의 성공 수)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[7, 8, 9], help="순서대로 시도할 시드")
    ap.add_argument("--allow-duplicates", action="store_true", help="4 순위: 이미 성공한 궤적을 다른 시드로 더 찍어 채움")
    ap.add_argument("--steps", type=int, default=220)
    ap.add_argument("--src", default=None, help="재생 소스 루트 (기본 replay_src 롤아웃; 사람 시연은 replay_src_demo)")
    ap.add_argument("--pose-root", default=None, help="시연별 배치 JSON 루트 (사람 시연: ~/isaac_libero/bridge/poses_demo)")
    ap.add_argument("--z-offset-m", type=float, default=None, help="재생 목표 z 오프셋 (기본 -0.006)")
    ap.add_argument("--status", action="store_true", help="실행 없이 현황만")
    a = ap.parse_args()
    global Z_OFFSET_M, SRC, POSE_ROOT
    if a.src:
        SRC = a.src
    if a.pose_root:
        POSE_ROOT = a.pose_root
    if a.z_offset_m is not None:
        Z_OFFSET_M = float(a.z_offset_m)

    for task in a.tasks:
        eps = sources(task)
        ok_by_ep, tried = tally(task, a.out, a.seeds)
        uniq = len(ok_by_ep); total_ok = sum(len(v) for v in ok_by_ep.values())
        print(f"[task {task}] 소스 {len(eps)}  성공 궤적 {uniq}  성공 판 {total_ok}  목표 {a.target}", flush=True)
        if a.status:
            continue
        # 1~3 순위: 시드 순서대로, 아직 성공하지 못한 궤적만
        for seed in a.seeds:
            for ep in eps:
                ok_by_ep, tried = tally(task, a.out, a.seeds)
                if len(ok_by_ep) >= a.target:
                    break
                if ep in ok_by_ep or seed in tried.get(ep, []):
                    continue
                t0 = time.time(); r = run_episode(task, ep, seed, a.out, a.steps)
                ok_by_ep, _ = tally(task, a.out, a.seeds)
                print(f"  task {task} e{ep:03d} seed {seed}  {'성공' if r else '실패'}  ({time.time()-t0:.0f}s)  성공 궤적 {len(ok_by_ep)}/{a.target}", flush=True)
            if len(tally(task, a.out, a.seeds)[0]) >= a.target:
                break
        # 4 순위: 중복 허용
        ok_by_ep, tried = tally(task, a.out, a.seeds)
        if a.allow_duplicates and len(ok_by_ep) < a.target:
            need = a.target - sum(len(v) for v in ok_by_ep.values())
            for seed in a.seeds:
                for ep in sorted(ok_by_ep):
                    if need <= 0:
                        break
                    if seed in tried.get(ep, []):
                        continue
                    r = run_episode(task, ep, seed, a.out, a.steps)
                    ok_by_ep, tried = tally(task, a.out, a.seeds)
                    if r:
                        need -= 1
                    print(f"  task {task} e{ep:03d} seed {seed} (중복)  {'성공' if r else '실패'}  남은 {max(0, need)}", flush=True)
                if need <= 0:
                    break
        ok_by_ep, _ = tally(task, a.out, a.seeds)
        print(f"[task {task}] 완료: 성공 궤적 {len(ok_by_ep)}  성공 판 {sum(len(v) for v in ok_by_ep.values())}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
