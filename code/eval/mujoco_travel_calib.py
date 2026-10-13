#!/usr/bin/env python3
"""MuJoCo-B 맞추기 (모델 없이): 사람 시범의 동작 명령을 MuJoCo 에 그대로 넣고, 명령 간격을 바꿨을 때
손이 기본(1초 20번)보다 얼마나 더 움직이는지 잰다. Isaac Sim 의 B 와 가장 비슷한 간격을 고른다.

  비교 기준 (Isaac Sim 의 B, 9/29 · 10/2 측정): 다가가는 동안 1.26배, 들어 올리는 동안 약 1.5배
  사람 시범: 과제 0 · 6, 시범 0 · 1 (Isaac Sim 측정과 같은 4개)
  구간: 다가감 = 처음 ~ 처음 집게를 닫는 명령 전, 들어 올림 = 닫은 뒤 손이 5 mm 넘게 오르기 시작한 때부터 20 명령

    python3 mujoco_travel_calib.py                  # 후보 20 · 10 · 8 · 6 · 5 Hz
    python3 mujoco_travel_calib.py --out ~/mujoco_b_freq.txt
읽기만 한다 (시범 파일을 읽고, 결과를 화면과 --out 파일에 씀).
"""
import argparse, glob, math, os, sys
import numpy as np

sys.path.insert(0, os.path.expanduser("~/LIBERO"))
import h5py
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

ap = argparse.ArgumentParser()
ap.add_argument("--freqs", type=int, nargs="+", default=[20, 10, 8, 6, 5])
ap.add_argument("--tasks", type=int, nargs="+", default=[0, 6])
ap.add_argument("--demos", type=int, nargs="+", default=[0, 1])
ap.add_argument("--target-approach", type=float, default=1.26)
ap.add_argument("--target-lift", type=float, default=1.5)
ap.add_argument("--out", default=os.path.expanduser("~/mujoco_b_freq.txt"))
a = ap.parse_args()

suite = benchmark.get_benchmark_dict()["libero_spatial"]()
droot = os.path.join(get_libero_path("datasets"), "libero_spatial")


def replay(task, states0, actions, cf):
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=64, camera_widths=64, control_freq=cf)
    env.seed(0); env.reset()
    obs = env.set_init_state(states0)
    p = [np.asarray(obs["robot0_eef_pos"], float)]
    for act in actions:
        obs, _, _, _ = env.step(np.asarray(act, float).tolist())
        p.append(np.asarray(obs["robot0_eef_pos"], float))
    env.close()
    return np.asarray(p)


rows = {cf: [] for cf in a.freqs}
for t in a.tasks:
    task = suite.get_task(t)
    fs = glob.glob(os.path.join(droot, f"{task.name}_demo.hdf5"))
    if not fs:
        print(f"과제 {t}: 시범 파일을 찾지 못함 ({droot}/{task.name}_demo.hdf5)"); continue
    with h5py.File(fs[0], "r") as f:
        for i in a.demos:
            g = f["data"][f"demo_{i}"]; acts = np.asarray(g["actions"]); s0 = np.asarray(g["states"])[0]
            traj = {cf: replay(task, s0, acts, cf) for cf in a.freqs}
            ref = traj[20] if 20 in traj else traj[a.freqs[0]]
            jc = next((j for j, x in enumerate(acts) if x[6] > 0), None)
            if jc is None or jc < 2:
                print(f"과제 {t} 시범 {i}: 집게를 닫는 명령을 찾지 못함"); continue
            j1 = next((j for j in range(jc, len(ref)) if ref[j, 2] - ref[jc, 2] > 0.005), None)
            j1 = (j1 - 1) if j1 else None
            j2 = min(j1 + 20, len(ref) - 1) if j1 is not None else None
            for cf in a.freqs:
                p = traj[cf]
                app = np.sum(np.linalg.norm(np.diff(p[:jc + 1], axis=0), axis=1)) / max(1e-9, np.sum(np.linalg.norm(np.diff(ref[:jc + 1], axis=0), axis=1)))
                lift = ((p[j2, 2] - p[j1, 2]) / (ref[j2, 2] - ref[j1, 2])) if j1 is not None and abs(ref[j2, 2] - ref[j1, 2]) > 0.003 else float("nan")
                rows[cf].append((app, lift))
                print(f"과제 {t} 시범 {i}  1초 {cf:2d}번: 다가감 {app:.2f}배, 들어 올림 {lift:.2f}배")

print("\n명령 간격      다가가는 동안   들어 올리는 동안   (Isaac Sim 의 B: %.2f · %.2f)" % (a.target_approach, a.target_lift))
best, best_cost = None, float("inf")
for cf in a.freqs:
    if not rows[cf]:
        continue
    R = np.nanmean(np.asarray(rows[cf], float), axis=0)
    cost = abs(math.log(R[0] / a.target_approach)) + (abs(math.log(R[1] / a.target_lift)) if not math.isnan(R[1]) and R[1] > 0 else 1.0)
    print("1초 %2d번 (%.3f초)   %.2f배          %.2f배" % (cf, 1 / cf, R[0], R[1]))
    if cf != 20 and cost < best_cost:
        best, best_cost = cf, cost
best = best or 6
print(f"\n고른 간격: 1초 {best}번 ({1 / best:.3f}초) — Isaac Sim 의 B 와 이동 비율이 가장 가까움")
with open(a.out, "w") as f:
    f.write(f"{best}\n")
