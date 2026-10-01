#!/usr/bin/env python3
"""LIBERO 원본 사람 시연(HDF5) → 재생 소스 npz.

롤아웃 소스(alltheTask_episode_YYY.npz)와 같은 키로 저장하므로 수집기가 그대로 쓴다.
  libero_actions (T,7)   환경 규약 (집게 −1 열림 / +1 닫힘)
  eef_pos (T,3)          obs/ee_pos
  eef_quat (T,4)         obs/ee_ori (축각) → 쿼터니언 xyzw  (브리지에 --libero-eef-quat-order xyzw)
  joint_states (T,7), gripper_states (T,2), instruction, task_id
  init_state_id          시연 번호 (LIBERO init_states 의 같은 번호와 배치가 같음을 검증해 기록)

    conda activate openvla_libero5090
    python3 libero_demo_to_replay_npz.py --tasks 0 1 2 3 4 5 6 7 8 9 --out /home/user/isaac_libero/runs/replay_src_demo
    python3 libero_demo_to_replay_npz.py --tasks 0 --out ... --drop-noops     # 정지 스텝 제거 (OpenVLA 의 no_noops 와 동일)
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import h5py
import numpy as np


def axis_angle_to_quat_xyzw(aa: np.ndarray) -> np.ndarray:
    aa = np.asarray(aa, dtype=np.float64)
    angle = np.linalg.norm(aa, axis=-1, keepdims=True)
    axis = np.where(angle > 1e-9, aa / np.maximum(angle, 1e-12), np.array([1.0, 0.0, 0.0]))
    half = angle / 2.0
    xyz = axis * np.sin(half)
    w = np.cos(half)
    return np.concatenate([xyz, w], axis=-1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--out", default="/home/user/isaac_libero/runs/replay_src_demo")
    ap.add_argument("--libero-root", default="/home/user/LIBERO")
    ap.add_argument("--drop-noops", action="store_true", help="위치·회전 명령이 0 이고 집게가 안 바뀐 스텝 제거")
    ap.add_argument("--verify-layout", action="store_true", help="시연 첫 상태의 물체 위치를 init_states 와 대조 (MuJoCo 필요)")
    a = ap.parse_args()
    sys.path.insert(0, a.libero_root)
    from libero.libero import benchmark, get_libero_path
    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    ds_root = os.path.join(get_libero_path("datasets"), "libero_spatial")

    total = 0
    for t in a.tasks:
        task = suite.get_task(t)
        h5 = os.path.join(ds_root, task.name + "_demo.hdf5")
        if not os.path.exists(h5):
            cands = glob.glob(os.path.join(ds_root, "*.hdf5"))
            print(f"[task {t}] 파일 없음: {h5}\n  후보: {[os.path.basename(c) for c in cands][:3]}"); continue
        instr = task.language
        out_dir = os.path.join(a.out, f"task_{t:02d}"); os.makedirs(out_dir, exist_ok=True)
        with h5py.File(h5, "r") as f:
            demos = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[-1]))
            n_written = 0; lens = []
            for name in demos:
                i = int(name.split("_")[-1])
                g = f["data"][name]
                acts = np.asarray(g["actions"], dtype=np.float32)
                ee_pos = np.asarray(g["obs"]["ee_pos"], dtype=np.float64)
                ee_ori = np.asarray(g["obs"]["ee_ori"], dtype=np.float64)
                grip = np.asarray(g["obs"]["gripper_states"], dtype=np.float32)
                joints = np.asarray(g["obs"]["joint_states"], dtype=np.float32)
                n = min(len(acts), len(ee_pos))
                acts, ee_pos, ee_ori, grip, joints = acts[:n], ee_pos[:n], ee_ori[:n], grip[:n], joints[:n]
                if a.drop_noops:
                    keep = np.ones(n, dtype=bool)
                    for k in range(n):
                        still = np.all(np.abs(acts[k, :6]) < 1e-6)
                        same_grip = k > 0 and acts[k, 6] == acts[k - 1, 6]
                        keep[k] = not (still and same_grip)
                    acts, ee_pos, ee_ori, grip, joints = acts[keep], ee_pos[keep], ee_ori[keep], grip[keep], joints[keep]
                quat = axis_angle_to_quat_xyzw(ee_ori)
                np.savez(os.path.join(out_dir, f"demo_episode_{i:03d}.npz"),
                         libero_actions=acts, eef_pos=ee_pos, eef_quat=quat, eef_axisangle=ee_ori,
                         joint_states=joints, gripper_states=grip,
                         instruction=np.array(instr), task_id=np.array([t], dtype=np.int32),
                         init_state_id=np.array([i], dtype=np.int32), source=np.array("libero_human_demo"))
                n_written += 1; lens.append(len(acts))
            print(f"[task {t}] {n_written} 시연 → {out_dir}   길이 {min(lens)}~{max(lens)} (중앙 {int(np.median(lens))})   집게 값 {sorted(set(np.round(acts[:, 6], 1)))}")
            total += n_written

        if a.verify_layout:
            from libero.libero.envs import OffScreenRenderEnv
            init = suite.get_task_init_states(t)
            with h5py.File(h5, "r") as f:
                s0 = np.asarray(f["data"]["demo_0"]["states"][0])
            env = OffScreenRenderEnv(bddl_file_name=suite.get_task_bddl_file_path(t), camera_heights=64, camera_widths=64); env.reset()
            def bowl_xy(state):
                env.set_init_state(state); sim = env.env.sim
                return np.array(sim.data.body_xpos[sim.model.body_name2id("akita_black_bowl_1_main")][:2])
            d = np.linalg.norm(bowl_xy(s0) - bowl_xy(init[0])) * 1000
            print(f"   배치 검증: demo_0 첫 상태 vs init_states[0] 그릇 xy 차이 {d:.1f} mm  ({'같은 배치' if d < 2 else '다름 — init_state_id 대응 확인 필요'})")
            env.close()
    print(f"합계 {total} 시연")
    return 0


if __name__ == "__main__":
    sys.exit(main())
