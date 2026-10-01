#!/usr/bin/env python3
"""LIBERO 사람 시연의 첫 상태(states[0])에서 물체 배치를 뽑아 시연별 pose JSON 을 만든다.

시연은 평가용 init_states 와 다른 배치에서 시작하므로(그릇 xy 9~35 mm 차이), 시연을 재생하려면
시연 자체의 배치가 필요하다.  poses_all/s00.json 을 템플릿으로 물체 이름·구조를 가져오고,
xy·yaw 는 시연 상태에서, z 는 템플릿 값(받침 정렬이 채움)을 쓴다.  task 5 는 정렬을 끈 전용
설정을 쓰므로 안정값 + 5 mm 로 z 를 채운 사본을 poses_demo_t5 에 따로 만든다.

    conda activate openvla_libero5090
    python3 make_demo_pose_jsons.py --tasks 0 1 2 3 4 5 6 7 8 9

출력  ~/isaac_libero/bridge/poses_demo/task_XX/demo_episode_YYY.json
      ~/isaac_libero/bridge/poses_demo_t5/task_05/demo_episode_YYY.json
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys

import h5py
import numpy as np

BR = "/home/user/isaac_libero/bridge"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--template", default=f"{BR}/poses_all/libero_spatial_task_poses_s00.json")
    ap.add_argument("--libero-root", default="/home/user/LIBERO")
    a = ap.parse_args()
    sys.path.insert(0, a.libero_root)
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    ds_root = os.path.join(get_libero_path("datasets"), "libero_spatial")
    template = json.load(open(a.template))

    for t in a.tasks:
        task = suite.get_task(t)
        h5 = os.path.join(ds_root, task.name + "_demo.hdf5")
        env = OffScreenRenderEnv(bddl_file_name=suite.get_task_bddl_file_path(t), camera_heights=64, camera_widths=64); env.reset()
        sim = env.env.sim
        out = f"{BR}/poses_demo/task_{t:02d}"; out5 = f"{BR}/poses_demo_t5/task_{t:02d}"
        os.makedirs(out, exist_ok=True)
        if t == 5:
            os.makedirs(out5, exist_ok=True)
        names = list(template["tasks"][str(t)]["poses"].keys())
        n = 0; moved = []
        with h5py.File(h5, "r") as f:
            for name in sorted(f["data"].keys(), key=lambda k: int(k.split("_")[-1])):
                i = int(name.split("_")[-1])
                s0 = np.asarray(f["data"][name]["states"][0])
                env.set_init_state(s0)
                for _ in range(10):
                    env.step([0, 0, 0, 0, 0, 0, -1])
                d = copy.deepcopy(template); poses = d["tasks"][str(t)]["poses"]
                d5 = copy.deepcopy(template); poses5 = d5["tasks"][str(t)]["poses"]
                for obj in names:
                    if "cabinet" in obj:
                        continue
                    try:
                        bid = sim.model.body_name2id(obj)
                    except Exception:
                        continue
                    p = sim.data.body_xpos[bid]; q = sim.data.body_xquat[bid]      # wxyz
                    for P, keep_z in ((poses, True), (poses5, False)):
                        z = P[obj]["position_xyz"][2] if keep_z else float(p[2]) + 0.005
                        P[obj]["position_xyz"] = [round(float(p[0]), 6), round(float(p[1]), 6), round(z, 6)]
                        for key in ("orientation_wxyz", "quaternion_wxyz", "rotation_wxyz"):
                            if key in P[obj]:
                                P[obj][key] = [round(float(v), 6) for v in q]
                    if "bowl_1" in obj:
                        moved.append(np.linalg.norm(np.array(p[:2]) - np.array(template["tasks"][str(t)]["poses"][obj]["position_xyz"][:2])) * 1000)
                d["note"] = f"layout from LIBERO human demo {name} first state (task {t}); z from template except task 5 variant"
                json.dump(d, open(f"{out}/demo_episode_{i:03d}.json", "w"), indent=2)
                if t == 5:
                    json.dump(d5, open(f"{out5}/demo_episode_{i:03d}.json", "w"), indent=2)
                n += 1
        env.close()
        print(f"[task {t}] {n} 개 → {out}   템플릿(s00) 대비 그릇 xy 이동 중앙 {np.median(moved):.0f} mm, 최대 {np.max(moved):.0f} mm")
    return 0


if __name__ == "__main__":
    sys.exit(main())
