#!/usr/bin/env python3
"""LIBERO 초기 상태 50개 전부를 Isaac 용 pose JSON 으로 뽑는다.

지금 브리지는 libero_spatial_task_poses.json (init_state_id=0 하나) 로 모든 판을 돌린다.
그래서 Isaac 의 50 판은 배치가 전부 같고, 다른 배치의 재생 궤적은 그릇을 빗나간다.

이 스크립트는 기존 JSON 을 템플릿으로 삼아 구조·물체 이름·좌표 규약을 그대로 두고,
init state k 마다 poses 만 바꾼 파일을 만든다.

    out_dir/libero_spatial_task_poses_s00.json  …  _s49.json

브리지에는 판마다 --pose-json-path <out_dir>/libero_spatial_task_poses_s{episode:02d}.json 을 준다.
자기 검증: k=0 의 결과가 템플릿과 같아야 한다 (추출 규약이 맞는지 확인).

    conda activate openvla_libero5090
    python3 make_libero_pose_jsons.py \\
        --template /home/user/isaac_libero/bridge/libero_spatial_task_poses.json \\
        --out-dir  /home/user/isaac_libero/bridge/poses_all
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--template", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--libero", default="/home/user/LIBERO")
    ap.add_argument("--n-states", type=int, default=50)
    ap.add_argument("--settle-steps", type=int, default=10,
                    help="init state 적용 뒤 물리 스텝 수. OpenVLA 평가와 같은 10 이면 물체가 떨어져 안정된 배치")
    ap.add_argument("--keep-template-z", action="store_true", default=True,
                    help="xy·회전은 안정된 값, z 는 템플릿 값(낙하 높이) 유지 — Isaac 이 제자리로 수직 낙하")
    a = ap.parse_args()

    sys.path.insert(0, a.libero)
    from libero.libero import benchmark                          # noqa: E402
    from libero.libero.envs import OffScreenRenderEnv            # noqa: E402

    tpl = json.loads(a.template.read_text())
    tasks = tpl["tasks"]
    suite = benchmark.get_benchmark_dict()[tpl.get("benchmark", "libero_spatial").replace("-", "_").lower()]() \
        if isinstance(tpl.get("benchmark"), str) and "libero" in tpl["benchmark"].lower() \
        else benchmark.get_benchmark_dict()["libero_spatial"]()
    a.out_dir.mkdir(parents=True, exist_ok=True)

    # 템플릿의 task 키가 문자열 인덱스인지 확인
    task_keys = list(tasks)
    print("tasks:", task_keys)

    per_state = {k: copy.deepcopy(tpl) for k in range(a.n_states)}
    for k in range(a.n_states):
        per_state[k]["init_state_id"] = k

    max_dev_k0 = 0.0
    for tkey in task_keys:
        tid = int(tkey) if str(tkey).isdigit() else task_keys.index(tkey)
        states = suite.get_task_init_states(tid)
        env = OffScreenRenderEnv(bddl_file_name=suite.get_task_bddl_file_path(tid),
                                 camera_heights=128, camera_widths=128)
        body_names = list(tasks[tkey]["poses"])
        env.reset()                                   # 리셋은 태스크당 한 번이면 충분
        for k in range(min(a.n_states, len(states))):
            env.set_init_state(states[k])
            for _ in range(a.settle_steps):
                env.step([0, 0, 0, 0, 0, 0, -1])
            sim = env.env.sim; sim.forward()
            poses = {}
            for name in body_names:
                try:
                    bid = sim.model.body_name2id(name)
                except Exception:
                    continue
                pos = np.asarray(sim.data.body_xpos[bid], dtype=float)
                quat = np.asarray(sim.data.body_xquat[bid], dtype=float)       # MuJoCo: WXYZ
                rmat = np.asarray(sim.data.body_xmat[bid], dtype=float).reshape(3, 3)
                entry = copy.deepcopy(tasks[tkey]["poses"][name])
                if a.keep_template_z and "position_xyz" in entry:
                    pos = pos.copy(); pos[2] = float(entry["position_xyz"][2])
                if "rotation_matrix" in entry:
                    entry["rotation_matrix"] = [[round(float(v), 9) for v in row] for row in rmat]
                # 템플릿이 쓰는 키 이름을 그대로 유지
                for pk in ("position_xyz", "position", "pos"):
                    if pk in entry: entry[pk] = [round(float(v), 9) for v in pos]; break
                for qk in ("orientation_wxyz", "quat_wxyz", "orientation", "quat"):
                    if qk in entry: entry[qk] = [round(float(v), 9) for v in quat]; break
                poses[name] = entry
                if k == 0 and not any(x in name for x in ("cabinet", "stove", "table")):
                    ref = tasks[tkey]["poses"][name]
                    rp = ref.get("position_xyz", ref.get("position", ref.get("pos")))
                    if rp is not None:
                        max_dev_k0 = max(max_dev_k0, float(np.linalg.norm(np.array(rp) - pos)))
            per_state[k]["tasks"][tkey]["poses"] = poses
        env.close()
        # 50 배치에서 위치가 안 변하는 물체(캐비닛·스토브 등 고정 가구)는 템플릿 값을 그대로 둔다.
        # 템플릿은 그런 물체를 yaml 의 보정된 고정값으로 넣었고, Isaac 장면도 그 값에 맞춰져 있다.
        fixed = []
        for name in body_names:
            ps = [per_state[k]["tasks"][tkey]["poses"].get(name, {}).get("position_xyz") for k in range(min(a.n_states, len(states)))]
            ps = [np.array(v) for v in ps if v is not None]
            if ps and float(np.max([np.linalg.norm(v - ps[0]) for v in ps])) < 1e-3:
                fixed.append(name)
                for k in range(min(a.n_states, len(states))):
                    per_state[k]["tasks"][tkey]["poses"][name] = copy.deepcopy(tasks[tkey]["poses"][name])
        print("task %s  %d 개 배치 추출  (물체 %d, 고정 유지: %s)" % (tkey, min(a.n_states, len(states)), len(body_names), ", ".join(fixed) or "없음"))

    for k in range(a.n_states):
        out = a.out_dir / ("libero_spatial_task_poses_s%02d.json" % k)
        out.write_text(json.dumps(per_state[k], indent=2))
    print("저장:", a.out_dir, "(%d 파일)" % a.n_states)
    print("참고  k=0 vs 템플릿 최대 위치 차이 %.4f m  (settle %d 스텝 뒤라 몇 mm 차이는 정상; 5 cm 넘는 물체는 낙하 중 미끄러진 것)"
          % (max_dev_k0, a.settle_steps))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
