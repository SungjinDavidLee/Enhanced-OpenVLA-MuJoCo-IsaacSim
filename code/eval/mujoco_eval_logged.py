#!/usr/bin/env python3
"""MuJoCo(LIBERO) 평가를 원래 도구 그대로 돌리면서, 두 가지만 덧붙인다.

  1) 명령 간격:  MJ_CONTROL_FREQ (기본 20 = 0.05초마다 새 명령). 10 이면 0.1초마다 = MuJoCo-B
  1-1) 역정규화 통계: MJ_UNNORM_KEY (없으면 원래 도구가 고르는 libero_spatial). 다시 학습시킨 모델(OpF4)은 isaac_libero_spatial
  2) 판 기록:   판마다 Isaac Sim 과 같은 형식의 episode_result.json 을 MJ_LOG_OUT/task_XX/eYYY/ 에 저장 (10/8 부터 과제별 구조)
                (손가락 폭 · 손 위치 · 목표 그릇 위치 · 접시 위치 · 다른 물체 위치를 매 스텝)
                → table1_metrics.py · compare_runs.py · 실패 종류 분류를 그대로 쓸 수 있음

원래 도구(~/openvla/experiments/robot/libero/run_libero_spatial_taskwise_benchmark.py)는 고치지 않는다.
화면 처리 · 동작 명령 변환 · 성공 판정(LIBERO 자체 판정)은 원래 도구 그대로다. 인자도 원래 도구와 같다.

    MJ_CONTROL_FREQ=20 MJ_LOG_OUT=~/LIBERO/experiments/MJ_OpZ_default_layout50 \\
      python3 mujoco_eval_logged.py --task-id all --num-trials 50 --seed 7 \\
      --model openvla/openvla-7b-finetuned-libero-spatial --output-root ~/LIBERO/experiments/mj_runs/default
"""
import csv
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np

RUNNER = Path(os.path.expanduser("~/openvla/experiments/robot/libero/run_libero_spatial_taskwise_benchmark.py"))
CF = int(os.environ.get("MJ_CONTROL_FREQ", "20"))
LOG_OUT = Path(os.path.expanduser(os.environ.get("MJ_LOG_OUT", "~/LIBERO/experiments/MJ_logged")))
LOG_OUT.mkdir(parents=True, exist_ok=True)

spec = importlib.util.spec_from_file_location("libero_runner", RUNNER)
runner = importlib.util.module_from_spec(spec)
sys.modules["libero_runner"] = runner
spec.loader.exec_module(runner)

BUF = []            # 현재 판의 스텝별 관측 기록
NAMES = {}          # 현재 환경의 목표 그릇 · 접시 이름


def _goal_names(env):
    try:
        for pred in env.env.parsed_problem["goal_state"]:
            if str(pred[0]).lower() == "on" and len(pred) >= 3:
                return str(pred[1]), str(pred[2])
    except Exception:
        pass
    return None, None


def _record(obs, env):
    if not NAMES:
        t, d = _goal_names(env)
        keys = [k for k in obs if k.endswith("_pos") and not k.startswith("robot0")]
        NAMES["target"] = t if t and f"{t}_pos" in obs else next((k[:-4] for k in keys if "bowl" in k), None)
        NAMES["dest"] = d if d and f"{d}_pos" in obs else next((k[:-4] for k in keys if "plate" in k), None)
        NAMES["others"] = [k[:-4] for k in keys if k[:-4] not in (NAMES["target"], NAMES["dest"])]
    g = np.abs(np.asarray(obs["robot0_gripper_qpos"], dtype=float).reshape(-1))
    tb = np.asarray(obs[f"{NAMES['target']}_pos"], dtype=float).reshape(-1)[:3].tolist() if NAMES["target"] else None
    pl = np.asarray(obs[f"{NAMES['dest']}_pos"], dtype=float).reshape(-1)[:3].tolist() if NAMES["dest"] else None
    BUF.append({
        "gripper_finger_mean_m": float(g.mean()),
        "gripper_joint_positions": g.tolist(),
        "eef_position_m": np.asarray(obs["robot0_eef_pos"], dtype=float).reshape(-1)[:3].tolist(),
        "target_bowl_aabb_center_m": tb,
        "target_bowl_aabb_min_m": tb,          # 그릇 중심 높이를 씀 (들어 올린 높이는 처음과의 차이로 판정하므로 같음)
        "plate_aabb_center_m": pl,
        "other_objects_pos": {n: np.asarray(obs[f"{n}_pos"], dtype=float).reshape(-1)[:3].tolist() for n in NAMES["others"]},
    })


def _wrap_env(env):
    step0, set0 = env.step, env.set_init_state

    def set_init_state(state):
        obs = set0(state)
        BUF.clear(); _record(obs, env)
        return obs

    def step(action):
        out = step0(action)
        _record(out[0], env)
        return out

    env.step, env.set_init_state = step, set_init_state
    return env


_make0 = runner.make_runtime_imports


def make_runtime_imports():
    rt = _make0()
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    def get_libero_env(task, model_family, resolution=256):
        bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=resolution, camera_widths=resolution, control_freq=CF)
        env.seed(0)                           # 원래 도구와 같음
        NAMES.clear()
        return _wrap_env(env), task.language

    rt["get_libero_env"] = get_libero_env
    return rt


runner.make_runtime_imports = make_runtime_imports

# 역정규화 통계 지정 (다시 학습시킨 모델은 자기 학습 데이터의 통계를 써야 함). 원래 도구의 설정 객체만 감싼다.
UNNORM = os.environ.get("MJ_UNNORM_KEY")
if UNNORM:
    _EC = runner.EvalConfig

    def EvalConfig(*args, **kwargs):
        cfg = _EC(*args, **kwargs)
        try:
            cfg.unnorm_key = UNNORM
        except Exception:
            object.__setattr__(cfg, "unnorm_key", UNNORM)
        return cfg

    runner.EvalConfig = EvalConfig


class LoggingWriter(csv.DictWriter):
    """원래 도구가 판마다 episodes.csv 에 한 줄을 쓸 때, 같은 판의 기록을 episode_result.json 으로 저장한다."""

    def writerow(self, row):
        r = super().writerow(row)
        # 제목 줄(writeheader 가 부르는 writerow)은 건너뛴다 — 값이 숫자인 줄만 판으로 본다
        if isinstance(row, dict) and str(row.get("episode_idx", "")).isdigit() and str(row.get("task_id", "")).isdigit():
            t, e = int(row["task_id"]), int(row["episode_idx"])
            d = LOG_OUT / f"task_{t:02d}" / f"e{e:03d}"
            d.mkdir(parents=True, exist_ok=True)
            res = {"task_success": bool(int(row["success"])), "termination_reason": row.get("termination_reason"),
                   "task_id": t, "episode_idx": e, "simulator": "mujoco", "control_freq_hz": CF,
                   "num_policy_steps": row.get("num_policy_steps"), "num_sim_steps": row.get("num_sim_steps"),
                   "target_object": NAMES.get("target"), "destination_object": NAMES.get("dest"),
                   "stage_trace": [{"policy_step": i, "observations": o} for i, o in enumerate(BUF)]}
            (d / "episode_result.json").write_text(json.dumps({"result": res}), encoding="utf-8")
        return r


csv.DictWriter = LoggingWriter

if __name__ == "__main__":
    with open(LOG_OUT / "RUN_CONFIG.txt", "a", encoding="utf-8") as f:
        import datetime
        f.write(f"--- {datetime.datetime.now():%Y-%m-%d %H:%M:%S}\n시뮬레이터: MuJoCo (LIBERO 원래 평가 도구 그대로)\n"
                f"명령 간격: 1초 {CF}번 ({1 / CF:.3f}초)\n역정규화 통계: {UNNORM or 'libero_spatial (원래 도구 기본)'}\n인자: {' '.join(sys.argv[1:])}\n")
    main = getattr(runner, "main", None)
    if main is None:
        sys.exit("원래 도구에서 main() 을 찾지 못함 — 원래 도구의 실행 진입점을 확인할 것")
    main()
