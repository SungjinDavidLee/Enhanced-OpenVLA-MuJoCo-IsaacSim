#!/usr/bin/env python3
"""묶음으로 동작 명령을 내는 모델(π0.5 · OFT)을 MuJoCo(LIBERO)에서 평가한다. Isaac Sim 과 같은 정책 서버(/predict_chunk)에 물어본다.
  π0.5 : pi05_server.py — 10개 중 5개를 돌려줌, 집게 값은 이어진 값 그대로 (openpi 예제와 같음)
  OFT  : oft_server_v2.py — 8개를 돌려줌, 집게 값을 0 · 1 로 딱 잘라 씀 (--binarize-gripper, OpenVLA-OFT 의 LIBERO 평가와 같음)
  서버가 돌려준 묶음은 모두 실행한 뒤 다시 물어본다.

공통 규약 (openpi · OpenVLA-OFT 의 LIBERO 예제와 같음)
  - 화면: agentview · 손목 카메라를 256 으로 찍어 180° 뒤집음 (MuJoCo 화면은 거꾸로 찍힘), 자르지 않고 224 로 줄임
          → 서버에 crop_area 1.0 을 보냄 (Isaac Sim 은 시야를 맞추려고 서버 기본값 0.389 로 자름)
  - 로봇 자세 8개: 손 위치 3 · 손 방향(축각) 3 · 손가락 2
  - 집게 값: 서버는 브리지 형식(0 닫음 · 1 엶)으로 돌려주므로 환경 형식(−1 엶 · +1 닫음)으로 되돌림
              이어진 값: g_env = 1 − 2g,  --binarize-gripper: g < 0.5 → +1(닫음), 아니면 −1(엶)
  - 화면 자르기: 서버에 crop_area 를 보냄 (기본 1.0 = 자르지 않음). OFT 는 서버 안에서 0.9 를 한 번 더 자름 (원래 OFT 평가와 같음)
LIBERO 평가 규약 (OpenVLA 의 MuJoCo 평가와 같음): 과제마다 초기 상태 50개(배치50), env.seed(0), 시작 뒤 「집게 열기」 10스텝 기다림,
동작 명령 최대 220번, 성공은 LIBERO 자체 판정(환경의 done).
판 기록은 Isaac Sim 과 같은 형식으로 <out>/task_XX/eYYY/episode_result.json 에 저장 (10/8 부터 과제별 구조, strict_outcomes.py 등에 그대로 씀).
이미 기록이 있는 판은 건너뜀 (멈췄다 다시 걸어도 이어서 진행).

    python3 mujoco_eval_chunk.py --out ~/LIBERO/experiments/MJ_Pi05Z_default_layout50 --model-name pi05_libero
    python3 mujoco_eval_chunk.py --out ~/LIBERO/experiments/MJ_OFT_default_layout50 --model-name openvla-oft --binarize-gripper
"""
import argparse, base64, io, json, math, os, sys, time, urllib.request
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.expanduser("~/LIBERO"))
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--server", default="http://127.0.0.1:8000")
ap.add_argument("--tasks", default="0-9")
ap.add_argument("--trials", type=int, default=50)
ap.add_argument("--max-steps", type=int, default=220)
ap.add_argument("--wait", type=int, default=10)
ap.add_argument("--resolution", type=int, default=256)
ap.add_argument("--crop-area", type=float, default=1.0)
ap.add_argument("--control-freq", type=int, default=20)
ap.add_argument("--no-video", action="store_true")
ap.add_argument("--model-name", default="pi05_libero")
ap.add_argument("--binarize-gripper", action="store_true", help="집게 값을 0 · 1 로 딱 잘라 씀 (OFT)")
a = ap.parse_args()
a.out = os.path.expanduser(a.out)
lo, hi = (int(x) for x in a.tasks.split("-")) if "-" in a.tasks else (int(a.tasks), int(a.tasks))


def quat2axisangle(q):            # robosuite 와 같은 변환 (x, y, z, w)
    q = np.asarray(q, float).copy(); q[3] = min(1.0, max(-1.0, q[3]))
    den = math.sqrt(1.0 - q[3] * q[3])
    return np.zeros(3) if math.isclose(den, 0.0) else q[:3] * 2.0 * math.acos(q[3]) / den


def png64(arr):
    buf = io.BytesIO(); Image.fromarray(np.ascontiguousarray(arr)).save(buf, format="PNG"); return base64.b64encode(buf.getvalue()).decode()


def ask(instruction, img, wimg, state):
    body = json.dumps({"instruction": instruction, "image_base64": png64(img), "wrist_base64": png64(wimg),
                       "state": [float(v) for v in state], "crop_area": a.crop_area}).encode()
    req = urllib.request.Request(a.server + "/predict_chunk", data=body, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=120) as r:
        res = json.loads(r.read())
    if not res.get("ok"):
        raise RuntimeError(str(res)[:300])
    acts = np.asarray(res["actions"], float)[:, :7]
    if a.binarize_gripper:
        acts[:, 6] = np.where(acts[:, 6] < 0.5, 1.0, -1.0)    # 0 닫음 · 1 엶 → +1 닫음 · −1 엶 (딱 잘라)
    else:
        acts[:, 6] = 1.0 - 2.0 * acts[:, 6]                   # 0 닫음 · 1 엶 → −1 엶 · +1 닫음 (이어진 값)
    return list(acts)


def goal_names(env, obs):
    t = d = None
    try:
        for p in env.env.parsed_problem["goal_state"]:
            if str(p[0]).lower() == "on" and len(p) >= 3:
                t, d = str(p[1]), str(p[2]); break
    except Exception:
        pass
    keys = [k for k in obs if k.endswith("_pos") and not k.startswith("robot0")]
    t = t if t and f"{t}_pos" in obs else next((k[:-4] for k in keys if "bowl" in k), None)
    d = d if d and f"{d}_pos" in obs else next((k[:-4] for k in keys if "plate" in k), None)
    others = [k[:-4] for k in keys if k[:-4] not in (t, d) and "to_robot0" not in k]
    return t, d, others


def record(buf, obs, names):
    t, d, others = names
    g = np.abs(np.asarray(obs["robot0_gripper_qpos"], float).reshape(-1))
    tb = np.asarray(obs[f"{t}_pos"], float)[:3].tolist(); pl = np.asarray(obs[f"{d}_pos"], float)[:3].tolist()
    buf.append({"gripper_finger_mean_m": float(g.mean()), "gripper_joint_positions": g.tolist(),
                "eef_position_m": np.asarray(obs["robot0_eef_pos"], float)[:3].tolist(),
                "target_bowl_aabb_center_m": tb, "target_bowl_aabb_min_m": tb, "plate_aabb_center_m": pl,
                "other_objects_pos": {n: np.asarray(obs[f"{n}_pos"], float)[:3].tolist() for n in others}})


os.makedirs(a.out, exist_ok=True)
with open(os.path.join(a.out, "RUN_CONFIG.txt"), "a", encoding="utf-8") as f:
    f.write(f"--- {time.strftime('%F %T')}\n시뮬레이터: MuJoCo (LIBERO), 모델: {a.model_name} (서버 {a.server}, 서버가 돌려준 묶음을 모두 실행, 집게 {'0·1 로 자름' if a.binarize_gripper else '이어진 값'})\n"
            f"명령 간격: 1초 {a.control_freq}번, 화면 {a.resolution} · 자르기 면적 {a.crop_area}, 기다림 {a.wait}, 최대 {a.max_steps}\n")
suite = benchmark.get_benchmark_dict()["libero_spatial"]()
done_n = ok_n = 0
for t in range(lo, hi + 1):
    task = suite.get_task(t); inits = suite.get_task_init_states(t)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=a.resolution, camera_widths=a.resolution, control_freq=a.control_freq)
    env.seed(0)
    for e in range(a.trials):
        path = os.path.join(a.out, f"task_{t:02d}", f"e{e:03d}", "episode_result.json")
        old = os.path.join(a.out, f"task_{t:02d}_e{e:03d}", "mujoco", "episode_result.json")   # 옮기기 전 구조
        if os.path.exists(path) or os.path.exists(old):
            continue
        t0 = time.time(); buf, frames = [], []
        try:
            env.reset(); obs = env.set_init_state(inits[e]); names = goal_names(env, obs); record(buf, obs, names)
            for _ in range(a.wait):
                obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1]); record(buf, obs, names)
            queue, success, steps, reason = [], False, 0, "timeout"
            while steps < a.max_steps:
                img = obs["agentview_image"][::-1, ::-1]; wimg = obs["robot0_eye_in_hand_image"][::-1, ::-1]
                if not a.no_video:
                    frames.append(np.ascontiguousarray(img))
                if not queue:
                    state = np.concatenate([obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"]])
                    queue = ask(task.language, img, wimg, state)
                obs, _, done, _ = env.step(queue.pop(0).tolist()); steps += 1; record(buf, obs, names)
                if done:
                    success, reason = True, "success"; break
            res = {"task_success": success, "termination_reason": reason, "task_id": t, "episode_idx": e, "simulator": "mujoco",
                   "control_freq_hz": a.control_freq, "num_policy_steps": steps, "model": a.model_name,
                   "target_object": names[0], "destination_object": names[1],
                   "stage_trace": [{"policy_step": i, "observations": o} for i, o in enumerate(buf)]}
            os.makedirs(os.path.dirname(path), exist_ok=True)
            json.dump({"result": res}, open(path, "w"))
            if frames:
                import imageio.v2 as imageio
                imageio.mimwrite(os.path.join(os.path.dirname(path), "raw_rgb.mp4"), frames, fps=20)
            done_n += 1; ok_n += success
            print(f"[task {t} ep {e:02d}] success={success} steps={steps} {time.time() - t0:.0f}s  누적 {ok_n}/{done_n}", flush=True)
        except Exception as ex:
            print(f"[task {t} ep {e:02d}] 오류: {type(ex).__name__}: {ex}", flush=True)
    env.close()
print(f"끝: 이번에 돌린 판 {done_n}, 성공 {ok_n}")
