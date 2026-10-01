#!/usr/bin/env python3
"""Isaac 캡처 → OpenVLA 미세조정 npz (v2).

원본 build_finetune_dataset.py 의 전처리(정사각 크롭 → 면적 0.35 중앙 크롭 → 224 BILINEAR)와
프레임 추출은 그대로 두고, 이번에 잡힌 문제들을 고쳤다.

  선별      요약 TSV 의 파지 기준  →  episode_result.json 의 task_success (과제 성공)
  액션      action_raw (정책 규약, 집게 1=열림)  →  libero_actions (환경 규약, −1 열림 / +1 닫힘)
            RLDS 변환기와 GR00T 공유 npz 가 기대하는 규약
  롤아웃    로그의 raw 7-D 에서 집게만 환경 규약으로 변환 (앞 6 개는 같은 값)
  state     0 채움  →  trace 의 실제 값 (EE 위치 3 · 축각 3 · 손가락 2).  회전 키가 없으면 축각은 0 이고 경고
  폴더      replay_all/task_XX_eYYY  (+ 2 차 replay_all_pass2),  롤아웃 zs_faithful_50ep/task_XX_eYYY
  메타      init_state_id, replay_pass, seed 추가

프레임 정렬은 원본 규칙을 유지한다: 캡처는 액션보다 프레임이 1 개 많고 frame[i] 는 액션 i 를 낸
관측이다. 프레임 수가 액션 수 +0~2 범위를 벗어나면 그 판은 건너뛰고 보고한다.

    python3 build_finetune_dataset_v2.py \\
        --replay-roots /home/user/LIBERO/experiments/replay_all /home/user/LIBERO/experiments/replay_all_pass2 \\
        --rollout-root /home/user/LIBERO/experiments/zs_faithful_50ep \\
        --output-root  /home/user/LIBERO/experiments/finetune_dataset_v2
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import subprocess
import tempfile

import numpy as np
from PIL import Image

SQUARE_CROP = True
CROP_AREA = 0.35
OUTPUT_SIZE = 224
STEP_ACTION_RE = re.compile(r"\[step (\d+)\] raw=\[([-0-9.eE, ]+)\]")
REPLAY_SRC = "/home/user/isaac_libero/runs/replay_src"


# ----------------------------------------------------------------- 전처리 (원본 그대로)
def preprocess(frame: Image.Image) -> np.ndarray:
    image = frame.convert("RGB")
    if SQUARE_CROP:
        width, height = image.size
        side = min(width, height)
        left = (width - side) // 2
        top = (height - side) // 2
        image = image.crop((left, top, left + side, top + side))
    side_fraction = math.sqrt(CROP_AREA)
    width, height = image.size
    crop_w = max(1, min(width, int(round(width * side_fraction))))
    crop_h = max(1, min(height, int(round(height * side_fraction))))
    left = (width - crop_w) // 2
    top = (height - crop_h) // 2
    image = image.crop((left, top, left + crop_w, top + crop_h))
    image = image.resize((OUTPUT_SIZE, OUTPUT_SIZE), Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.uint8)


def extract_frames(video_path: str) -> list[np.ndarray]:
    with tempfile.TemporaryDirectory() as tmp:
        result = subprocess.run(
            ["ffmpeg", "-loglevel", "error", "-i", video_path, "-vsync", "0", f"{tmp}/f_%05d.png"],
            capture_output=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {result.stderr.decode()[:200]}")
        return [preprocess(Image.open(p)) for p in sorted(glob.glob(f"{tmp}/f_*.png"))]


def parse_rollout_actions(log_path: str) -> np.ndarray:
    by_step: dict[int, list[float]] = {}
    with open(log_path, errors="ignore") as handle:
        for line in handle:
            if "] raw=[" not in line or "candidate" in line:
                continue
            m = STEP_ACTION_RE.search(line)
            if m:
                vals = [float(v) for v in m.group(2).split(",")]
                if len(vals) == 7:
                    by_step[int(m.group(1))] = vals
    if not by_step:
        return np.zeros((0, 7), dtype=np.float32)
    return np.asarray([by_step[s] for s in sorted(by_step)], dtype=np.float32)


def raw_to_env(raw: np.ndarray) -> np.ndarray:
    """정책 규약(집게 1=열림, 0=닫힘) → 환경 규약(−1 열림, +1 닫힘). 앞 6 개는 동일."""
    env = raw.astype(np.float32).copy()
    env[:, 6] = np.where(raw[:, 6] > 0.5, -1.0, 1.0)
    return env


# ----------------------------------------------------------------- state
_ROT_KEYS = ("eef_rotation", "eef_rotation_matrix", "eef_orientation_quat_wxyz", "eef_quaternion_wxyz", "eef_quat_wxyz")
_missing_rot_warned = [False]


def _axis_angle_from(obs: dict) -> np.ndarray | None:
    for k in _ROT_KEYS:
        v = obs.get(k)
        if v is None:
            continue
        v = np.asarray(v, dtype=np.float64)
        if v.shape == (9,):
            v = v.reshape(3, 3)
        if v.shape == (3, 3):
            R = v
        elif v.shape == (4,):
            w, x, y, z = v
            R = np.array([
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ])
        else:
            continue
        angle = math.acos(max(-1.0, min(1.0, (np.trace(R) - 1) / 2)))
        if angle < 1e-8:
            return np.zeros(3)
        axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * math.sin(angle))
        return axis * angle
    return None


def state_from_trace(trace: list[dict], n: int) -> np.ndarray:
    """LIBERO proprio 8: eef_pos(3) + eef axis-angle(3) + gripper_qpos(2: +w/2, −w/2)."""
    obs_rows = [s.get("observations") or {} for s in trace if s.get("observations")]
    state = np.zeros((n, 8), dtype=np.float32)
    for i in range(min(n, len(obs_rows))):
        o = obs_rows[i]
        pos = o.get("eef_position_m")
        if pos is not None:
            state[i, 0:3] = pos
        aa = _axis_angle_from(o)
        if aa is not None:
            state[i, 3:6] = aa
        elif not _missing_rot_warned[0]:
            print("  [경고] trace 에 EE 회전 키가 없어 state[3:6] 은 0 으로 둡니다:", sorted(k for k in o if "eef" in k))
            _missing_rot_warned[0] = True
        w = float(o.get("gripper_finger_mean_m", 0.0))
        state[i, 6], state[i, 7] = w / 2.0, -w / 2.0
    return state


# ----------------------------------------------------------------- 쓰기
def write_episode(out_dir, name, images, actions, state, meta) -> int:
    n = len(actions)
    if n < 5:
        return 0
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(
        f"{out_dir}/{name}.npz",
        image=np.asarray(images[:n], dtype=np.uint8),
        action=np.asarray(actions[:n], dtype=np.float32),
        state=np.asarray(state[:n], dtype=np.float32),
        instruction=str(meta["instruction"]),
        task_id=np.asarray([meta["task_id"]], dtype=np.int32),
        source=str(meta["source"]),
        init_state_id=np.asarray([meta["init_state_id"]], dtype=np.int32),
        replay_pass=np.asarray([meta.get("replay_pass", 0)], dtype=np.int32),
        seed=np.asarray([meta.get("seed", 0)], dtype=np.int32),
    )
    return n


def _episode_dirs(root: str):
    for d in sorted(glob.glob(f"{root}/task_*_e*")):
        if not os.path.isdir(d):
            continue
        base = os.path.basename(d)
        try:
            task = int(base.split("_")[1]); ep = int(base.split("_e")[-1])
        except ValueError:
            continue
        yield d, task, ep


def _load_result(d: str):
    fs = glob.glob(d + "/**/episode_result.json", recursive=True)
    if not fs:
        return None
    return json.load(open(fs[0])).get("result", {})


def _check_frames(images, n_actions, tag) -> bool:
    extra = len(images) - n_actions
    if extra < 0 or extra > 2:
        print(f"  SKIP {tag}: 프레임 {len(images)} 개, 액션 {n_actions} 개 (정렬 불가)")
        return False
    return True


def build_replay(root: str, out_root: str, replay_pass: int) -> tuple[int, int]:
    written = steps = 0
    for d, task, ep in _episode_dirs(root):
        r = _load_result(d)
        if not r or not r.get("task_success"):
            continue
        video = f"{d}/raw_rgb.mp4"
        npz = f"{REPLAY_SRC}/task_{task:02d}/alltheTask_episode_{ep:03d}.npz"
        if not (os.path.exists(video) and os.path.exists(npz)):
            print(f"  SKIP task {task} e{ep}: 파일 없음"); continue
        try:
            data = np.load(npz, allow_pickle=True)
            actions = np.asarray(data["libero_actions"], dtype=np.float32)
            images = extract_frames(video)
            if not _check_frames(images, len(actions), f"replay task {task} e{ep}"):
                continue
            state = state_from_trace(r.get("stage_trace") or [], len(actions))
            n = write_episode(
                f"{out_root}/task_{task:02d}", f"replay{replay_pass}_e{ep:03d}",
                images, actions, state,
                dict(instruction=str(data["instruction"]), task_id=task, source="replay",
                     init_state_id=ep, replay_pass=replay_pass, seed=int(r.get("seed", 0) or 0)),
            )
            if n:
                written += 1; steps += n
        except Exception as exc:
            print(f"  FAIL replay task {task} e{ep}: {exc!r}")
    return written, steps


def build_rollout(root: str, out_root: str) -> tuple[int, int]:
    written = steps = 0
    for d, task, ep in _episode_dirs(root):
        r = _load_result(d)
        if not r or not r.get("task_success"):
            continue
        video = f"{d}/raw_rgb.mp4"
        log = d + ".log" if os.path.exists(d + ".log") else f"{d}/episode.log"
        if not (os.path.exists(video) and os.path.exists(log)):
            print(f"  SKIP rollout task {task} e{ep}: 파일 없음"); continue
        try:
            raw = parse_rollout_actions(log)
            if raw.shape[0] < 5:
                continue
            actions = raw_to_env(raw)
            images = extract_frames(video)
            if not _check_frames(images, len(actions), f"rollout task {task} e{ep}"):
                continue
            state = state_from_trace(r.get("stage_trace") or [], len(actions))
            n = write_episode(
                f"{out_root}/task_{task:02d}", f"rollout_e{ep:03d}",
                images, actions, state,
                dict(instruction=str(r.get("instruction", "")), task_id=int(r.get("task_id", task)),
                     source="rollout", init_state_id=ep, replay_pass=0, seed=int(r.get("seed", 0) or 0)),
            )
            if n:
                written += 1; steps += n
        except Exception as exc:
            print(f"  FAIL rollout task {task} e{ep}: {exc!r}")
    return written, steps


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--replay-roots", nargs="*", default=["/home/user/LIBERO/experiments/replay_all"],
                    help="1 차, 2 차 … 재생 결과 루트. 순서대로 replay_pass 1, 2, … 로 기록")
    ap.add_argument("--rollout-root", default="/home/user/LIBERO/experiments/zs_faithful_50ep")
    ap.add_argument("--output-root", default="/home/user/LIBERO/experiments/finetune_dataset_v2")
    ap.add_argument("--skip-rollout", action="store_true")
    a = ap.parse_args()

    total_ep = total_steps = 0
    for i, root in enumerate(a.replay_roots, start=1):
        print(f"\n[replay pass {i}] {root}")
        n, s = build_replay(root, a.output_root, i)
        print(f"  {n} episodes, {s} steps"); total_ep += n; total_steps += s
    if not a.skip_rollout:
        print(f"\n[rollout] {a.rollout_root}")
        n, s = build_rollout(a.rollout_root, a.output_root)
        print(f"  {n} episodes, {s} steps"); total_ep += n; total_steps += s

    print(f"\ntotal {total_ep} episodes, {total_steps} steps\n")
    for task in range(10):
        files = glob.glob(f"{a.output_root}/task_{task:02d}/*.npz")
        r1 = len([f for f in files if "/replay1_" in f]); r2 = len([f for f in files if "/replay2_" in f])
        ro = len([f for f in files if "/rollout_" in f])
        print(f"  task_{task:02d}: {len(files):3d}  (replay1 {r1}, replay2 {r2}, rollout {ro})")

    # 집게 규약 자기 검증: 첫 스텝은 열림(−1), 어느 시점엔 닫힘(+1) 이어야 한다
    bad = 0
    for f in glob.glob(f"{a.output_root}/task_*/*.npz")[:200]:
        g = np.load(f)["action"][:, 6]
        if not (g[0] < 0 and (g > 0).any()):
            bad += 1
    print(f"\n집게 규약 검사 (표본 200): 이상 {bad} 개  (0 이어야 함: 시작 −1 열림, 이후 +1 닫힘)")


if __name__ == "__main__":
    main()
