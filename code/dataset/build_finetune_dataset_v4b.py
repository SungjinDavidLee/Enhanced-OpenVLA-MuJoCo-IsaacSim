#!/usr/bin/env python3
"""Isaac 사람 시연 재생 → OpenVLA 미세조정 npz (v4b: 화면 한 스텝 보정, 검사 내장).

    conda activate openvla_libero5090
    python3 build_finetune_dataset_v4b.py --check-only                 # 검사만 (몇 분)
    python3 build_finetune_dataset_v4b.py                             # 검사 통과 시 변환 → finetune_dataset_v4b

입력
  재생       ~/LIBERO/experiments/replay_demo/seed*/task_XX_eYYY/   (성공 판만, 시연당 하나)
  액션       ~/isaac_libero/runs/replay_src_demo/task_XX/demo_episode_YYY.npz 의 libero_actions (no-op 제거본, 환경 규약)
  화면       raw_rgb.mp4 (agentview D405 640×480), 있으면 wrist_rgb.mp4
출력 (--out, 기본 ~/LIBERO/experiments/finetune_dataset_v4)
  task_XX/demoYYY[_repK].npz    image (N,224,224,3) · action (N,7) · state (N,8) · [wrist_image] · 메타
  dataset_report.json / .md     검사 결과 · 태스크별 개수 · 액션 통계 · 학습 · 평가 크롭 설정

이미지 전처리   정사각(짧은 변 기준 가운데) → 면적 --crop-area 가운데 → 224 (BILINEAR)
  기본 0.389:  학습 증강(면적 90 % 무작위 크롭) 뒤 평균 시야 = 0.389 × 0.9 = 0.35 = 평가 서버 crop 0.35 (세로 약 46°)
  증강 없이 학습할 거면 --crop-area 0.35

검사 (하나라도 실패하면 변환하지 않음.  --force 로 무시 가능하지만 권하지 않음)
  C1 집게 규약     액션 집게 값이 {−1, +1} 이고 원본 시연 npz 와 같음
  C2 이미지 방향   변환한 첫 화면을 같은 시연의 LIBERO HDF5 첫 화면(OpenVLA RLDS 방향 = 180° 회전)과 비교.
                   그대로 · 좌우 · 상하 · 180° 네 방향 중 「그대로」 가 가장 닮아야 통과
  C3 정지 프레임   위치 · 회전 6 성분이 모두 0 이고 집게가 직전과 같은 스텝 비율 < 1 %
  C4 정렬          실행된 스텝 수 대비 영상 프레임 수 차이 0–2
  C5 액션 통계     태스크별 q01 · q99 를 원본 LIBERO 시연(같은 시연 목록)과 비교해 기록 (학습 후 체크포인트 통계와 대조용)
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys

import cv2
import numpy as np
from PIL import Image

HOME = os.path.expanduser("~")
REPLAY = f"{HOME}/LIBERO/experiments/replay_demo"
SRC = f"{HOME}/isaac_libero/runs/replay_src_demo"
OUT = f"{HOME}/LIBERO/experiments/finetune_dataset_v4b"
SIZE = 224
IMAGE_OFFSET = 1          # 09-30 확인: 재생 데이터의 화면은 상태 · 액션보다 한 스텝 늦게 찍힘 → 화면 k+1 을 스텝 k 에 씀


# ---------------------------------------------------------------- 전처리
def preprocess(rgb: np.ndarray, area: float, size: int = SIZE) -> np.ndarray:
    img = Image.fromarray(rgb)
    w, h = img.size; s = min(w, h)
    img = img.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
    c = max(1, int(round(s * math.sqrt(area)))); o = (s - c) // 2
    img = img.crop((o, o, o + c, o + c)).resize((size, size), Image.Resampling.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def read_video(path: str) -> list[np.ndarray]:
    cap = cv2.VideoCapture(path); out = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        out.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
    cap.release()
    return out


# ---------------------------------------------------------------- 입력 모으기
def successful_replays(tasks):
    """(task, demo) 마다 성공한 재생 하나 (시드 번호가 작은 것)."""
    pick = {}
    for d in sorted(glob.glob(f"{REPLAY}/seed*/task_*_e*")):
        if not os.path.isdir(d):
            continue
        base = os.path.basename(d)
        try:
            t = int(base.split("_")[1]); e = int(base.split("_e")[-1]); seed = int(d.split("/seed")[-1].split("/")[0])
        except ValueError:
            continue
        if t not in tasks:
            continue
        fs = glob.glob(d + "/**/episode_result.json", recursive=True)
        if not fs:
            continue
        r = json.load(open(fs[0])).get("result", {})
        if not r.get("task_success"):
            continue
        if (t, e) not in pick or seed < pick[(t, e)][0]:
            pick[(t, e)] = (seed, d, r)
    return pick


def state_from_obs(o: dict) -> np.ndarray:
    st = np.zeros(8, np.float32)
    st[0:3] = o.get("eef_position_m") or [0, 0, 0]
    R = o.get("eef_rotation")
    if R is not None:
        R = np.asarray(R, float).reshape(3, 3)
        ang = math.acos(max(-1.0, min(1.0, (np.trace(R) - 1) / 2)))
        if ang > 1e-6:
            ax = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * math.sin(ang))
            st[3:6] = ax * ang
    g = o.get("gripper_joint_positions")
    if g:
        st[6], st[7] = float(g[0]), float(g[1])
    else:
        w = float(o.get("gripper_finger_mean_m") or 0.0); st[6], st[7] = w / 2, -w / 2
    return st


def load_episode(t, e, d, r, area, with_wrist):
    src = np.load(f"{SRC}/task_{t:02d}/demo_episode_{e:03d}.npz", allow_pickle=True)
    acts = np.asarray(src["libero_actions"], np.float32)
    instr = str(src["instruction"]) if "instruction" in src.files else str(r.get("instruction") or "")
    by = {}
    for s in r.get("stage_trace") or []:
        ps = s.get("policy_step")
        if s.get("observations") and isinstance(ps, int) and ps >= 0 and ps not in by:
            by[ps] = s["observations"]
    executed = (max(by) + 1) if by else 0
    vids = glob.glob(d + "/**/raw_rgb.mp4", recursive=True)
    frames = read_video(vids[0]) if vids else []
    # 재생 모드는 스텝 k 동안 로봇을 시연의 k 번째 자세로 옮기고, 화면 k 는 그 스텝 시작 때 찍힌다.
    # 그래서 시연 k 번째 자세의 화면은 화면 k+1 이다 (IMAGE_OFFSET = 1).  상태 기록 k 는 스텝 k 끝에 찍혀 이미 k 번째 자세.
    n = min(executed, len(acts), len(frames) - IMAGE_OFFSET)
    info = {"task": t, "demo": e, "dir": d, "actions_src": len(acts), "executed": executed, "frames": len(frames), "n": n, "image_offset": IMAGE_OFFSET}
    if n < 5:
        return None, info
    images = np.stack([preprocess(frames[i + IMAGE_OFFSET], area) for i in range(n)])
    state = np.stack([state_from_obs(by.get(i) or by[max(k for k in by if k <= i)]) for i in range(n)])
    ep = {"image": images, "action": acts[:n], "state": state, "instruction": instr, "raw_first": frames[0]}
    if with_wrist:
        wv = glob.glob(d + "/**/wrist_rgb.mp4", recursive=True)
        if wv:
            wf = read_video(wv[0])
            if len(wf) >= n + IMAGE_OFFSET:
                ep["wrist_image"] = np.stack([preprocess(wf[i + IMAGE_OFFSET], 1.0) for i in range(n)])
    return ep, info


# ---------------------------------------------------------------- 검사
def check_gripper(ep, src_acts):
    g = ep["action"][:, 6]
    ok_vals = bool(np.all(np.isin(np.round(g, 3), [-1.0, 1.0])))
    same = bool(np.allclose(ep["action"], src_acts[:len(ep["action"])]))
    return ok_vals and same, {"values_pm1": ok_vals, "matches_source": same, "close_ratio": float(np.mean(g > 0)), "first": float(g[0])}


def noop_ratio(a):
    mov = np.all(np.abs(a[:, :6]) < 1e-6, axis=1)
    same_g = np.concatenate([[False], a[1:, 6] == a[:-1, 6]])
    return float(np.mean(mov & same_g))


def orientation_check(ep, t, e):
    """LIBERO HDF5 첫 화면(180° 회전 = OpenVLA RLDS 방향)과 네 방향 비교. 가장 닮은 방향을 돌려준다."""
    try:
        import h5py
        sys.path.insert(0, f"{HOME}/LIBERO")
        from libero.libero import benchmark, get_libero_path
        task = benchmark.get_benchmark_dict()["libero_spatial"]().get_task(t)
        with h5py.File(os.path.join(get_libero_path("datasets"), "libero_spatial", task.name + "_demo.hdf5"), "r") as f:
            ref = np.asarray(f["data"]["demo_%d" % e]["obs"]["agentview_rgb"][0])[::-1, ::-1]
    except Exception as exc:  # noqa: BLE001
        return None, {"error": repr(exc)[:160]}
    def feat(img):
        g = cv2.cvtColor(cv2.resize(np.ascontiguousarray(img), (64, 64), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY).astype(np.float32)
        g = cv2.GaussianBlur(g, (5, 5), 0); g -= g.mean(); return g / (np.linalg.norm(g) + 1e-6)
    r = feat(ref); x = ep["image"][0]
    cands = {"그대로": x, "좌우": x[:, ::-1], "상하": x[::-1], "180°": x[::-1, ::-1]}
    score = {k: float(np.sum(feat(v) * r)) for k, v in cands.items()}
    best = max(score, key=score.get)
    return best, {"scores": {k: round(v, 3) for k, v in score.items()}, "best": best}


def libero_action_stats(tasks):
    """원본 LIBERO 시연 액션(no-op 제거본 npz) 의 q01 · q99 — 비교 기준."""
    allacts = [np.asarray(np.load(f, allow_pickle=True)["libero_actions"], np.float32)
               for t in tasks for f in sorted(glob.glob(f"{SRC}/task_{t:02d}/demo_episode_*.npz"))]
    A = np.concatenate(allacts) if allacts else np.zeros((1, 7))
    return np.quantile(A, 0.01, axis=0), np.quantile(A, 0.99, axis=0)


# ---------------------------------------------------------------- 메인
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--crop-area", type=float, default=0.389)
    ap.add_argument("--oversample", nargs="*", default=["5=3"], help="task=배수 (기본 task 5 를 3 배)")
    ap.add_argument("--no-wrist", action="store_true")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--orient-samples", type=int, default=2, help="태스크당 방향 검사할 시연 수")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--image-offset", type=int, default=1, help="화면 번호 이동 (재생 데이터 1, 0 이면 v4 와 같음)")
    a = ap.parse_args()
    over = {int(k): int(v) for k, v in (x.split("=") for x in a.oversample)}
    global IMAGE_OFFSET
    IMAGE_OFFSET = a.image_offset
    print("화면 번호 이동:", IMAGE_OFFSET, "(스텝 k 에 화면 k+%d 을 씀)" % IMAGE_OFFSET)

    pick = successful_replays(set(a.tasks))
    per_task = {t: sorted(e for (tt, e) in pick if tt == t) for t in a.tasks}
    print("성공 시연 수:", {t: len(v) for t, v in per_task.items()}, " 합계", sum(len(v) for v in per_task.values()))

    report = {"crop_area": a.crop_area, "eval_crop_with_aug": round(a.crop_area * 0.9, 4), "eval_crop_without_aug": a.crop_area,
              "oversample": over, "episodes": [], "checks": {}}
    fails = {"C1": [], "C2": [], "C3": [], "C4": []}
    orient_done = {t: 0 for t in a.tasks}
    episodes = []
    for (t, e), (seed, d, r) in sorted(pick.items()):
        try:
            ep, info = load_episode(t, e, d, r, a.crop_area, not a.no_wrist)
        except Exception as exc:  # noqa: BLE001
            print("  읽기 실패 task %d demo %d: %r" % (t, e, exc)); fails["C4"].append((t, e)); continue
        info["seed"] = seed
        if ep is None:
            fails["C4"].append((t, e)); report["episodes"].append(info); continue
        src_acts = np.asarray(np.load(f"{SRC}/task_{t:02d}/demo_episode_{e:03d}.npz", allow_pickle=True)["libero_actions"], np.float32)
        ok1, g = check_gripper(ep, src_acts); info["gripper"] = g
        if not ok1:
            fails["C1"].append((t, e))
        info["noop_ratio"] = noop_ratio(ep["action"])
        if info["noop_ratio"] >= 0.01:
            fails["C3"].append((t, e))
        if not (IMAGE_OFFSET <= info["frames"] - info["n"] <= 2 + IMAGE_OFFSET or info["executed"] < info["actions_src"]):
            fails["C4"].append((t, e))
        if orient_done[t] < a.orient_samples:
            best, o = orientation_check(ep, t, e); info["orientation"] = o; orient_done[t] += 1
            if best is not None and best != "그대로":
                fails["C2"].append((t, e))
            print("  방향 검사 task %d demo %d: %s" % (t, e, o.get("scores") or o))
        report["episodes"].append(info); episodes.append((t, e, ep))

    q01, q99 = libero_action_stats(a.tasks)
    A = np.concatenate([ep["action"] for _, _, ep in episodes]) if episodes else np.zeros((1, 7))
    report["checks"] = {
        "C1_gripper_fail": fails["C1"], "C2_orientation_fail": fails["C2"], "C3_noop_fail": fails["C3"], "C4_alignment_fail": fails["C4"],
        "C5_action_q01": np.quantile(A, 0.01, axis=0).round(4).tolist(), "C5_action_q99": np.quantile(A, 0.99, axis=0).round(4).tolist(),
        "C5_libero_q01": q01.round(4).tolist(), "C5_libero_q99": q99.round(4).tolist(),
        "noop_ratio_overall": float(np.mean([i.get("noop_ratio", 0) for i in report["episodes"] if "noop_ratio" in i] or [0])),
    }
    orient_err = [i for i in report["episodes"] if "orientation" in i and "error" in i["orientation"]]
    orient_ok = [i for i in report["episodes"] if "orientation" in i and "best" in i["orientation"]]
    if orient_err and not orient_ok:
        print("\n[경고] C2 이미지 방향 검사를 하지 못했습니다 (LIBERO HDF5 · h5py 확인):", orient_err[0]["orientation"]["error"])
        print("       방향을 확인하지 않은 채 변환하려면 --force.  권하지 않습니다.")
        fails["C2"].append(("검사 불가", -1))
    hard = fails["C1"] or fails["C2"] or fails["C3"]
    print("\n검사 결과")
    print("  C1 집게 규약     실패 %d 판" % len(fails["C1"]))
    print("  C2 이미지 방향   실패 %d 판  (그대로가 가장 닮아야 함)" % len(fails["C2"]))
    print("  C3 정지 프레임   1 %% 넘는 판 %d,  전체 비율 %.2f %%" % (len(fails["C3"]), 100 * report["checks"]["noop_ratio_overall"]))
    print("  C4 정렬 · 읽기   제외 %d 판" % len(fails["C4"]))
    print("  C5 액션 q01  재생 %s\n              원본 %s" % (report["checks"]["C5_action_q01"], report["checks"]["C5_libero_q01"]))
    print("     액션 q99  재생 %s\n              원본 %s" % (report["checks"]["C5_action_q99"], report["checks"]["C5_libero_q99"]))

    os.makedirs(a.out, exist_ok=True)
    counts = {}
    if not a.check_only and (not hard or a.force):
        for t, e, ep in episodes:
            reps = over.get(t, 1)
            for k in range(reps):
                name = "demo%03d" % e + ("" if reps == 1 else "_rep%d" % k)
                os.makedirs(f"{a.out}/task_{t:02d}", exist_ok=True)
                extra = {"wrist_image": ep["wrist_image"]} if "wrist_image" in ep else {}
                np.savez_compressed(f"{a.out}/task_{t:02d}/{name}.npz", image=ep["image"], action=ep["action"], state=ep["state"],
                                    instruction=ep["instruction"], task_id=np.asarray([t], np.int32), source="isaac_replay_human_demo",
                                    init_state_id=np.asarray([e], np.int32), crop_area=np.asarray([a.crop_area], np.float32), **extra)
                counts[t] = counts.get(t, 0) + 1
        print("\n변환 완료:", a.out, " 태스크별 파일 수", counts, " 합계", sum(counts.values()))
    elif hard and not a.force:
        print("\n검사 실패가 있어 변환하지 않았습니다.  dataset_report.md 를 확인하세요.")
    report["written_per_task"] = counts
    json.dump(report, open(f"{a.out}/dataset_report.json", "w"), indent=1, ensure_ascii=False, default=str)
    md = ["# 학습셋 v4b 보고", "", "- 화면 번호 이동 %d (재생 모드는 스텝 k 동안 시연 k 번째 자세로 이동, 화면 k 는 스텝 시작 때 찍힘)" % IMAGE_OFFSET, "- 크롭 면적 %.3f → 학습 증강(0.9) 사용 시 평가 서버 crop **%.3f**, 증강 없이 학습 시 **%.3f**" % (a.crop_area, a.crop_area * 0.9, a.crop_area),
          "- 성공 시연: " + ", ".join("T%d %d" % (t, len(v)) for t, v in per_task.items()), "- 반복: %s" % over,
          "- 검사: C1 %d · C2 %d · C3 %d · C4 %d 실패" % tuple(len(fails[k]) for k in ("C1", "C2", "C3", "C4")),
          "- 액션 q01 재생 %s / 원본 %s" % (report["checks"]["C5_action_q01"], report["checks"]["C5_libero_q01"]),
          "- 액션 q99 재생 %s / 원본 %s" % (report["checks"]["C5_action_q99"], report["checks"]["C5_libero_q99"]),
          "", "## 학습 후 확인할 것", "",
          "1. 체크포인트의 dataset_statistics.json 에서 이 데이터셋의 q01 · q99 가 위 「재생」 값과 같은지.",
          "2. 평가 서버의 역정규화 키(unnorm_key)가 이 데이터셋 이름인지 (libero_spatial 이 아님).",
          "3. 데이터 변환(RLDS) 설정이 LIBERO 규약(집게 −1/+1 → [0,1] 로 자르고 뒤집기)을 쓰는지."]
    open(f"{a.out}/dataset_report.md", "w").write("\n".join(md) + "\n")
    print("보고서:", f"{a.out}/dataset_report.md")
    return 1 if (hard and not a.force) else 0


if __name__ == "__main__":
    raise SystemExit(main())
