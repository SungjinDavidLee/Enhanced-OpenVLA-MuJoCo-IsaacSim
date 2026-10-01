#!/usr/bin/env python3
"""학습셋 v4 를 학습 전에 한 번 더 검증한다 — 특히 화면 · 상태 · 액션이 한 스텝씩 밀리지 않았는지.

    conda activate openvla_libero5090
    python3 verify_dataset_v4.py                       # ~/LIBERO/experiments/finetune_dataset_v4

검사
  V1 파일        모든 npz 가 열리고, image (N,224,224,3) uint8 · action (N,7) · state (N,8) 길이가 같고, 값이 유한
  V2 지시문       태스크마다 지시문이 하나이고 LIBERO 태스크 문장과 같음
  V3 집게 값      액션 집게가 −1 / +1 만
  V4 멈춘 화면    연속 두 화면이 사실상 같은 비율 (영상 멈춤 · 중복 기록 확인)
  V5 화면–액션      화면 i → i+1 변화량과 액션 크기 · 손끝 속도의 지연별 상관을, 같은 시연의 원본 LIBERO(MuJoCo)
                     화면 · 액션에서 똑같이 계산한 곡선과 비교.  곡선이 평평하므로 칸이 아니라 포물선 봉우리 위치로 비교:
                     V5b(화면 ↔ 자기 상태) 차이 ≤ 0.5 스텝, V5(화면 ↔ 액션) 차이 − 상태 추종 지연 ≤ 0.5 스텝
  V6 상태–MuJoCo    시작점을 맞춘 뒤 Isaac 손끝 k 번째와 MuJoCo 손끝 k+L 번째의 거리.  L = 0 에서 가장 작아야 함
  V7 집게 시점     닫기 명령 첫 스텝 대비 손가락이 실제로 닫히기 시작한 스텝 (0–3 이 정상, 음수면 밀림)
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from collections import defaultdict

import numpy as np

HOME = os.path.expanduser("~")
ROOT = f"{HOME}/LIBERO/experiments/finetune_dataset_v4"
SRC = f"{HOME}/isaac_libero/runs/replay_src_demo"
LAGS = (-2, -1, 0, 1, 2)


def lag_corr(x, y, lag):
    """x_i 와 y_{i+lag} 의 상관."""
    if lag >= 0:
        a, b = x[: len(x) - lag], y[lag: lag + len(x) - lag]
    else:
        a, b = x[-lag:], y[: len(y) + lag]
    n = min(len(a), len(b)); a, b = a[:n], b[:n]
    if n < 8 or np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def mujoco_reference(pairs, per_task=3):
    """같은 시연의 LIBERO HDF5 (MuJoCo) 에서 화면 변화 – 액션 · 화면 변화 – 손끝 속도 지연 곡선."""
    import h5py
    sys.path.insert(0, f"{HOME}/LIBERO")
    from libero.libero import benchmark, get_libero_path
    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    ref_a, ref_s, used = defaultdict(list), defaultdict(list), 0
    by_task = defaultdict(list)
    for t, e in sorted(pairs):
        by_task[t].append(e)
    for t, eps in by_task.items():
        path = os.path.join(get_libero_path("datasets"), "libero_spatial", suite.get_task(t).name + "_demo.hdf5")
        with h5py.File(path, "r") as f:
            for e in eps[:per_task]:
                g = f["data"]["demo_%d" % e]; A = np.asarray(g["actions"], np.float32)
                img = np.asarray(g["obs"]["agentview_rgb"]).astype(np.float32)
                ob = g["obs"]; pos = np.asarray(ob["ee_pos"] if "ee_pos" in ob else ob["ee_states"][:, :3], float)
                keep, prev = [], None                       # OpenVLA 방식 no-op 제거 (위치 · 회전 노름 < 1e-4 이고 집게 같음)
                for i in range(len(A)):
                    if prev is not None and np.linalg.norm(A[i, :6]) < 1e-4 and A[i, 6] == prev:
                        continue
                    keep.append(i); prev = A[i, 6]
                A, img, pos = A[keep], img[keep], pos[keep]
                d = np.abs(img[1:] - img[:-1]).mean(axis=(1, 2, 3)); mag = np.linalg.norm(A[:, :3], axis=1)
                sp = np.linalg.norm(np.diff(pos, axis=0), axis=1)
                for L in LAGS:
                    ref_a[L].append(lag_corr(d, mag, L)); ref_s[L].append(lag_corr(d, sp, L))
                used += 1
    return {L: float(np.nanmean(v)) for L, v in ref_a.items()}, {L: float(np.nanmean(v)) for L, v in ref_s.items()}, used


def vertex(curve, find_max=True):
    """지연별 값에서 봉우리(또는 골) 위치를 칸 사이까지 추정 (가장 높은 칸과 양옆으로 포물선)."""
    Ls = sorted(curve); v = np.array([curve[L] for L in Ls], float)
    if not find_max:
        v = -v
    i = int(np.nanargmax(v))
    if i == 0 or i == len(v) - 1:
        return float(Ls[i])
    a, b, c = v[i - 1], v[i], v[i + 1]; den = a - 2 * b + c
    return float(Ls[i]) + (0.5 * (a - c) / den if abs(den) > 1e-12 else 0.0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--max-per-task", type=int, default=0, help="태스크당 검사할 파일 수 (0 = 전부)")
    a = ap.parse_args()
    files = sorted(glob.glob(f"{a.root}/task_*/*.npz"))
    if not files:
        print("파일 없음:", a.root); return 1
    try:
        sys.path.insert(0, f"{HOME}/LIBERO")
        from libero.libero import benchmark
        suite = benchmark.get_benchmark_dict()["libero_spatial"]()
        lang = {t: suite.get_task(t).language.strip().lower() for t in range(10)}
    except Exception as exc:  # noqa: BLE001
        lang = {}; print("(LIBERO 태스크 문장을 못 읽음: %r — V2 는 지시문 일관성만 봄)" % (exc,))

    bad = defaultdict(list); instr = defaultdict(set); frozen = []; lag_img = defaultdict(list); lag_st = defaultdict(list); grip_delay = []
    lag_imgst = defaultdict(list); pos_err = defaultdict(list)
    per_task = defaultdict(int); seen_src = set()
    for f in files:
        t = int(f.split("task_")[1][:2])
        if a.max_per_task and per_task[t] >= a.max_per_task:
            continue
        per_task[t] += 1
        try:
            d = np.load(f, allow_pickle=True)
            img, act, st = d["image"], d["action"], d["state"]
        except Exception as exc:  # noqa: BLE001
            bad["V1"].append((f, repr(exc)[:80])); continue
        n = len(act)
        ok = (img.ndim == 4 and img.shape[1:] == (224, 224, 3) and img.dtype == np.uint8 and len(img) == n and st.shape == (n, 8)
              and act.shape == (n, 7) and np.isfinite(act).all() and np.isfinite(st).all() and n >= 10)
        if "wrist_image" in d.files:
            ok = ok and len(d["wrist_image"]) == n
        if not ok:
            bad["V1"].append((f, "모양 %s %s %s %s" % (img.shape, img.dtype, act.shape, st.shape))); continue
        instr[t].add(str(d["instruction"]).strip().lower())
        if not np.all(np.isin(np.round(act[:, 6], 3), [-1.0, 1.0])):
            bad["V3"].append(f)
        im = img.astype(np.float32)
        diff = np.abs(im[1:] - im[:-1]).mean(axis=(1, 2, 3))
        frozen.append(float(np.mean(diff < 0.5)))
        mag = np.linalg.norm(act[:, :3], axis=1)
        spd = np.linalg.norm(np.diff(st[:, :3], axis=0), axis=1)
        for L in LAGS:
            lag_img[L].append(lag_corr(diff, mag, L)); lag_imgst[L].append(lag_corr(diff, spd, L))
        # 상태–MuJoCo 정렬 (같은 원본 시연은 한 번만)
        e = int(d["init_state_id"][0]) if "init_state_id" in d.files else None
        key = (t, e)
        if e is not None and key not in seen_src and os.path.exists(f"{SRC}/task_{t:02d}/demo_episode_{e:03d}.npz"):
            seen_src.add(key)
            pm = np.asarray(np.load(f"{SRC}/task_{t:02d}/demo_episode_{e:03d}.npz", allow_pickle=True)["eef_pos"], float)
            vi = np.diff(st[:, :3], axis=0); vm = np.diff(pm[: n + 3], axis=0)
            for L in LAGS:                                   # 위치 기준: 시작점을 맞춘 뒤 k 번째 Isaac 과 k+L 번째 MuJoCo 의 거리
                ks = [k for k in range(max(0, -L), min(n, len(pm) - max(L, 0)))]
                if len(ks) > 8:
                    di = (st[ks, :3] - st[ks[0], :3]) - (pm[[k + L for k in ks]] - pm[ks[0] + L])
                    pos_err[L].append(float(np.median(np.linalg.norm(di, axis=1))) * 1000)
            for L in LAGS:
                cs = []
                for c in range(3):
                    cs.append(lag_corr(vi[:, c], vm[:, c], L))
                lag_st[L].append(np.nanmean(cs))
        # 집게 시점
        k0 = next((k for k in range(n) if act[k, 6] > 0), None)
        if k0 is not None:
            wdt = st[:, 6] - st[:, 7]
            k1 = next((k for k in range(max(0, k0 - 5), n) if wdt[k] < wdt[max(0, k0 - 1)] - 0.002), None)
            if k1 is not None:
                grip_delay.append(k1 - k0)

    print("검사한 파일 %d 개 (태스크별 %s)\n" % (sum(per_task.values()), dict(sorted(per_task.items()))))
    print("V1 파일          문제 %d 개%s" % (len(bad["V1"]), ("  예: %s" % (bad["V1"][:2],)) if bad["V1"] else ""))
    v2 = []
    for t in sorted(instr):
        if len(instr[t]) != 1 or (lang and lang.get(t) and next(iter(instr[t])) != lang[t]):
            v2.append((t, sorted(instr[t])[:2], lang.get(t)))
    print("V2 지시문         문제 태스크 %d 개%s" % (len(v2), ("  %s" % v2[:2]) if v2 else ""))
    print("V3 집게 값        문제 %d 개" % len(bad["V3"]))
    fz = np.array(frozen)
    print("V4 멈춘 화면      판당 중앙 %.1f %%, 가장 많은 판 %.1f %%  (시연의 정지 구간이 있어 0 은 아님, 30 %% 넘는 판 %d 개)" % (
        100 * np.median(fz), 100 * fz.max(), int(np.sum(fz > 0.3))))
    mi = {L: float(np.nanmean(v)) for L, v in lag_img.items()}; bi = max(mi, key=lambda L: mi[L])
    mis = {L: float(np.nanmean(v)) for L, v in lag_imgst.items()}; bis = max(mis, key=lambda L: mis[L])
    try:
        ra, rs, used = mujoco_reference(seen_src)
        rba, rbs = max(ra, key=lambda L: ra[L]), max(rs, key=lambda L: rs[L])
    except Exception as exc:  # noqa: BLE001
        ra = rs = None; used = 0; print("(MuJoCo 기준 곡선을 못 만듦: %r)" % (exc,))
    fmt = lambda m: "  ".join("%+d:%.3f" % (L, m[L]) for L in LAGS)
    print("V5 화면–액션      Isaac   %s  → 봉우리 %+d" % (fmt(mi), bi))
    if ra:
        print("                 MuJoCo  %s  → 봉우리 %+d   (같은 시연 %d 개)" % (fmt(ra), rba, used))
    print("V5b 화면–손끝속도 Isaac   %s  → 봉우리 %+d" % (fmt(mis), bis))
    if rs:
        print("                 MuJoCo  %s  → 봉우리 %+d" % (fmt(rs), rbs))
    v5_ok = False
    if ra is not None:
        pe_tmp = {L: float(np.median(v)) for L, v in pos_err.items()} if pos_err else None
        track = vertex(pe_tmp, find_max=False) if pe_tmp else 0.0            # Isaac 상태가 MuJoCo 를 따라가는 지연 (음수 = 늦음)
        d5b = vertex(mis) - vertex(rs); d5 = vertex(mi) - vertex(ra)
        print("   봉우리 위치 (칸 사이까지):  V5b Isaac %+.2f · MuJoCo %+.2f → 차이 %+.2f   |   V5 Isaac %+.2f · MuJoCo %+.2f → 차이 %+.2f   |   상태 추종 지연 %+.2f" % (
            vertex(mis), vertex(rs), d5b, vertex(mi), vertex(ra), d5, track))
        v5_ok = abs(d5b) <= 0.5 and abs(d5 - track) <= 0.5
        print("   → %s" % ("화면 · 상태 · 액션의 짝이 MuJoCo 학습 데이터와 같은 관계 (V5b 차이 0.5 스텝 이내, V5 차이는 상태 추종 지연으로 설명됨)" if v5_ok
                          else "봉우리 위치 차이가 0.5 스텝을 넘음 ← 밀림 의심.  0.5 스텝 넘게 음수면 화면이 늦고, 양수면 화면이 빠름"))
    else:
        print("   → MuJoCo 기준 없이 판정 불가")
    if pos_err:
        pe = {L: float(np.median(v)) for L, v in pos_err.items()}; bs = min(pe, key=lambda L: pe[L])
        print("V6 상태–MuJoCo    위치 오차 중앙값 (mm)  %s  → 가장 작은 지연 %+d %s  (원본 시연 %d 개)" % (
            "  ".join("%+d:%.1f" % (L, pe[L]) for L in LAGS), bs, "정상" if bs == 0 else "← 밀림 의심", len(seen_src)))
    else:
        bs = 0; print("V6 상태–MuJoCo 원본 시연을 못 찾아 건너뜀")
    if grip_delay:
        g = np.array(grip_delay)
        print("V7 집게 시점      닫기 명령 → 손가락 닫히기 시작  중앙 %+d 스텝, 범위 %+d ~ %+d, 음수 %d 판" % (
            int(np.median(g)), g.min(), g.max(), int(np.sum(g < 0))))
    fail = bad["V1"] or v2 or bad["V3"] or (not v5_ok) or bs != 0 or (grip_delay and np.median(grip_delay) < 0)
    print("\n판정:", "문제 없음 — 학습에 써도 됨" if not fail else "문제 있음 — 위 줄을 확인")
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
