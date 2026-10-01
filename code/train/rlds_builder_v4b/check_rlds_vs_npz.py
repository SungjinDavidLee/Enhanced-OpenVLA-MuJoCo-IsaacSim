#!/usr/bin/env python3
"""빌드한 RLDS 를 npz 와 대조한다 (에피소드 수 · 스텝 수 · 앞쪽 에피소드의 이미지 · 액션 · 지시문).

    python3 check_rlds_vs_npz.py --data-dir ~/tensorflow_datasets_v4b --npz-root ~/datasets/finetune_dataset_v4b
"""
import argparse, glob, os
import numpy as np
import tensorflow_datasets as tfds

ap = argparse.ArgumentParser()
ap.add_argument("--data-dir", default=os.path.expanduser("~/tensorflow_datasets_v4b"))
ap.add_argument("--npz-root", default=os.path.expanduser("~/datasets/finetune_dataset_v4b"))
ap.add_argument("--n-check", type=int, default=5)
a = ap.parse_args()
paths = sorted(glob.glob(f"{a.npz_root}/task_*/*.npz"))
npz_steps = sum(len(np.load(p, allow_pickle=True)["action"]) for p in paths)
b = tfds.builder("isaac_libero_spatial", data_dir=a.data_dir)
ds = b.as_dataset(split="train")
n_ep, n_st, bad, checked = 0, 0, [], 0
for ep in tfds.as_numpy(ds):
    n_ep += 1
    steps = list(ep["steps"]); n_st += len(steps)
    if checked < a.n_check:
        key = ep["episode_metadata"]["file_path"].decode()
        d = np.load(os.path.join(a.npz_root, key), allow_pickle=True)
        img = np.stack([s["observation"]["image"] for s in steps]); act = np.stack([s["action"] for s in steps])
        lang = steps[0]["language_instruction"].decode()
        ok = (img.shape == d["image"].shape and np.array_equal(img, d["image"]) and np.allclose(act, d["action"]) and lang == str(d["instruction"])
              and steps[0]["is_first"] and steps[-1]["is_last"])
        print("  %-26s 스텝 %3d  이미지 %s  액션 %s  지시문 %s  %s" % (key, len(steps), "같음" if np.array_equal(img, d["image"]) else "다름",
              "같음" if np.allclose(act, d["action"]) else "다름", "같음" if lang == str(d["instruction"]) else "다름", "OK" if ok else "← 문제"))
        if not ok:
            bad.append(key)
        checked += 1
print("RLDS 에피소드 %d (npz %d) · 스텝 %d (npz %d)" % (n_ep, len(paths), n_st, npz_steps))
print("판정:", "문제 없음" if (not bad and n_ep == len(paths) and n_st == npz_steps) else "문제 있음")
