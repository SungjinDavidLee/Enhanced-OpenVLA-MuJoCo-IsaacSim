#!/usr/bin/env python3
"""캡처 폴더의 agentview 와 손목 화면을 OpenCV 창으로 나란히 보여 준다.

    python3 view_wrist_capture.py ~/LIBERO/experiments/wrist_test/e0            # 첫/마지막 프레임 정지 화면
    python3 view_wrist_capture.py ~/LIBERO/experiments/wrist_test/e0 --play     # 두 영상을 같이 재생 (q 로 종료, space 일시정지)

확인할 것: 손목 화면에 두 손가락이 아래쪽에 보이고, 접근할 때 그릇 테두리가 손가락 사이로 들어오는지.
"""

import argparse
import os
import sys

import cv2
import numpy as np


def side_by_side(a, b, h=360):
    def fit(img):
        if img is None:
            return np.zeros((h, int(h * 4 / 3), 3), np.uint8)
        s = h / img.shape[0]
        return cv2.resize(img, (int(img.shape[1] * s), h))
    return np.concatenate([fit(a), fit(b)], axis=1)


def wait_close(win: str) -> None:
    """아무 키 · 창 닫기 · Ctrl+C 모두로 끝난다 (waitKey(0) 는 Ctrl+C 를 못 받음)."""
    try:
        n = 0
        while True:
            n += 1
            if cv2.waitKey(100) & 0xFF != 255:
                break
            if n > 5 and cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                break
    except KeyboardInterrupt:
        pass
    cv2.destroyAllWindows()


def label(img, text):
    cv2.putText(img, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return img


def _frames_at(path: str, fracs):
    cap = cv2.VideoCapture(path); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); out = []
    for f in fracs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, min(n - 1, int(round(f * (n - 1))))); ok, img = cap.read(); out.append(img if ok else None)
    return out


def save_grid(d1: str, d2, out: str, n: int) -> int:
    """영상에서 n 개 시점을 뽑아 [agentview | wrist] (× 두 폴더) 를 세로로 쌓은 PNG 로 저장."""
    fracs = [i / max(1, n - 1) for i in range(n)]
    cols = [os.path.join(d1, "raw_rgb.mp4"), os.path.join(d1, "wrist_rgb.mp4")]
    if d2:
        cols += [os.path.join(d2, "raw_rgb.mp4"), os.path.join(d2, "wrist_rgb.mp4")]
    seqs = [_frames_at(c, fracs) for c in cols]
    rows = []
    for i, f in enumerate(fracs):
        row = side_by_side(seqs[0][i], seqs[1][i], h=240)
        if d2:
            row = np.concatenate([row, side_by_side(seqs[2][i], seqs[3][i], h=240)], axis=1)
        rows.append(label(row, "%d%%   [%s] agentview | wrist%s" % (int(f * 100), os.path.basename(d1), ("   ||   [%s] agentview | wrist" % os.path.basename(d2)) if d2 else "")))
    out = os.path.expanduser(out); cv2.imwrite(out, np.concatenate(rows, axis=0)); print("저장:", out); return 0


def compare(d1: str, d2: str, play: bool) -> int:
    """두 폴더(예: Isaac / MuJoCo)의 agentview 와 손목을 2x2 로 보여 준다."""
    names = ("raw_rgb", "wrist_rgb")
    if not play:
        rows = []
        for tag in ("first", "last"):
            imgs = [cv2.imread(os.path.join(dd, "%s_%s.png" % (n, tag))) for dd in (d1, d2) for n in names]
            rows.append(label(side_by_side(side_by_side(imgs[0], imgs[1]), side_by_side(imgs[2], imgs[3])), "%s:  [%s] agentview | wrist   ||   [%s] agentview | wrist" % (tag, os.path.basename(d1), os.path.basename(d2))))
        cv2.imshow("compare (any key to close)", np.concatenate(rows, axis=0)); wait_close("compare (any key to close)"); return 0
    caps = [cv2.VideoCapture(os.path.join(dd, n + ".mp4")) for dd in (d1, d2) for n in names]
    n = int(min(c.get(cv2.CAP_PROP_FRAME_COUNT) for c in caps)); i = 0; paused = False
    print("프레임 수:", [int(c.get(cv2.CAP_PROP_FRAME_COUNT)) for c in caps])
    while True:
        if not paused:
            fr = [c.read() for c in caps]
            if not all(ok for ok, _ in fr):
                break
            i += 1
            f = [x for _, x in fr]
            cv2.imshow("compare", label(side_by_side(side_by_side(f[0], f[1]), side_by_side(f[2], f[3])), "step %d/%d  [%s] | [%s]   (space pause, q quit)" % (i, n, os.path.basename(d1), os.path.basename(d2))))
        try:
            k = cv2.waitKey(80) & 0xFF
        except KeyboardInterrupt:
            break
        if k == ord("q") or (i > 4 and cv2.getWindowProperty("compare", cv2.WND_PROP_VISIBLE) < 1):
            break
        if k == ord(" "):
            paused = not paused
    cv2.destroyAllWindows(); return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("capture_dir")
    ap.add_argument("--play", action="store_true")
    ap.add_argument("--compare", default=None, help="비교할 두 번째 캡처 폴더 (예: MuJoCo 렌더).  손목끼리 · agentview 끼리 나란히")
    ap.add_argument("--save", default=None, help="창 대신 PNG 로 저장 (예: ~/wrist_compare.png).  --grid 와 함께 쓰면 여러 시점")
    ap.add_argument("--grid", type=int, default=6, help="--save 때 영상에서 고르게 뽑을 시점 수")
    a = ap.parse_args()
    d = a.capture_dir
    if a.save:
        return save_grid(d, a.compare, a.save, a.grid)
    if a.compare:
        return compare(d, a.compare, a.play)
    files = {k: os.path.join(d, k) for k in ("raw_rgb.mp4", "wrist_rgb.mp4", "raw_rgb_first.png", "wrist_rgb_first.png", "raw_rgb_last.png", "wrist_rgb_last.png")}
    have = {k: os.path.exists(v) for k, v in files.items()}
    print("파일:", "  ".join("%s=%s" % (k, "O" if v else "x") for k, v in have.items()))
    if not have["wrist_rgb.mp4"] and not have["wrist_rgb_first.png"]:
        print("손목 캡처가 없습니다. yaml 의 wrist_camera.enabled 와 패치 적용을 확인하세요."); return 1

    if not a.play:
        rows = []
        for tag in ("first", "last"):
            av = cv2.imread(files["raw_rgb_%s.png" % tag]); wr = cv2.imread(files["wrist_rgb_%s.png" % tag])
            rows.append(label(side_by_side(av, wr), "agentview %s | wrist %s" % (tag, tag)))
        img = np.concatenate(rows, axis=0)
        cv2.imshow("wrist check (any key to close)", img); wait_close("wrist check (any key to close)")
        return 0

    ca, cw = cv2.VideoCapture(files["raw_rgb.mp4"]), cv2.VideoCapture(files["wrist_rgb.mp4"])
    n = int(min(ca.get(cv2.CAP_PROP_FRAME_COUNT), cw.get(cv2.CAP_PROP_FRAME_COUNT)))
    print("프레임 수: agentview %d  wrist %d" % (ca.get(cv2.CAP_PROP_FRAME_COUNT), cw.get(cv2.CAP_PROP_FRAME_COUNT)))
    i = 0; paused = False
    while True:
        if not paused:
            oa, fa = ca.read(); ow, fw = cw.read()
            if not (oa and ow):
                break
            i += 1
            frame = label(side_by_side(fa, fw), "step %d / %d   (space: pause, q: quit)" % (i, n))
            cv2.imshow("wrist check", frame)
        try:
            k = cv2.waitKey(60) & 0xFF
        except KeyboardInterrupt:
            break
        if k == ord("q") or (i > 4 and cv2.getWindowProperty("wrist check", cv2.WND_PROP_VISIBLE) < 1):
            break
        if k == ord(" "):
            paused = not paused
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
