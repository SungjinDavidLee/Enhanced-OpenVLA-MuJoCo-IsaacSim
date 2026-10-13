"""판(에피소드) 폴더 찾기 — 두 구조를 모두 읽는다. 모든 분석 · 평가 도구가 이 파일 하나를 같이 쓴다.

  새 구조 (2026-10-08 부터):   <결과 폴더>/task_XX/eYYY/        판 로그는 <결과 폴더>/task_XX/eYYY.log
  옛 구조 (2026-09-24 ~ 10-08): <결과 폴더>/task_XX_eYYY/        판 로그는 <결과 폴더>/task_XX_eYYY.log

같은 판이 두 구조에 다 있으면 새 구조를 쓴다. 판 번호는 표 · CSV · 영상 이름에서 어디서나 「task_XX_eYYY」로 부른다.
"""
import glob
import os
import re

_ID = re.compile(r"task_(\d\d)(?:_e|/e)(\d\d\d)")


def ep_id(t, e):
    return f"task_{int(t):02d}_e{int(e):03d}"


def parse(s):
    """경로나 이름에서 (과제, 판) 을 읽는다. 못 읽으면 None."""
    m = _ID.search(str(s).replace(os.sep, "/"))
    return (int(m.group(1)), int(m.group(2))) if m else None


def new_dir(base, t, e):
    return os.path.join(base, f"task_{int(t):02d}", f"e{int(e):03d}")


def old_dir(base, t, e):
    return os.path.join(base, ep_id(t, e))


def ep_dir(base, t, e):
    """있는 판 폴더 (새 구조 먼저). 없으면 None."""
    for d in (new_dir(base, t, e), old_dir(base, t, e)):
        if os.path.isdir(d):
            return d
    return None


def ep_dirs(base):
    """결과 폴더 안의 모든 판 폴더를 (과제, 판, 폴더) 로, 과제 · 판 순서대로."""
    found = {}
    for d in sorted(glob.glob(os.path.join(base, "task_??", "e???"))) + sorted(glob.glob(os.path.join(base, "task_??_e???"))):
        k = parse(d)
        if k and os.path.isdir(d):
            found.setdefault(k, d)
    return [(t, e, found[(t, e)]) for (t, e) in sorted(found)]


def result_path(d):
    fs = glob.glob(os.path.join(d, "**", "episode_result.json"), recursive=True) if d else []
    return fs[0] if fs else None


def ep_results(base):
    """판 기록이 있는 판만 (과제, 판, episode_result.json 경로) 로."""
    out = []
    for t, e, d in ep_dirs(base):
        p = result_path(d)
        if p:
            out.append((t, e, p))
    return out


def video(base, t, e, name="raw_rgb.mp4"):
    d = ep_dir(base, t, e)
    fs = glob.glob(os.path.join(d, "**", name), recursive=True) if d else []
    return fs[0] if fs else None


def log_path(base, t, e):
    for p in (new_dir(base, t, e) + ".log", old_dir(base, t, e) + ".log"):
        if os.path.exists(p):
            return p
    return None
