"""판 기록(episode_result.json)에서 손가락 · 그릇 신호를 읽는 공통 규칙. Isaac Sim 과 MuJoCo 모두에 쓴다.

table1_metrics.py · compare_runs.py · analyze_eight.py · recheck_classes.py · calibrate_grasp_rule.py · strict_outcomes.py 가
이 파일 하나를 같이 쓴다. 규칙을 바꿀 때는 여기(또는 같은 폴더의 grasp_rule.json)만 고친다.

판 4 (2026-10-07)
  - 「집게를 닫았다」: 손가락 폭이 한 번이라도 35 mm 아래로 내려감 (두 시뮬레이터 같음). 열었을 때의 폭은 둘 다 약 40 mm
    (MuJoCo 는 「집게 열기」 10스텝 뒤 38.7 mm, LIBERO 사람 시범 최대 39.7 mm, Isaac Sim 40 mm). 판마다 가장 많이 닫힌 폭의
    분포에서 35–38 mm 가 모든 칸에서 비어 있어(0–4판) 그 사이에 둠. 20–35 mm 는 닫다가 만 판(시도는 함).
  - 실패 종류에 「집게를 닫지 않음」을 더함 (판 내내 35 mm 아래로 내려가지 않음).
  - 「그릇을 물음」: 손가락 폭이 범위 안에서 3스텝 연속 + (멈춤 기준이 있으면) 그 3스텝 동안 폭 변화가 기준 미만.
    범위 · 멈춤 기준은 grasp_rule.json (calibrate_grasp_rule.py 가 정답을 아는 판으로 채점해 고른 값). 시뮬레이터마다 다른 것은
    그릇을 물었을 때 손가락이 멈추는 폭(MuJoCo 약 2 mm, Isaac Sim 2.5–12 mm)과 집게가 닫히는 빠르기다.
  - 기준점: MuJoCo 는 시작 뒤 10스텝 동안 물체가 자리 잡기를 기다리므로 그 뒤를 기준으로 삼는다. Isaac Sim 은 첫 기록.
  - undisturbed: 목표가 아닌 물체의 실제 위치만 쓴다 (「물체에서 손까지의 거리」 항목 *_to_robot0_eef 는 뺀다).
"""
import json
import os
import numpy as np

CLOSED_BELOW = 0.035
RULE = {"isaac": {"held": [0.0025, 0.012], "stall": None}, "mujoco": {"held": [0.0010, 0.006], "stall": None}}
_rule_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "grasp_rule.json")
if os.path.exists(_rule_file):
    for _k, _v in json.load(open(_rule_file)).items():
        if _k in RULE:
            RULE[_k].update({kk: vv for kk, vv in _v.items() if kk in ("held", "stall")})
for _k in RULE:
    RULE[_k]["closed_below"] = CLOSED_BELOW
FINGER = RULE
MUJOCO_WAIT = 10
LIFT = 0.04
NEAR_PLATE = 0.10
KINDS = ["성공", "물었지만 들지 못함", "물지 못하고 닫음", "집게를 닫지 않음", "들었지만 접시까지 못 감", "접시 근처에서 실패"]


def sim(r):
    return "mujoco" if r.get("simulator") == "mujoco" else "isaac"


def observations(r):
    return [s["observations"] for s in r.get("stage_trace") or [] if s.get("observations")]


def start(r, obs):
    return min(MUJOCO_WAIT, max(0, len(obs) - 1)) if sim(r) == "mujoco" else 0


def finger(obs):
    return [o.get("gripper_finger_mean_m") if o.get("gripper_finger_mean_m") is not None else 0.04 for o in obs]


def closed(r, obs):
    return any(x < CLOSED_BELOW for x in finger(obs)[start(r, obs):])


def held_w(w, band, stall):
    lo, hi = band
    for i in range(len(w) - 2):
        win = w[i:i + 3]
        if all(lo < x < hi for x in win) and (stall is None or max(win) - min(win) < stall):
            return True
    return False


def held(r, obs, stall="rule", band=None):
    rule = RULE[sim(r)]
    return held_w(finger(obs)[start(r, obs):], band or rule["held"], rule["stall"] if stall == "rule" else stall)


def lift(r, obs):
    bz = [o["target_bowl_aabb_min_m"][2] for o in obs[start(r, obs):]]
    return max(bz) - bz[0] if bz else 0.0


def final_xy_error_mm(obs):
    last = obs[-1] if obs else {}
    b, p = last.get("target_bowl_aabb_center_m"), last.get("plate_aabb_center_m")
    if not b or not p:
        return None
    return 1000 * float(np.linalg.norm(np.asarray(b[:2]) - np.asarray(p[:2])))


def others_moved_m(r, obs):
    """목표 그릇과 접시가 아닌 물체가 기준점부터 끝까지 가장 많이 움직인 거리. 기록이 없으면 None."""
    if not obs or "other_objects_pos" not in obs[0]:
        return None
    a, b = obs[start(r, obs)]["other_objects_pos"], obs[-1]["other_objects_pos"]
    d = [float(np.linalg.norm(np.asarray(b[k][:3]) - np.asarray(a[k][:3]))) for k in a if k in b and "to_robot0" not in k]
    return max(d) if d else 0.0


def classify(r, stall="rule", band=None):
    if r.get("task_success"):
        return KINDS[0]
    obs = observations(r)
    if not obs:
        return None
    if lift(r, obs) < LIFT:
        if held(r, obs, stall, band):
            return KINDS[1]
        return KINDS[2] if closed(r, obs) else KINDS[3]
    last = obs[-1]
    b, p = last.get("target_bowl_aabb_center_m"), last.get("plate_aabb_center_m") or [9, 9]
    return KINDS[5] if b and np.linalg.norm(np.asarray(b[:2]) - np.asarray(p[:2])) < NEAR_PLATE else KINDS[4]
