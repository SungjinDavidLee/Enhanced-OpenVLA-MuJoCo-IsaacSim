#!/usr/bin/env bash
# 제로샷 500 판 (태스크 10 × 초기 상태 50) — 판별 LIBERO 배치, 관문 해제, 오프셋 0.
# 동역학은 S=10 · gain 1.0 (작은 명령도 그대로 실행).  S=3 + gain 0.65 는 먼 태스크(task 6)에서
# 잡은 뒤 들기 단계의 작은 명령이 0.4 배로 줄어 정책이 멈춤 (s00 task 6: 3/42 → S=10 에서 6/10).
# IK 위치-only fallback 은 정책 평가에서 끔 (재생 전용).
# 결과가 있는 판은 건너뛴다.  서버(포트 8000)가 떠 있어야 한다.
#
#   단일:  nohup bash eval_zs_all_isaac.sh > ~/zs_w1.log 2>&1 &
#   병렬 2: TASKS="0 1 2 3 4" SUM_FILE=~/zs_w1.txt nohup bash eval_zs_all_isaac.sh > ~/zs_w1.log 2>&1 &
#          TASKS="5 6 7 8 9" SUM_FILE=~/zs_w2.txt nohup bash eval_zs_all_isaac.sh > ~/zs_w2.log 2>&1 &
#   집계:  python3 summarize_isaac_eval.py ~/LIBERO/experiments/zs_faithful_50ep
#
# 환경변수로 바꿀 수 있는 것
#   OUT       결과 루트 (기본 ~/LIBERO/experiments/zs_faithful_50ep)
#   FIXED_STATE=00   모든 판을 배치 s00 으로 고정 (논문 16.8 % 와 같은 조건).  OUT 을 따로 주어야 함
#   SERVER    정책 서버 URL (기본 http://127.0.0.1:8000)
#   CFGBASE   yaml 이름 (기본 libero_spatial_environment;  OFT 는 libero_spatial_environment_oft)
#   EXTRA     러너에 더 붙일 인자 (예: "--convergence-gain 1.0 --sim-steps-per-action 10")

set -u
BR=/home/user/isaac_libero/bridge
OUT=${OUT:-/home/user/LIBERO/experiments/zs_faithful_50ep}
SUM=${SUM_FILE:-/home/user/zs_summary.txt}
SERVER=${SERVER:-http://127.0.0.1:8000}
TASKS=${TASKS:-"0 1 2 3 4 5 6 7 8 9"}
EPISODES=${EPISODES:-"$(seq 0 49)"}
EXTRA=${EXTRA:-}
mkdir -p "$OUT"; touch "$SUM"

curl -s --max-time 5 "$SERVER/health" >/dev/null || { echo "정책 서버 없음: $SERVER" | tee -a "$SUM"; exit 1; }

for t in $TASKS; do
  for e in $EPISODES; do
    D="$OUT/task_$(printf %02d $t)_e$(printf %03d $e)"
    if [ -n "$(find "$D" -name episode_result.json -print -quit 2>/dev/null)" ]; then continue; fi
    P=$(printf "%s/poses_all/libero_spatial_task_poses_s%02d.json" "$BR" $e)
    # FIXED_STATE=00 이면 모든 판을 그 배치 하나로 (논문의 단일 배치 프로토콜과 직접 비교용)
    if [ -n "${FIXED_STATE:-}" ]; then P=$(printf "%s/poses_all/libero_spatial_task_poses_s%02d.json" "$BR" $((10#$FIXED_STATE))); fi
    CFG="$BR/${CFGBASE:-libero_spatial_environment}.yaml"
    # task 5 (라메킨 위 그릇): 받침 AABB 를 공중에서 재는 결함 때문에 정렬을 끈 전용 yaml 과 안정값+5 mm JSON 사용
    if [ "$t" = "5" ] && [ -d "$BR/poses_t5" ]; then P=$(printf "%s/poses_t5/libero_spatial_task_poses_s%02d.json" "$BR" ${FIXED_STATE:-$e}); CFG="$BR/${CFGBASE:-libero_spatial_environment}_t5.yaml"; fi
    mkdir -p "$D"
    /home/user/isaacsim/python.sh "$BR/run_openvla_isaac_bridge.py" \
      --env-script "$BR/libero_spatial_env_isaac601.py" \
      --config "$CFG" \
      --pose-json-path "$P" --libero-world-offset-m 0 0 0 \
      --task-id $t --episode-id $e --seed 7 --server-url "$SERVER" \
      --steps 220 --no-cv-display --stage-telemetry \
      --sim-steps-per-action 10 --convergence-gain 1.0 --ik-backtrack-factors 1.0 0.5 --no-ik-position-fallback \
      --close-approach-settle-steps 0 --gripper-settle-steps 1 \
      --target-position-tolerance-m 0.5 --target-rotation-tolerance-deg 90 \
      $EXTRA \
      --capture-dir "$D" > "$D.log" 2>&1
    python3 - "$D" "$t" "$e" << 'EOF' 2>/dev/null | tee -a "$SUM" || echo "task $t e$e  결과 없음" | tee -a "$SUM"
import json, glob, sys
D, t, e = sys.argv[1:4]
r = json.load(open(glob.glob(D + '/**/episode_result.json', recursive=True)[0]))['result']
print('task %s e%s  성공=%s  스텝=%d  S%s  종료=%s' % (t, e, r['task_success'], r['processed_steps'], r['stage_level'], r['termination_reason']))
EOF
  done
done
echo "=== 완료 $(date) ===" | tee -a "$SUM"
