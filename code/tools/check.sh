#!/usr/bin/env bash
# 진행 확인: 워커 · 서버 · GPU · 판별 로그 갱신 · 오류 · 제로샷 집계 · 재생 수집 집계.  맨 위 세 줄만 바꿔 쓴다.
ZS=/home/user/LIBERO/experiments/zs_s00_S10          # 제로샷 결과 폴더
RP=/home/user/LIBERO/experiments/replay_demo         # 재생 수집 폴더 (롤아웃은 replay_all)
PORT=8000                                            # 정책 서버 포트 (OpenVLA 8000, OFT 8001)
echo "=== $(date +%H:%M) ==="
printf "  평가 워커 %d   수집 워커 %d   서버 %s   GPU %s\n" \
  $(pgrep -af "eval_zs_all_isaac" | grep -v pgrep | wc -l) \
  $(pgrep -af "collect_success_per_task" | grep -v pgrep | wc -l) \
  "$(curl -s --max-time 3 http://127.0.0.1:$PORT/health >/dev/null && echo OK || echo 없음)" \
  "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
pgrep -af "isaacsim/python.sh.*run_openvla_isaac_bridge" | grep -oE "task-id [0-9]+ --episode-id [0-9]+|action-replay" | paste -sd' ' | sed 's/^/  도는 판: /'
echo "--- 판별 로그 갱신 (5 분 넘으면 멈춤 의심) ---"
for f in $(ls -t $ZS/*.log $RP/seed*/*.log 2>/dev/null | head -3); do
  printf "  %-30s %3d 분 전\n" "$(echo $f | sed 's#.*/experiments/##')" "$(( ($(date +%s) - $(stat -c %Y $f)) / 60 ))"; done
echo "--- 오류 ---"; E=$(grep -lE "OutOfMemory|bridge failed|Connection refused|exit code 139" $ZS/*.log $RP/seed*/*.log 2>/dev/null | tail -3); [ -n "$E" ] && echo "$E" || echo "  없음"
ZS=$ZS RP=$RP python3 - << 'PY'
import glob, json, os
from collections import defaultdict
ZS, RP = os.environ['ZS'], os.environ['RP']
s = defaultdict(lambda: [0, 0])
for f in glob.glob(ZS + '/task_*_e*/episode_result.json'):
    t = f.split('/')[-2].split('_')[1]; r = json.load(open(f)).get('result', {}); s[t][1] += 1; s[t][0] += bool(r.get('task_success'))
n = sum(v[1] for v in s.values()); k = sum(v[0] for v in s.values())
print('--- 제로샷 %s: 완료 %d/500   성공 %d (%.1f%%)   ' % (os.path.basename(ZS), n, k, 100*k/n if n else 0) + '  '.join('t%s:%d/%d' % (t, *s[t]) for t in sorted(s)))
c = defaultdict(lambda: [set(), 0, 0])
for d in glob.glob(RP + '/seed*/task_*_e*'):
    fs = glob.glob(d + '/**/episode_result.json', recursive=True)
    if not fs: continue
    t = d.split('/')[-1].split('_')[1]; ep = d.split('_e')[-1]; ok = bool(json.load(open(fs[0])).get('result', {}).get('task_success'))
    c[t][2] += 1; c[t][1] += ok; ok and c[t][0].add(ep)
tot = sum(len(v[0]) for v in c.values())
print('--- 수집 %s (성공 궤적 / 성공 판 / 시도):  합계 %d/500   ' % (os.path.basename(RP), tot) + '  '.join('t%s:%d/%d/%d' % (t, len(c[t][0]), c[t][1], c[t][2]) for t in sorted(c)))
PY
