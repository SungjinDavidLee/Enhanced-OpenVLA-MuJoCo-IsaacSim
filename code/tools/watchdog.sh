#!/usr/bin/env bash
# 15 분 넘게 로그가 갱신되지 않은 브리지 프로세스를 끊는다 (워커는 다음 판으로 넘어감).  5 분마다 검사.
# 결과가 없는 판은 폴더를 지워 재시도되게 한다.  기록: ~/watchdog.log
while true; do
  for pid in $(pgrep -f "isaacsim/python.sh.*run_openvla_isaac_bridge"); do
    d=$(tr '\0' ' ' < /proc/$pid/cmdline 2>/dev/null | grep -oE "capture-dir [^ ]+" | awk '{print $2}')
    [ -n "$d" ] && [ -f "$d.log" ] || continue
    age=$(( ($(date +%s) - $(stat -c %Y "$d.log")) / 60 ))
    if [ "$age" -ge 15 ]; then
      echo "$(date +%H:%M) 굳은 판 끊음: $(basename $d) (로그 ${age}분 정지)" >> ~/watchdog.log
      pkill -P $pid 2>/dev/null; kill $pid 2>/dev/null; sleep 3
      [ -f "$d/episode_result.json" ] || rm -rf "$d" "$d.log"
    fi
  done
  sleep 300
done
