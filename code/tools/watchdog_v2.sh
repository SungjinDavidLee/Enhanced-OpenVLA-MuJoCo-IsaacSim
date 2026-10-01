#!/usr/bin/env bash
# 멈춘 Isaac 판을 끊는다.  로그가 STALE_MIN 분 넘게 안 바뀌면 끊고, CHECK_S 초마다 확인한다.
# 결과가 없는 판은 폴더를 지워 재시도되게 한다.  기록: ~/watchdog.log
STALE_MIN=${STALE_MIN:-5}
CHECK_S=${CHECK_S:-60}
while true; do
  for pid in $(pgrep -f "isaacsim/python.sh.*run_openvla_isaac_bridge"); do
    d=$(tr '\0' ' ' < /proc/$pid/cmdline 2>/dev/null | grep -oE "capture-dir [^ ]+" | awk '{print $2}')
    [ -n "$d" ] && [ -f "$d.log" ] || continue
    age=$(( ($(date +%s) - $(stat -c %Y "$d.log")) / 60 ))
    if [ "$age" -ge "$STALE_MIN" ]; then
      echo "$(date +%H:%M) 굳은 판 끊음: $d (로그 ${age}분 정지)" >> ~/watchdog.log
      pkill -P $pid 2>/dev/null; kill $pid 2>/dev/null; sleep 3
      [ -n "$(find "$d" -name episode_result.json 2>/dev/null | head -1)" ] || rm -rf "$d" "$d.log"
    fi
  done
  sleep $CHECK_S
done
