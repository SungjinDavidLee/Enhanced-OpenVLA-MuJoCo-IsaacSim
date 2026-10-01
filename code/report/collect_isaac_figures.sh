#!/usr/bin/env bash
# 실제 Isaac Sim 실행 장면을 뽑아 README v2 가 참조하는 이름으로 figures/isaac/ 에 넣는다.
#   bash code/report/collect_isaac_figures.sh            (레포 맨 위에서 실행, CPU 만, 10–20 분)
# 평가가 진행 중인 결과 폴더가 있으면 그 판까지만 반영된다. 500 판이 끝난 뒤 실행할 것.
set -u
HERE=$(cd "$(dirname "$0")/../.." && pwd); OUT=$HERE/figures/isaac; A=~/report_assets
python3 "$HERE/code/report/make_report_assets.py" || exit 1
mkdir -p "$OUT"
c() { [ -f "$A/$1" ] && cp "$A/$1" "$OUT/$2" && echo "  $2" || echo "  (없음) $1"; }
c moments/ovla_fast/overview.png            moments_ovla_fast.png
c moments/oft_cal/overview.png              moments_oft_cal.png
c moments/ovla_fast_modelview/overview.png  moments_ovla_fast_modelview.png
for t in 02 03 06 08 09; do c gifs/oft_cal_task_${t}_success.gif success_t${t}.gif; done
c filmstrips/oft_cal_task_03.png            filmstrip_oft_cal_t03.png
c filmstrips/oft_cal_task_09.png            filmstrip_oft_cal_t09.png
c gifs/fail_overshoot_oft_fast_t0.gif       fail_overshoot.gif
c gifs/fail_tilt_oft_cal_t0.gif             fail_tilt.gif
c gifs/fail_task5_oft_cal.gif               fail_task5.gif
c failures/fail_overshoot.png               fail_overshoot.png
c failures/fail_tilt.png                    fail_tilt.png
c failures/fail_task5.png                   fail_task5.png
c failures/bench_tilt.png                   bench_tilt.png
c camera/d405_cropbox.png                   d405_cropbox.png
c camera/model_view.png                     model_view.png
c camera/wrist_view.png                     wrist_view.png
cp "$A/manifest.md" "$OUT/manifest.md" 2>/dev/null
echo "완료: $OUT"
