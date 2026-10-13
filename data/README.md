# 데이터 · Data

## results_2026-10-13

보고서 2 · 3 · 4와 README(총 정리)의 모든 표와 그림이 이 폴더의 `episodes_all.csv` 하나에서 나왔다. `code/report/make_report_assets_v2.py --reuse`에 이 파일을 주면 표와 그림을 그대로 다시 만들 수 있다. 판 기록 원본(판마다 수 MB의 `episode_result.json`)은 용량 때문에 올리지 않았다.

All tables and figures of Reports 2, 3, 4 and the README come from the single file `episodes_all.csv` in this folder. Running `code/report/make_report_assets_v2.py --reuse` on it rebuilds them. The raw per-episode logs (`episode_result.json`, several MB each) are not uploaded because of their size.

### episodes_all.csv — 판마다 한 줄 · one row per episode (24 cells × 500 = 12,000 rows)

| 열 · column | 뜻 | meaning |
| --- | --- | --- |
| `model` | `OpZ` OpenVLA 원본, `OpF4` OpenVLA 재학습, `OFT` OpenVLA-OFT, `Pi05` π0.5 | `OpZ` OpenVLA zero-shot, `OpF4` OpenVLA retrained, `OFT` OpenVLA-OFT, `Pi05` π0.5 |
| `cell` | `mj` MuJoCo 기본, `mjb` MuJoCo-B, `M1` / `M50` Isaac M · 배치 고정 / 50, `B1` / `B50` Isaac B · 배치 고정 / 50 | `mj` MuJoCo default, `mjb` MuJoCo-B, `M1` / `M50` Isaac M · fixed / 50 layouts, `B1` / `B50` Isaac B · fixed / 50 layouts |
| `folder` | 워크스테이션의 결과 폴더 이름 | result folder name on the workstation |
| `sim` | `mujoco` 또는 `isaac` | `mujoco` or `isaac` |
| `task`, `ep`, `ep_id` | 과제 번호(0–9), 판 번호(0–49) | task (0–9), episode (0–49) |
| `success` | 기존 판정 성공 1 / 실패 0 | success by the original criterion (1/0) |
| `trace`, `steps` | 매 스텝 기록이 있는지, 기록된 스텝 수 | whether per-step logs exist, number of logged steps |
| `outcome` | 다섯 가지 중 하나: `정석 성공` 놓치지 않고 한번에 옮김, `회귀 성공` 놓친 뒤 다시 집어 성공, `판정 의심 성공` 판정 의심, `회귀 실패` 놓친 뒤 끝내 실패, `그 밖의 실패` | one of five: `정석 성공` one-pass success, `회귀 성공` success after regrasp, `판정 의심 성공` unverified success, `회귀 실패` failure after a lost grasp, `그 밖의 실패` other failure |
| `reason_code`, `reason` | 판정 의심의 이유: 1 들어 올린 적 없음, 2 접시 위 순간 없음, 3 놓을 때 쥐고 있지 않음 (0 = 해당 없음) | reason for unverified success: 1 never lifted with the hand, 2 no on-plate moment, 3 not held at release (0 = n/a) |
| `attempts` | 집게를 닫았다 연 횟수 (그릇에 닿지 않은 것 포함) | gripper close-and-reopen count (including closes that did not touch the bowl) |
| `touched` | 그중 그릇에 닿은 시도 수 | attempts that touched the bowl |
| `drops` | 그릇을 떨어뜨린 횟수 | number of drops |
| `release_mm` | 마지막으로 쥐고 있던 순간 그릇 바닥의 접시 윗면 위 높이 (mm) | height of the bowl bottom above the plate top at the last held moment (mm) |
| `kind` | 실패 종류 (`성공`, `물었지만 들지 못함`, `물지 못하고 닫음`, `집게를 닫지 않음`, `들었지만 접시까지 못 감`, `접시 근처에서 실패`) | failure kind (success, grasped but not lifted, closed without grasping, never closed, lifted but not carried to plate, failed near plate) |
| `closed`, `held` | 집게를 닫은 적이 있는지, 그릇을 문 적이 있는지 (Table 1 지표) | gripper ever closed, bowl ever grasped (Table 1 metrics) |
| `final_err_mm` | 끝났을 때 그릇 중심 – 접시 중심 수평 거리 (mm) | final horizontal bowl-to-plate center distance (mm) |
| `others_moved_m` | 다른 물체가 가장 많이 움직인 거리 (m, MuJoCo만 기록) | largest displacement of other objects (m, MuJoCo only) |
| `note` | 읽기 · 판정 오류 메모 (모두 비어 있음) | read / judging error note (all empty) |

### 그 밖의 파일 · other files

| 파일 · file | 내용 · contents |
| --- | --- |
| `screen.csv` | OpenVLA 재학습 단계 고르기: 단계별로 과제마다 0–4번 판 · fine-tuning step screen: episodes 0–4 of every task per step |
| `tables/t1_success.csv` | 성공률, 95 % 구간, 한번에 옮김, MuJoCo 대비 남은 비율 · success rate, 95 % CI, one-pass, retention vs MuJoCo |
| `tables/t2_outcomes.csv` | 다섯 가지 나누기, 평균 닫은 횟수 · 닿은 시도 · five-way grouping, mean closings and touching attempts |
| `tables/t3_unverified.csv` | 판정 의심 이유와 놓은 높이 구간 · reasons for unverified successes and release-height bands |
| `tables/t4_failure_kinds.csv` | 실패 종류 · failure kinds |
| `tables/t5_metrics.csv` | Table 1 지표 · Table 1 metrics |
| `tables/t6_per_task.csv` | 과제별 성공 · 한번에 옮김 · per-task success and one-pass |
| `tables/t7_within_model.csv` | 한 모델 안 비교와 OpenVLA 재학습 효과 (차이, z, p, 95 % 구간, 판정 의심 범위) · within-model comparisons and fine-tuning effect |
| `tables/t8_between_models.csv` | 같은 칸에서 모델끼리 비교 · between-model comparisons in the same cell |
| `tables/t9_finetune_step_screen.csv` | 재학습 단계별 성공 · successes per fine-tuning step |
