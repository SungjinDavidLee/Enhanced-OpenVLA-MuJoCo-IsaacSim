# Evaluating VLA robot policies in Isaac Sim — moving a model trained in MuJoCo to another simulator

한국어 (main): [README.md](README.md)

This repository records how an AI model (the OpenVLA family) trained in the MuJoCo simulator to make a robot arm "pick up the black bowl and place it on the plate" performs when moved, without additional training, to the Isaac Sim simulator. The goals are to find out how much and why the success rate drops, and to recover it by retraining on Isaac Sim images. Results are published report by report, and this page is updated with each report.

## Reports

| No. | Title | Status |
| --- | --- | --- |
| 1-1 | [Moving OpenVLA to Isaac Sim — from 16.8 % to 22.8 % success, and why it did not go higher](reports/1-1_openvla_zero_shot/README_en.md) | Published (version 1-1a, 2026-10-01) |
| 1-2 | [Effects of the execution setting and object placement](reports/1-2_execution_setting_and_layout/README.md) | In preparation |
| 2 | [What OpenVLA alone could not show — an evaluation setup for several models, and OpenVLA-OFT](reports/2_multi_model_environment_and_oft/README.md) | In preparation |
| 3 | [OpenVLA vs OpenVLA-OFT](reports/3_openvla_vs_oft/README.md) | In preparation |

## Results so far

| Report | Key result |
| --- | --- |
| 1-1 | After fixing defects in the evaluation code, the success rate counting only real grasps rose from 12.6 % to 22.8 % under the same conditions (execution setting, object placement, 500 episodes). Most remaining failures were grasping the bowl without lifting it or closing the hand outside the bowl; in Isaac Sim, grasping one side of the bowl makes it tilt strongly (38–54°, MuJoCo 2–15°) |

## Repository layout

| Path | Contents |
| --- | --- |
| `reports/` | One folder per report: Korean text (`README.md`), English (`README_en.md`), PDFs, figures |
| `code/` | Code used for evaluation, analysis, training data and training. See each report's "How to reproduce" section |

## Update log

| Date | Change |
| --- | --- |
| 2026-10-01 | Report 1-1 (version 1-1a) published. The earlier full record (v2) remains under git tag `v2` |
