# Report 1-1. Moving OpenVLA to Isaac Sim — from 16.8 % to 22.8 % success, and why it did not go higher

Written 2026-10-01 · Version 1-1a · 한국어 (main): [README.md](README.md) · PDF: [English](report_1-1_en.pdf) · [한국어](report_1-1_ko.pdf)

> This version contains only results measured under the same conditions with the same number of episodes. The effect of the execution setting is being re-measured with matching episode counts; this report will be updated when it finishes, and Report 1-2 covers it in detail. Figure labels are in Korean in this version; English figures will follow in the update.

## 0. What this report is about

OpenVLA is an AI model that controls a robot arm to "pick up the black bowl and place it on the plate". It was trained on human demonstrations recorded in a simulator called MuJoCo. This report tests the same model, without any additional training, in a different simulator, Isaac Sim. It describes the success rate, how we fixed problems in the evaluation code that distorted that rate, and why the rate still did not rise much after the fixes.

## Conclusions at a glance

After fixing defects in the evaluation code, the success rate under the same conditions rose **from 12.6 % (counting only episodes where the robot actually grasped the bowl) to 22.8 %**. Most remaining failures were not about finding the wrong place: the robot **grasped the bowl but could not lift it, or closed its hand in the air outside the bowl**. When the same grasp is performed without the model, the bowl in Isaac Sim tilts strongly and rises less.

![Figure 1](figures/fig01_success_rate.png)

*Figure 1. Success rate of OpenVLA in Isaac Sim without additional training. Dots: success rate; vertical lines: 95 % confidence range. Left: value reported in the paper (old success criterion). Middle: the same run counted with the current criterion. Right: after fixing the evaluation code (same execution setting and object placement as the paper).*

| Question | Answer | Evidence |
| --- | --- | --- |
| How much did the success rate rise? | The paper reported 16.8 % (84 of 500 episodes). In 21 of those, the robot never grasped the bowl but the episode was counted as a success; counting only real grasp-and-place successes gives 12.6 %. After fixing the evaluation code, the same criterion gives 22.8 % (114 of 500) | Figs 1, 2 |
| What raised it? | Fixing defects in the evaluation code. All three values were measured with the same execution setting, the same object placement and 500 episodes | Sections 2, 3 |
| Is this execution setting the same as the training environment? | No. For the same command, the robot hand moves more than in MuJoCo (1.3× while approaching the bowl, more than 3× right after starting). Results with the MuJoCo-matched speed are being re-measured with the same episode count | Fig 5, Section 5 |
| Why didn't it go higher? | Of 500 episodes, 105 closed the hand in the air outside the bowl, 164 grasped the bowl but could not lift it, and 110 lifted it but did not reach the plate. In a grasp test without the model, the bowl in Isaac Sim tilts 38–54° (MuJoCo: 2–15°) | Figs 6, 7, 8 |
| What was ruled out | Image processing, camera field of view, object size. Bowl weight has only a partial effect | Table in Section 6 |
| What is still unknown | How the success rate changes under the same conditions when the execution setting matches MuJoCo's speed | Section 8 |
| Next | 500 episodes at MuJoCo-matched speed with the same object placement (running, Report 1-2). Retraining the model on human demonstrations that were actually lifted in Isaac Sim (running) | Section 8 |

## 1. What is evaluated

**Tasks.** The 10 tasks of the LIBERO-Spatial benchmark. Every task is "pick up the black bowl and place it on the plate"; tasks differ in where the bowl starts.

| Task | Where the bowl starts | Task | Where the bowl starts |
| --- | --- | --- | --- |
| 0 | between the plate and the ramekin (small bowl) | 5 | on the ramekin |
| 1 | next to the ramekin | 6 | next to the cookie box |
| 2 | at the center of the table | 7 | on the stove |
| 3 | on the cookie box | 8 | next to the plate |
| 4 | in the top drawer of the wooden cabinet | 9 | on the wooden cabinet |

**Episodes and object placement.** One trial is an "episode". Each task runs 50 episodes, 500 in total. All tests in this report start with the objects (bowl, plate, boxes) in the first placement defined by LIBERO (the same protocol as the paper).

**Episode length.** An episode fails if it does not succeed within 220 commands from the model.

**Success criterion.** Success requires that the bowl's bottom rests on the plate, its height matches the plate's top surface, it has stopped moving, and it came down from more than 3 cm above the plate. In addition, for every successful episode we check the finger record to confirm **the robot actually grasped the bowl**.

![Figure 9](figures/fig09_screens_mujoco_vs_isaac.png)

*Figure 9. Starting views of the 10 tasks. From left: MuJoCo, Isaac Sim (the view the model receives), and the two overlaid.*

## 2. Was the starting number accurate?

**Observation.** The paper reported 16.8 %, but almost half of the successes (39 of 84) came from task 6 alone.

**What we suspected and why.** Two things. First, the success criterion at the time was "the bowl is near the plate when the episode ends", so pushing the bowl without grasping it could count as a success. Second, the evaluation code had safety rules such as "stop if the gripper closes more than 6 times", which might take retries away from the model.

**How we checked and what we found.** Checking the finger records of all 84 successes, **21 had never grasped the bowl** (18 of them in task 6). Counting why episodes ended, **395 of 500 episodes had been cut off by the safety rules** before using their 220 commands.

**What we fixed.**

- Removed the rules that cut episodes short; an episode fails only when its 220 commands are used up.
- Changed how the gripper closes. The old code stopped the arm and then closed the gripper; in MuJoCo the arm keeps moving while the gripper closes.
- Removed a 15 mm offset in the target hand height, which sometimes made the hand close above the bowl rim.
- When the arm's joint computation (inverse kinematics) fails, the robot now holds its current pose instead of ending the episode.
- Adopted the stricter success criterion above and checked every success for an actual grasp.

![Figure 2](figures/fig02_evaluation_code.png)

*Figure 2. Two problems in the old evaluation code. One square = one episode. (a) Episodes cut off early: 395 of 500 before the fix, 0 after. (b) Successes without a grasp: 21 of 84 before, 0 of 114 after.*

## 3. Results after fixing the evaluation code

With the same model, the same execution setting and the same object placement, 500 episodes gave **22.8 % (114 successes)**, with no success lacking a grasp. Under the same criterion this is a rise from 12.6 % to 22.8 %, and the 95 % confidence ranges do not overlap (Figure 1).

| Task | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | Total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Successes / 50 | 10 | 13 | 28 | 9 | 2 | 0 | 40 | 3 | 7 | 2 | 114 / 500 |
| Success rate | 20 % | 26 % | 56 % | 18 % | 4 % | 0 % | 80 % | 6 % | 14 % | 4 % | 22.8 % |

## 4. How the model moves the robot

The model does not turn the robot's joints directly. It looks at the camera image and the task sentence, issues one short movement command such as "move the hand 2 cm forward and 1 cm down", looks at the image again after the robot moves, and issues the next command. This repeats hundreds of times per episode.

![Figure 3](figures/fig03_how_model_controls.png)

*Figure 3. The model issues short movement commands while watching the camera image.*

**Duration of one command.** The human demonstrations the model learned from were recorded 20 times per second, i.e. one command every 0.05 s. So one command means "the movement during 0.05 s". In 0.05 s the MuJoCo robot does not cover the full commanded distance: from rest it moves about 9 % of it on the first command, and about 23 % per command while moving. The model learned to issue commands that fit this response.

**Model computation time.** "OpenVLA runs at 5–10 Hz" refers to how fast a GPU computes commands. Like LIBERO's original evaluation, our evaluation pauses simulated time while the model computes and advances time after a command arrives. In our setup one command took 0.73 s on average to compute, but from the robot's point of view commands still arrive at fixed intervals.

## 5. Execution setting — what we used and how it differs from the training environment

How far the Isaac Sim robot moves per command is set by two values: how many physics steps run before the next command (physics runs 60 times per second), and what percentage of the commanded distance is used as the hand's target.

![Figure 4](figures/fig04_two_settings.png)

*Figure 4. The two execution settings. Physics (60 steps per second) is the same; what differs is the interval between commands and how far the robot moves in between.*

| | Setting that moves more per command | MuJoCo-matched speed |
| --- | --- | --- |
| Interval between commands | 0.17 s (10 physics steps) | 0.05 s (3 physics steps, same as MuJoCo) |
| Hand target | the full commanded distance | 65 % of the commanded distance |
| Actual movement per command | almost the full commanded distance | about 23 % (same as MuJoCo) |
| Tests in this report | **both the paper's test and the fixed-code test** | being re-measured with the same episode count |

**What happened.**

- **Observation.** The paper's test and the fixed-code test both used the setting that moves more per command (the recorded number of physics steps per command is 10 in both).
- **What we suspected and why.** On 25 September, suspecting that this setting moves the robot differently from MuJoCo, we matched it to MuJoCo's speed. With the same object placement, task 6's success rate then dropped sharply (the robot grasped the bowl but could not lift it), so on 27 September we went back to the original setting and measured 22.8 %. That MuJoCo-speed test was stopped at 216 episodes, so it cannot be used for an equal-count comparison.
- **How we checked and what we found.** On 29 September we fed the same human-demonstration commands to both simulators and compared how far the hand moved (4 demonstrations). The MuJoCo-matched speed moved almost the same distance as MuJoCo (102), while the other setting moved 1.3× more while approaching the bowl (126) and more than 3× more right after starting (318). It was also uneven by direction (1.73× forward-backward, 0.65× left-right, 0.96× up-down), and at the moment of grasping the hand position differed from MuJoCo by 159 mm and its orientation by 136° (21 mm and 8° at the MuJoCo-matched speed).

![Figure 5](figures/fig05_distance_same_commands.png)

*Figure 5. Distance moved by the hand when the same human-demonstration commands were given (MuJoCo = 100, average of 4 demonstrations).*

**What we can say now.** Both 16.8 % and 22.8 % were measured with an execution setting that moves differently from the training environment. The difference between them is the effect of fixing the evaluation code. How the success rate changes with the MuJoCo-matched setting and the same object placement is being measured again with 500 episodes.

## 6. Why it did not go higher

**Observation.** After the fixes, 386 of 500 episodes still failed.

**How we checked.** We classified failures from the finger-width and bowl-height records: "grasped the bowl" if the fingers stopped at about the bowl-wall thickness, "lifted" if the bowl's bottom rose more than 4 cm, and "brought near the plate" if the bowl ended within 10 cm of the plate.

![Figure 6](figures/fig06_where_episodes_dropped.png)

*Figure 6. At which stage the 500 episodes dropped out (fixed evaluation code, setting that moves more per command, one object placement).*

**Results.** Failures were spread across three stages.

- **Closed the hand in the air outside the bowl (105 episodes).** In successful episodes the center between the fingers was 42–45 mm from the bowl center, i.e. over the bowl wall; in failed ones it was 67–123 mm, outside the bowl (Figure 7). The robot moving more than in MuJoCo per command, and unevenly by direction (Section 5), may prevent it from stopping exactly over the bowl. This will be checked when the MuJoCo-speed test finishes.
- **Grasped but could not lift (164 episodes).** The model commanded a lift, but the bowl did not come up.
- **Lifted but did not reach the plate (110 episodes).** After lifting the bowl, the robot did not move toward the plate.

![Figure 7](figures/fig07_where_hand_closed.png)

*Figure 7. Where the hand first closed, seen from above (median distances from the bowl center).*

**A test to find the cause of "could not lift".** To separate the model's decisions from the physics, we made the robot perform a fixed motion **without the model** (grasp the bowl rim 10, 20 and 30 mm below the top and lift 8 cm) in both simulators (task 0, once per depth).

![Figure 8](figures/fig08_grasp_test.png)

*Figure 8. Grasp test without the model (task 0). Left: bowl shape at the measured angles when grasped 10 mm below the rim. Middle: how far the bowl's bottom rose. Right: bowl tilt.*

With the same motion, in MuJoCo the bowl rose 65–77 mm almost level (2–15°); in Isaac Sim it **tilted 38–54° around the grasp point and rose only 35–43 mm**. In Isaac Sim, grasping one side of the bowl makes it tilt strongly and hard to lift. Changing five bowl-side physics settings (solver iterations, friction material, contact offset, torsional friction, etc.) did not change this, and making the bowl as light as in MuJoCo (85 g → 6 g) helped only for shallow grasps. Finger-side settings have not been tested yet.

**What was ruled out.**

| Suspect | How we checked | Result |
| --- | --- | --- |
| Image processing (cropping, flipping, task sentence, command decoding) | Watched whether the model brings the hand to 3 cm in front of the bowl early in an episode | It does → not the cause |
| Camera field of view | Computed the vertical field of view of the image the model receives | About 46°, nearly the same as MuJoCo training images (45°) |
| Object size | Compared bowl and plate shapes and collision shapes in both simulators | Same |
| Bowl weight (MuJoCo 6 g, Isaac Sim 85 g) | Weight in successful episodes; grasp test with 6 g | Successful episodes also used 85 g; 6 g helped only shallow grasps → not the main cause |

## 7. What now appears to be the cause

- **Grasping stage: a difference in physics.** In Isaac Sim, grasping one side of the bowl makes it tilt strongly, so the grasp the model learned in MuJoCo does not work. Bowl-side settings did not fix this.
- **Positioning stage: a difference in execution setting (being checked).** If the robot moves differently per command than in the training environment, it is hard to stop the hand exactly over the bowl.

## 8. Limitations and next experiments

**Limitations.**

- All tests in this report used an execution setting that moves differently from the training environment. 500 episodes with the MuJoCo-matched speed and the same object placement are running, and this report will be updated with the result.
- Only one object placement was used. LIBERO's standard protocol (50 placements, changing every episode) is compared with equal episode counts in Report 1-2.
- Every test was run once. The grasp test without the model was run once per depth.
- This evaluation pauses simulated time while the model computes. The case where time keeps running during computation, as on a real robot, is not covered.

**Next experiments.**

1. MuJoCo-matched speed, same object placement, 500 episodes — difference due to the execution setting only (running, Report 1-2)
2. Setting that moves more per command, 50 object placements, 500 episodes — difference due to object placement only (Report 1-2)
3. Retraining the model on 429 human demonstrations that were actually lifted in Isaac Sim — whether the model can overcome the physics difference at the grasping stage (running)
4. Finger-side physics settings — whether bowl tilt can be reduced

## 9. How to reproduce

Code names and values appear only in this section.

- Simulator: Isaac Sim 6.0.1, physics 60 Hz. GPU: RTX 5090 32 GB.
- Model: `openvla/openvla-7b-finetuned-libero-spatial`, unnormalization key `libero_spatial`; the server crops the image to a square, takes the central 0.35 of the area and resizes it to 224×224.
- Setting that moves more per command: `--sim-steps-per-action 10 --convergence-gain 1.0`. MuJoCo-matched speed: `--sim-steps-per-action 3 --convergence-gain 0.65`.
- Main arguments of the fixed evaluation code: `--steps 220 --ik-backtrack-factors 1.0 0.5 --no-ik-position-fallback --close-approach-settle-steps 0 --gripper-settle-steps 1 --libero-world-offset-m 0 0 0 --stage-telemetry`.
- One object placement: `FIXED_STATE=00` in the evaluation script.
- Checking the execution setting: `physics_steps_accounted` in the episode record (the number of physics steps actually run per command) is 10 for the setting that moves more per command and 3 for the MuJoCo-matched speed.

## Appendix. Terms

| Term | Meaning |
| --- | --- |
| Episode | One trial |
| Command | A short movement instruction the model issues after looking at the image: change of hand position and orientation, and open / close the gripper |
| Evaluation code | The program that turns the model's commands into robot motion and runs, ends and judges episodes |
| Execution setting | Values that set how far the robot moves per command (Section 5) |
| 95 % confidence range | The range the success rate would roughly fall in if the same test were repeated; about ±3–4 percentage points for 500 episodes |
