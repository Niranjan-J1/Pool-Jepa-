# Shot planner results: Stage 1 aim + Stage 2 force (sim)

Runs on the cluster, 2026-11-04/05, branch `feature/cue-setup`. Frozen encoder `encoder_best.pt`
(6 cue keypoints, readout R^2 0.993, ~1.3-1.5 cm readout error) throughout.

Test setup (`planning/full_shot_test.py`, 100 trials, seed 0): straight-line reachable targets,
i.e. the rest point of a secret force fired from the exact firing pose, with a cue ball that
touched nothing. Targets are short on this 66 x 33 cm table (mean 9.6 cm, median 5.6 cm), so
results are also given for targets >= 5 cm away (56 of 100). `record_shot.py --scan 60` counts
"clean" seeds: aligned, nothing disturbed, strike connected, miss <= 2 cm, target >= 8 cm.

## Shot model: before vs after

| model | data | size | loss |
|---|---|---|---|
| before (`shot_model_best`) | ~5k shots, force 0.05-0.3, old sim | 256 x 3 | full-latent MSE |
| `_big` | 5k new-sim base (0.05-0.3) + 5k straight shots >= 8 cm (0.1-0.3, bias 2) | 512 x 5 | full-latent MSE |
| `_bigcue` | same as `_big` | 512 x 5 | + 5 x MSE on the cue keypoints |

Held-out cue rest error (`train_shot_model.py`, 1000 held-out shots; ceiling = encoder readout of
the true rest frame, 1.3 cm):

| model | all, mean | travel >= 8 cm | >= 8 cm straight | >= 8 cm cushion/ball | travel r |
|---|---|---|---|---|---|
| before | ~5 cm (not re-evaluated on this held-out set) | | | | |
| `_big` | 5.9 cm (25% < 3 cm) | 6.2 cm | 5.8 cm | 7.4 cm | 0.78 |
| `_bigcue` | **3.4 cm** (53% < 3 cm) | **3.5 cm** | **3.2 cm** (55% < 3 cm) | 4.6 cm | 0.93 |

More data and a bigger model alone did not help (`_big` 5.9 cm). Weighting the loss on the cue
keypoints, the only dims the planner scores, did (3.4 cm). The full-latent MSE was spending the
model's capacity on robot/other-ball dims. (Not run: `_small`, new data at the original size,
which would separate the data effect from the size effect.)

## Full loop: before vs after (`full_shot_test.py --trials 100`)

All targets:

| strategy | before mean / <3 / <5 | `_bigcue` mean / <3 / <5 |
|---|---|---|
| stage 2 only (perfect aim) | 3.9 cm / 47% / 76% | 2.1 cm / 79% / 92% |
| fixed force 0.175 | 7.5 cm / 18% / 37% | 7.5 cm / 18% / 38% |
| full loop (Stage 1 on decoded cue) | 4.1 cm / 43% / 72% | **2.4 cm / 78% / 94%** |
| full loop, true cue position | 3.9 cm / 45% / 74% | 2.0 cm / 80% / 95% |

Targets >= 5 cm away (56):

| strategy | before mean / <3 / <5 | `_bigcue` mean / <3 / <5 |
|---|---|---|
| stage 2 only | 5.2 cm / 16% / 62% | 2.4 cm / 71% / 89% |
| fixed force 0.175 | 6.2 cm / 25% / 50% | 6.2 cm / 25% / 52% |
| full loop | 5.0 cm / 21% / 61% | **2.4 cm / 73% / 95%** |

Before, the planner did not beat a fixed force on targets >= 5 cm (16% vs 25% within 3 cm). Now
it clearly does (71-73% vs 25%). Clean seeds in the 60-seed scan: 1 before, **7** with
`_bigcue` (0, 2, 17, 18, 24, 33, 59).

## Cue-decode drift (`experiments/cue_decode_drift.py`)

Not robot drift. Decode error is 1.1-1.7 cm at every robot distance (front 2-15 cm behind the
ball, or far away: 1.42 cm); the same cue position decodes only 0.15 cm differently when the
robot drives up to it (no consistent direction); a calibration with near-robot scenes changes
nothing. The 1.25 cm at hand-off is the readout's overall accuracy. On short targets that alone
is a large aim error (mean heading error 11.6 deg, median 4.9 deg). `--calib-near-robot` stays off.

## Open issues (unchanged by the shot-model work)

- **Stage 1 timeouts:** 8% with the decoded cue vs 0% with the true cue (scan seeds 4, 20, 25,
  36, 37, 39, 49, 51, 58). Not explained by drift; next suspect is the decoded estimate
  jumping between re-decodes and resetting the controller's final-turn phase.
- **Overconfident Stage 2:** predicted miss is often far below the actual one on long shots
  (seed 9: 0.1 vs 6.8 cm; seeds 16, 45, 55 similar). Model error 3.2 cm vs 1.3 cm ceiling on
  straight long shots.
- **Aim accuracy is capped by the readout** (~1.25 cm): improving the readout/encoder is the
  lever for short targets.
- **Balls disturbed** before the shot in 18% of full-loop trials (ball avoidance off in that
  test; 4/60 in the scan with avoidance on). **Strike missed** the ball in 6%.
