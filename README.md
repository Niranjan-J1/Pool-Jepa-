# Pool-JEPA

**A small robot that learns to play pool by imagining its shots before it takes them.**

A differential-drive robot with a solenoid striker finds the cue ball through an overhead camera.
(In the current sim results the robot's own pose comes from the simulator, not the camera; a
marker-based pose is planned.) A self-supervised encoder (JEPA-style, with no labels and no pixel reconstruction) turns the camera
frames into a compact latent state. A learned shot model then predicts where each possible strike
would leave the cue ball. The robot drives into position, imagines every strike strength, picks
the one predicted to land closest to the target, and fires.

> **Status:** everything below runs in a 2D physics simulator (pymunk) of a 26-inch table.
> The physical robot (ESP32 + solenoid) and the C++ inference path are planned. See
> [`project_proposal.md`](project_proposal.md).

<!-- Demo: add a GIF here, e.g. ![demo](docs/demo.gif)
     (render with `python planning/record_shot.py --montage --pace social --seed 59 33`) -->

## Results (simulation)

100 test scenes, each with a reachable straight-line target. The miss is the distance between
where the cue ball stops and the target.

| Strategy | Mean miss | Within 3 cm | Within 5 cm |
|---|---|---|---|
| Fixed strike strength (baseline) | 7.5 cm | 18% | 38% |
| **Full loop: aim from camera pixels, then imagine and fire** | **2.4 cm** | **78%** | **94%** |
| Same, but given the true cue-ball position (diagnostic) | 2.0 cm | 80% | 95% |

On targets at least 5 cm away, the full loop lands 73% of shots within 3 cm, against 25% for the
fixed-strength baseline. The detailed write-up, including what did and didn't help, is in
[`experiments/results_shot_planner.md`](experiments/results_shot_planner.md).

The biggest single improvement came from weighting the shot model's loss on the cue-ball
keypoints, the only part of the latent the planner actually scores. That cut the held-out error
from 5.9 cm to 3.4 cm. More data and a bigger model on their own did not help.

## How it works

```
camera frame ──► encoder ──► latent z (32 learned keypoints)
                                 │
                                 ├─► cue-ball readout ──► Stage 1: drive to the firing pose
                                 │                         (closed-loop, path planned around balls)
                                 │
                                 └─► shot model(z, heading, force) ──► predicted cue rest point
                                          Stage 2: try every force on a fine grid,
                                          fire the one landing closest to the target
```

1. **Perception.** A small CNN with a spatial-softmax head encodes two stacked frames, with the
   table background subtracted, into 32 keypoints. It is trained JEPA-style: predict the next
   latent from an EMA target encoder, with no decoder and no negatives. A linear readout of a few
   keypoints gives the cue-ball position (R² 0.993, about 1.3 cm error).
2. **Stage 1, aim.** A closed-loop controller drives the robot behind the cue ball and lines it
   up with the target, re-planning a collision-free path every step from the decoded position.
3. **Stage 2, imagine.** The shot model maps (latent before the strike, robot heading, force) to
   the latent once everything has stopped. With one continuous decision left, the planner scores
   a fine grid of forces in a single batch instead of running CEM.

## Repository layout

| Path | What's there |
|---|---|
| `env/` | pymunk simulator: table, balls, diff-drive robot, solenoid strike, renderer |
| `data/` | data collection (`collect.py`: random driving, strike outcomes, shot pairs) and checks |
| `perception/` | encoder (spatial-softmax keypoint head) |
| `models/` | shot-outcome model |
| `planning/` | Stage 1 aim controller, Stage 2 shot planner, full-shot test, demo video renderer |
| `experiments/` | training scripts, evaluations, diagnostics, results write-ups |

## Quick start

Requires Python 3.10+.

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Datasets and checkpoints are not in the repo (they're in `.gitignore`), so you train your own.
Data goes to `data/datasets/` (or `$POOLJEPA_DATA`) and checkpoints to `data/checkpoints/`.

```bash
# 1. Collect simulator data and train the encoder
python data/collect.py
python data/collect.py --mode strike_outcomes --episodes 500
python experiments/train_encoder.py

# 2. Collect shot data (normal + long straight shots) and train the shot model
python data/collect.py --mode shot_pairs --episodes 5000 --force-range 0.05 0.3 --prefix shots_v2
python data/collect.py --mode shot_pairs --episodes 5000 --force-range 0.1 0.3 --force-bias 2 \
    --min-travel 0.08 --straight-only --prefix shots_long --seed 1
python experiments/train_shot_model.py --pattern "shots_v2_*.npz" "shots_long_*.npz" \
    --cue-loss-weight 5 --out-suffix _bigcue

# 3. Evaluate the full loop and record a demo video
python planning/full_shot_test.py --trials 100 --suffix _bigcue
python planning/record_shot.py --montage --pace social --seed 59 33
```

Videos are saved to `videos/` as 1080p60 MP4 plus a GIF. `record_shot.py --scan 60` lists seeds
where the whole loop succeeds cleanly.

## Known limitations

- Simulation only so far; nothing has been run on hardware yet.
- Stage 1 uses the simulator's true robot pose; only the cue-ball position is read from pixels.
- Aim accuracy is capped by the cue-ball readout (about 1.25 cm), which matters most on short
  shots.
- Stage 1 times out in about 8% of trials when it uses the decoded cue position.
- The shot model is sometimes overconfident on long shots. Its predicted miss can be far below
  the real one.
- Targets are straight shots without cushions or other balls. Multi-ball and cushion shots are
  next.

## Roadmap

From the [proposal](project_proposal.md): the real robot (ESP32, solenoid, overhead webcam),
multi-step latent rollouts for ball-to-ball shots, ensemble uncertainty, active data collection,
language-specified goals, and a hand-written C++/AVX2 inference path.

## License

[MIT](LICENSE)
