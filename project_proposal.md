# Pool-JEPA: Learning to Play Pool Through Imagination

**A Self-Supervised Latent World Model for Physics Prediction and Planning on a Tabletop Billiards Domain**

Niranjan Jaishankar + collaborator | 6 months | ~C$200 budget

---

## 1. Research Question

Can a small, self-supervised latent world model — trained from scratch on a few hours of real tabletop billiards interactions — learn multi-body collision dynamics (including indirect cue-ball-to-target-ball chains) well enough to plan goal-directed shots through imagined rollouts, and does planning through imagination generalize better than a reactive policy of equivalent model capacity?

## 2. Sub-Questions

**SQ1 — Latent vs. pixel prediction.** Does predicting in learned latent space maintain accuracy over longer rollout horizons than predicting in pixel space?

**SQ2 — Imagination vs. reaction.** Does a planner that simulates futures through a world model outperform a reactive policy (same parameters, same data) on novel shot configurations?

**SQ3 — Active vs. passive data collection.** Does uncertainty-driven exploration (choosing shots the model is most uncertain about) improve world model accuracy faster than random shot collection?

**SQ4 — Uncertainty calibration.** Does ensemble disagreement reliably predict actual prediction error, and can it be used to distinguish confident shots from risky ones?

**SQ5 — Inference speed vs. task performance.** Does a hand-optimized C++ inference path on a laptop CPU (enabling faster replanning) measurably improve shot success compared to a Python/PyTorch baseline?

## 3. Background & Motivation

Recent work in Physical AI has converged on world models as the key missing component for robust robotic manipulation. Systems like DreamerV3 (Hafner et al., 2023), TD-MPC2 (Hansen et al., 2024), DINO-WM (Zhou et al., 2025), and V-JEPA 2 (Assran et al., 2025) demonstrate that predicting in latent space — rather than reconstructing pixels — yields world models capable of zero-shot planning on real hardware.

However, these systems use billion-parameter models trained on millions of hours of data. Nobody has systematically studied the lower boundary: how small can the model be, how little data does it need, and at what point does imagination-based planning stop outperforming reactive control?

Tabletop billiards is an ideal domain for this investigation because:

- The physics are rich — multi-body collisions, cushion bounces, friction, momentum transfer — making the world model load-bearing rather than decorative.
- Indirect manipulation through a cue ball creates multi-step causal chains (robot strikes cue ball → cue ball hits target ball → target ball bounces off cushion) that a reactive policy fundamentally cannot solve — only a system that imagines futures can plan through these chains.
- The two-phase action structure (continuous positioning + committed strike) creates a natural hierarchical planning problem that tests the world model at multiple timescales.
- Outcomes are easily evaluated automatically via ball positions from the overhead camera.
- Experimental throughput is high (~60–100 trials/hour) because the arena is in-room and permanently set up.

## 4. System Architecture

### Physical Setup

- 26-inch tabletop billiards table (in-room, permanently set up)
- Small differential-drive robot with an onboard solenoid striker, operating on the table surface
- Overhead USB webcam (~1m above surface)
- ESP32 on the robot, receiving commands over WiFi and streaming encoder telemetry

### The Robot — Hybrid Mobile Striker

A small diff-drive platform (~14cm × 20cm) with two geared DC motors (with encoders) at the rear, a ball caster at the front, and a solenoid-and-plunger assembly mounted on the front, aimed along the robot's heading. The robot does NOT push balls with its body — it drives into position behind the cue ball, rotates until its heading aligns with the desired shot line, then fires the solenoid to strike the cue ball with a controlled impulse. The cue ball then travels across the table and collides with target balls.

This hybrid approach gives you:

- **Continuous closed-loop control during positioning** — the robot drives around the table, adjusting angle and distance to the cue ball using camera feedback at every timestep.
- **Committed single-impulse strike** — once aligned, the solenoid fires and the cue ball is on its own, creating the indirect multi-contact chain that makes the prediction problem genuinely hard.
- **Real pool** — a cue ball, target balls, pockets, cushion bounces, and an agent choosing shots. Not object rearrangement in a box.

### Two-Phase Action Space

The robot's actions are hierarchical:

- **Phase 1 — Positioning (continuous):** (left_pwm, right_pwm) at every control timestep (~10Hz). The robot navigates to a firing position behind the cue ball and rotates to align its heading with the desired shot line. This phase is fully closed-loop — the overhead camera provides real-time feedback and the planner replans at every step.
- **Phase 2 — Strike (committed):** (fire: true, fire_force: float). The solenoid fires, imparting an impulse to the cue ball. Once fired, the robot has no further control over the outcome — the cue ball's trajectory and all subsequent collisions are determined by physics. This is the moment where the world model's prediction accuracy matters most.

### Software Stack

| Layer | Domain | Technology |
|-------|--------|------------|
| 1. Embedded | Motor control, solenoid trigger, encoder telemetry, UDP interface, safety watchdog | C++, ESP32 |
| 2. Perception | Ball + robot detection by color/marker, camera-to-table homography, position extraction | Python, OpenCV |
| 3. Visual encoder | Small CNN (~1–2M params) compressing arena frames into compact embeddings | PyTorch |
| 4. Latent dynamics predictor | Ensemble of 5 MLPs (~500K params each), (latent, action) → next latent | PyTorch |
| 5. Hierarchical planner | Stage 1: MPC for robot positioning. Stage 2: CEM for strike force/angle selection | Python |
| 6. Language conditioning | Encode instructions ("sink the red ball in the corner pocket") into goal representations | Python, API or local model |
| 7. Optimized inference | Hand-written C++ with AVX2 SIMD, batched CEM loop, zero framework overhead | C++ |

### Encoder Training Objective

JEPA-style self-supervised: given two frames separated by a few timesteps, the encoder produces embeddings whose difference is predictable from the actions taken between them. No pixel reconstruction (no decoder), no contrastive negatives. The encoder learns to compress away visual noise and retain only physically relevant state.

### Why Latent Prediction, Not Pixel Prediction

Pixel-space prediction wastes model capacity on irrelevant details — shadows, lighting variation, surface texture, camera noise. Latent prediction forces the encoder to learn a compact representation of the physics, enabling accurate multi-step rollouts with a tiny model. This is the core JEPA insight (LeCun, 2022; Assran et al., 2025).

### Hierarchical Planning

At each turn, the planner operates in two stages:

**Stage 1 — Positioning (MPC, closed-loop):**
1. Determine the ideal firing position: where should the robot be, and at what heading, to send the cue ball toward the target?
2. Plan a trajectory to reach that pose using CEM/MPPI through the world model's positioning dynamics
3. Execute one step, observe, replan — standard model-predictive control at ~10Hz
4. Continue until the robot reports it's in position and aligned

**Stage 2 — Shot selection (CEM, single decision):**
1. Given the current robot pose (aligned behind the cue ball), sample 200 candidate strike forces
2. For each, roll the world model forward: predict robot-to-cue-ball contact → cue ball trajectory → cue-ball-to-target-ball collision → target ball final position
3. Score each imagined outcome against the language-specified goal ("red ball in the corner pocket")
4. Keep top 20, resample, repeat 3 CEM iterations
5. Fire the solenoid at the best force

This decomposition means the world model is tested at two timescales: fast, incremental positioning dynamics (Phase 1) and slow, multi-contact chain predictions (Phase 2).

## 5. Inference Optimization (SQ5)

### Why This Is Non-Trivial Despite Small Models

Individual forward passes through ~500K-param MLPs are microseconds on any hardware. The bottleneck is the planner: Stage 2 alone requires 200 candidates × 5 models × 10 prediction steps × 3 CEM iterations = ~30,000 forward passes per shot decision. In Python/PyTorch with naive looping, interpreter overhead per call dominates actual compute.

### Optimization Strategy (CPU-first, no GPU required)

1. **Export weights.** Dump trained PyTorch models to raw float arrays. No framework, no ONNX runtime.
2. **Hand-write the forward pass in C++.** MLP forward pass = matrix-vector multiply + bias + ReLU. Use AVX2 SIMD intrinsics for the matmuls.
3. **Batch the CEM loop.** Stack all 200 candidate states into a (200 × dim) matrix. Run all 200 through each predictor in one batched multiply instead of 200 individual calls. This collapses 30,000 small operations into ~150 large batched operations.
4. **Fuse the ensemble.** Stack all 5 predictor weight matrices and run the ensemble as a single (5 × 200 × dim) batched operation.

### What Gets Measured

Run the same CEM planner (200 candidates, 10 steps, 3 iterations, 5 models) on three backends:

- Python + PyTorch (naive loop)
- Python + PyTorch (batched tensor operations)
- C++ with AVX2 SIMD (batched, no framework)

Optional fourth: C++ with CUDA/Triton (when GPU access is available — not at dorm, available at home).

Measure planning latency in ms. Then test whether faster planning actually improves task success.

### The Thesis

If a laptop CPU with AVX2 runs CEM planning at 100+ Hz for models this small, GPU acceleration adds nothing — and that is itself a result worth reporting. The contribution is demonstrating that frontier Physical AI planning doesn't require expensive hardware when the world model is appropriately sized.

## 6. Methodology

### Team Structure — Two Parallel Tracks

This is a two-person project with similar technical backgrounds. Work splits into parallel tracks that share a common interface and merge at defined checkpoints.

**Shared contract (defined in Week 0):** `env/base_env.py` — the interface that both the simulator and the real robot implement. Action format, observation format, and data logging format are agreed upon before any code is written.

**Track 1 (ML track):** Simulator, encoder, predictor, planner, active exploration, evaluation harness — entirely software, zero hardware dependency.

**Track 2 (Hardware track):** Robot chassis assembly, ESP32 firmware, solenoid striker, camera mounting and calibration, WiFi command interface, `env/real_env.py` implementing the same `BaseEnv` interface.

Tracks are independent for weeks 1–8, merge at week 8 when the simulator is swapped for real hardware.

### Phase 0 — Interface Contract (Week 0)

Both team members agree on:
- Action format: `{left_pwm: float, right_pwm: float, fire: bool, fire_force: float}`
- Observation format: `{frame: ndarray, robot_pose: (x,y,θ), ball_positions: [(x,y), ...], ball_settled: bool}`
- Data logging format: `(frame_t, action_t, frame_{t+1})` tuples stored as HDF5 or numpy archives
- `env/base_env.py` written and committed

### Phase 1 — Simulation + Hardware Build (Weeks 1–4, parallel)

**ML track:**
- Build 2D billiards simulator using pymunk (robot, cue ball, target balls, cushions, solenoid strike)
- Collect random interaction data in simulation
- Train first encoder and predictor, verify basic prediction accuracy

**Hardware track:**
- Order and assemble the diff-drive robot (ESP32, motors, driver, chassis, caster, battery)
- Mount and wire the solenoid striker
- Flash ESP32 firmware: motor control, solenoid trigger, UDP command interface, safety watchdog
- Milestone: robot drives on command and solenoid fires reliably

### Phase 2 — Full ML Pipeline + Perception (Weeks 4–8, parallel)

**ML track:**
- Finish ensemble predictor, CEM planner (both stages), active exploration
- Validate all 5 sub-questions against the simulator
- Produce draft figures for every experimental comparison

**Hardware track:**
- Mount and calibrate overhead camera (homography, ball/robot tracking via OpenCV)
- Implement `env/real_env.py` with the same interface as `sim_env.py`
- Milestone: send commands from laptop → robot moves → camera reports accurate ball positions

### Phase 3 — Merge: Sim-to-Real (Weeks 8–10)

Swap `sim_env.py` for `real_env.py` behind the same `BaseEnv` interface. Collect real-world data (~5,000–10,000 shots via random actions). Retrain world model on real data. Measure sim-to-real prediction gap. Optionally pretrain on sim data and fine-tune on real data.

### Phase 4 — Real-World Planning + Language Conditioning (Weeks 10–14)

Deploy hierarchical planner on real table. Test goal-directed shots with natural language instructions ("sink the red ball in the corner pocket," "bank the blue off the far cushion"). Measure task success rate.

### Phase 5 — Active Exploration & Ablations (Weeks 14–18)

Run the four-way comparison:

- Random shots (no model)
- Scripted heuristic (aim directly at target, no model)
- Learned model + hierarchical planner, trained on random data
- Learned model + hierarchical planner, trained on actively collected data

Collect all five experimental figures.

### Phase 6 — Systems Optimization (Weeks 18–24, stretch)

Port encoder + predictor + planner to C++/AVX2. Benchmark latency. Test whether faster replanning improves shot accuracy. Optional GPU comparison if hardware access is available.

## 7. Evaluation Metrics

| Metric | Sub-Question | What It Measures |
|--------|-------------|------------------|
| Prediction error vs. rollout horizon (latent vs. pixel) | SQ1 | World model accuracy over multi-step imagined futures |
| Task success rate: reactive vs. world-model planner | SQ2 | Whether imagination helps |
| Learning curve: active vs. passive data collection | SQ3 | Whether uncertainty-driven exploration improves sample efficiency |
| Calibration plot: ensemble disagreement vs. actual error | SQ4 | Whether the model knows what it doesn't know |
| Planning latency: Python vs. C++ | SQ5 | Systems optimization impact |
| Success rate vs. control frequency | SQ5 | Whether faster replanning improves physical outcomes |

### Task Success Criteria

- **Direct shots:** cue ball strikes target ball, target ball ends within 5cm of specified position (e.g., near a pocket). Automatically evaluated via overhead camera.
- **Multi-collision shots ("hit red so it knocks blue into the corner"):** final target ball reaches target regardless of intermediate trajectory. Tests multi-step causal prediction through the cue ball.
- **Positioning accuracy:** robot reaches intended firing pose within 2cm / 5° of target. Tests Phase 1 MPC quality independently of shot outcome.

## 8. Hardware Bill of Materials

| Component | Est. Cost (C$) |
|-----------|----------------|
| Tabletop billiards table (26") | $26 |
| ESP32-DevKitC V4 | $12 |
| 2× JGA25-370 DC gear motors with encoders | $22 |
| TB6612FNG motor driver | $8 |
| 65mm rubber wheels (×2) | $6 |
| Ball caster (15mm) | $4 |
| 2S 7.4V 1500mAh LiPo + charger | $28 |
| LM2596 buck converter (7.4V → 5V) | $4 |
| Solenoid (12V push-type, ~20mm stroke) | $15 |
| MOSFET/transistor for solenoid switching | $3 |
| Solenoid mounting bracket (3D-printed) | $5 |
| Chassis plate (3D-printed or laser-cut) | $10 |
| USB webcam (1080p) | $25 |
| Camera stand / desk clamp arm | $15 |
| Wires, connectors, standoffs, misc | $10 |
| **Total** | **~C$193** |

Remaining ~C$7 is tight — source parts from AliExpress where possible for cheaper pricing. University 3D printers (WARG shop or engineering machine shop) eliminate the chassis/bracket cost (~$15 saved).

## 9. Repo Structure

```
pool-jepa/
├── project_proposal.md          # this document
├── env/
│   ├── base_env.py              # shared interface — the contract
│   ├── sim_env.py               # 2D pymunk simulator
│   └── real_env.py              # (later) real robot + camera
├── perception/
│   ├── encoder.py               # visual encoder (CNN, JEPA training)
│   └── tracking.py              # (later) OpenCV ball/robot tracking
├── models/
│   ├── predictor.py             # latent dynamics MLP
│   └── ensemble.py              # ensemble wrapper + uncertainty
├── planning/
│   ├── cem.py                   # CEM planner (shot selection)
│   ├── mpc.py                   # MPC controller (robot positioning)
│   └── active_explore.py        # uncertainty-driven action selection
├── data/
│   ├── collect.py               # data collection loops
│   └── datasets/                # logged tuples (gitignored)
├── firmware/                    # ESP32 C++ code
├── inference_cpp/               # optimized C++ inference (stretch)
├── experiments/
│   ├── train_encoder.py
│   ├── train_predictor.py
│   ├── run_planner.py
│   └── ablations.py            # four-way comparison + figures
├── configs/                     # hyperparameters
└── notebooks/                   # exploration (not production)
```

## 10. Key References

### Foundational — World Models & Latent Dynamics

- Hafner, D., Pasukonis, J., Ba, J., & Lillicrap, T. (2023). *Mastering Diverse Domains through World Models.* (DreamerV3). arXiv:2301.04104
- Hansen, N., Su, H., & Wang, X. (2024). *TD-MPC2: Scalable, Robust World Models for Continuous Control.* ICLR 2024. arXiv:2310.16828

### Visual World Models & Zero-Shot Planning

- Zhou, G., Pan, H., LeCun, Y., & Pinto, L. (2025). *DINO-WM: World Models on Pre-trained Visual Features Enable Zero-shot Planning.* ICML 2025. arXiv:2411.04983
- Assran, M. et al. (2025). *V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning.* arXiv:2506.09985

### Vision-Language-Action Models

- Kim, M.J. et al. (2024). *OpenVLA: An Open-Source Vision-Language-Action Model.* arXiv:2406.09246
- Sun, Y. et al. (2026). *VLA-JEPA: Enhancing Vision-Language-Action Models with Latent World Model.* arXiv:2602.10098

### Surveys

- (2026). *World Model for Robot Learning: A Comprehensive Survey.* arXiv:2605.00080
- (2026). *From World Models to World Action Models: A Concise Tutorial for Robotics.* arXiv:2607.00836

## 11. Expected Contributions

1. An empirical study mapping the minimum model scale and data requirements for imagination-based planning to outperform reactive control in indirect, multi-contact manipulation on real hardware
2. A self-contained, reproducible Physical AI benchmark on a ~C$193 platform where the robot actually plays pool — positions itself, aims, and strikes through a cue ball
3. A demonstration that active, uncertainty-driven exploration improves world model sample efficiency in a real multi-body collision domain with compounding prediction error through causal chains
4. Evidence on whether CPU-optimized inference (no GPU) is sufficient for real-time latent world model planning at this model scale
5. An open-source codebase spanning embedded control, self-supervised vision, latent dynamics modeling, hierarchical planning, and optimized C++ inferences