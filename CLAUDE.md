# Project
Pool-JEPA: a small robot plays pool on a 26" tabletop table using a self-supervised JEPA-style encoder and a learned world model, planning shots by imagining outcomes. Validated in a 2D pymunk sim first; real hardware and real footage come later. Not a VLA until language goals exist.

# Layout
- env/ : sim_env.py (pymunk sim, legacy mode + multiball mode), base_env.py (interface)
- data/ : collect.py (data collection modes)
- perception/ : encoder.py (VisualEncoder, spatial-softmax keypoint head)
- models/ : shot_model.py (single-jump shot outcome model)
- planning/ : shot_planner.py, aim_controller.py
- experiments/ : training and eval scripts
- tests/ : regression_baseline.py, test_multiball.py
- project_proposal.md : the full plan. README.md : results so far.

# Commands
- Regression: python tests/regression_baseline.py (must print IDENTICAL on the laptop; on the cluster "OK: within 1e-9" is acceptable, from last-bit float rounding)
- Sim tests: python tests/test_multiball.py
- Encoder training: experiments/train_encoder.py; shot model: experiments/train_shot_model.py

# Environment
- Laptop (Windows) is for code. Training and big collection run on the WATcloud SLURM cluster (VPN required, rtx_3090, 2h job limit).
- Big data and checkpoints live on /mnt/wato-drive2 via the POOLJEPA_DATA env var, never in git or /home (quota).
- Checkpoints (encoder_best.pt, shot_model_bigcue.pt) and frames_background.npy exist only on the cluster. The background must be regenerated for multiball renders.
- Workflow: edit on laptop, push to GitHub, pull on the cluster. Always commit and push before I run anything on the cluster.
- Code must run on Python 3.10 (cluster). Do not use 3.11+ features.

# Conventions
- Units: sim in metres internally; multiball obs also gives ball_positions_cm.
- Ball ids: 0 = cue, 1..7 = object balls. Use ball_slots() for fixed-size masked arrays.

# Working agreement
- Work in buckets. After each bucket (or any step I mark as a checkpoint), STOP and report. Do not start the next one until I say so.
- Reports are plain text and numbers. No images, montages or visualizations unless I ask.
- Do not optimize performance unless it blocks the work. Report speed numbers, don't chase them.
- If an approach fails twice, stop and tell me what you tried instead of trying a third variant.
- Do not run long jobs, collections or SLURM jobs. Give me the exact commands and I run them on the cluster.
- Legacy behavior is protected: SimEnv() with no args and existing collect.py modes must stay unchanged. Run tests/regression_baseline.py after touching env/ and expect IDENTICAL.
- One commit per logical change. Before reporting "done", show `git status` (clean) and `git log origin/<branch>..HEAD` (empty). Never claim something is committed or pushed without showing this.
- Push back when a request conflicts with the project goals, and ask me when my intent is unclear. Prefer a short question over a guess.
- Honesty constraints for anything public: sim only until real footage exists; this is not a VLA until language goals exist.
- No Co-Authored-By trailers in commits.

# Decisions already made (do not re-litigate without new evidence)
- Sim first, then real footage test, then hardware.
- Self-supervised JEPA-style encoder with a spatial-softmax keypoint head, no decoder.
- Single-jump shot model plus force grid search for planning. Step-by-step latent prediction with CEM was tried and abandoned. Only build multi-step if single-jump demonstrably fails on multi-ball data.
- Data and checkpoints live on the cluster (POOLJEPA_DATA), not in git.