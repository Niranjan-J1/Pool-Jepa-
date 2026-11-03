# Dynamics Probe — How to Read the Results

The probe trains the predictor on GROUND-TRUTH state (perfect ball/robot
positions), not learned latents. This isolates "can the dynamics model learn
the physics?" from "is the encoder any good?". It's the ceiling the learned-
latent version will later be measured against.

## Pass/fail: one-step ball-position error
- < 3 cm   → GOOD. Dynamics model learns collisions. Proceed to ensemble.
- 3–8 cm   → MARGINAL. Works but noisy; check if it's the strike steps dragging
             the average. Proceed with caution.
- > 8 cm   → BROKEN. Debug before building anything on top. Likely causes:
             not enough collision data, or predictor too small for contact
             dynamics.

Robot-pose error should be much smaller than ball error — driving is the easy
part. If robot error is also high, something is wrong with the whole setup.

## Pass/fail: multi-step rollout (error vs horizon)
- Error stays < 5 cm through 5+ steps → PLANNABLE. The CEM planner can imagine
  multi-step futures through this model. This is what Sub-Question 1 needs.
- Error explodes after 2–3 steps → the model is only good one step ahead.
  Not fatal, but flags that we may need a recurrent predictor (DreamerV3 RSSM)
  before the planner works well.

## Position-only vs with-velocity
- The --with-velocity run SHOULD be clearly better, especially right after a
  strike, because a single state snapshot can't tell how fast a ball is moving.
- If velocity barely helps → either the data lacks fast motion, or momentum
  isn't being captured correctly. Investigate.
- This confirms a known limitation: one frame ≠ velocity. It justifies feeding
  the predictor a short history (or recurrence) on real hardware.

## What this sets up for later (Sub-Question 1)
- This ground-truth-state error is the BEST CASE.
- When we later train the predictor on LEARNED LATENTS from the encoder, it will
  be worse. The gap between the two = "how much does imperfect perception cost
  us?" That comparison is a reportable result, not a failure.