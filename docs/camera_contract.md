# Camera contract: what the sim assumes and what the real camera must meet

The encoder and every probe are trained on sim frames. A real camera only "fits" if, after
calibration and warping, its frames look geometrically like these. Numbers below come from
`env/sim_env.py` unless marked as an estimate.

## 1. What the sim camera is (fixed by the code)

| Property | Value | Source |
|---|---|---|
| Frame size | 432 x 240 px, RGB uint8 | `RENDER_WIDTH/HEIGHT` |
| View | 0.7204 m x 0.4002 m (playfield + 3 cm margin each side in x) | `RENDER_MARGIN` |
| Scale | ~600 px/m (1.67 mm/px), same in x and y | derived |
| Projection | perfectly top-down orthographic, no lens distortion, no perspective | `render()` |
| Playfield | 0.6604 x 0.3302 m (26" x 13") | `TABLE_LENGTH/WIDTH` |
| Ball | 2.0 cm diameter -> ~12 px | `BALL_RADIUS = 0.01` |
| Robot | 8 x 6 cm -> ~48 x 36 px | `ROBOT_LENGTH/WIDTH` |
| Rail (multiball) | 6 mm collision radius; felt drawn to hl - 6 mm | `MULTIBALL_CUSHION_RADIUS` |
| Pockets | 6, capture radius 2.8 cm at the corners and long-rail midpoints | `POCKET_RADIUS` |
| Time step | 1/60 s per control step | `DT` |
| Max ball speed | 2.5 m/s (max impulse 0.05 N*s on a 20 g ball) | `MAX_STRIKE_IMPULSE` |
| Lighting | flat, constant colour, no shadows, no noise | `render()` |
| Encoder input | two stacked frames, empty-table background subtracted | `experiments/train_encoder.py` |

## 2. Real-frame pipeline (to be built in bucket R)

1. **Lens calibration**, once per camera: checkerboard plus `cv2.calibrateCamera`. Target
   reprojection error < 0.5 px. Undistort every frame.
2. **Homography**, once per mounting: click the 4 playfield corners (the corner pocket
   centres), then warp onto the canonical 432 x 240 view at 600 px/m.
3. **Background:** median of empty-table frames taken under the real lighting, subtracted
   exactly as in the sim.

This maps any adequate camera onto the sim's geometry. The raw resolution then only has to
be high enough to resample down cleanly.

## 3. Minimum spec for buying the camera

| Requirement | Minimum | Why |
|---|---|---|
| Resolution over the view | >= 600 px/m raw; aim for >= 1200 px/m | so the warp to 600 px/m downsamples, never upsamples |
| Coverage | the whole playfield + rails in frame, with margin | the homography needs all 4 corners |
| Frame rate | >= 30 fps; 60 fps preferred | the sim runs at 60 Hz; Stage 1 replans each step |
| Exposure | manually lockable, <= 1/250 s | at 2.5 m/s, 1/250 s gives 1 cm (6 px) of blur; auto-exposure changes the background |
| Focus, white balance | manually lockable or fixed | background subtraction assumes a stable image |
| Interface | UVC (plain USB webcam) | works with OpenCV on Linux and Windows without vendor drivers |
| Mount | rigid, optical axis within ~5 deg of vertical | keeps the homography valid between sessions |

**Estimate (verify against the real camera spec):** a 1080p webcam with about 70 deg
horizontal FOV, mounted at 1 m, covers about 1.40 m, i.e. about 1370 px/m. That clears both
the minimum and the target. A wider lens (about 90 deg) gives about 960 px/m, still above the
minimum but with more distortion to calibrate out.

**Perspective residual (estimate):** ball centres sit 1 cm above the felt. Near the table
edge (about 18 deg off-axis at 1 m) they appear shifted by about 3 mm. That is small next to
the current 1.3 cm readout error, but it is not zero; a higher mount reduces it.

**BOM note:** the proposal budgets C$25 for the webcam. Cameras with manual exposure and
focus control often cost more, so check the controls (e.g. `v4l2-ctl --list-ctrls` on Linux)
before buying on price alone.

## 4. Physical tolerances: measure these when the table arrives

The sim constants are assumptions, not measurements. Measure the real table and balls, then:

| Quantity | Sim value | If the real value differs by more than... | Then |
|---|---|---|---|
| Ball diameter | 2.0 cm | 10% | update `BALL_RADIUS`, re-collect B1 data, retrain |
| Playfield L x W | 66.0 x 33.0 cm | 3% | update `TABLE_LENGTH/WIDTH`, re-collect |
| Pocket mouth | 2.8 cm capture radius | 20% | update `POCKET_RADIUS` |
| Ball mass | 20 g | 25% | update `BALL_MASS`; this rescales the force-impulse calibration |

Open question: does "26 inch" refer to the playfield or the outer frame? The sim assumes the
playfield. If it's the outer frame, the playfield is smaller and the ball-to-table ratio
changes.
