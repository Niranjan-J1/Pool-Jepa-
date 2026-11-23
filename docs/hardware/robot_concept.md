# Robot concept v0 (rough design)

**Status:** DRAFT for your review, 2026-11-23.
- Built on D-001a (variant B envelope, approved) and D-001b (Soozier 38" table, estimated 86 × 43 cm playfield).
- Exact part choices are still open; they're made in the walkthroughs D-003 (striker), D-004 (drive), D-005 (compute), D-006 (power) and D-007 (marker).
- Numbers marked "est." come from `hardware/analysis/` scripts or listings, not measurements.

## 1. What it looks like
About **6.7 cm long × 9.3 cm wide × 4.5 cm tall**: a small, wide puck on two wheels, with a striker poking out of the front.

**Top view** (front = the direction it shoots):
```
              ◄──────────── 9.3 cm ────────────►
            ┌─────────────────────────────────────┐   ▲
            │ ▐█▌                             ▐█▌ │   │
   wheel ──►│ ▐█▌ [ motor L ]═══╤═══[ motor R ] ▐█▌ │   │
   (34 mm)  │ ▐█▌               │ axle line   ▐█▌ │  6.7 cm
            │        (caster)   │                 │   │
            │                [solenoid]──► plunger│   │
            └─────────────────────────────────────┘   ▼
                      ▲ rear                 front ▲ (strikes the cue ball here)
  (upper deck, above: LiPo · ESP32-C3 · motor driver · boost + capacitor · MARKER on the very top)
```

**Side view:**
```
   4.5 cm ┬  ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄  ← marker sticker (top plate)
          │  [LiPo][ESP32][drv][cap]   upper deck (electronics)
          │  ━━━━━━━━━━━━━━━━━━━━━━━  standoffs
          │    ◯ wheel   [solenoid]══► plunger axis at BALL-CENTRE height
   felt ──┴────◯────(caster)──────────────────●  cue ball
              axle 1.7 cm up         1.3–1.9 cm up (set by measured ball size)
```

Key geometry:
- **The striker axis sits exactly at ball-centre height**: 1.27 / 1.59 / 1.90 cm for 1" / 1.25" / 1.5" balls. It strikes horizontally, so there's no spin and no hopping, and the 2D sim stays valid (from the full-table discussion). The solenoid mount gets a height-adjustable slot, or shims, until the balls are measured.
- **Two wheels on one axle, plus a rear ball caster** for balance (differential drive, the same as the sim).
- **Motors end to end across the robot.** That's why it's wider than it is long (D-001a).
- **The marker on top** is how the camera tracks the robot's position and heading.

## 2. How it moves
**Differential drive:** two independently powered wheels, like a tank.
- Both forward means straight; opposite directions means spin in place; unequal speeds means a curve. These are the sim's actions, `left_pwm` and `right_pwm`.

| Quantity | Value | Source |
|---|---|---|
| Top speed | 0.3 m/s | sim `MAX_WHEEL_SPEED` |
| Wheel speed needed | about 169 rpm with 34 mm wheels | calc; N20 motors at about 1:100 gearing are in this range (D-004 confirms) |
| Wheel track | about 8.0 cm | from the envelope |
| Max spin rate | about 7.5 rad/s (430°/s) | 2 × 0.3 / 0.080 (the sim uses 6 cm, so 10 rad/s; updated in the re-target step) |
| Crossing the table | about 2.9 s for 86 cm | calc |
| Positioning target | within 2 cm / 5° of the firing pose | proposal, success metric |

**Who steers:** the **laptop**, not the robot. The robot just obeys "left wheel X, right wheel Y" commands at about 30 Hz. The laptop closes the loop using the camera: it sees the marker, computes where the robot is, and corrects. This is the existing `planning/aim_controller.py` (`AimController`, with `firing_pose_reachable` already checking that the robot fits behind the ball). Encoders are optional because the camera supplies the pose; D-004 decides.

**Parallax correction:** the marker sits about 4.5 cm above the felt. Seen from 0.66 m up, near the table edge, it appears **about 3 cm off** from where the robot really is. That's easy to correct exactly, because the marker height is known. It must be included in the pose code (D-007).

## 3. How it operates (one shot)
```
 LAPTOP                                                ROBOT (ESP32)
 1. camera frame ─► perception ─► planner imagines shots,
    drops infeasible ones, picks (heading, force)
 2. computes the firing pose behind the cue ball
 3. loop @30 Hz: see marker ─► pose ─► AimController ─► DRIVE(left, right) ─►  spin wheels
    (stops ~5 cm behind the cue ball, aligned along the shot)                  watchdog: no command
 4. robot stopped ─► re-check the still frame (Stage 2: imagine forces)        for 200 ms ⇒ STOP
 5. ─────────────────────────────────────────────── FIRE(force) ────────────►  pulse solenoid for t(force)
                                                                               (from a calibration curve),
                                                                               recharge capacitor
 6. wait until all balls settle (camera) ─► read the result ─► log it ─► next shot
```
- **Force control:** "force" (0–1 in the sim) maps to a **solenoid pulse length**, or to a capacitor charge voltage (D-003). A calibration curve, measured once with slow-motion phone video, maps the sim force to real ball speed. The sim's `MAX_STRIKE_IMPULSE` (0.05 N·s) is the upper end.
- **After firing,** the robot stays put. Balls can bounce off it, which the sim already models because the robot is a body there.
- **Safety:** firmware watchdog (stop if commands stop), solenoid duty-cycle limit (no overheating), and a capacitor bleed resistor (D-006).

## 4. What's still open (decided in the walkthroughs)
| Question | Decision | Why it matters |
|---|---|---|
| Is the mini solenoid strong enough (≤0.05 J) for 0.06–0.08 J shots? Capacitor boost? | D-003 | the hardest-shot speed |
| Encoders or not; N20 gear ratio | D-004 | width (9.3 vs 11.1 cm), speed control |
| ESP32-C3 + Wi-Fi UDP, delay budget | D-005 | Stage 1 alignment accuracy |
| 1S LiPo + boost vs 2S; runtime; safe charging | D-006 | solenoid energy, safety |
| Marker type and size (ArUco vs two-dot) | D-007 | pose accuracy; the top plate is about 6 × 8 cm usable |
| Camera (parked mid-walkthrough) | D-002 | step 1 approved; step 2 waits on your phone and mount answers |

## 5. Next: CAD
Once you're happy with this concept, the Onshape layout follows. Two plates (lower and upper deck), motor clips, a solenoid cradle with a height slot, a caster mount and standoffs, laid out to the D-001a part dimensions. You'd build it, with my step-by-step guidance and dimension checks.
