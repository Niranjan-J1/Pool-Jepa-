# D-001a: Robot packing study (smallest realistic footprint)

**Status:** PROPOSED. Waiting for your approval.
**Date:** 2026-11-23
**Why:** you chose to design the robot first and then pick a table to fit it. This study takes the parts the robot needs, lays them out in two decks, and gets the smallest footprint that actually holds them. The exact part choices come later, in D-003 to D-007; here only their sizes matter.

## Parts and sizes (mm)
| Part | L × W × H | Source |
|---|---|---|
| N20 gear motor + encoder | 12 × 41.5 × 10 | [SparkFun N20 pair](https://www.sparkfun.com/n20-motor-with-encoder-and-cable-pair.html) |
| N20 without encoder | 12 × 32.5 × 10 | Pololu micro metal gearmotor (32.5 overall) |
| Wheel | 34 × 7 | ESTIMATE (common N20 wheel) |
| Mini push solenoid (JF-0530B) | 30 × 13 × 15, 10 mm stroke, ~5 N | [listing](https://www.oomipood.ee/en/product/solenoid_klapp_12v_2a_5n_13x165x20mm_h10mm); specs vary by seller ([Funduino](https://funduinoshop.com/en/electronic-modules/other/electromagnets/electromagnet-opening/closing/pulling-jf-0530b-5v-0-7a)) |
| Medium solenoid | 40 × 20 × 20 | ESTIMATE (next size up) |
| Ball caster | 15 × 15 | ESTIMATE |
| ESP32-C3 SuperMini | 22.5 × 18 | [mischianti](https://mischianti.org/esp32-c3-super-mini-high-resolution-pinout-datasheet-and-specs/) |
| DRV8833 motor driver | 20.3 × 12.7 | [Pololu 2130](https://www.pololu.com/product/2130) |
| 1S 500 mAh LiPo | 39 × 25 × 8.5 or 58 × 18 × 8 | [Turnigy](https://hobbyking.com/en_us/turnigy-nano-tech-500mah-1s-25-50c-lipo-pack-losi-mini-compatible.html), [Tattu](https://www.getfpv.com/batteries/micro-quad-flight-batteries/tattu-r-line-version-1-0-500mah-1s1p-95c-lipo-battery.html) |
| Boost module, capacitor + MOSFET | 36 × 17; 20 × 15 | ESTIMATE |

## Layout (top-down; x = forward along the shot)
```
 lower deck                                   upper deck (over the lower deck)
 ┌───────────────────────────────┐           ┌───────────────────────────────┐
 │ wheel│ motor ══╪══ motor │wheel│  axle     │ LiPo   ESP32-C3   DRV8833     │
 │      │   [caster]  [solenoid]►│  row      │ boost  cap+MOSFET   MARKER ▣   │
 └───────────────────────────────┘           └───────────────────────────────┘
   rear                         front (striker)
```
The two motors sit end to end across the robot, so **motor length sets the width**: the robot comes out wider than it is long.

## Results (`python hardware/analysis/packing.py`)
| Variant | Footprint L × W | Upper deck fits | Reach on 66×33: rails only / with 5 other balls | Smallest 2:1 table for ≥60% (with 5 balls) |
|---|---|---|---|---|
| A: N20 + encoders, mini solenoid | 6.7 × 11.1 cm | yes (47% used) | 54.0% / 38.9% | 100 × 50 cm |
| **B: N20 without encoders, mini solenoid** | **6.7 × 9.3 cm** | yes (57%) | **57.4% / 43.3%** | 92 × 46 cm |
| C: N20 + encoders, medium solenoid | 7.7 × 11.1 cm | yes (41%) | 51.6% / 35.8% | 106 × 53 cm |
| For reference: sim robot 8 × 6 cm | — | — | 58.4% / 45.4% | — |

## What this means
1. **Variant B roughly matches the sim robot** on a 66 × 33 table (57/43% vs 58/45%). The planner would have about the same shot freedom on the real table as in sim. Variant B drops the wheel encoders, which is reasonable because the robot's pose comes from the camera marker (D-007). Whether we need encoders is a D-004 question.
2. **The ≥60% target counted with balls on the table is unrealistic in this budget.** It needs a playfield of about 92–106 cm, a 40"-class table, and those cost **about C$170–370 in Canada**:
   - [Canadian Tire Hathaway 40", C$169.99](https://www.canadiantire.ca/en/pdp/hathaway-breakout-table-top-pool-billiards-table-w-balls-cues-triangle-blue-40-in-3742683p.html)
   - [Walmart.ca, C$268](https://www.walmart.ca/en/c/kp/pool-table)
   - [Wayfair.ca, C$370](https://www.wayfair.ca/furniture/pdp/jeteho-40-pool-table-tabletop-set-for-game-compact-blue-with-cue-sticks-ball-kit-triangle-chalk-brush-jteh2266.html)
   
   That's the whole C$200 cap spent on the table alone.
3. **Even the sim doesn't reach 60% once balls are counted** (45%). So the target as I first stated it was stricter than the sim itself, which was my mistake. A fair target is **"sim parity"**: at least the sim robot's reach on the sim table, about **≥55% rails only and ≥43% with 5 balls**.

## Recommendation
- **Robot: variant B, about 6.7 × 9.3 cm** (N20 motors, mini solenoid, 1S LiPo, ESP32-C3), with encoders left open for D-004. If D-004 says encoders are needed, variant A (11.1 cm wide) still works, at a small loss of reach.
- **Change the requirement R-REACH to sim parity:** ≥55% rails only and ≥43% with 5 other balls, measured with `footprint.py` on the real table and robot dimensions.
- **Table (D-001b):** a mini table with a playfield near 66 × 33 cm is enough. Prefer one whose playfield is at least 60 × 30 cm. If 26" turns out to be the outer frame (about 56 × 28 cm), variant B gets about 37% with balls, so look for a slightly bigger table (about 28–32") instead.

**Flagged for D-003:** at 5 N over 10 mm, the mini solenoid gives at most about 0.05 J. A 20–25 g ball at 2.5 m/s needs about 0.06–0.08 J before any transfer losses. Either real shots need less speed than the sim's maximum (likely: a 66 cm table rarely needs 2.5 m/s), or we need a capacitor-boosted drive or the medium solenoid (variant C). D-003 will check this properly.

**Reasons to revisit:**
- a real part is more than 10% bigger than listed;
- D-004 requires encoders;
- D-003 requires the medium solenoid.

## Your decision
- [ ] Approve variant B as the design envelope (encoders open until D-004)
- [ ] Approve R-REACH = sim parity (≥55% rails only, ≥43% with 5 balls)
- [ ] Approve the table guidance (playfield ≥60 × 30 cm, mini table, about C$30–60)
- Overrides / notes:
