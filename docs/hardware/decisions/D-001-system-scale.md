# D-001: System scale (robot footprint, table, balls)

**Status:** SPLIT (2026-11-23). By your choice, the robot is designed first: see [D-001a](D-001a-robot-packing.md). This file's analysis and measurement checklist carry over into D-001b (table choice). R-REACH changed to "sim parity", see D-001a.
**Date:** 2026-11-23
**Why decide this first:** the robot's size, the table's size and the ball size set the sim constants, the camera's field of view, how big the marker has to be, how strong the striker has to be, and which shots the robot can physically take. Every later decision (D-002 to D-007) depends on this one.

## The problem
| | Sim (what B1/B2 data was made with) | Proposal hardware |
|---|---|---|
| Playfield | 66 × 33 cm (`TABLE_LENGTH/WIDTH`) | "26-inch tabletop table" (unmeasured) |
| Balls | 2.0 cm diameter (`BALL_RADIUS` 1 cm) | not specified |
| Robot | 8 × 6 cm (`ROBOT_LENGTH/WIDTH`) | about 20 × 14 cm, 65 mm wheels |

Three facts:
1. **Real mini-table balls are probably 1 inch (2.54 cm), not 2.0 cm.** Mini tabletop sets list 1" balls ([Hey! Play listing](https://www.walmart.com/ip/Hey-Play-Mini-Table-Top-Pool-Table-Set/582659193), [Toy Time listing](https://www.lowes.com/pd/Toy-Time-Trademark-Global-Mini-Pool-Table-Set-Tabletop-Party-Game/1002622258)). That's 27% off the sim, beyond the 10% tolerance in `docs/camera_contract.md`.
2. **"26 inch" may be the outer frame, not the playfield.** Listings rarely give the playfield size. For example, a 40" table has a 37 × 18" playfield ([Hathaway](https://www.lowes.com/pd/Hathaway-Sharp-Shooter-Indoor-Tabletop-Pool-Table/50285963)). If 26" is the frame, the playfield is only about 56 × 28 cm.
3. **The proposal robot is far too big for this table.** See the analysis below.

## Analysis (`python hardware/analysis/footprint.py`)
"Reachable" means: for a random ball position and a random required shot direction, the robot can park behind the ball (front edge 5 cm from the ball centre, pointing along the shot) with its whole body inside the rails. 200k random cases.

| Playfield | Balls | Robot 8×6 cm | Robot 10×8 cm | Robot 20×14 cm (proposal) |
|---|---|---|---|---|
| 66 × 33 cm (sim) | 2.0 cm | 57.7% | 49.2% | 18.7% |
| 66 × 33 cm | 2.54 cm | 58.4% | 49.8% | 18.8% |
| 56 × 28 cm (if 26" is the frame) | 2.54 cm | 52.6% | 43.4% | 10.7% |
| 94 × 46 cm (40" table) | 2.54 cm | 70.9% | 64.7% | — |
| 120 × 60 cm (DIY) | 2.54 cm | 77.4% | 72.5% | 52.4% |

How to read it:
- Even the sim robot can't take about 40% of arbitrary shots on a 66 cm table. The planner already copes by choosing a different target or pocket, so that's acceptable.
- The proposal robot can't take about 81%, and it covers 13.6% of the table. **Ruled out** for any table under about 1 m.
- Ball size barely changes reachability. **Table size** is what matters.

## Options
| | (a) Small robot, buy a mini table | (b) Small robot, bigger table (40" commercial or DIY) | (c) Proposal robot, DIY 120 cm table |
|---|---|---|---|
| Robot | 8–10 cm long; N20 micro gear motors, ESP32-C3 class board | same as (a) | 20 × 14 cm, as in the proposal |
| Reachable shots | 50–58% | 65–77% | 52% |
| Sim change needed | update constants to the measured table and balls, recollect B1 (about 1.5 h cluster), retrain B2 (about 1.3 h) | same, plus a bigger playfield; the camera needs a wider view (D-002) | same, plus building pockets and cushions yourself (hard to make consistent) |
| Cost (estimate) | table C$30–60 | 40" table C$60–100 (may push over C$150); DIY C$30–50 plus time | DIY table C$30–50; bigger motors and battery |
| Risk | packing parts into about 9 × 8 cm is tight (see below) | camera must cover about 1 m; table cost and space | homemade pockets and cushions behave inconsistently; it no longer looks like real pool |

**Packing check for (a):**
- Two N20 motors with encoders are each about 12 mm wide and 34 mm long. Mounted end to end across the robot, they need about 70 mm, so a 6 cm wide robot doesn't fit; about **8 cm wide** does.
- N20 motors with magnetic encoders cost about US$12.50 each ([Adafruit 4638](https://www.adafruit.com/product/4638?gQT=2)), or less as a pair ([SparkFun pair](https://www.sparkfun.com/n20-motor-with-encoder-and-cable-pair.html)) or from cheaper suppliers ([Waveshare](https://www.waveshare.com/product/robotics/motors-servos/motors/dcgm-n20-12v-en-200rpm.htm)). D-004 confirms prices in CAD.

## Recommendation
**(a): a small robot with a design envelope of at most 10 × 8 cm (target about 9 × 8 cm), on a bought mini table.** Plus two process changes:
1. **Buy the table early: now, not after H3.** It's cheap, and it's the biggest unknown: the playfield size, ball size and mass, pocket mouths and cushion bounce all feed the sim constants. Measure everything with the checklist below.
2. **Make the sim match the measured hardware once, before B3.** That means updating `TABLE_*`, `BALL_RADIUS`, `BALL_MASS`, `POCKET_RADIUS` and `ROBOT_*` in a new multiball-only constants profile (legacy stays protected), then recollecting B1 and retraining B2. The cost is about 3 cluster hours.
   - **Why before B3:** then B3/B4 results and the flagship post are on real-geometry sim. Doing it after B3 would mean redoing B3 too.
   - **The trade-off:** B3 waits for the table to arrive (about 1 week). In the meantime B4-prep, H1 and R1 continue.

**Why not (b) now:** reachability improves 50% → 65%, but at higher cost and space, and the camera has to cover twice the area. That halves px/m, and px/m is the perception accuracy we just trained for. We'd revisit (b) if the planner loses too many scenes to unreachable shots in B4.

**Reasons to revisit this decision:**
- the measured table is under about 56 cm long;
- the packed robot comes out over 10 cm;
- B4 shows more than 30% of test scenes have no reachable good shot.

## Table measurement checklist (when it arrives)
- Playfield length and width, cushion nose to cushion nose (mm).
- Ball diameter (calipers, 3 balls) and mass (kitchen scale, all balls / count).
- Pocket mouth width (corner and side) and pocket depth.
- Cushion height above the felt (it must sit above the ball centre for normal rebounds, and it limits how low the striker can go).
- A rough cushion-bounce test: roll a ball into the rail on video and compare speed in vs out. This sets `MULTIBALL_CUSHION_ELASTICITY`.
- Felt colour (photo under your room lighting), for B6.

## Your decision
- [ ] Approve (a) with the at-most-10×8 cm envelope
- [ ] Approve buying the table now
- [ ] Approve "match sim to the measured hardware before B3"
- Overrides / notes:
