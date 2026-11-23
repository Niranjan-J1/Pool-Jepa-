# Hardware design (H1)

Specs for the real robot, table and camera, derived from the simulator and the project goals.
How decisions are made:
- Every requirement traces back to a goal or a sim constant.
- Each choice is a decision record: options, criteria, numbers with sources, a recommendation, the owner's decision, and what would make us revisit it.
- Analysis scripts in `hardware/analysis/` print every number the documents quote.

Budget: target C$150, up to C$200 when the extra is justified in writing.
CAD: Onshape.

## System at a glance
```
 overhead camera ──frames──► laptop: homography → encoder → planner ──UDP──► robot (ESP32)
   (D-002)                     (R1, B2-B4)                                    drive (D-004)
                                                                              striker (D-003)
 marker on robot top (D-007) ◄── pose from camera                             power (D-006)
```

## Decisions
| ID | Topic | Status |
|---|---|---|
| [D-001](decisions/D-001-system-scale.md) | System scale: robot footprint, table, balls | SPLIT into D-001a/b |
| [D-001a](decisions/D-001a-robot-packing.md) | Robot packing study (footprint) | PROPOSED |
| D-001b | Table choice (from D-001a footprint) | not started |
| D-002 | Camera | not started |
| D-003 | Striker (solenoid, driver, strike height) | not started |
| D-004 | Drive (motors, encoders, wheels) | not started |
| D-005 | Compute and comms (ESP32 variant, UDP, latency) | not started |
| D-006 | Power | not started |
| D-007 | Robot localization marker | not started |

Still to come: `requirements.md`, `budgets.md` + `bom.csv`, `icd.md`, the Onshape layout plan and the bench-test plan.
