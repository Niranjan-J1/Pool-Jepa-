# D-001b: Table choice

**Status:** PROPOSED. Waiting for your approval.
**Date:** 2026-11-23
**Inputs:**
- Robot envelope 6.7 × 9.3 cm (D-001a, variant B).
- R-REACH = sim parity: ≥55% of shots reachable counting rails only, and ≥43% with 5 other balls.
- Budget: C$150 target, C$200 stretch.

**Caveat:** retailers list the **outer** size, almost never the playfield (cushion nose to cushion nose). The playfields below are ESTIMATES: outer size minus about 1.5" of rail per side, plus a pessimistic case. Confirm the real playfield before you buy, or buy where returns are easy.

## Candidates
| Table | Outer size | Price (CAD) | Playfield (est. / pessimistic) | Reach rails / +5 balls (est.) | Reach (pessimistic) | Meets parity? |
|---|---|---|---|---|---|---|
| Toys"R"Us 27" tabletop | 27" (68.5 cm) | **C$69.99** ([Toys"R"Us](https://www.toysrus.ca/en/27---68.5cm--Table-Tabletop-Pool/248A7AB9.html)) | 61×29 / 58×26 cm | 52% / 37% | 48% / 31% | **No** |
| RAYKEEP 32" (removable legs) | 32 × 15.5" | about C$70, unverified ([Wayfair.ca listing](https://www.wayfair.ca/furniture/pdp/raychee-2-bych1045.html); [reseller](https://bltv.de/products/raykeep-32-mini-pool-table-removable-portable-billiards-tabl/216684153/), currency unclear) | 74×32 / 71×29 cm | **58% / 44%** | 54% / 40% | **Yes (estimate), borderline pessimistic** |
| Soozier 38" folding tabletop | 38" | **C$93.99** ([Staples.ca](https://www.staples.ca/products/3146613-en-soozier-38-folding-mini-pool-table-set-tabletop-billiards-game-with-cues)), C$94.99 ([The Brick](https://www.thebrick.com/products/soozier-38-folding-mini-pool-table-set-tabletop-billiards-game-with-cues)) | 86×43 / 81×38 cm | **67% / 57%** | 63% / 52% | **Yes, with margin** |
| 40" Hathaway (reference) | 40" | C$169.99 ([Canadian Tire](https://www.canadiantire.ca/en/pdp/hathaway-breakout-table-top-pool-billiards-table-w-balls-cues-triangle-blue-40-in-3742683p.html)) | ~94×46 cm | ~71% / ~65% | — | yes, but it's the whole budget |

Ball size is unknown for all of them: listings say anything from 1" to 1.5" ([guide](https://bargames101.com/how-to-find-the-best-mini-pool-table-with-5-top-picks-for-all-ages/)). It barely changes reach (rows computed for 2.54 cm; 3.8 cm balls differ by 2 points at most), but it does set the striker height (D-003) and the sim's `BALL_RADIUS`.

## Trade-offs between the two that qualify
| | RAYKEEP 32" | Soozier 38" |
|---|---|---|
| Reach margin over parity | about +1 point (none if pessimistic) | about +14 points |
| Cost | ~C$70 | ~C$94 (+C$24) |
| Sim change | playfield 74×32 is near the current 66×33, a small change | 86×43 is a bigger change: more px needed or a lower px/m (see below) |
| Camera (D-002) | about 0.9 m across including margin; easy at 1 m height | about 1.0 m across; still fine with a 70° webcam at 1 m (covers about 1.4 m) |
| Encoder input | at 600 px/m the frame grows to about 480×230 px | at 600 px/m about 552×294 px; or keep 432×240 and drop to ~470 px/m (balls about 12 px vs 15 px) |
| Budget left for robot + camera | ~C$80–130 | ~C$56–106 |

On the encoder input: today's frame is 432×240 at 600 px/m for a 66 cm table. A bigger table means either a bigger frame, which is a B2 retrain with a new input size and is cheap since we retrain anyway after the measured-sim update, or fewer px per ball. Either is fine. It's a choice we make at the sim update.

## Recommendation
**Soozier 38" folding tabletop (about C$94).**
- It's the only cheap option that meets parity **with margin** even in the pessimistic case (52% with 5 balls vs the 43% target).
- The RAYKEEP's estimate only just passes and could fail once it's measured.
- The extra C$24 buys about 14 points of shot freedom. That's the justification for spending above the cheapest option.
- It's sold by Canadian retailers with normal return policies.

**Budget check, estimate only (D-002 to D-006 will firm it up):**
- table C$94;
- robot parts about C$70–90 (ESP32-C3, 2× N20, DRV8833, mini solenoid, 1S LiPo, boost, caster and wheels, misc);
- camera C$0 (phone) to C$35.

Total about **C$165–220**. Inside the stretch range if the camera is your phone or a cheap webcam. D-002 decides this.

**Before buying:**
1. Check the listing's ball size and the product photos for pockets (some tabletop sets have no pocket nets or have shallow pockets).
2. Make sure it can sit flat on a table, legs folded or removed. The robot needs a flat, rigid playfield.
3. Keep the receipt and box, and measure everything with the D-001 checklist as soon as it arrives.

**Reasons to revisit:**
- the measured playfield is under 80 × 38 cm;
- the felt is too soft or uneven for the robot's wheels (check in the first week);
- the budget for the rest of the parts comes in over C$110.

## Your decision
- [ ] Approve the Soozier 38" (buy now)
- [ ] Or choose the RAYKEEP 32" (cheaper; very little reach margin)
- Overrides / notes:
