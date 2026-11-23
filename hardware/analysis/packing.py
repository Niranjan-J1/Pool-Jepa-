"""D-001a packing study: smallest robot footprint that holds the parts, before picking a table.

Two-deck layout (top-down, x = forward along the shot, y = left):
  lower deck: striker solenoid at the front centre, the motor axle row (two motors end to end
              across the robot, wheels outside them), a ball caster at the rear.
  upper deck: battery, microcontroller, motor driver, solenoid boost/capacitor. Marker on top.
The footprint is the bounding box of the lower deck (wheels included); the upper deck only has
to fit inside it. Dimensions are datasheet/listing values (mm, sources in D-001a) or ESTIMATES.

Run: python hardware/analysis/packing.py   (instant)
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

# ---- part dimensions, mm: (length along x, width along y, height); source tag
PARTS = {
    "n20_enc":    ((12.0, 41.5, 10.0), "SparkFun N20 + encoder, 10x12x41.5"),
    "n20_noenc":  ((12.0, 32.5, 10.0), "Pololu micro metal gearmotor, 32.5 overall"),
    "wheel34":    ((34.0, 7.0, 34.0), "ESTIMATE: common 34 x 7 mm N20 wheel"),
    "sol_mini":   ((30.0, 13.0, 15.0), "JF-0530B body ~30 x 13 x 15, stroke 10"),
    "sol_medium": ((40.0, 20.0, 20.0), "ESTIMATE: next size up (~10-20 N class)"),
    "caster":     ((15.0, 15.0, 10.0), "ESTIMATE: 3/8 in ball caster"),
    "esp32c3":    ((22.5, 18.0, 5.0), "ESP32-C3 SuperMini 22.5 x 18"),
    "drv8833":    ((20.3, 12.7, 3.0), "Pololu DRV8833 carrier 0.8 x 0.5 in"),
    "lipo_short": ((39.0, 25.0, 8.5), "Turnigy nano-tech 1S 500 mAh 39 x 25 x 8.5"),
    "lipo_long":  ((58.0, 18.0, 8.0), "Tattu R-Line 1S 500 mAh 58 x 18 x 8"),
    "boost":      ((36.0, 17.0, 6.0), "ESTIMATE: MT3608-class boost module"),
    "cap_mosfet": ((20.0, 15.0, 13.0), "ESTIMATE: 470-1000 uF cap + MOSFET + diode"),
}
GAP = 2.0          # mm clearance between parts
SHAFT = 3.0        # mm from motor end to wheel inner face (hub)
EDGE = 3.0         # mm chassis edge margin
STROKE = 10.0      # mm plunger travel ahead of the solenoid body (front of robot = body front)
MARKER_MIN = 30.0  # mm square needed on top for a fiducial (refined in D-007)


def L(p):
    return PARTS[p][0][0]


def W(p):
    return PARTS[p][0][1]


def layout(motor, solenoid, battery):
    """Return (length_x, width_y, notes) of the lower-deck bounding box, plus an upper-deck fit check."""
    # Width: wheel | shaft | motor | gap | motor | shaft | wheel  (+ edges)
    width = 2 * (W("wheel34") + SHAFT + W(motor)) + GAP + 2 * EDGE
    # Length along x, rear -> front: caster | gap | axle row | gap | solenoid body | edges.
    # The wheels (34 mm) straddle the axle; they sit outside the centre column, so they only
    # matter if they stick out past the rear or front of the centre stack.
    centre = L("caster") + GAP + L(motor) + GAP + L(solenoid)
    axle_x = EDGE + L("caster") + GAP + L(motor) / 2
    wheel_rear, wheel_front = axle_x - L("wheel34") / 2, axle_x + L("wheel34") / 2
    length = max(centre + 2 * EDGE, wheel_front + EDGE) - min(0.0, wheel_rear - EDGE)
    # Upper deck: parts laid flat side by side must fit in length x width (area check + longest side).
    upper = [battery, "esp32c3", "drv8833", "boost", "cap_mosfet"]
    area = sum((L(p) + GAP) * (W(p) + GAP) for p in upper)
    deck = (length - 2 * EDGE) * (width - 2 * EDGE)
    longest = max(max(L(p), W(p)) for p in upper)
    fits = area <= 0.85 * deck and longest <= max(length, width) - 2 * EDGE
    marker_ok = min(length, width) - 2 * EDGE >= MARKER_MIN
    return length, width, area / deck, fits, marker_ok


def main():
    from footprint import TABLE_LENGTH, TABLE_WIDTH, reachable_fraction
    variants = [
        ("A: N20+encoders, mini solenoid", "n20_enc", "sol_mini", "lipo_short"),
        ("B: N20 no encoders, mini solenoid", "n20_noenc", "sol_mini", "lipo_short"),
        ("C: N20+encoders, medium solenoid", "n20_enc", "sol_medium", "lipo_short"),
        ("D: N20+encoders, mini sol, long LiPo", "n20_enc", "sol_mini", "lipo_long"),
    ]
    print("parts (mm, L x W x H):")
    for k, ((l, w, h), src) in PARTS.items():
        print(f"  {k:<11} {l:>5.1f} x {w:>5.1f} x {h:>4.1f}   {src}")
    print(f"\n{'variant':<40} {'L x W (cm)':>12} {'upper deck use':>15} {'fits':>5} {'marker':>7} "
          f"{'reach 66x33 rails/+5':>21}")
    ball_r = 0.0127
    for name, motor, sol, batt in variants:
        length, width, use, fits, marker = layout(motor, sol, batt)
        f0 = reachable_fraction(length / 1000, width / 1000, TABLE_LENGTH, TABLE_WIDTH, ball_r, n=100_000)
        f5 = reachable_fraction(length / 1000, width / 1000, TABLE_LENGTH, TABLE_WIDTH, ball_r, n=100_000, n_other=5)
        print(f"{name:<40} {length / 10:>5.1f} x {width / 10:<5.1f} {100 * use:>14.0f}% {str(fits):>5} "
              f"{str(marker):>7} {100 * f0:>11.1f}% / {100 * f5:.1f}%")
    print(f"\n(the striker plunger travels {STROKE:.0f} mm ahead of the body when fired; reach uses 2.54 cm balls)")

    # Smallest 2:1 playfield reaching >= 60% with 5 other balls, for each variant.
    print("\nsmallest 2:1 playfield with >= 60% reachable (rails + 5 other balls):")
    for name, motor, sol, batt in variants:
        length, width, *_ = layout(motor, sol, batt)
        best = None
        for tl_cm in range(56, 131, 2):
            tl = tl_cm / 100
            if reachable_fraction(length / 1000, width / 1000, tl, tl / 2, ball_r, n=40_000, n_other=5) >= 0.60:
                best = tl_cm
                break
        print(f"  {name:<40} {best} x {best / 2:.0f} cm" if best else f"  {name:<40} > 130 cm")


if __name__ == "__main__":
    main()
