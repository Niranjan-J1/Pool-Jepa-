"""D-001: how robot footprint and ball/table scale limit which shots the robot can take.

For a random ball position on the playfield and a random desired shot heading, the robot
must park behind the ball: its front edge at the setup distance d from the ball centre,
pointing along the heading, with its whole rectangle inside the rails. "Reachable" = that
pose fits. We report the fraction of (ball, heading) pairs that are reachable, and the
fraction of the playfield the robot covers.

Run: python hardware/analysis/footprint.py   (a few seconds, mostly importing the sim)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from env.sim_env import (BALL_RADIUS, MULTIBALL_CUSHION_RADIUS, ROBOT_LENGTH, ROBOT_WIDTH,  # noqa: E402
                         SETUP_DISTANCE_RANGE, TABLE_LENGTH, TABLE_WIDTH)

N = 200_000
SETUP_D = float(np.mean(SETUP_DISTANCE_RANGE))  # front edge -> ball centre (m)


def reachable_fraction(robot_l, robot_w, table_l, table_w, ball_r, d=SETUP_D, n=N, seed=0, n_other=0):
    """n_other: other object balls placed uniformly on the playfield (one redraw if they
    overlap the cue ball). The robot rectangle must not overlap any of them either."""
    rng = np.random.default_rng(seed)
    hl, hw = table_l / 2, table_w / 2
    # Ball centres anywhere a ball can rest (inside the rails).
    bx = rng.uniform(-hl + ball_r, hl - ball_r, n)
    by = rng.uniform(-hw + ball_r, hw - ball_r, n)
    th = rng.uniform(-np.pi, np.pi, n)
    c, s = np.cos(th), np.sin(th)
    # Robot centre sits (d + L/2) behind the ball along the heading.
    cx, cy = bx - (d + robot_l / 2) * c, by - (d + robot_l / 2) * s
    ok = np.ones(n, bool)
    for lx in (-robot_l / 2, robot_l / 2):
        for ly in (-robot_w / 2, robot_w / 2):
            x = cx + lx * c - ly * s
            y = cy + lx * s + ly * c
            ok &= (np.abs(x) <= hl) & (np.abs(y) <= hw)
    for _ in range(n_other):
        ox = rng.uniform(-hl + ball_r, hl - ball_r, n)
        oy = rng.uniform(-hw + ball_r, hw - ball_r, n)
        clash = np.hypot(ox - bx, oy - by) < 2 * ball_r
        ox[clash] = rng.uniform(-hl + ball_r, hl - ball_r, clash.sum())
        oy[clash] = rng.uniform(-hw + ball_r, hw - ball_r, clash.sum())
        # Other ball in the robot frame; closest point of the rectangle to its centre.
        dx, dy = ox - cx, oy - cy
        rx, ry = dx * c + dy * s, -dx * s + dy * c
        qx = np.clip(rx, -robot_l / 2, robot_l / 2)
        qy = np.clip(ry, -robot_w / 2, robot_w / 2)
        ok &= np.hypot(rx - qx, ry - qy) > ball_r
    return float(ok.mean())


def main():
    sim_table = (TABLE_LENGTH - 2 * MULTIBALL_CUSHION_RADIUS, TABLE_WIDTH - 2 * MULTIBALL_CUSHION_RADIUS)
    robots = {
        "sim robot 8x6 cm": (ROBOT_LENGTH, ROBOT_WIDTH),
        "small 10x8 cm": (0.10, 0.08),
        "proposal 20x14 cm": (0.20, 0.14),
    }
    tables = {
        "sim playfield 66x33 cm, 2.0 cm balls": (*sim_table, BALL_RADIUS),
        "same table, 2.54 cm (1 in) balls": (*sim_table, 0.0127),
        "smaller 56x28 cm (if 26 in = frame), 2.54 cm balls": (0.56, 0.28, 0.0127),
        "DIY 120x60 cm, 2.54 cm balls": (1.20, 0.60, 0.0127),
    }
    print(f"setup distance (front edge -> ball centre) {SETUP_D * 100:.1f} cm; {N:,} random (ball, heading) pairs")
    print("reachable = robot fits behind the cue ball; 'rails' ignores other balls, '+5 balls' also "
          "avoids 5 other object balls (typical mid-game)\n")
    print(f"{'table':<52} {'robot':<20} {'rails':>7} {'+5 balls':>9} {'covers':>7}")
    for tname, (tl, tw, br) in tables.items():
        for rname, (rl, rw) in robots.items():
            f0 = reachable_fraction(rl, rw, tl, tw, br)
            f5 = reachable_fraction(rl, rw, tl, tw, br, n_other=5)
            cover = rl * rw / (tl * tw)
            print(f"{tname:<52} {rname:<20} {100 * f0:>6.1f}% {100 * f5:>8.1f}% {100 * cover:>6.1f}%")
        print()
    print("ball/table ratio (ball diameter / playfield length):")
    for tname, (tl, tw, br) in tables.items():
        print(f"  {tname:<52} {2 * br / tl:.4f}")


if __name__ == "__main__":
    main()
