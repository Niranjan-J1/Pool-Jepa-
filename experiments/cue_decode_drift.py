"""Does the decoded cue position drift when the robot is close to the cue ball?

    python experiments/cue_decode_drift.py --background $POOLJEPA_DATA/frames_background.npy

In the full-shot test the decoded cue was ~1.24 cm off at hand-off (robot right behind the
ball), which on short targets is a ~14 deg aim error. This script separates "the readout is
just ~1.2 cm accurate everywhere" from "the robot pulls the cue keypoints":

  1. Decode error vs robot distance: cue ball placed uniformly, robot teleported so its front
     is g cm behind it (g = 2..15), or left far away; mean / 90th-percentile error per bin.
  2. Paired shift: the SAME cue position rendered with the robot far away and then right
     behind it (front 5 cm away). The decoded cue moves by delta; reported as its size and its
     component along robot->cue (+ = pushed away from the robot, - = pulled toward it).
  3. Which keypoints move: per selected cue keypoint, the paired shift expressed as an implied
     cue displacement (cm) through that keypoint's own affine map.

Each is reported for the original calibration and for one with near-robot scenes
(cem.calibrate_cue_keypoints(near_robot_fraction=...)). If the near-robot calibration closes
the near-vs-far gap, run the planners with --calib-near-robot.
"""
import argparse
import json
import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from env.sim_env import BALL_RADIUS, CUSHION_THICKNESS, SimEnv, TABLE_LENGTH, TABLE_WIDTH
from planning import cem

GAPS_CM = [2, 3, 5, 7, 10, 15]
FAR = 0.25  # m, robot centre -> cue: "far"


def encoder_file(args):
    if args.encoder:
        return args.encoder
    meta = os.path.join(args.checkpoint_dir, f"shot_model{args.suffix}.json")
    if os.path.exists(meta):
        return json.load(open(meta)).get("encoder_file", f"encoder{args.suffix}.pt")
    return f"encoder{args.suffix}.pt"


def latent(enc, env, bg, dev):
    f = env.render()
    return cem.encode(enc, f, f, bg, dev)


def main(args):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    env = SimEnv()
    enc_name = encoder_file(args)
    enc = cem.make_spatial_encoder(dev)
    enc.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, enc_name), map_location=dev))
    enc.eval()
    bg = cem.load_background(env, args.background)
    print(f"Encoder {enc_name}")

    cals = {}
    for name, frac in (("original", 0.0), (f"near-robot {args.near_fraction:g}", args.near_fraction)):
        kp = cem.calibrate_cue_keypoints(enc, env, bg, dev, n_scenes=args.calib_scenes, seed=0,
                                         near_robot_fraction=frac)
        cals[name] = kp
        print(f"  calibration {name:>16}: {kp.selected.sum()} keypoints, R^2 {kp.readout_r2:.3f}, "
              f"held-out err robot far {kp.err_far_cm:.2f} cm"
              + (f", robot near {kp.err_near_cm:.2f} cm" if frac else ""))

    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)
    lim = np.array([TABLE_LENGTH / 2, TABLE_WIDTH / 2]) - CUSHION_THICKNESS - BALL_RADIUS - 0.005

    def fresh_scene():
        """Reset, put the cue ball at a uniform spot clear of the other balls and the robot."""
        while True:
            obs = env.reset(setup="behind_cue")
            p = rng.uniform(-lim, lim)
            rx, ry, th = obs["robot_pose"]
            if env._overlaps_robot(tuple(p), (rx, ry), th, BALL_RADIUS + 0.005):
                continue
            if any(np.hypot(*(np.array(b.position) - p)) < 2 * BALL_RADIUS + 0.005
                   for n, b in env._ball_bodies.items() if n != "cue"):
                continue
            env._ball_bodies["cue"].position = tuple(p)
            return p, np.array([rx, ry])

    # 1. error vs robot distance
    bins = {f"front {g:2d} cm": (g / 100, g / 100) for g in GAPS_CM}
    rows = {b: {n: [] for n in cals} for b in list(bins) + ["robot far"]}
    for b, gap in list(bins.items()) + [("robot far", None)]:
        got = 0
        while got < args.per_bin:
            p, robot = fresh_scene()
            if gap is None:
                if np.linalg.norm(robot - p) < FAR:
                    continue
            elif not cem.place_robot_behind(env, p, rng, gap_range=gap):
                continue
            z = latent(enc, env, bg, dev)
            for n, kp in cals.items():
                rows[b][n].append(100 * np.linalg.norm(kp.decode_cue(z) - p))
            got += 1
    print(f"\n1. Decoded cue error (cm) vs robot position, {args.per_bin} scenes per row, mean / p90")
    print(f"  {'robot':>14}" + "".join(f"{n:>26}" for n in cals))
    for b, r in rows.items():
        print(f"  {b:>14}" + "".join(f"{np.mean(v):>17.2f} / {np.percentile(v, 90):5.2f}" for v in r.values()))

    # 2 + 3. paired: same cue position, robot far -> robot 5 cm behind
    shift = {n: [] for n in cals}
    along = {n: [] for n in cals}
    kp_shift = {n: [] for n in cals}
    got = 0
    while got < args.pairs:
        p, robot = fresh_scene()
        if np.linalg.norm(robot - p) < FAR:
            continue
        z_far = latent(enc, env, bg, dev)
        if not cem.place_robot_behind(env, p, rng, gap_range=(0.05, 0.05)):
            continue
        z_near = latent(enc, env, bg, dev)
        r_now = np.array(env._robot_body.position)
        u = (p - r_now) / np.linalg.norm(p - r_now)        # robot -> cue
        for n, kp in cals.items():
            d = kp.decode_cue(z_near) - kp.decode_cue(z_far)
            shift[n].append(100 * np.linalg.norm(d))
            along[n].append(100 * float(d @ u))
            dk = (z_near - z_far).cpu().numpy().reshape(-1, 2)
            implied = []
            for k in np.flatnonzero(kp.selected):
                M = kp.affine[k][:2]                          # cue (x, y) -> keypoint (x, y)
                implied.append(100 * np.linalg.norm(dk[k] @ np.linalg.pinv(M)))
            kp_shift[n].append(implied)
        got += 1
    print(f"\n2. Same cue position, robot far -> front 5 cm behind ({args.pairs} pairs)")
    for n in cals:
        s, a = np.array(shift[n]), np.array(along[n])
        print(f"  {n:>16}: decoded cue moves {s.mean():.2f} cm mean (p90 {np.percentile(s, 90):.2f}); "
              f"along robot->cue {a.mean():+.2f} cm mean ({'pushed away from' if a.mean() > 0 else 'pulled toward'} the robot)")
    print(f"\n3. Per cue keypoint (original calibration): implied cue shift when the robot arrives, cm")
    kp0 = list(cals.values())[0]
    ks = np.array(kp_shift[list(cals)[0]])
    for j, k in enumerate(np.flatnonzero(kp0.selected)):
        print(f"  keypoint {k:2d} (R^2_cue {kp0.r2_cue[k]:.2f}, R^2_robot {kp0.r2_robot[k]:.2f}): "
              f"mean {ks[:, j].mean():.2f}, p90 {np.percentile(ks[:, j], 90):.2f}")
    print("\nReading: if row 'front 5 cm' >> 'robot far' and section 2 shows a consistent along-shift,\n"
          "the robot pulls the cue keypoints; a near-robot calibration that closes the gap fixes it.\n"
          "If all rows are similar (~ the far error), the 1.2 cm is the readout's overall accuracy,\n"
          "not robot drift, and the fix is a better readout/encoder rather than calibration.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Cue-decode drift vs robot proximity")
    p.add_argument("--checkpoint-dir", default=cem.CHECKPOINT_DIR)
    p.add_argument("--suffix", default="_best")
    p.add_argument("--encoder", default=None, help="encoder file (default: the one the shot model names)")
    p.add_argument("--background", default=cem.BACKGROUND_PATH)
    p.add_argument("--calib-scenes", type=int, default=400)
    p.add_argument("--near-fraction", type=float, default=0.5)
    p.add_argument("--per-bin", type=int, default=60)
    p.add_argument("--pairs", type=int, default=100)
    p.add_argument("--seed", type=int, default=1)
    main(p.parse_args())
