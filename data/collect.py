"""Collect simulator transitions.

Modes:
    random_drive     (default) 50-step episodes of random driving with 3-5 strikes at random
                     steps in [5, 45]. Saved as batch_v3_XXX.npz.
    strike_outcomes  Short episodes that strike on the very first step and then record the
                     cue ball / target balls rolling out, so nearly every transition has real
                     ball motion. Saved as batch_strike_XXX.npz.
    shot_pairs       One strike per episode, stored as (before, at rest) only: two frames
                     before the strike, the frame once every ball has stopped, the force, and
                     ground-truth positions before/after. Training data for the shot-outcome
                     model. Physics between the strike and rest is not rendered, so this is
                     fast (~0.1 s/shot). Saved as shots_XXX.npz.
                     Long-shot data: --force-bias 2 favours hard shots, --min-travel 0.08
                     --straight-only keeps cue balls that roll >= 8 cm without touching a
                     cushion or ball (rejected shots are re-drawn), e.g.
                         --mode shot_pairs --episodes 5000 --force-range 0.1 0.3 --force-bias 2
                         --min-travel 0.08 --straight-only --prefix shots_long --seed 1

Output directory: $POOLJEPA_DATA if set, else data/datasets.

Usage:
    python data/collect.py
    python data/collect.py --mode strike_outcomes --episodes 500
    python data/collect.py --mode shot_pairs --episodes 5000
"""
import argparse
import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from env.base_env import Action
from env.sim_env import SimEnv

MODES = {
    "random_drive": dict(
        episodes=1000, steps=50, fires_per_episode=(3, 5), fire_step_range=(5, 45),
        prefix="batch_v3",
    ),
    "strike_outcomes": dict(
        episodes=500, steps=20, fires_per_episode=(1, 1), fire_step_range=(0, 0),
        prefix="batch_strike",
    ),
    "shot_pairs": dict(episodes=5000, prefix="shots"),
}
FIRE_FORCE_RANGE = (0.3, 1.0)
SHOT_FORCE_RANGE = (0.05, 1.0)  # include soft shots: short targets need them
SHOT_MAX_SETTLE_STEPS = 900     # 15 s; shots settle in <2 s with rolling resistance
SHOTS_PER_FILE = 1000
BATCH_SIZE = 100  # episodes per saved file
RESET_SETUP = "behind_cue"
DATASETS_DIR = os.environ.get("POOLJEPA_DATA", os.path.join(os.path.dirname(__file__), "datasets"))

FIELDS = [
    "frames_before", "frames_after",
    "actions_left_pwm", "actions_right_pwm", "actions_fire", "actions_fire_force",
    "robot_pose_before", "robot_pose_after",
    "ball_positions_before", "ball_positions_after",
    "episode_idx", "step_idx",
    "strike_connected", "setup_distance", "setup_aim_offset",
]
DTYPES = {
    "frames_before": np.uint8, "frames_after": np.uint8,
    "actions_fire": np.bool_, "strike_connected": np.bool_,
    "episode_idx": np.int32, "step_idx": np.int32,
}


def save_batch(prefix, batch_idx, buf):
    path = os.path.join(DATASETS_DIR, f"{prefix}_{batch_idx:03d}.npz")
    arrays = {k: np.array(v, dtype=DTYPES.get(k, np.float32)) for k, v in buf.items()}
    np.savez_compressed(path, **arrays)
    size_mb = os.path.getsize(path) / (1024 * 1024)
    return path, size_mb


def sample_fire_steps(cfg):
    n_fires = random.randint(*cfg["fires_per_episode"])
    lo, hi = cfg["fire_step_range"]
    return set(random.sample(range(lo, hi + 1), n_fires))


def collect(mode="random_drive", episodes=None, prefix=None):
    cfg = dict(MODES[mode])
    if episodes is not None:
        cfg["episodes"] = episodes
    if prefix is not None:
        cfg["prefix"] = prefix
    n_episodes, n_steps, prefix = cfg["episodes"], cfg["steps"], cfg["prefix"]
    print(f"Mode: {mode}  episodes={n_episodes}  steps/episode={n_steps}  -> {prefix}_XXX.npz")

    os.makedirs(DATASETS_DIR, exist_ok=True)
    env = SimEnv()
    buf = {k: [] for k in FIELDS}

    t0 = time.time()
    batch_idx = 0
    total_saved_mb = 0.0
    total_fires = 0
    total_connected = 0

    for ep in range(n_episodes):
        obs = env.reset(setup=RESET_SETUP)
        setup = env.last_setup
        fire_steps = sample_fire_steps(cfg)
        total_fires += len(fire_steps)

        for step in range(n_steps):
            prev = obs
            fire = step in fire_steps
            action = Action(
                left_pwm=random.uniform(-1, 1),
                right_pwm=random.uniform(-1, 1),
                fire=fire,
                fire_force=random.uniform(*FIRE_FORCE_RANGE) if fire else 0.0,
            )

            obs = env.step(action)

            buf["frames_before"].append(prev["frame"])
            buf["frames_after"].append(obs["frame"])
            buf["actions_left_pwm"].append(action["left_pwm"])
            buf["actions_right_pwm"].append(action["right_pwm"])
            buf["actions_fire"].append(action["fire"])
            buf["actions_fire_force"].append(action["fire_force"])
            buf["robot_pose_before"].append(prev["robot_pose"])
            buf["robot_pose_after"].append(obs["robot_pose"])
            buf["ball_positions_before"].append(prev["ball_positions"])
            buf["ball_positions_after"].append(obs["ball_positions"])
            buf["episode_idx"].append(ep)
            buf["step_idx"].append(step)
            buf["strike_connected"].append(env.last_strike_connected)
            buf["setup_distance"].append(setup.get("distance", np.nan))
            buf["setup_aim_offset"].append(setup.get("aim_offset", np.nan))
            total_connected += env.last_strike_connected

        if (ep + 1) % BATCH_SIZE == 0 or ep + 1 == n_episodes:
            path, size_mb = save_batch(prefix, batch_idx, buf)
            total_saved_mb += size_mb
            batch_idx += 1

            elapsed = time.time() - t0
            eps_per_sec = (ep + 1) / elapsed
            print(f"Episode {ep + 1}/{n_episodes}  "
                  f"({eps_per_sec:.1f} ep/s, {elapsed:.0f}s elapsed)  "
                  f"Saved {path} ({size_mb:.1f} MB)", flush=True)

            for v in buf.values():
                v.clear()

    elapsed = time.time() - t0
    n_trans = n_episodes * n_steps
    print(f"\nDone. {n_episodes} episodes, {batch_idx} batches, "
          f"{total_saved_mb:.1f} MB total, {elapsed:.0f}s")
    print(f"Fire transitions: {total_fires}/{n_trans} ({100 * total_fires / n_trans:.1f}%)")
    print(f"Strikes that connected with the cue ball: {total_connected}/{total_fires} "
          f"({100 * total_connected / max(1, total_fires):.1f}%)")


STRAIGHT_LATERAL_TOL = 0.001  # m: cue ball counts as rolling straight while within 1 mm of its line


def settle_tracking(env, idle, heading, max_steps=SHOT_MAX_SETTLE_STEPS):
    """env.settle() that also reports whether the cue ball rolled out straight along `heading`
    (the strike direction) without touching a cushion or another ball: it never leaves the
    line by more than STRAIGHT_LATERAL_TOL, never moves backwards, and no other ball moves.
    Returns (rendered final obs, steps, straight)."""
    start = np.array(env._get_observation(render=False)["ball_positions"])
    u = np.array([np.cos(heading), np.sin(heading)])
    n_vec = np.array([-u[1], u[0]])
    straight, last = True, 0.0
    for n in range(1, max_steps + 1):
        obs = env.step(idle, render=False)
        if straight:
            b = np.array(obs["ball_positions"])
            d = b[0] - start[0]
            along = float(d @ u)
            if abs(float(d @ n_vec)) > STRAIGHT_LATERAL_TOL or along < last - 1e-4 or \
                    (np.linalg.norm(b[1:] - start[1:], axis=1) > 1e-4).any():
                straight = False
            last = max(last, along)
        if obs["ball_settled"]:
            break
    return env._get_observation(), n, straight


def sample_force(force_range, bias):
    """bias = 1: uniform. bias > 1: density rises toward the top of the range
    (f = lo + (hi - lo) * u^(1/bias); bias = 2 -> linear ramp, 3 -> quadratic)."""
    lo, hi = force_range
    return lo + (hi - lo) * random.random() ** (1.0 / bias)


def collect_shots(episodes=None, prefix=None, force_range=SHOT_FORCE_RANGE, force_bias=1.0,
                  min_travel=0.0, straight_only=False, max_tries=200):
    """shot_pairs mode: (two frames before, frame at rest) per strike.

    force_bias > 1 favours harder shots. min_travel / straight_only reject shots whose cue ball
    travels less than min_travel (m) / does not roll out straight (cushion or ball contact); a
    rejected shot is re-drawn (new scene and force), up to max_tries per kept shot. Every saved
    shot records `cue_travel` (m) and `straight`, so training can report long shots separately."""
    n_episodes = episodes or MODES["shot_pairs"]["episodes"]
    prefix = prefix or MODES["shot_pairs"]["prefix"]
    print(f"Mode: shot_pairs  shots={n_episodes}  force range {force_range} bias {force_bias}  "
          f"min travel {min_travel} m  straight only {straight_only}  -> {prefix}_XXX.npz in {DATASETS_DIR}")
    os.makedirs(DATASETS_DIR, exist_ok=True)
    env = SimEnv()
    idle = Action(left_pwm=0.0, right_pwm=0.0, fire=False, fire_force=0.0)
    keys = ["frame_pre_prev", "frame_pre", "frame_post", "fire_force", "strike_connected",
            "robot_pose_pre", "ball_positions_pre", "ball_positions_post", "settle_steps",
            "setup_distance", "setup_aim_offset", "cue_travel", "straight"]
    buf = {k: [] for k in keys}
    t0, file_idx, unsettled, attempts = time.time(), 0, 0, 0

    for ep in range(n_episodes):
        for _ in range(max_tries):
            attempts += 1
            obs = env.reset(setup=RESET_SETUP)
            f_prev = obs["frame"]
            obs = env.step(idle)  # robot at rest: encoder input before the shot is (f_prev, f_pre)
            force = sample_force(force_range, force_bias)
            pre = obs
            env.step(Action(left_pwm=0.0, right_pwm=0.0, fire=True, fire_force=force), render=False)
            connected = env.last_strike_connected
            post, n_steps, straight = settle_tracking(env, idle, pre["robot_pose"][2])
            straight = straight and connected
            travel = float(np.linalg.norm(np.subtract(post["ball_positions"][0], pre["ball_positions"][0])))
            if travel >= min_travel and (straight or not straight_only):
                break
        else:
            raise RuntimeError(f"no shot met min_travel={min_travel} straight_only={straight_only} "
                               f"in {max_tries} tries; loosen the filter or raise the force range")
        unsettled += not post["ball_settled"]

        buf["frame_pre_prev"].append(f_prev)
        buf["frame_pre"].append(pre["frame"])
        buf["frame_post"].append(post["frame"])
        buf["fire_force"].append(force)
        buf["strike_connected"].append(connected)
        buf["robot_pose_pre"].append(pre["robot_pose"])
        buf["ball_positions_pre"].append(pre["ball_positions"])
        buf["ball_positions_post"].append(post["ball_positions"])
        buf["settle_steps"].append(n_steps)
        buf["setup_distance"].append(env.last_setup.get("distance", np.nan))
        buf["setup_aim_offset"].append(env.last_setup.get("aim_offset", np.nan))
        buf["cue_travel"].append(travel)
        buf["straight"].append(straight)

        if (ep + 1) % SHOTS_PER_FILE == 0 or ep + 1 == n_episodes:
            path = os.path.join(DATASETS_DIR, f"{prefix}_{file_idx:03d}.npz")
            arrays = {k: np.array(v, dtype=np.uint8 if k.startswith("frame") else
                                  np.bool_ if k in ("strike_connected", "straight") else
                                  np.int32 if k == "settle_steps" else np.float32)
                      for k, v in buf.items()}
            np.savez_compressed(path, **arrays)
            print(f"Shot {ep + 1}/{n_episodes}  ({(ep + 1) / (time.time() - t0):.1f} shots/s, "
                  f"{100 * (ep + 1) / attempts:.0f}% of attempts kept)  "
                  f"Saved {path} ({os.path.getsize(path) / 2**20:.1f} MB)", flush=True)
            file_idx += 1
            for v in buf.values():
                v.clear()

    print(f"\nDone. {n_episodes} shots in {time.time() - t0:.0f}s; "
          f"{unsettled} did not settle within {SHOT_MAX_SETTLE_STEPS} steps")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=list(MODES), default="random_drive")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--prefix", default=None, help="Override the output file prefix")
    parser.add_argument("--force-range", type=float, nargs=2, default=SHOT_FORCE_RANGE,
                        metavar=("MIN", "MAX"), help="shot_pairs: strike force range")
    parser.add_argument("--force-bias", type=float, default=1.0,
                        help="shot_pairs: >1 favours harder shots (2 = linear ramp toward MAX)")
    parser.add_argument("--min-travel", type=float, default=0.0,
                        help="shot_pairs: keep only shots whose cue ball travels at least this far (m)")
    parser.add_argument("--straight-only", action="store_true",
                        help="shot_pairs: keep only shots where the cue ball rolls out straight "
                             "(no cushion or ball contact)")
    parser.add_argument("--seed", type=int, default=None,
                        help="seed the scene/force RNG (use different seeds for different runs)")
    args = parser.parse_args()
    if args.seed is not None:
        random.seed(args.seed)
    if args.mode == "shot_pairs":
        collect_shots(args.episodes, args.prefix, tuple(args.force_range), args.force_bias,
                      args.min_travel, args.straight_only)
    else:
        collect(args.mode, args.episodes, args.prefix)
