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


def collect_shots(episodes=None, prefix=None, force_range=SHOT_FORCE_RANGE):
    """shot_pairs mode: (two frames before, frame at rest) per strike."""
    n_episodes = episodes or MODES["shot_pairs"]["episodes"]
    prefix = prefix or MODES["shot_pairs"]["prefix"]
    print(f"Mode: shot_pairs  episodes={n_episodes}  force range {force_range}  "
          f"-> {prefix}_XXX.npz in {DATASETS_DIR}")
    os.makedirs(DATASETS_DIR, exist_ok=True)
    env = SimEnv()
    idle = Action(left_pwm=0.0, right_pwm=0.0, fire=False, fire_force=0.0)
    keys = ["frame_pre_prev", "frame_pre", "frame_post", "fire_force", "strike_connected",
            "robot_pose_pre", "ball_positions_pre", "ball_positions_post", "settle_steps",
            "setup_distance", "setup_aim_offset"]
    buf = {k: [] for k in keys}
    t0, file_idx, unsettled = time.time(), 0, 0

    for ep in range(n_episodes):
        obs = env.reset(setup=RESET_SETUP)
        f_prev = obs["frame"]
        obs = env.step(idle)  # robot at rest: encoder input before the shot is (f_prev, f_pre)
        force = random.uniform(*force_range)
        pre = obs
        env.step(Action(left_pwm=0.0, right_pwm=0.0, fire=True, fire_force=force), render=False)
        connected = env.last_strike_connected
        post, n_steps = env.settle(SHOT_MAX_SETTLE_STEPS)
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

        if (ep + 1) % SHOTS_PER_FILE == 0 or ep + 1 == n_episodes:
            path = os.path.join(DATASETS_DIR, f"{prefix}_{file_idx:03d}.npz")
            arrays = {k: np.array(v, dtype=np.uint8 if k.startswith("frame") else
                                  np.bool_ if k == "strike_connected" else
                                  np.int32 if k == "settle_steps" else np.float32)
                      for k, v in buf.items()}
            np.savez_compressed(path, **arrays)
            print(f"Shot {ep + 1}/{n_episodes}  ({(ep + 1) / (time.time() - t0):.1f} shots/s)  "
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
    args = parser.parse_args()
    if args.mode == "shot_pairs":
        collect_shots(args.episodes, args.prefix, tuple(args.force_range))
    else:
        collect(args.mode, args.episodes, args.prefix)
