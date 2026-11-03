"""Collect simulator transitions.

Modes:
    random_drive     (default) 50-step episodes of random driving with 3-5 strikes at random
                     steps in [5, 45]. Saved as batch_v3_XXX.npz.
    strike_outcomes  Short episodes that strike on the very first step and then record the
                     cue ball / target balls rolling out, so nearly every transition has real
                     ball motion. Saved as batch_strike_XXX.npz.

Usage:
    python data/collect.py
    python data/collect.py --mode strike_outcomes --episodes 500
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
}
FIRE_FORCE_RANGE = (0.3, 1.0)
BATCH_SIZE = 100  # episodes per saved file
RESET_SETUP = "behind_cue"
DATASETS_DIR = os.path.join(os.path.dirname(__file__), "datasets")

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=list(MODES), default="random_drive")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--prefix", default=None, help="Override the output file prefix")
    args = parser.parse_args()
    collect(args.mode, args.episodes, args.prefix)
