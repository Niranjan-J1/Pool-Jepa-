import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from env.base_env import Action
from env.sim_env import SimEnv

NUM_EPISODES = 1000
STEPS_PER_EPISODE = 50
FIRES_PER_EPISODE = (3, 5)      # inclusive range
FIRE_STEP_RANGE = (5, 45)       # inclusive range of steps eligible to fire
BATCH_SIZE = 100
FILE_PREFIX = "batch_v3"
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


def save_batch(batch_idx, buf):
    path = os.path.join(DATASETS_DIR, f"{FILE_PREFIX}_{batch_idx:03d}.npz")
    arrays = {k: np.array(v, dtype=DTYPES.get(k, np.float32)) for k, v in buf.items()}
    np.savez_compressed(path, **arrays)
    size_mb = os.path.getsize(path) / (1024 * 1024)
    return path, size_mb


def sample_fire_steps():
    n_fires = random.randint(*FIRES_PER_EPISODE)
    return set(random.sample(range(FIRE_STEP_RANGE[0], FIRE_STEP_RANGE[1] + 1), n_fires))


def collect():
    os.makedirs(DATASETS_DIR, exist_ok=True)
    env = SimEnv()
    buf = {k: [] for k in FIELDS}

    t0 = time.time()
    batch_idx = 0
    total_saved_mb = 0.0
    total_fires = 0
    total_connected = 0

    for ep in range(NUM_EPISODES):
        obs = env.reset(setup=RESET_SETUP)
        setup = env.last_setup
        fire_steps = sample_fire_steps()
        total_fires += len(fire_steps)

        for step in range(STEPS_PER_EPISODE):
            prev = obs
            fire = step in fire_steps
            action = Action(
                left_pwm=random.uniform(-1, 1),
                right_pwm=random.uniform(-1, 1),
                fire=fire,
                fire_force=random.uniform(0.3, 1.0) if fire else 0.0,
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

        if (ep + 1) % BATCH_SIZE == 0:
            path, size_mb = save_batch(batch_idx, buf)
            total_saved_mb += size_mb
            batch_idx += 1

            elapsed = time.time() - t0
            eps_per_sec = (ep + 1) / elapsed
            print(f"Episode {ep + 1}/{NUM_EPISODES}  "
                  f"({eps_per_sec:.1f} ep/s, {elapsed:.0f}s elapsed)  "
                  f"Saved {path} ({size_mb:.1f} MB)", flush=True)

            for v in buf.values():
                v.clear()

    elapsed = time.time() - t0
    n_trans = NUM_EPISODES * STEPS_PER_EPISODE
    print(f"\nDone. {NUM_EPISODES} episodes, {batch_idx} batches, "
          f"{total_saved_mb:.1f} MB total, {elapsed:.0f}s")
    print(f"Fire transitions: {total_fires}/{n_trans} ({100 * total_fires / n_trans:.1f}%)")
    print(f"Strikes that connected with the cue ball: {total_connected}/{total_fires} "
          f"({100 * total_connected / max(1, total_fires):.1f}%)")


if __name__ == "__main__":
    collect()
