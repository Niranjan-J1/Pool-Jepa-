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
FIRE_STEP = 25
BATCH_SIZE = 100
DATASETS_DIR = os.path.join(os.path.dirname(__file__), "datasets")


def save_batch(batch_idx, frames_before, frames_after, left, right, fire, force):
    path = os.path.join(DATASETS_DIR, f"batch_{batch_idx:03d}.npz")
    np.savez_compressed(
        path,
        frames_before=np.array(frames_before, dtype=np.uint8),
        frames_after=np.array(frames_after, dtype=np.uint8),
        actions_left_pwm=np.array(left, dtype=np.float32),
        actions_right_pwm=np.array(right, dtype=np.float32),
        actions_fire=np.array(fire, dtype=np.bool_),
        actions_fire_force=np.array(force, dtype=np.float32),
    )
    size_mb = os.path.getsize(path) / (1024 * 1024)
    return path, size_mb


def collect():
    os.makedirs(DATASETS_DIR, exist_ok=True)
    env = SimEnv()

    frames_before = []
    frames_after = []
    actions_left_pwm = []
    actions_right_pwm = []
    actions_fire = []
    actions_fire_force = []

    t0 = time.time()
    batch_idx = 0
    total_saved_mb = 0.0

    for ep in range(NUM_EPISODES):
        obs = env.reset()

        for step in range(STEPS_PER_EPISODE):
            frame_before = obs["frame"]

            fire = step == FIRE_STEP
            action = Action(
                left_pwm=random.uniform(-1, 1),
                right_pwm=random.uniform(-1, 1),
                fire=fire,
                fire_force=random.uniform(0.3, 1.0) if fire else 0.0,
            )

            obs = env.step(action)

            frames_before.append(frame_before)
            frames_after.append(obs["frame"])
            actions_left_pwm.append(action["left_pwm"])
            actions_right_pwm.append(action["right_pwm"])
            actions_fire.append(action["fire"])
            actions_fire_force.append(action["fire_force"])

        if (ep + 1) % BATCH_SIZE == 0:
            path, size_mb = save_batch(
                batch_idx, frames_before, frames_after,
                actions_left_pwm, actions_right_pwm,
                actions_fire, actions_fire_force,
            )
            total_saved_mb += size_mb
            batch_idx += 1

            elapsed = time.time() - t0
            eps_per_sec = (ep + 1) / elapsed
            print(f"Episode {ep + 1}/{NUM_EPISODES}  "
                  f"({eps_per_sec:.1f} ep/s, {elapsed:.0f}s elapsed)  "
                  f"Saved {path} ({size_mb:.1f} MB)")

            frames_before.clear()
            frames_after.clear()
            actions_left_pwm.clear()
            actions_right_pwm.clear()
            actions_fire.clear()
            actions_fire_force.clear()

    elapsed = time.time() - t0
    print(f"\nDone. {NUM_EPISODES} episodes, {batch_idx} batches, "
          f"{total_saved_mb:.1f} MB total, {elapsed:.0f}s")


if __name__ == "__main__":
    collect()
