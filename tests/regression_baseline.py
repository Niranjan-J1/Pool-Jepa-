"""Backward-compatibility regression harness for env.sim_env.SimEnv.

Captures the legacy (no-args) behaviour of the simulator as a set of fixed-seed
rollouts and compares a fresh run against a saved baseline. The baseline is
generated once from the pre-multiball code; after the multiball changes land,
``SimEnv()`` with no constructor arguments must reproduce it exactly.

Usage:
    python tests/regression_baseline.py --save     # write tests/baseline_legacy.json
    python tests/regression_baseline.py             # compare against the saved baseline
"""
import argparse
import hashlib
import json
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from env.base_env import Action
from env.sim_env import SimEnv

N_SEEDS = 20
BASELINE_PATH = os.path.join(os.path.dirname(__file__), "baseline_legacy.json")


def _shot_rollout(seed):
    """behind_cue setup, one strike, settle to rest. Exercises cue + object-ball
    collisions and cushion rebounds."""
    random.seed(seed)
    env = SimEnv()
    env.reset(setup="behind_cue")
    force = random.uniform(0.3, 1.0)
    env.step(Action(left_pwm=0.0, right_pwm=0.0, fire=True, fire_force=force), render=False)
    obs, steps = env.settle()
    return {
        "robot_pose": list(obs["robot_pose"]),
        "ball_positions": [list(p) for p in obs["ball_positions"]],
        "settle_steps": steps,
    }


def _drive_rollout(seed):
    """default setup, 50 steps of random driving with a few strikes. Exercises the
    robot pushing balls around."""
    random.seed(10_000 + seed)
    env = SimEnv()
    obs = env.reset(setup="default")
    fire_steps = set(random.sample(range(5, 46), random.randint(3, 5)))
    for step in range(50):
        action = Action(
            left_pwm=random.uniform(-1, 1),
            right_pwm=random.uniform(-1, 1),
            fire=step in fire_steps,
            fire_force=random.uniform(0.3, 1.0) if step in fire_steps else 0.0,
        )
        obs = env.step(action, render=False)
    return {
        "robot_pose": list(obs["robot_pose"]),
        "ball_positions": [list(p) for p in obs["ball_positions"]],
    }


def _frame_hashes():
    """SHA-256 of rendered legacy frames, so a render regression (pixel drift, a
    stray pocket drawn in legacy mode) is caught even though the rollouts use
    render=False."""
    random.seed(0)
    env = SimEnv()
    out = {}
    obs = env.reset(setup="default")
    out["default_reset"] = hashlib.sha256(obs["frame"].tobytes()).hexdigest()
    for i in range(30):
        obs = env.step(Action(left_pwm=0.5, right_pwm=-0.3, fire=(i == 10), fire_force=0.7))
    out["default_step30"] = hashlib.sha256(obs["frame"].tobytes()).hexdigest()
    return out


def compute():
    return {
        "shot": [_shot_rollout(s) for s in range(N_SEEDS)],
        "drive": [_drive_rollout(s) for s in range(N_SEEDS)],
        "frame_hashes": _frame_hashes(),
    }


def _max_abs_diff(a, b):
    """Largest absolute difference between two nested JSON structures of numbers."""
    if isinstance(a, list):
        if len(a) != len(b):
            return float("inf")
        return max([_max_abs_diff(x, y) for x, y in zip(a, b)], default=0.0)
    return abs(float(a) - float(b))


def compare(current, baseline):
    worst = 0.0
    if "frame_hashes" in baseline:
        for name, h in baseline["frame_hashes"].items():
            if current["frame_hashes"].get(name) != h:
                print(f"  RENDER MISMATCH in frame '{name}'")
                worst = float("inf")
            else:
                print(f"  render '{name}': identical")
    for scene in ("shot", "drive"):
        for i, (cur, base) in enumerate(zip(current[scene], baseline[scene])):
            for key in base:
                d = _max_abs_diff(cur[key], base[key])
                if d > worst:
                    worst = d
                if d > 0:
                    print(f"  diff in {scene}[{i}].{key}: {d:.3e}")
    return worst


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", action="store_true", help="write the baseline instead of comparing")
    args = ap.parse_args()

    data = compute()
    if args.save:
        with open(BASELINE_PATH, "w") as f:
            json.dump(data, f, indent=1)
        print(f"Wrote baseline for {N_SEEDS} seeds x 2 scenes -> {BASELINE_PATH}")
    else:
        with open(BASELINE_PATH) as f:
            baseline = json.load(f)
        worst = compare(data, baseline)
        print(f"max abs diff vs baseline: {worst:.3e}")
        if worst == 0.0:
            print("IDENTICAL: legacy behaviour is byte-for-byte unchanged.")
        elif worst < 1e-9:
            print("OK: within 1e-9 float tolerance.")
        else:
            print("MISMATCH: legacy behaviour changed!")
            sys.exit(1)
