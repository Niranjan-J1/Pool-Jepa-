"""Render a montage of multiball shots (collision, cushion bounce, a pot) for a visual sanity
check. Writes a PNG strip to an output directory (default: a scratch folder, NOT the repo).

    python tests/make_montage.py [out_dir]
"""
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from env.base_env import Action
from env.sim_env import SimEnv, pocket_centers

IDLE = Action(left_pwm=0.0, right_pwm=0.0, fire=False, fire_force=0.0)


def run_capture(env, snaps):
    """Step the env (idle) and grab a rendered frame at each step index in `snaps`."""
    snaps = sorted(snaps)
    frames = []
    last = snaps[-1]
    for i in range(last + 1):
        if i in snaps:
            frames.append(env.render())
        if i < last:
            env.step(IDLE, render=False)
    return frames


def shot_collision():
    env = SimEnv(n_object_balls=2)
    env.reset(layout={0: (-0.12, 0.0), 1: (0.05, 0.0), 2: (0.12, 0.12)})
    env._ball_bodies[0].velocity = (1.4, 0.0)
    return "collision (cue breaks into object balls)", run_capture(env, {0, 6, 16, 40})


def shot_cushion():
    env = SimEnv(n_object_balls=0)
    env.reset(layout={0: (-0.10, -0.02)})
    env._ball_bodies[0].velocity = (1.2 * math.cos(math.radians(50)), 1.2 * math.sin(math.radians(50)))
    return "cushion bounce (~mirror angle, slower)", run_capture(env, {0, 10, 22, 45})


def shot_pot():
    env = SimEnv(n_object_balls=1)
    px, py = pocket_centers()[3]  # top-right corner
    start = (px * 0.5, py * 0.5)
    env.reset(layout={0: (-0.05, 0.10), 1: start})
    d = (px - start[0], py - start[1]); n = math.hypot(*d)
    env._ball_bodies[1].velocity = (0.9 * d[0]/n, 0.9 * d[1]/n)
    return "pot (object ball into the corner pocket)", run_capture(env, {0, 6, 12, 30})


def main(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    shots = [shot_collision(), shot_cushion(), shot_pot()]
    ncols = 4
    fig, axes = plt.subplots(len(shots), ncols, figsize=(ncols * 3.0, len(shots) * 1.9))
    for r, (title, frames) in enumerate(shots):
        for c in range(ncols):
            ax = axes[r][c]
            ax.imshow(frames[c])
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            ax.set_title(f"t={c}", fontsize=7)
        axes[r][0].set_ylabel(title, fontsize=8, rotation=90, labelpad=6)
    fig.suptitle("Pool-JEPA multiball — collision / cushion / pot", fontsize=11)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.9, bottom=0.02, wspace=0.05, hspace=0.2)
    out = os.path.join(out_dir, "multiball_montage.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    default_out = os.environ.get("POOLJEPA_SCRATCH", os.path.join(os.path.dirname(__file__), "_scratch_montage"))
    main(sys.argv[1] if len(sys.argv) > 1 else default_out)
