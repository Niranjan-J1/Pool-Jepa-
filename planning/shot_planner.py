"""Stage 2 of the hierarchical planner: choose the strike force for an already-aimed robot.

The robot fires along its heading, so once it is aimed (Stage 1) the only remaining
decision is how hard to hit. With one continuous variable, an exhaustive search over a
fine force grid replaces CEM: every candidate is scored at once through the shot model,

    z_rest = ShotModel(z_pre, force)    ->    cue rest position = cue readout(z_rest)

and the force whose predicted rest position is closest to the target wins.

Test (pre-aimed scenes, Stage 1 skipped):
    python planning/shot_planner.py --trials 50
For each scene a secret force f* is simulated and the cue ball's true rest position becomes
the target, so every target is reachable. The planner must recover a force that lands the
cue ball there. The simulator is deterministic, so the same scene is replayed for each
candidate (planner's choice, fixed 0.5, random) and the miss distances are compared.
"""
import argparse
import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from models.shot_model import load_shot_model
from planning import cem

def force_grid(lo=0.05, hi=1.0, step=0.0025):
    """Candidate forces. Keep [lo, hi] inside the range the shot model was trained on
    (stored in its .json); outside it the model is extrapolating."""
    return torch.arange(lo, hi + 1e-9, step)


class ShotPlanner:
    def __init__(self, shot_model, cue_kp: "cem.CueKeypoints", device="cpu", forces=None):
        self.model = shot_model.to(device).eval()
        self.kp = cue_kp
        self.device = device
        self.forces = (force_grid() if forces is None else forces).to(device)

    @torch.no_grad()
    def plan(self, z_pre: torch.Tensor, target_xy):
        """z_pre: (D,) latent before the strike. Returns (force, info)."""
        z = z_pre.to(self.device).flatten().unsqueeze(0).expand(len(self.forces), -1)
        rest = self.kp.decode_cue_batch(self.model(z, self.forces))  # (F, 2) m
        cost = np.linalg.norm(rest - np.asarray(target_xy)[None], axis=1)
        i = int(cost.argmin())
        return float(self.forces[i]), {"predicted_rest": rest[i], "predicted_miss": float(cost[i]),
                                       "all_rest": rest, "all_cost": cost}


# --------------------------------------------------------------------------------------
# Test
# --------------------------------------------------------------------------------------

IDLE = dict(left_pwm=0.0, right_pwm=0.0, fire=False, fire_force=0.0)


def replay_shot(env, scene_state, force, render_pre=False):
    """Recreate the scene from a saved `random` state, fire with `force`, let it settle.
    Returns (cue rest position, (prev_frame, pre_frame) if render_pre)."""
    random.setstate(scene_state)
    obs = env.reset(setup="behind_cue")
    f_prev = obs["frame"] if render_pre else None
    obs = env.step(IDLE, render=render_pre)
    frames = (f_prev, obs["frame"]) if render_pre else None
    env.step(dict(left_pwm=0.0, right_pwm=0.0, fire=True, fire_force=force), render=False)
    post, _ = env.settle()
    return np.array(post["ball_positions"][0]), frames


def run_test(args):
    from env.sim_env import SimEnv

    device = "cuda" if torch.cuda.is_available() else "cpu"
    shot_path = os.path.join(args.checkpoint_dir, f"shot_model{args.suffix}.pt")
    shot_model, meta = load_shot_model(shot_path, device)
    enc_file = meta.get("encoder_file", f"encoder{args.suffix}.pt")
    encoder = cem.make_spatial_encoder(device)
    encoder.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, enc_file), map_location=device))
    encoder.eval()
    print(f"Loaded {os.path.basename(shot_path)} with its frozen encoder {enc_file}")

    env = SimEnv()
    background = cem.load_background(env, args.background)
    kp = cem.calibrate_cue_keypoints(encoder, env, background, device, n_scenes=args.calib_scenes,
                                     seed=args.seed)
    print(f"{kp.selected.sum()} cue keypoints, readout R^2 {kp.readout_r2:.3f}\n")
    f_lo, f_hi = meta.get("force_range", [0.05, 1.0])
    planner = ShotPlanner(shot_model, kp, device, forces=force_grid(f_lo, f_hi))
    f_mid = round((f_lo + f_hi) / 2, 3)
    print(f"Force range (from training data): {f_lo:.2f}-{f_hi:.2f}; secret forces drawn from the same range")

    random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    miss = {"planner": [], f"fixed {f_mid}": [], "random force": []}
    force_err, travel = [], []
    for trial in range(args.trials):
        scene = random.getstate()
        f_star = float(rng.uniform(f_lo, f_hi))
        target, _ = replay_shot(env, scene, f_star)
        start, (f_prev, f_pre) = replay_shot(env, scene, f_star, render_pre=True)  # same scene, frames
        z_pre = cem.encode(encoder, f_prev, f_pre, background, device)

        f_plan, info = planner.plan(z_pre, target)
        for name, f in (("planner", f_plan), (f"fixed {f_mid}", f_mid),
                        ("random force", float(rng.uniform(f_lo, f_hi)))):
            rest, _ = replay_shot(env, scene, f)
            miss[name].append(100 * np.linalg.norm(rest - target))
        force_err.append(abs(f_plan - f_star))
        # cue start = position before the shot (re-read from a fresh replay without firing)
        random.setstate(scene)
        cue0 = np.array(env.reset(setup="behind_cue")["ball_positions"][0])
        travel.append(100 * np.linalg.norm(target - cue0))
        if trial < args.show:
            print(f"[{trial:2d}] target {100 * np.linalg.norm(target - cue0):5.1f} cm away | f*={f_star:.2f} "
                  f"chosen {f_plan:.2f} | predicted miss {100 * info['predicted_miss']:4.1f} cm, "
                  f"actual miss {miss['planner'][-1]:4.1f} cm")
        random.setstate(scene)
        random.random()  # advance so the next trial gets a new scene

    print(f"\n========== {args.trials} pre-aimed shots (target = true rest point of a secret force) ==========")
    print(f"Target distance from cue start: mean {np.mean(travel):.1f} cm")
    print(f"{'strategy':>14} {'mean miss cm':>13} {'median':>8} {'<3 cm':>7} {'<5 cm':>7}")
    for name, m in miss.items():
        m = np.array(m)
        print(f"{name:>14} {m.mean():>13.1f} {np.median(m):>8.1f} {100 * (m < 3).mean():>6.0f}% "
              f"{100 * (m < 5).mean():>6.0f}%")
    print(f"Planner force error |f - f*|: mean {np.mean(force_err):.2f} "
          f"(different forces can reach nearby points, so this is a soft check)")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Stage-2 shot planner test on pre-aimed scenes")
    p.add_argument("--checkpoint-dir", default=cem.CHECKPOINT_DIR)
    p.add_argument("--suffix", default="_best")
    p.add_argument("--background", default=cem.BACKGROUND_PATH)
    p.add_argument("--trials", type=int, default=50)
    p.add_argument("--show", type=int, default=10, help="print the first N trials")
    p.add_argument("--calib-scenes", type=int, default=400)
    p.add_argument("--seed", type=int, default=0)
    run_test(p.parse_args())
