"""Record full autonomous shots (Stage 1 aim -> Stage 2 force -> fire) as 1080p MP4 + GIF.

    python planning/record_shot.py --suffix _bigcue --scan 60          # list clean seeds
    python planning/record_shot.py --suffix _bigcue --seed 0 59        # -> videos/shot_seed0.mp4/.gif, ...

The scene for a seed is built exactly as in full_shot_test.py: a behind_cue scene and a
straight-line reachable target (the rest point of a secret force fired from the exact firing
pose). Stage 1 runs on the decoded cue position with ball avoidance on (it is given the other
balls' true positions to path around), Stage 2 picks the force through the shot model, and the
robot fires.

Perception is unchanged: the encoder reads the plain 432x240 env.render() frames it was trained
on (shown as an inset). The clip itself is drawn by planning/demo_render.py from the simulator
state at 1920x1080, 60 fps:
  Perceiving       the cue-ball reticle fades in where the encoder locates it
  Aiming           closed-loop drive to the firing pose (one frame per 60 Hz physics step)
  Imagining shots  every candidate force's predicted landing point appears, green (on target)
                   to red (far), then the chosen one is ringed
  Firing           the shot, with a fading trail; --shot-slowmo interpolates between physics
                   steps (default 1.5x, labelled on screen)
  Result card      imagined vs actual landing point and the miss distance
The table's corner pockets are decorative (the simulated table has none); --no-pockets drops them.

A seed is "clean" when Stage 1 aligned in time, no ball moved before the shot, the strike
connected, the cue ball stopped within --clean-miss of the target, and the target is at least
--min-travel away (so the shot is visible).
Needs: pip install imageio imageio-ffmpeg
"""
import argparse
import math
import os
import random
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from env.sim_env import SimEnv
from planning import cem
from planning.aim_controller import AimController
from planning.demo_render import DemoRenderer, interpolate
from planning.full_shot_test import (BALL_MOVED, DECODE_EVERY, MAX_SETTLE, STAGE1_MAX_STEPS,
                                     Stage2, sample_target)
from planning.shot_planner import IDLE

FPS = 60


def snap(env):
    rb = env._robot_body
    return {"robot": (float(rb.position.x), float(rb.position.y), float(rb.angle)),
            "balls": np.array([(b.position.x, b.position.y) for b in env._ball_bodies.values()])}


# --------------------------------------------------------------------------------------
# One episode
# --------------------------------------------------------------------------------------

def run_episode(env, s2, ctl, scene, t, timeline=None, n_show=48, slowmo=1.5):
    """Stage 1 -> Stage 2 -> fire on a sampled scene. With a timeline list, appends
    (state, overlay) items for demo_render (one per output frame)."""
    record = timeline is not None
    target = np.asarray(t["target"])
    random.setstate(scene)
    obs = env.reset(setup="behind_cue")
    balls0 = np.array(obs["ball_positions"])

    def add(state, **ov):
        if record:
            timeline.append((state, dict(target=target, **ov)))

    # Perceiving: the first decode, shown before the robot moves
    if record:
        raw = env.render()
        cue_first = s2.decode_cue(raw)
        s0 = snap(env)
        for i in range(int(1.8 * FPS)):
            add(s0, stage="Perceiving", sub="the encoder locates the cue ball from raw camera pixels",
                target_alpha=min(1.0, i / 25), cue_est=cue_first,
                cue_alpha=float(np.clip((i - 30) / 30, 0, 1)), inset=raw)

    # Stage 1 (Aiming)
    ctl.reset()
    cue_est, done, n, last_raw = None, False, 0, None
    for n in range(STAGE1_MAX_STEPS):
        if n % DECODE_EVERY == 0:
            last_raw = env.render()
            cue_est = s2.decode_cue(last_raw)
        left, right, done, info = ctl.step(obs["robot_pose"], cue_est, target,
                                           np.array(obs["ball_positions"][1:]))
        add(snap(env), stage="Aiming", sub="closed-loop drive to the firing pose (cue position from vision)",
            cue_est=cue_est, aim_line=True, inset=last_raw,
            hud=f"heading error {math.degrees(info['head_err']):+6.1f} deg")
        if done:
            break
        obs = env.step(dict(IDLE, left_pwm=left, right_pwm=right), render=False)
    for _ in range(MAX_SETTLE):   # anything nudged comes to rest before the shot
        if env._balls_settled():
            break
        env.step(IDLE, render=False)
        add(snap(env), stage="Aiming", sub="closed-loop drive to the firing pose (cue position from vision)",
            cue_est=cue_est, aim_line=True, inset=last_raw)
    obs = env._get_observation(render=False)
    balls = np.array(obs["ball_positions"])
    disturbed = bool((np.linalg.norm(balls[1:] - balls0[1:], axis=1) > BALL_MOVED).any())
    cue_moved = bool(np.linalg.norm(balls[0] - balls0[0]) > BALL_MOVED)

    # Stage 2 (Imagining): two rest frames -> z_pre -> every candidate force through the shot model
    f_prev = env.render()
    obs = env.step(IDLE, render=True)
    z_pre = cem.encode(s2.encoder, f_prev, obs["frame"], s2.background, s2.device)
    heading = obs["robot_pose"][2] if s2.model.use_heading else None
    force, pinfo = s2.planner.plan(z_pre, target, heading)
    if record:
        st = snap(env)
        cue_now = s2.kp.decode_cue(z_pre)
        allf = s2.planner.forces.cpu().numpy()
        i_best = int(np.argmin(pinfo["all_cost"]))
        idx = np.unique(np.r_[np.linspace(0, len(allf) - 1, n_show).astype(int), i_best])
        pts, costs = pinfo["all_rest"][idx], pinfo["all_cost"][idx]
        reveal = int(2.2 * FPS)
        for i in range(reveal):
            k = int(math.ceil((i + 1) / reveal * len(idx)))
            add(st, stage="Imagining shots", sub="the world model predicts where each force leaves the cue ball",
                cue_est=cue_now, candidates=(pts[:k], costs[:k]), inset=obs["frame"],
                hud=f"force {allf[idx[k - 1]]:.2f}  ->  predicted miss {100 * costs[k - 1]:4.1f} cm")
        for i in range(int(1.3 * FPS)):
            add(st, stage="Imagining shots", sub=f"best of {len(allf)} imagined forces: {force:.2f}",
                cue_est=cue_now, candidates=(pts, costs), chosen=pinfo["predicted_rest"],
                chosen_text=f"force {force:.2f}  ·  predicted miss {100 * pinfo['predicted_miss']:.1f} cm",
                inset=obs["frame"])

    # Fire and roll out
    states = [snap(env)] if record else None
    obs = env.step(dict(IDLE, fire=True, fire_force=force), render=False)
    connected = env.last_strike_connected
    for k in range(MAX_SETTLE):
        if record:
            states.append(snap(env))
        if obs["ball_settled"] and k > 0:
            break
        obs = env.step(IDLE, render=False)
    rest = np.array(env._get_observation(render=False)["ball_positions"][0])
    miss = float(np.linalg.norm(rest - target))
    travel = float(np.linalg.norm(target - t["cue0"]))
    if record:
        sub = "solenoid strike" + (f"  ·  {slowmo:g}x slow motion" if slowmo > 1 else "")
        frames = interpolate(states, slowmo)
        for i, s in enumerate(frames):
            add(s, stage="Firing", sub=sub, trail=True, chosen=pinfo["predicted_rest"],
                burst=(i / 14) if i < 14 else None)
        end = frames[-1]
        n_forces = len(s2.planner.forces)
        for i in range(int(3.5 * FPS)):
            add(end, stage=None, result=dict(predicted=pinfo["predicted_rest"], actual=rest, miss=miss,
                                              travel=travel, force=force, n_forces=n_forces,
                                              alpha=min(1.0, i / 20)))

    stats = dict(aligned=done, steps=n, disturbed=disturbed, cue_moved=cue_moved, connected=connected,
                 miss=miss, force=force, predicted_miss=pinfo["predicted_miss"], travel=travel)
    return stats


def scene_for_seed(env, s2, seed):
    random.seed(seed)
    scene = random.getstate()
    rng = np.random.default_rng(seed)
    return scene, sample_target(env, scene, rng, s2.f_lo, s2.f_hi)


def is_clean(st, args):
    return (st["aligned"] and not st["disturbed"] and not st["cue_moved"] and st["connected"]
            and st["miss"] <= args.clean_miss and st["travel"] >= args.min_travel)


def describe(seed, st, clean):
    return (f"seed {seed:4d} | target {100 * st['travel']:5.1f} cm | aligned {st['aligned']!s:5} "
            f"({st['steps'] / 60:4.1f} s) | balls disturbed {st['disturbed']!s:5} | connected "
            f"{st['connected']!s:5} | force {st['force']:.2f} | miss {100 * st['miss']:4.1f} cm "
            f"(predicted {100 * st['predicted_miss']:4.1f}) {'CLEAN' if clean else ''}")


def render_video(timeline, out_base, args):
    """Stream 1080p frames to the MP4 (never all in memory); keep a downscaled copy for the GIF."""
    import imageio.v2 as imageio
    renderer = DemoRenderer(pockets=not args.no_pockets)
    mp4, gif = out_base + ".mp4", out_base + ".gif"
    gif_every = max(1, round(FPS / args.gif_fps))
    gif_h = round(args.gif_width * 9 / 16)
    gif_frames, t0 = [], time.time()
    with imageio.get_writer(mp4, fps=FPS, codec="libx264", quality=None, pixelformat="yuv420p",
                            macro_block_size=8, ffmpeg_params=["-crf", "18", "-preset", "slow",
                                                               "-movflags", "+faststart"]) as w:
        for i, (state, ov) in enumerate(timeline):
            f = renderer.frame(state, ov)
            w.append_data(f)
            if i % gif_every == 0:
                gif_frames.append(np.asarray(Image.fromarray(f).resize((args.gif_width, gif_h), Image.LANCZOS)))
    imageio.mimsave(gif, gif_frames, duration=1000 * gif_every / FPS, loop=0)
    print(f"  {len(timeline)} frames ({len(timeline) / FPS:.1f} s) rendered in {time.time() - t0:.0f} s")
    for p in (mp4, gif):
        print(f"  saved {p} ({os.path.getsize(p) / 2**20:.1f} MB)")


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = SimEnv()
    s2 = Stage2(SimpleNamespace(checkpoint_dir=args.checkpoint_dir, suffix=args.suffix,
                                background=args.background, calib_scenes=args.calib_scenes,
                                calib_near_robot=args.calib_near_robot, seed=0), env, device)
    ctl = AimController()

    if args.scan:
        clean = []
        for seed in range(args.seed[0], args.seed[0] + args.scan):
            scene, t = scene_for_seed(env, s2, seed)
            if t is None:
                print(f"seed {seed:4d} | no straight reachable target")
                continue
            st = run_episode(env, s2, ctl, scene, t)
            ok = is_clean(st, args)
            clean += [(seed, st)] if ok else []
            print(describe(seed, st, ok), flush=True)
        print(f"\n{len(clean)} clean seeds (aligned, nothing disturbed, connected, miss <= "
              f"{100 * args.clean_miss:.0f} cm, target >= {100 * args.min_travel:.0f} cm away):")
        for seed, st in sorted(clean, key=lambda x: -x[1]["travel"]):
            print(f"  --seed {seed:<5d} target {100 * st['travel']:5.1f} cm, miss {100 * st['miss']:.1f} cm")
        return

    os.makedirs(args.out_dir, exist_ok=True)
    for seed in args.seed:
        scene, t = scene_for_seed(env, s2, seed)
        if t is None:
            print(f"seed {seed}: no straight reachable target; skipped")
            continue
        timeline = []
        st = run_episode(env, s2, ctl, scene, t, timeline=timeline, slowmo=args.shot_slowmo)
        print(describe(seed, st, is_clean(st, args)))
        if not is_clean(st, args):
            print("  note: this seed is not clean (see criteria in the docstring); rendering anyway")
        render_video(timeline, os.path.join(args.out_dir, f"shot_seed{seed}{args.suffix}"), args)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Record full autonomous shots as 1080p MP4 + GIF")
    p.add_argument("--seed", type=int, nargs="+", default=[0],
                   help="scene seed(s) to record (with --scan: the first seed scanned)")
    p.add_argument("--scan", type=int, default=0, help="try this many seeds without recording; list clean ones")
    p.add_argument("--out-dir", default=os.path.join(cem.REPO, "videos"))
    p.add_argument("--shot-slowmo", type=float, default=1.5,
                   help="frames per physics step during the shot (interpolated; 1 = real time)")
    p.add_argument("--no-pockets", action="store_true", help="omit the decorative corner pockets")
    p.add_argument("--gif-width", type=int, default=640)
    p.add_argument("--gif-fps", type=int, default=20)
    p.add_argument("--clean-miss", type=float, default=0.02, help="m; max miss for a clean seed")
    p.add_argument("--min-travel", type=float, default=0.08, help="m; min target distance for a clean seed")
    p.add_argument("--checkpoint-dir", default=cem.CHECKPOINT_DIR)
    p.add_argument("--suffix", default="_bigcue")
    p.add_argument("--background", default=cem.BACKGROUND_PATH)
    p.add_argument("--calib-scenes", type=int, default=400)
    p.add_argument("--calib-near-robot", type=float, default=0.0,
                   help="extra share of calibration scenes with the robot right behind the cue ball")
    main(p.parse_args())
