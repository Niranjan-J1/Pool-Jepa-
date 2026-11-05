"""Record one full autonomous shot (Stage 1 aim -> Stage 2 force -> fire) as MP4 + GIF.

    python planning/record_shot.py --scan 40               # try seeds 0..39, list clean ones
    python planning/record_shot.py --seed 7                # record seed 7 -> videos/shot_seed7.{mp4,gif}

The scene for a seed is built exactly as in full_shot_test.py: a behind_cue scene and a
straight-line reachable target (the rest point of a secret force fired from the exact firing
pose). Stage 1 runs on the decoded cue position with ball avoidance on (it is given the other
balls' true positions to path around), Stage 2 picks the force through the shot model, and the
robot fires. Every simulator step is rendered with env.render().

Overlay:
  - the target (crosshair), the decoded cue position and the aim line while the robot aims;
  - during force planning, the candidate forces' predicted cue rest points appearing one by one,
    coloured by predicted miss (green = on target, red = far), then the chosen one ringed:
    the robot "imagining" shots before it fires;
  - after the shot, the predicted and actual rest points and the miss distance.

A seed is "clean" when Stage 1 aligned in time, no other ball moved before the shot, the
strike connected, the cue ball stopped within --clean-miss of the target, and the target is
at least --min-travel away (so the shot is visible).
Needs: pip install imageio imageio-ffmpeg
"""
import argparse
import math
import os
import random
import sys
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from env.sim_env import (BALL_RADIUS, RENDER_HEIGHT, RENDER_MARGIN, RENDER_WIDTH, SimEnv,
                         TABLE_LENGTH)
from planning import cem
from planning.aim_controller import AimController
from planning.full_shot_test import (BALL_MOVED, DECODE_EVERY, MAX_SETTLE, STAGE1_MAX_STEPS,
                                     Stage2, sample_target)
from planning.shot_planner import IDLE

FPS = 30                 # output frame rate; the sim runs at 60 Hz, so every 2nd sim frame is kept
SIM_STRIDE = 2
HALF_W = TABLE_LENGTH / 2 + RENDER_MARGIN
HALF_H = HALF_W * RENDER_HEIGHT / RENDER_WIDTH


# --------------------------------------------------------------------------------------
# Overlay drawing
# --------------------------------------------------------------------------------------

class Painter:
    def __init__(self, scale: int):
        self.s = scale
        try:
            self.font = ImageFont.load_default(size=7 * scale)
        except TypeError:  # Pillow < 10.1
            self.font = ImageFont.load_default()

    def px(self, xy):
        x, y = float(xy[0]), float(xy[1])
        return ((x + HALF_W) / (2 * HALF_W) * RENDER_WIDTH * self.s,
                (HALF_H - y) / (2 * HALF_H) * RENDER_HEIGHT * self.s)

    def frame(self, raw, target, title, caption="", aim=None, cue_est=None, candidates=None,
              chosen=None, actual=None):
        """raw: (H, W, 3) uint8 render. candidates: (points (N, 2), costs (N,)) to draw."""
        img = Image.fromarray(raw).resize((RENDER_WIDTH * self.s, RENDER_HEIGHT * self.s),
                                          Image.LANCZOS)
        d = ImageDraw.Draw(img, "RGBA")
        s = self.s
        if aim is not None:
            d.line([self.px(aim[0]), self.px(aim[1])], fill=(255, 255, 255, 110), width=max(1, s))
        if candidates is not None:
            pts, costs = candidates
            for p, c in zip(pts, costs):
                t = float(np.clip(c / 0.10, 0, 1))     # 0 cm -> green, >= 10 cm -> red
                col = (int(60 + 195 * t), int(220 - 170 * t), 60, 200)
                x, y = self.px(p)
                r = 1.6 * s
                d.ellipse([x - r, y - r, x + r, y + r], fill=col)
        # target crosshair
        tx, ty = self.px(target)
        r = 4 * s
        d.ellipse([tx - r, ty - r, tx + r, ty + r], outline=(255, 215, 0, 255), width=max(1, s // 2 + 1))
        d.line([tx - 1.6 * r, ty, tx + 1.6 * r, ty], fill=(255, 215, 0, 255), width=max(1, s // 2))
        d.line([tx, ty - 1.6 * r, tx, ty + 1.6 * r], fill=(255, 215, 0, 255), width=max(1, s // 2))
        if cue_est is not None:   # where the encoder thinks the cue ball is
            x, y = self.px(cue_est)
            r = BALL_RADIUS / (2 * HALF_W) * RENDER_WIDTH * s * 1.6
            d.ellipse([x - r, y - r, x + r, y + r], outline=(0, 220, 255, 230), width=max(1, s // 2))
        if chosen is not None:
            x, y = self.px(chosen)
            r = 3.2 * s
            d.ellipse([x - r, y - r, x + r, y + r], outline=(255, 255, 255, 255), width=s)
        if actual is not None:
            x, y = self.px(actual)
            r = 2.2 * s
            d.ellipse([x - r, y - r, x + r, y + r], outline=(255, 80, 255, 255), width=s)
        pad = 3 * s
        for text, y0, anchor_bottom in ((title, pad, False), (caption, RENDER_HEIGHT * s - pad, True)):
            if not text:
                continue
            box = d.textbbox((0, 0), text, font=self.font)
            w, h = box[2] - box[0], box[3] - box[1]
            y = y0 - h - pad if anchor_bottom else y0
            d.rectangle([pad, y - pad / 2, pad + w + 2 * pad, y + h + pad * 1.5], fill=(0, 0, 0, 150))
            d.text((2 * pad, y), text, font=self.font, fill=(255, 255, 255, 255))
        return np.asarray(img)


# --------------------------------------------------------------------------------------
# One episode
# --------------------------------------------------------------------------------------

def run_episode(env, s2, ctl, scene, t, painter=None, n_show=48):
    """Stage 1 -> Stage 2 -> fire on a sampled scene. With a painter, returns video frames too."""
    record = painter is not None
    frames = []
    target = t["target"]
    random.setstate(scene)
    obs = env.reset(setup="behind_cue")
    balls0 = np.array(obs["ball_positions"])

    # Opening hold: the scene and the target
    if record:
        raw = env.render()
        frames += [painter.frame(raw, target, "Target: send the cue ball to the gold mark")] * FPS

    # Stage 1
    ctl.reset()
    cue_est, done, n = None, False, 0
    for n in range(STAGE1_MAX_STEPS):
        decode = n % DECODE_EVERY == 0
        raw = env.render() if (record or decode) else None
        if decode:
            cue_est = s2.decode_cue(raw)
        left, right, done, info = ctl.step(obs["robot_pose"], cue_est, target,
                                           np.array(obs["ball_positions"][1:]))
        if record and n % SIM_STRIDE == 0:
            frames.append(painter.frame(raw, target, "Stage 1: aiming (closed loop on decoded cue position)",
                                        f"{info['phase']}  |  heading error {math.degrees(info['head_err']):+6.1f} deg",
                                        aim=(cue_est, target), cue_est=cue_est))
        if done:
            break
        obs = env.step(dict(IDLE, left_pwm=left, right_pwm=right), render=False)
    for _ in range(MAX_SETTLE):   # anything nudged comes to rest before the shot
        if env._balls_settled():
            break
        env.step(IDLE, render=False)
    obs = env._get_observation(render=False)
    balls = np.array(obs["ball_positions"])
    disturbed = bool((np.linalg.norm(balls[1:] - balls0[1:], axis=1) > BALL_MOVED).any())
    cue_moved = bool(np.linalg.norm(balls[0] - balls0[0]) > BALL_MOVED)

    # Stage 2: two rest frames -> z_pre -> score every candidate force through the shot model
    f_prev = env.render()
    obs = env.step(IDLE, render=True)
    z_pre = cem.encode(s2.encoder, f_prev, obs["frame"], s2.background, s2.device)
    heading = obs["robot_pose"][2] if s2.model.use_heading else None
    force, pinfo = s2.planner.plan(z_pre, target, heading)
    i_best = int(np.argmin(pinfo["all_cost"]))
    if record:
        raw = obs["frame"]
        allf = s2.planner.forces.cpu().numpy()
        idx = np.unique(np.r_[np.linspace(0, len(allf) - 1, n_show).astype(int), i_best])
        pts, costs = pinfo["all_rest"][idx], pinfo["all_cost"][idx]
        for k in range(1, len(idx) + 1):
            frames.append(painter.frame(raw, target, "Stage 2: imagining shots with the world model",
                                        f"candidate force {allf[idx[k - 1]]:.2f} -> predicted miss "
                                        f"{100 * costs[k - 1]:4.1f} cm",
                                        candidates=(pts[:k], costs[:k])))
        frames += [painter.frame(raw, target, "Stage 2: best imagined shot",
                                 f"fire at force {force:.2f}  (predicted miss {100 * pinfo['predicted_miss']:.1f} cm)",
                                 candidates=(pts, costs), chosen=pinfo["predicted_rest"])] * int(1.2 * FPS)

    # Fire and roll out
    obs = env.step(dict(IDLE, fire=True, fire_force=force), render=record)
    connected = env.last_strike_connected
    for k in range(MAX_SETTLE):
        if record and k % SIM_STRIDE == 0:
            frames.append(painter.frame(obs["frame"], target, f"Fire! force {force:.2f}",
                                        chosen=pinfo["predicted_rest"]))
        if obs["ball_settled"] and k > 0:
            break
        obs = env.step(IDLE, render=record)
    rest = np.array(env._get_observation(render=False)["ball_positions"][0])
    miss = float(np.linalg.norm(rest - target))
    if record:
        raw = env.render()
        frames += [painter.frame(raw, target, f"Result: {100 * miss:.1f} cm from the target",
                                 "white ring = imagined rest point, magenta = actual",
                                 chosen=pinfo["predicted_rest"], actual=rest)] * int(1.5 * FPS)

    stats = dict(aligned=done, steps=n, disturbed=disturbed, cue_moved=cue_moved, connected=connected,
                 miss=miss, force=force, predicted_miss=pinfo["predicted_miss"],
                 travel=float(np.linalg.norm(target - t["cue0"])))
    return stats, frames


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


def save_videos(frames, out_base, gif_scale):
    import imageio.v2 as imageio
    mp4, gif = out_base + ".mp4", out_base + ".gif"
    with imageio.get_writer(mp4, fps=FPS, codec="libx264", quality=8, pixelformat="yuv420p",
                            macro_block_size=16) as w:
        for f in frames:
            w.append_data(f)
    small = [np.asarray(Image.fromarray(f).resize((RENDER_WIDTH * gif_scale, RENDER_HEIGHT * gif_scale),
                                                  Image.LANCZOS)) for f in frames]
    imageio.mimsave(gif, small, duration=1000 / FPS, loop=0)
    for p in (mp4, gif):
        print(f"saved {p} ({os.path.getsize(p) / 2**20:.1f} MB)")


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = SimEnv()
    s2 = Stage2(SimpleNamespace(checkpoint_dir=args.checkpoint_dir, suffix=args.suffix,
                                background=args.background, calib_scenes=args.calib_scenes,
                                seed=0), env, device)
    ctl = AimController()

    if args.scan:
        clean = []
        for seed in range(args.seed, args.seed + args.scan):
            scene, t = scene_for_seed(env, s2, seed)
            if t is None:
                print(f"seed {seed:4d} | no straight reachable target")
                continue
            st, _ = run_episode(env, s2, ctl, scene, t)
            ok = is_clean(st, args)
            clean += [(seed, st)] if ok else []
            print(describe(seed, st, ok), flush=True)
        print(f"\n{len(clean)} clean seeds (aligned, nothing disturbed, connected, miss <= "
              f"{100 * args.clean_miss:.0f} cm, target >= {100 * args.min_travel:.0f} cm away):")
        for seed, st in sorted(clean, key=lambda x: -x[1]["travel"]):
            print(f"  --seed {seed:<5d} target {100 * st['travel']:5.1f} cm, miss {100 * st['miss']:.1f} cm")
        return

    scene, t = scene_for_seed(env, s2, args.seed)
    if t is None:
        raise SystemExit(f"seed {args.seed}: no straight reachable target; try another seed")
    st, frames = run_episode(env, s2, ctl, scene, t, painter=Painter(args.scale))
    print(describe(args.seed, st, is_clean(st, args)))
    if not is_clean(st, args):
        print("note: this seed is not clean (see criteria in the docstring); saving anyway")
    os.makedirs(args.out_dir, exist_ok=True)
    save_videos(frames, os.path.join(args.out_dir, f"shot_seed{args.seed}"), args.gif_scale)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Record one full autonomous shot as MP4 + GIF")
    p.add_argument("--seed", type=int, default=0, help="scene seed (with --scan: first seed)")
    p.add_argument("--scan", type=int, default=0, help="try this many seeds without recording; list clean ones")
    p.add_argument("--out-dir", default=os.path.join(cem.REPO, "videos"))
    p.add_argument("--scale", type=int, default=2, help="MP4 upscale factor (2 -> 864x480)")
    p.add_argument("--gif-scale", type=int, default=1, help="GIF upscale factor (keep small for file size)")
    p.add_argument("--clean-miss", type=float, default=0.02, help="m; max miss for a clean seed")
    p.add_argument("--min-travel", type=float, default=0.08, help="m; min target distance for a clean seed")
    p.add_argument("--checkpoint-dir", default=cem.CHECKPOINT_DIR)
    p.add_argument("--suffix", default="_best")
    p.add_argument("--background", default=cem.BACKGROUND_PATH)
    p.add_argument("--calib-scenes", type=int, default=400)
    main(p.parse_args())
