"""Record full autonomous shots (Stage 1 aim -> Stage 2 force -> fire) as 1080p MP4 + GIF.

    python planning/record_shot.py --scan 60                 # list clean seeds
    python planning/record_shot.py --seed 59                 # one clip per seed
    python planning/record_shot.py --montage --seed 0 59 33 18   # one video, shots back to back

The scene for a seed is built exactly as in full_shot_test.py: a behind_cue scene and a
straight-line reachable target (the rest point of a secret force fired from the exact firing
pose). Stage 1 runs on the decoded cue position with ball avoidance on (it is given the other
balls' true positions to path around), Stage 2 picks the force through the shot model, and the
robot fires.

The clip (planning/demo_render.py, 1920x1080, 60 fps) shows the work as it happens:
  camera inset     throughout: the encoder's actual input (432x240 camera frame minus the empty
                   table) with its learned keypoints, the cue-ball keypoints highlighted, and
                   the decoded cue-ball position. During aiming it updates at the controller's
                   10 Hz decode rate; during the roll-out it is re-encoded every physics step for
                   display only (the shot is open loop).
  Finding the cue ball   the reticle fades in where the encoder locates it
  Aiming                 closed-loop drive to the firing pose, one frame per physics step
  Imagining N shots      every candidate strength in the planner's grid (N of them) as a ray
                         from the cue ball to its predicted landing point, green (close to the
                         target) to red; the chosen one brightens and locks in. The shot model
                         predicts landing points, not paths, and every candidate goes along the
                         same heading, so the rays are straight.
  Firing                 the shot with a fading trail; --shot-slowmo interpolates between
                         physics steps (default 1.5x, labelled)
  Result                 imagined vs actual landing point and the miss distance
--montage puts several seeds in one video with a title card before each shot and a closing card.
The corner pockets are decorative (the simulated table has none); --no-pockets drops them.

A seed is "clean" when Stage 1 aligned in time, no ball moved before the shot, the strike
connected, the cue ball stopped within --clean-miss of the target, and the target is at least
--min-travel away.
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


def perceive(s2, raw, with_inset=False):
    """Encode one camera frame (the (frame, frame) pair the readout was calibrated on) and decode
    the cue ball. Same computation as Stage2.decode_cue. Optionally also the inset data."""
    z = cem.encode(s2.encoder, raw, raw, s2.background, s2.device)
    cue = s2.kp.decode_cue(z)
    if not with_inset:
        return cue, None
    return cue, inset_data(s2, raw, z, cue)


def inset_data(s2, raw, z, cue):
    """The encoder input as an image: (frame - background) / 255 in [-1, 1] -> grey levels."""
    img = np.clip(128 + 0.5 * (raw.astype(np.float32) - s2.background), 0, 255).astype(np.uint8)
    return dict(image=img, kps=z.detach().cpu().numpy().reshape(-1, 2), cue_sel=s2.kp.selected, cue=cue)


# --------------------------------------------------------------------------------------
# One episode
# --------------------------------------------------------------------------------------

def run_episode(env, s2, ctl, scene, t, timeline=None, slowmo=1.5):
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

    # Finding the cue ball: the first decode, shown before the robot moves
    if record:
        cue_first, ins = perceive(s2, env.render(), with_inset=True)
        s0 = snap(env)
        for i in range(int(2.0 * FPS)):
            add(s0, stage="Finding the cue ball", sub="located from raw camera pixels by a learned encoder",
                target_alpha=min(1.0, i / 25), cue_est=cue_first,
                cue_alpha=float(np.clip((i - 30) / 30, 0, 1)), inset=ins)

    # Aiming (Stage 1)
    ctl.reset()
    cue_est, done, n, ins = None, False, 0, None
    for n in range(STAGE1_MAX_STEPS):
        if n % DECODE_EVERY == 0:
            cue_est, ins = perceive(s2, env.render(), with_inset=record)
        left, right, done, info = ctl.step(obs["robot_pose"], cue_est, target,
                                           np.array(obs["ball_positions"][1:]))
        add(snap(env), stage="Aiming", sub="driving behind the cue ball and lining up with the target",
            cue_est=cue_est, aim_line=True, inset=ins)
        if done:
            break
        obs = env.step(dict(IDLE, left_pwm=left, right_pwm=right), render=False)
    for _ in range(MAX_SETTLE):   # anything nudged comes to rest before the shot
        if env._balls_settled():
            break
        env.step(IDLE, render=False)
        add(snap(env), stage="Aiming", sub="driving behind the cue ball and lining up with the target",
            cue_est=cue_est, aim_line=True, inset=ins)
    obs = env._get_observation(render=False)
    balls = np.array(obs["ball_positions"])
    disturbed = bool((np.linalg.norm(balls[1:] - balls0[1:], axis=1) > BALL_MOVED).any())
    cue_moved = bool(np.linalg.norm(balls[0] - balls0[0]) > BALL_MOVED)

    # Imagining (Stage 2): two rest frames -> z_pre -> every candidate force through the shot model
    f_prev = env.render()
    obs = env.step(IDLE, render=True)
    z_pre = cem.encode(s2.encoder, f_prev, obs["frame"], s2.background, s2.device)
    heading = obs["robot_pose"][2] if s2.model.use_heading else None
    force, pinfo = s2.planner.plan(z_pre, target, heading)
    n_forces = len(s2.planner.forces)
    if record:
        st = snap(env)
        cue_now = s2.kp.decode_cue(z_pre)
        ins = inset_data(s2, obs["frame"], z_pre, cue_now)
        ends, costs = pinfo["all_rest"], pinfo["all_cost"]
        chosen = int(np.argmin(costs))
        label = f"Imagining {n_forces} shots"
        sub = "predicting where each strike strength would leave the cue ball"
        reveal, spray, lock_n, hold = int(1.4 * FPS), int(0.7 * FPS), int(0.8 * FPS), int(0.9 * FPS)
        fan = lambda shown, lock, lf=None: dict(start=cue_now, ends=ends, costs=costs, shown=shown,
                                                 chosen=chosen, lock=lock, lock_frame=lf)
        for i in range(reveal):     # sweep from soft to hard strikes
            add(st, stage=label, sub=sub, cue_est=cue_now, inset=ins,
                fan=fan(int(math.ceil((i + 1) / reveal * n_forces)), 0.0))
        for i in range(spray):      # the full spray
            add(st, stage=label, sub=sub, cue_est=cue_now, inset=ins, fan=fan(n_forces, 0.0))
        for i in range(lock_n):     # the best one brightens
            k = (i + 1) / lock_n
            add(st, stage=label, sub=f"best: strength {force:.2f}, predicted to land "
                                     f"{100 * pinfo['predicted_miss']:.1f} cm from the target",
                cue_est=cue_now, inset=ins, fan=fan(n_forces, k * k * (3 - 2 * k)))
        lock_at = len(timeline)
        for i in range(hold):       # locked in
            add(st, stage=label, sub=f"best: strength {force:.2f}, predicted to land "
                                     f"{100 * pinfo['predicted_miss']:.1f} cm from the target",
                cue_est=cue_now, inset=ins, fan=fan(n_forces, 1.0, lock_at))

    # Firing
    states, insets = ([snap(env)], [ins]) if record else (None, None)
    obs = env.step(dict(IDLE, fire=True, fire_force=force), render=False)
    connected = env.last_strike_connected
    for k in range(MAX_SETTLE):
        if record:
            states.append(snap(env))
            # display only: the shot is open loop, nothing is decoded for control here
            insets.append(perceive(s2, env.render(), with_inset=True)[1])
        if obs["ball_settled"] and k > 0:
            break
        obs = env.step(IDLE, render=False)
    rest = np.array(env._get_observation(render=False)["ball_positions"][0])
    miss = float(np.linalg.norm(rest - target))
    travel = float(np.linalg.norm(target - t["cue0"]))
    if record:
        sub = "solenoid strike" + (f"  ·  {slowmo:g}x slow motion" if slowmo > 1 else "")
        frames = interpolate(states, slowmo)
        for i, (s, j) in enumerate(frames):
            add(s, stage="Firing", sub=sub, trail=True, inset=insets[j],
                fan=dict(start=cue_now, ends=ends, costs=costs, shown=n_forces, chosen=chosen, lock=1.0,
                         lock_frame=lock_at, only_chosen=True),
                burst=(i / 14) if i < 14 else None)
        end = frames[-1][0]
        for i in range(int(3.5 * FPS)):
            add(end, stage="Result", inset=insets[-1],
                result=dict(predicted=pinfo["predicted_rest"], actual=rest, miss=miss, travel=travel,
                            force=force, n_forces=n_forces, alpha=min(1.0, i / 20)))

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


# --------------------------------------------------------------------------------------
# Video output
# --------------------------------------------------------------------------------------

class VideoOut:
    """Streams 1080p frames to an H.264 MP4 (never all in memory); keeps a small copy for the GIF."""

    def __init__(self, base, args):
        import imageio.v2 as imageio
        self.imageio, self.base, self.args = imageio, base, args
        self.w = imageio.get_writer(base + ".mp4", fps=FPS, codec="libx264", quality=None,
                                    pixelformat="yuv420p", macro_block_size=8,
                                    ffmpeg_params=["-crf", "18", "-preset", "slow", "-movflags", "+faststart"])
        self.gif_every = max(1, round(FPS / args.gif_fps))
        self.gif, self.n, self.t0 = [], 0, time.time()

    def add(self, f):
        self.w.append_data(f)
        if self.n % self.gif_every == 0:
            size = (self.args.gif_width, round(self.args.gif_width * 9 / 16))
            self.gif.append(np.asarray(Image.fromarray(f).resize(size, Image.LANCZOS)))
        self.n += 1

    def close(self):
        self.w.close()
        self.imageio.mimsave(self.base + ".gif", self.gif, duration=1000 * self.gif_every / FPS, loop=0)
        print(f"  {self.n} frames ({self.n / FPS:.1f} s) rendered in {time.time() - self.t0:.0f} s")
        for ext in (".mp4", ".gif"):
            print(f"  saved {self.base + ext} ({os.path.getsize(self.base + ext) / 2**20:.1f} MB)")


def add_card(out, renderer, title, sub="", note="", seconds=1.6):
    n = int(seconds * FPS)
    fade = int(0.3 * FPS)
    for i in range(n):
        a = min(1.0, (i + 1) / fade, (n - i) / fade)
        out.add(renderer.card(title, sub, a, note))


def add_shot(out, renderer, timeline):
    renderer.reset()
    for state, ov in timeline:
        out.add(renderer.frame(state, ov))


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
    renderer = DemoRenderer(pockets=not args.no_pockets)

    def episode(seed):
        scene, t = scene_for_seed(env, s2, seed)
        if t is None:
            print(f"seed {seed}: no straight reachable target; skipped")
            return None, None
        timeline = []
        st = run_episode(env, s2, ctl, scene, t, timeline=timeline, slowmo=args.shot_slowmo)
        print(describe(seed, st, is_clean(st, args)))
        if not is_clean(st, args):
            print("  note: this seed is not clean (see criteria in the docstring); rendering anyway")
        return st, timeline

    if args.montage:
        seeds = "_".join(map(str, args.seed))
        out = VideoOut(os.path.join(args.out_dir, f"montage{args.suffix}_{seeds}"), args)
        add_card(out, renderer, "A robot that imagines its shots",
                 "it finds the cue ball from camera pixels and predicts each strike before taking it",
                 seconds=2.4)
        misses = []
        for i, seed in enumerate(args.seed):
            st, timeline = episode(seed)
            if st is None:
                continue
            misses.append(st["miss"])
            add_card(out, renderer, f"Shot {i + 1} of {len(args.seed)}",
                     f"target {100 * st['travel']:.0f} cm away")
            add_shot(out, renderer, timeline)
        add_card(out, renderer, f"{len(misses)} shots, average miss {100 * np.mean(misses):.1f} cm",
                 "learned perception + a learned model of where each strike leaves the cue ball",
                 note=args.montage_note, seconds=3.0)
        out.close()
        return

    for seed in args.seed:
        st, timeline = episode(seed)
        if st is None:
            continue
        out = VideoOut(os.path.join(args.out_dir, f"shot_seed{seed}{args.suffix}"), args)
        add_shot(out, renderer, timeline)
        out.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Record full autonomous shots as 1080p MP4 + GIF")
    p.add_argument("--seed", type=int, nargs="+", default=[0],
                   help="scene seed(s) to record (with --scan: the first seed scanned)")
    p.add_argument("--scan", type=int, default=0, help="try this many seeds without recording; list clean ones")
    p.add_argument("--montage", action="store_true", help="record all --seed shots into one video")
    p.add_argument("--montage-note", default="simulation · selected successful shots from a 60-scene scan",
                   help="small print on the closing card")
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
