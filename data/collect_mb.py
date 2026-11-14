"""Multiball data collection (bucket B1). Called from data/collect.py; legacy modes are untouched.

Modes:
    mb_shots   One strike per record: two frames before, the frame at rest, fixed-slot ball state
               before/after, which pocket each ball went into, scratch, heading, force.
               Training/eval data for the multiball shot model. Saved as <prefix>_XXX.npz.
    mb_frames  Random driving with 3-5 strikes per 50-step episode on 1..N object balls, every
               step rendered. Encoder training frames. Saved as <prefix>_XXX.npz.

Splits (scene seeds never overlap between splits or shards):
    train   n_object_balls in 1..5, seeds  0e8 + shard*1e6 + i
    test    n_object_balls in 1..5, seeds  1e8 + shard*1e6 + i
    novel   n_object_balls in 6..7, seeds  2e8 + shard*1e6 + i   ("never seen" layouts)
--n-balls LO HI overrides the ball-count range.

mb_shots aim mixture (per shot, from the scene seed):
    ghost   50%  heading through the ghost-ball point of a random (object ball, pocket) with a
                 cut <= 60 deg and clear paths, +-1 deg, force 0.4-1.0
    ball    30%  heading at a random object ball, anywhere within its angular width (a hit)
    uniform 20%  any heading
The robot's heading is the shot direction (the solenoid fires along it), so the reset aim is
set to heading - aim_offset with the seed's own aim_offset draw.
"""
import math
import os
import random
import time

import numpy as np

from env.base_env import Action
from env.sim_env import (SimEnv, BALL_RADIUS, MAX_OBJECT_BALLS, ROBOT_LENGTH, pocket_centers)

SPLITS = {"train": (0, (1, 5)), "test": (1, (1, 5)), "novel": (2, (6, 7))}
SPLIT_SEED_STRIDE = 100_000_000
SHARD_SEED_STRIDE = 1_000_000
AIM_MIX = (("ghost", 0.5), ("ball", 0.3), ("uniform", 0.2))
AIM_KIND_ID = {"ghost": 0, "ball": 1, "uniform": 2}
GHOST_JITTER = math.radians(1.0)   # 3 deg was ~1.6 cm at the ghost point over 30 cm: too much
GHOST_MAX_CUT = math.radians(60.0)   # skip pockets the ball cannot realistically be cut into
MB_FORCE_RANGE = (0.05, 1.0)
GHOST_FORCE_RANGE = (0.4, 1.0)
MB_MAX_SETTLE_STEPS = 900
MB_SHOTS_PER_FILE = 1000
MB_FRAME_EPISODES_PER_FILE = 100
MB_FRAME_STEPS = 50
MB_FRAME_FIRES = (3, 5)
MB_FRAME_FIRE_STEPS = (5, 45)
MB_FRAME_FORCE_RANGE = (0.3, 1.0)
MAX_SCENE_TRIES = 50
IDLE = Action(left_pwm=0.0, right_pwm=0.0, fire=False, fire_force=0.0)
N_SLOTS = MAX_OBJECT_BALLS + 1


def scene_seed(split, shard, i):
    return SPLITS[split][0] * SPLIT_SEED_STRIDE + shard * SHARD_SEED_STRIDE + i


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def _path_blocked(a, b, obstacles):
    """True if a ball rolling from a to b would touch any obstacle ball (centre within 2R of
    the segment)."""
    ax, ay = a
    dx, dy = b[0] - ax, b[1] - ay
    L2 = dx * dx + dy * dy or 1e-12
    for ox, oy in obstacles:
        t = max(0.0, min(1.0, ((ox - ax) * dx + (oy - ay) * dy) / L2))
        if math.hypot(ax + t * dx - ox, ay + t * dy - oy) < 2 * BALL_RADIUS:
            return True
    return False


def choose_heading(rng, cue, balls):
    """Pick the shot direction for one scene. Returns (heading, kind, target_ball, target_pocket)."""
    u, acc = rng.random(), 0.0
    kind = AIM_MIX[-1][0]
    for name, p in AIM_MIX:
        acc += p
        if u < acc:
            kind = name
            break
    ids = sorted(balls)
    if kind == "uniform" or not ids:
        return rng.uniform(-math.pi, math.pi), "uniform", -1, -1
    b = rng.choice(ids)
    bx, by = balls[b]
    if kind == "ball":
        d = math.hypot(bx - cue[0], by - cue[1])
        half = math.asin(min(1.0, 2 * BALL_RADIUS / d))
        return _wrap(math.atan2(by - cue[1], bx - cue[0]) + rng.uniform(-half, half)), "ball", b, -1
    # Ghost ball: only pockets with a makeable cut (<= GHOST_MAX_CUT) and clear paths (cue ->
    # ghost point, target ball -> pocket, no other ball in the way); none -> aim at the ball.
    others = [p for i, p in balls.items() if i != b]
    options = []
    for k, (px, py) in enumerate(pocket_centers()):
        n = math.hypot(px - bx, py - by) or 1.0
        gx, gy = bx - 2 * BALL_RADIUS * (px - bx) / n, by - 2 * BALL_RADIUS * (py - by) / n
        shot = math.atan2(gy - cue[1], gx - cue[0])
        if abs(_wrap(shot - math.atan2(py - by, px - bx))) > GHOST_MAX_CUT:
            continue
        if _path_blocked(cue, (gx, gy), others) or _path_blocked((bx, by), (px, py), others):
            continue
        options.append((k, shot))
    if not options:
        d = math.hypot(bx - cue[0], by - cue[1])
        half = math.asin(min(1.0, 2 * BALL_RADIUS / d))
        return _wrap(math.atan2(by - cue[1], bx - cue[0]) + rng.uniform(-half, half)), "ball", b, -1
    k, shot = rng.choice(options)
    return _wrap(shot + rng.uniform(-GHOST_JITTER, GHOST_JITTER)), "ghost", b, k


def setup_shot(seed, n_range):
    """Build one aimed scene. Returns (env, obs, info) or None if the aimed robot pose does not
    fit this layout (the caller moves to the next seed and counts the reject)."""
    rng = random.Random(seed)
    n = rng.randint(*n_range)
    env = SimEnv(n_object_balls=n)
    obs = env.reset(setup="behind_cue", seed=seed)
    layout = {i: tuple(p) for i, p in enumerate(obs["ball_positions"])}
    offset = env.last_setup["aim_offset"]
    heading, kind, tb, tp = choose_heading(rng, layout[0], {i: p for i, p in layout.items() if i})
    # Same seed + aim override: the cue position and distance/offset draws are unchanged
    # (tests/test_multiball.py::test_reset_aim); the layout is passed back verbatim.
    obs = env.reset(setup="behind_cue", seed=seed, aim=_wrap(heading - offset), layout=layout)
    rb = env._robot_body
    front = rb.local_to_world((ROBOT_LENGTH / 2, 0))
    gap = math.hypot(layout[0][0] - front.x, layout[0][1] - front.y)
    if abs(gap - env.last_setup["distance"]) > 1e-6:
        return None  # sampler redrew the cue to fit the aimed robot, so it is not behind layout's cue
    if any(env._overlaps_robot(p, (rb.position.x, rb.position.y), rb.angle, BALL_RADIUS + 0.002)
           for i, p in layout.items() if i):
        return None  # robot would sit on a ball
    info = dict(n=n, heading=float(rb.angle), kind=kind, target_ball=tb, target_pocket=tp,
                distance=env.last_setup["distance"], offset=offset)
    return env, obs, info


def _pocket_array(events):
    """(N_SLOTS,) int8: pocket index each ball went into (last event for the cue), -1 if none."""
    out = np.full(N_SLOTS, -1, dtype=np.int8)
    for ball_id, k in events:
        out[ball_id] = k
    return out


def _save(path, buf, dtypes):
    arrays = {k: np.asarray(v, dtype=dtypes.get(k, np.float32)) for k, v in buf.items()}
    np.savez_compressed(path, **arrays)
    return os.path.getsize(path) / 2 ** 20


SHOT_DTYPES = {
    "frame_pre_prev": np.uint8, "frame_pre": np.uint8, "frame_post": np.uint8,
    "mask_pre": np.bool_, "mask_post": np.bool_, "potted_pockets": np.int8,
    "scratch": np.bool_, "strike_connected": np.bool_, "n_object_balls": np.int8,
    "aim_kind": np.int8, "aim_target_ball": np.int8, "aim_target_pocket": np.int8,
    "settle_steps": np.int32, "scene_seed": np.int64, "settled": np.bool_,
}


def collect_mb_shots(out_dir, prefix, shots, split="train", shard=0, n_range=None,
                     force_range=MB_FORCE_RANGE):
    n_range = tuple(n_range or SPLITS[split][1])
    print(f"Mode: mb_shots  split={split} shard={shard} n_object_balls={n_range}  shots={shots}  "
          f"force {force_range}  -> {prefix}_XXX.npz in {out_dir}", flush=True)
    os.makedirs(out_dir, exist_ok=True)
    keys = list(SHOT_DTYPES) + ["slots_pre", "slots_post", "robot_pose_pre", "heading", "fire_force",
                                "cue_travel", "setup_distance", "setup_aim_offset"]
    buf = {k: [] for k in keys}
    t0, i, rejects, file_idx = time.time(), 0, 0, 0
    totals = dict(pot=0, scratch=0, connected=0, unsettled=0)
    for kept in range(shots):
        for _ in range(MAX_SCENE_TRIES):
            seed = scene_seed(split, shard, i)
            i += 1
            built = setup_shot(seed, n_range)
            if built is not None:
                break
            rejects += 1
        else:
            raise RuntimeError(f"{MAX_SCENE_TRIES} consecutive scenes rejected near seed {seed}")
        env, obs, info = built
        f_prev = obs["frame"]
        pre = env.step(IDLE)                      # encoder input before the shot: (f_prev, f_pre)
        slots_pre, mask_pre = env.ball_slots()
        # Ghost-ball shots are meant to pot, and soft ones never reach the pocket (0% in the
        # smoke run), so they get the harder GHOST_FORCE_RANGE; other aims keep the full range.
        f_range = GHOST_FORCE_RANGE if info["kind"] == "ghost" else force_range
        force = random.Random(seed ^ 0x5EED).uniform(*f_range)
        env.step(Action(left_pwm=0.0, right_pwm=0.0, fire=True, fire_force=force), render=False)
        connected = env.last_strike_connected
        post, steps = env.settle(max_steps=MB_MAX_SETTLE_STEPS)
        slots_post, mask_post = env.ball_slots()
        pockets = _pocket_array(post["potted_pockets"])

        row = dict(frame_pre_prev=f_prev, frame_pre=pre["frame"], frame_post=post["frame"],
                   mask_pre=mask_pre, mask_post=mask_post, potted_pockets=pockets,
                   scratch=post["scratch"], strike_connected=connected, n_object_balls=info["n"],
                   aim_kind=AIM_KIND_ID[info["kind"]], aim_target_ball=info["target_ball"],
                   aim_target_pocket=info["target_pocket"], settle_steps=steps, scene_seed=seed,
                   settled=post["ball_settled"], slots_pre=slots_pre, slots_post=slots_post,
                   robot_pose_pre=pre["robot_pose"], heading=info["heading"], fire_force=force,
                   cue_travel=float(np.linalg.norm(slots_post[0] - slots_pre[0])),
                   setup_distance=info["distance"], setup_aim_offset=info["offset"])
        for k, v in row.items():
            buf[k].append(v)
        totals["pot"] += bool((pockets[1:] >= 0).any())
        totals["scratch"] += bool(post["scratch"])
        totals["connected"] += bool(connected)
        totals["unsettled"] += not post["ball_settled"]

        if (kept + 1) % MB_SHOTS_PER_FILE == 0 or kept + 1 == shots:
            path = os.path.join(out_dir, f"{prefix}_{file_idx:03d}.npz")
            mb = _save(path, buf, SHOT_DTYPES)
            print(f"Shot {kept + 1}/{shots}  ({(kept + 1) / (time.time() - t0):.2f} shots/s)  "
                  f"Saved {path} ({mb:.1f} MB)", flush=True)
            file_idx += 1
            for v in buf.values():
                v.clear()
    el = time.time() - t0
    print(f"\nSUMMARY mb_shots split={split} shard={shard}: {shots} shots in {el:.0f}s "
          f"({shots / el:.2f}/s), scene rejects {rejects}, "
          f"pot>=1 {100 * totals['pot'] / shots:.1f}%, scratch {100 * totals['scratch'] / shots:.1f}%, "
          f"connected {100 * totals['connected'] / shots:.1f}%, unsettled {totals['unsettled']}")


FRAME_DTYPES = {
    "frames_before": np.uint8, "frames_after": np.uint8, "actions_fire": np.bool_,
    "mask_before": np.bool_, "mask_after": np.bool_, "strike_connected": np.bool_,
    "episode_idx": np.int32, "step_idx": np.int32, "n_object_balls": np.int8,
    "scene_seed": np.int64,
}


def collect_mb_frames(out_dir, prefix, episodes, split="train", shard=0, n_range=None):
    n_range = tuple(n_range or SPLITS[split][1])
    print(f"Mode: mb_frames  split={split} shard={shard} n_object_balls={n_range}  episodes={episodes}"
          f"  -> {prefix}_XXX.npz in {out_dir}", flush=True)
    os.makedirs(out_dir, exist_ok=True)
    keys = list(FRAME_DTYPES) + ["actions_left_pwm", "actions_right_pwm", "actions_fire_force",
                                 "robot_pose_before", "robot_pose_after", "slots_before", "slots_after"]
    buf = {k: [] for k in keys}
    t0, file_idx, fires, connected = time.time(), 0, 0, 0
    for ep in range(episodes):
        seed = scene_seed(split, shard, ep)
        rng = random.Random(seed)
        env = SimEnv(n_object_balls=rng.randint(*n_range))
        obs = env.reset(setup="behind_cue", seed=seed)
        fire_steps = set(rng.sample(range(MB_FRAME_FIRE_STEPS[0], MB_FRAME_FIRE_STEPS[1] + 1),
                                    rng.randint(*MB_FRAME_FIRES)))
        slots, mask = env.ball_slots()
        for step in range(MB_FRAME_STEPS):
            fire = step in fire_steps
            action = Action(left_pwm=rng.uniform(-1, 1), right_pwm=rng.uniform(-1, 1), fire=fire,
                            fire_force=rng.uniform(*MB_FRAME_FORCE_RANGE) if fire else 0.0)
            prev, prev_slots, prev_mask = obs, slots, mask
            obs = env.step(action)
            slots, mask = env.ball_slots()
            row = dict(frames_before=prev["frame"], frames_after=obs["frame"],
                       actions_fire=fire, mask_before=prev_mask, mask_after=mask,
                       strike_connected=env.last_strike_connected, episode_idx=ep, step_idx=step,
                       n_object_balls=env._n_object_balls, scene_seed=seed,
                       actions_left_pwm=action["left_pwm"], actions_right_pwm=action["right_pwm"],
                       actions_fire_force=action["fire_force"], robot_pose_before=prev["robot_pose"],
                       robot_pose_after=obs["robot_pose"], slots_before=prev_slots, slots_after=slots)
            for k, v in row.items():
                buf[k].append(v)
            fires += fire
            connected += env.last_strike_connected
        if (ep + 1) % MB_FRAME_EPISODES_PER_FILE == 0 or ep + 1 == episodes:
            path = os.path.join(out_dir, f"{prefix}_{file_idx:03d}.npz")
            mb = _save(path, buf, FRAME_DTYPES)
            print(f"Episode {ep + 1}/{episodes}  ({(ep + 1) / (time.time() - t0):.2f} ep/s)  "
                  f"Saved {path} ({mb:.1f} MB)", flush=True)
            file_idx += 1
            for v in buf.values():
                v.clear()
    el = time.time() - t0
    print(f"\nSUMMARY mb_frames split={split} shard={shard}: {episodes} episodes "
          f"({episodes * MB_FRAME_STEPS} frames) in {el:.0f}s ({episodes / el:.2f} ep/s), "
          f"strikes connected {connected}/{fires}")
