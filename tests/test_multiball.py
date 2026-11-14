"""Verification suite for the multiball simulator (Bucket A, Phase 1).

Run directly:  python tests/test_multiball.py
Covers every item in the task's "Verification" list. Each check prints PASS/FAIL and a
line of evidence; the process exits non-zero if anything fails.
"""
import math
import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from env.base_env import Action
from env.sim_env import (
    SimEnv, random_layout, pocket_centers, BALL_RADIUS, TABLE_LENGTH, TABLE_WIDTH,
    CUSHION_THICKNESS, POCKET_RADIUS, MAX_STRIKE_IMPULSE, BALL_MASS,
)

IDLE = Action(left_pwm=0.0, right_pwm=0.0, fire=False, fire_force=0.0)
_results = []


def check(name, ok, evidence=""):
    _results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  —  {evidence}" if evidence else ""))


def _place_cue_and_roll(env, start, velocity, max_steps=2000):
    """Put the cue ball at `start` with `velocity` (m/s) and let the table settle."""
    env._ball_bodies[0].position = start
    env._ball_bodies[0].velocity = velocity
    for n in range(max_steps):
        env.step(IDLE, render=False)
        if env._balls_settled():
            break
    return n


# --- 1. Regression: n_object_balls=0 reproduces the saved baseline -----------------------
def test_regression():
    import regression_baseline as rb
    with open(rb.BASELINE_PATH) as f:
        import json
        baseline = json.load(f)
    worst = rb.compare(rb.compute(), baseline)
    check("regression: legacy SimEnv() reproduces baseline", worst == 0.0, f"max abs diff {worst:.1e}")


# --- 2. Head-on collision (equal mass): cue stops, object carries on ---------------------
def test_head_on():
    env = SimEnv(n_object_balls=1)
    # Cue to the left of an object ball, both on y=0; aim +x straight into it.
    env.reset(layout={0: (-0.10, 0.0), 1: (0.0, 0.0)})
    v0 = 1.0
    env._ball_bodies[0].velocity = (v0, 0.0)
    # Step a few control steps to just past contact (not full settle, to read the exchange).
    for _ in range(12):
        env.step(IDLE, render=False)
    cue_speed = env._ball_bodies[0].velocity.length
    obj_speed = env._ball_bodies[1].velocity.length
    # Equal-mass head-on: the cue should transfer almost all its speed to the object ball.
    ok = cue_speed < 0.15 * v0 and obj_speed > 0.6 * v0
    check("head-on collision transfers momentum (equal mass)", ok,
          f"cue {cue_speed:.3f} m/s, object {obj_speed:.3f} m/s (v0={v0})")


# --- 3. Cushion: ball at 45 deg leaves at ~45 deg, slower -------------------------------
def test_cushion_angle():
    env = SimEnv(n_object_balls=0, cushion_elasticity=0.75)
    # Aim at the top long rail (y = +hw) from below at 45 deg, away from pockets/corners.
    start = (-0.08, 0.0)
    speed_in = 1.0
    vx, vy = speed_in * math.cos(math.radians(45)), speed_in * math.sin(math.radians(45))
    env.reset(layout={0: (0.0, 0.0)})
    env._ball_bodies[0].position = start
    env._ball_bodies[0].velocity = (vx, vy)
    # Watch vy flip sign (the bounce). The incoming angle/speed are measured from the last
    # velocity before the flip (after rolling resistance/damping), not assumed.
    before = None
    after = None
    prev_v = env._ball_bodies[0].velocity
    for _ in range(400):
        env.step(IDLE, render=False)
        v = env._ball_bodies[0].velocity
        if v.y < 0 and prev_v.y >= 0 and v.length > 1e-3:
            before = (abs(math.degrees(math.atan2(prev_v.y, prev_v.x))), prev_v.length)
            after = (abs(math.degrees(math.atan2(v.y, v.x))), v.length)
            break
        prev_v = v
    ok = after is not None and abs(after[0] - before[0]) < 12.0 and after[1] < before[1]
    check("cushion rebound angle ~ incoming angle, with reduced speed", ok,
          (f"in {before[0]:.1f} deg @ {before[1]:.3f} m/s -> out {after[0]:.1f} deg @ {after[1]:.3f} m/s"
           if after else "no bounce seen"))


# --- render: drawn felt edge matches the physical rail ----------------------------------
def test_render_felt_edge():
    from env.sim_env import (RENDER_WIDTH, RENDER_HEIGHT, RENDER_MARGIN, FELT_COLOR,
                             MULTIBALL_CUSHION_RADIUS)
    env = SimEnv(n_object_balls=0)
    env.reset(layout={0: (0.0, 0.0)})
    env._ball_bodies[0].position = (0.0, 0.0)
    env._robot_body.position = (0.0, -TABLE_WIDTH / 4)   # keep the robot off the scan lines
    env._robot_body.angle = 0.0
    img = env.render().astype(int)
    half_w = TABLE_LENGTH / 2 + RENDER_MARGIN
    half_h = half_w * RENDER_HEIGHT / RENDER_WIDTH
    px_per_m = RENDER_WIDTH / (2 * half_w)
    felt = np.array([int(FELT_COLOR[i:i + 2], 16) for i in (1, 3, 5)])
    is_felt = np.abs(img - felt).sum(axis=2) <= 6
    lim_x = TABLE_LENGTH / 2 - MULTIBALL_CUSHION_RADIUS   # where a ball surface meets the rail
    lim_y = TABLE_WIDTH / 2 - MULTIBALL_CUSHION_RADIUS
    # Expected boundaries in pixel-edge coordinates; measured = extent of the felt run.
    row = int(round((half_h - TABLE_WIDTH / 4) * px_per_m))        # y = +hw/2
    col = int(round((TABLE_LENGTH / 4 + half_w) * px_per_m))       # x = +hl/2
    xs = np.flatnonzero(is_felt[row]); ys = np.flatnonzero(is_felt[:, col])
    errs = [
        abs(xs[0] - (half_w - lim_x) * px_per_m), abs(xs[-1] + 1 - (half_w + lim_x) * px_per_m),
        abs(ys[0] - (half_h - lim_y) * px_per_m), abs(ys[-1] + 1 - (half_h + lim_y) * px_per_m),
    ] if len(xs) and len(ys) else [float("inf")]
    worst = max(errs)
    check("render: felt edge within 1 px of the physical rail (multiball)", worst <= 1.0,
          f"worst edge error {worst:.2f} px (left/right/top/bottom {', '.join(f'{e:.2f}' for e in errs)})")


# --- 4. Pockets: a ball rolled into each of the 6 is potted once and removed ------------
def test_pockets_all_six():
    pockets = pocket_centers()
    all_ok = True
    detail = []
    for pi, (px, py) in enumerate(pockets):
        env = SimEnv(n_object_balls=1)
        # Start the object ball just inside the rail, aim it straight at the pocket.
        inward = (-math.copysign(1, px) if abs(px) > 1e-6 else 0.0,
                  -math.copysign(1, py) if abs(py) > 1e-6 else 0.0)
        start = (px + inward[0] * 0.05 if inward[0] else px,
                 py + inward[1] * 0.05 if inward[1] else py)
        # Nudge start fully inside the cushions.
        sx = max(-(TABLE_LENGTH/2 - CUSHION_THICKNESS - BALL_RADIUS - 1e-3),
                 min(TABLE_LENGTH/2 - CUSHION_THICKNESS - BALL_RADIUS - 1e-3, start[0]))
        sy = max(-(TABLE_WIDTH/2 - CUSHION_THICKNESS - BALL_RADIUS - 1e-3),
                 min(TABLE_WIDTH/2 - CUSHION_THICKNESS - BALL_RADIUS - 1e-3, start[1]))
        env.reset(layout={0: (0.0, 0.0), 1: (sx, sy)})
        # Velocity straight toward the pocket.
        d = (px - sx, py - sy)
        norm = math.hypot(*d) or 1.0
        env._ball_bodies[1].velocity = (0.8 * d[0]/norm, 0.8 * d[1]/norm)
        potted_seen = 0
        events = None
        for _ in range(400):
            obs = env.step(IDLE, render=False)
            if 1 in obs["potted"]:
                potted_seen = obs["potted"].count(1)
                events = obs["potted_pockets"]
                break
        removed = 1 not in env._ball_bodies
        ok = potted_seen == 1 and removed and events == [(1, pi)]
        all_ok &= ok
        detail.append(f"p{pi}:{'ok' if ok else 'FAIL'}")
    check("all 6 pockets register a pot exactly once, in the right pocket, and remove the ball",
          all_ok, " ".join(detail))


def test_pocket_once_only():
    # The potted list must contain an id exactly once even after many further steps.
    env = SimEnv(n_object_balls=1)
    px, py = pocket_centers()[0]
    start = (px * 0.6, py * 0.6)
    env.reset(layout={0: (0.0, 0.0), 1: start})
    d = (px - start[0], py - start[1]); n = math.hypot(*d) or 1.0
    env._ball_bodies[1].velocity = (0.8 * d[0]/n, 0.8 * d[1]/n)
    for _ in range(400):
        env.step(IDLE, render=False)
    check("potted id recorded exactly once", env._potted.count(1) == 1,
          f"potted={env._potted}")


# --- cue pot => scratch + respawn --------------------------------------------------------
def test_scratch():
    env = SimEnv(n_object_balls=0, cue_spawn=(0.0, 0.0))
    env.reset(layout={0: (0.0, 0.0)})
    px, py = pocket_centers()[3]  # a corner
    start = (px * 0.5, py * 0.5)
    env._ball_bodies[0].position = start
    d = (px - start[0], py - start[1]); n = math.hypot(*d) or 1.0
    env._ball_bodies[0].velocity = (0.8 * d[0]/n, 0.8 * d[1]/n)
    scratched = False
    for _ in range(400):
        obs = env.step(IDLE, render=False)
        if obs["scratch"]:
            scratched = True
            break
    cue_back = 0 in env._ball_bodies
    near_spawn = math.hypot(*env._ball_bodies[0].position) < 0.05 if cue_back else False
    check("cue pot sets scratch and respawns the cue", scratched and cue_back and near_spawn,
          f"scratch={scratched}, cue_on_table={cue_back}, respawn={tuple(round(c,3) for c in env._ball_bodies[0].position)}")


# --- 5. No tunneling at max force, including into corners/pockets ------------------------
def test_no_tunneling():
    # Max impulse on a ball: 0.05 N·s / 0.02 kg = 2.5 m/s. Fire straight at corners and rails.
    vmax = MAX_STRIKE_IMPULSE / BALL_MASS
    hl, hw = TABLE_LENGTH/2, TABLE_WIDTH/2
    targets = pocket_centers() + [(hl, 0.05), (-0.05, hw)]  # pockets + mid-rail spots
    escaped = 0
    for tx, ty in targets:
        env = SimEnv(n_object_balls=1)
        env.reset(layout={0: (0.0, 0.0), 1: (0.0, 0.08)})
        start = (-tx * 0.4, -ty * 0.4) if (tx or ty) else (0.0, 0.0)
        env._ball_bodies[0].position = start
        d = (tx - start[0], ty - start[1]); nrm = math.hypot(*d) or 1.0
        env._ball_bodies[0].velocity = (vmax * d[0]/nrm, vmax * d[1]/nrm)
        for _ in range(60):
            env.step(IDLE, render=False)
            # If the cue is still on the table it must stay within the cushions (+ small skin).
            if 0 in env._ball_bodies:
                x, y = env._ball_bodies[0].position
                if abs(x) > hl + 0.015 or abs(y) > hw + 0.015:
                    escaped += 1
                    break
    check("no tunneling through cushions/corners at max force", escaped == 0,
          f"{escaped}/{len(targets)} balls escaped the table (vmax={vmax:.2f} m/s)")


# --- 6. Layout generator: 1000 layouts all legal ----------------------------------------
def test_layout_generator():
    rng = random.Random(0)
    bad_overlap = bad_cushion = bad_pocket = 0
    clearance = BALL_RADIUS  # ball edge must stay off the cushion
    for _ in range(1000):
        n = rng.randint(1, 7)
        lay = random_layout(n, rng)
        pts = list(lay.values())
        # non-overlap
        for i in range(len(pts)):
            for j in range(i+1, len(pts)):
                if math.hypot(pts[i][0]-pts[j][0], pts[i][1]-pts[j][1]) < 2*BALL_RADIUS:
                    bad_overlap += 1
        for x, y in pts:
            if abs(x) > TABLE_LENGTH/2 - CUSHION_THICKNESS - clearance or \
               abs(y) > TABLE_WIDTH/2 - CUSHION_THICKNESS - clearance:
                bad_cushion += 1
            if any(math.hypot(x-px, y-py) <= POCKET_RADIUS for px, py in pocket_centers()):
                bad_pocket += 1
    ok = bad_overlap == 0 and bad_cushion == 0 and bad_pocket == 0
    check("1000 random layouts: no overlap / cushion / pocket violations", ok,
          f"overlap={bad_overlap}, cushion={bad_cushion}, pocket={bad_pocket}")


# --- 7. Settle: every shot terminates with all balls at rest ----------------------------
def test_settle_terminates():
    worst_steps = 0
    unsettled = 0
    for seed in range(30):
        env = SimEnv(n_object_balls=5)
        env.reset(setup="behind_cue", seed=seed)
        env.step(Action(left_pwm=0.0, right_pwm=0.0, fire=True, fire_force=1.0), render=False)
        obs, steps = env.settle(max_steps=1200)
        worst_steps = max(worst_steps, steps)
        if not obs["ball_settled"]:
            unsettled += 1
    check("every multiball shot settles (all balls at rest)", unsettled == 0,
          f"{unsettled}/30 unsettled, worst settle {worst_steps} steps")


# --- 8. Speed: render=False stays fast; report steps/sec --------------------------------
def test_speed():
    import statistics

    def sps(n, runs=3, N=4000):
        vals = []
        for _ in range(runs):
            env = SimEnv() if n is None else SimEnv(n_object_balls=n)
            env.reset() if n is None else env.reset(seed=0)
            env.step(Action(left_pwm=0.0, right_pwm=0.0, fire=True, fire_force=0.8), render=False)
            t0 = time.time()
            for _ in range(N):
                env.step(IDLE, render=False)
            vals.append(N / (time.time() - t0))
        return statistics.mean(vals)

    s_legacy = sps(None)  # old code path, 4 balls
    s0 = sps(0)           # cue only
    s3 = sps(3)           # multiball with 4 balls total -> same workload as legacy
    s5 = sps(5)           # 6 balls
    # Fair "did multiball regress the sim?" bar: same ball count (legacy 4 vs n=3's 4). This
    # isolates the new per-substep pocket check and extra observation keys from the cost of
    # simply simulating more balls.
    overhead = s_legacy / s3 if s3 else float("inf")
    ratio_n5 = s_legacy / s5 if s5 else float("inf")
    print(f"       steps/sec: legacy(4)={s_legacy:,.0f}  n=0={s0:,.0f}  n=3(4)={s3:,.0f}  n=5(6)={s5:,.0f}")
    print(f"       multiball overhead at equal ball count (legacy/n=3) = {overhead:.2f}x")
    if ratio_n5 > 2.0:
        print(f"       FLAG: n=5 is {ratio_n5:.2f}x slower than legacy (2 extra balls + pockets)")
    check("render=False stays fast (multiball overhead < 2x at equal ball count)", overhead < 2.0,
          f"legacy {s_legacy:,.0f} vs n=3 {s3:,.0f} sps = {overhead:.2f}x; n=5 = {ratio_n5:.2f}x vs legacy")
    return dict(legacy=s_legacy, n0=s0, n3=s3, n5=s5)


def _scratch_cue(env):
    """Roll the cue into corner pocket 3 until a scratch registers."""
    px, py = pocket_centers()[3]
    start = (px * 0.5, py * 0.5)
    env._ball_bodies[0].position = start
    d = (px - start[0], py - start[1]); n = math.hypot(*d)
    env._ball_bodies[0].velocity = (0.8 * d[0] / n, 0.8 * d[1] / n)
    for _ in range(400):
        if env.step(IDLE, render=False)["scratch"]:
            return True
    return False


def _legal_cue(env):
    """Return (ok, detail): cue overlaps no ball/robot/pocket/cushion."""
    from env.sim_env import _center_in_play, _clear_of_pockets
    cx, cy = env._ball_bodies[0].position
    for i, b in env._ball_bodies.items():
        if i and math.hypot(cx - b.position.x, cy - b.position.y) < 2 * BALL_RADIUS:
            return False, f"overlaps ball {i}"
    rb = env._robot_body
    if env._overlaps_robot((cx, cy), (rb.position.x, rb.position.y), rb.angle, BALL_RADIUS):
        return False, "overlaps robot"
    if not _clear_of_pockets(cx, cy, env._pocket_radius):
        return False, "in pocket mouth"
    if not _center_in_play(cx, cy, BALL_RADIUS):
        return False, "on cushion"
    return True, ""


def test_respawn_legal():
    # Object ball sitting exactly on the spawn point.
    env = SimEnv(n_object_balls=2, cue_spawn=(0.0, 0.0))
    env.reset(layout={0: (0.0, 0.1), 1: (0.0, 0.0), 2: (0.012, 0.0)})
    scratched = _scratch_cue(env)
    ok, why = _legal_cue(env)
    cx, cy = env._ball_bodies[0].position
    moved = math.hypot(cx, cy)
    check("scratch respawn avoids a ball on the spawn point", scratched and ok and 0.0 < moved < 0.08,
          f"scratch={scratched}, legal={ok} {why}, respawn=({cx:.4f},{cy:.4f}), moved {moved*1000:.1f} mm")
    # Spawn point under the robot.
    env = SimEnv(n_object_balls=1, cue_spawn=(-TABLE_LENGTH / 4, 0.0))
    env.reset(layout={0: (0.0, 0.1), 1: (0.1, -0.1)})
    scratched = _scratch_cue(env)
    ok, why = _legal_cue(env)
    check("scratch respawn avoids the robot", scratched and ok, f"scratch={scratched}, legal={ok} {why}")
    # Free spawn point is used verbatim.
    env = SimEnv(n_object_balls=1, cue_spawn=(0.05, 0.05))
    env.reset(layout={0: (0.0, 0.1), 1: (0.1, -0.1)})
    _scratch_cue(env)
    cx, cy = env._ball_bodies[0].position
    check("scratch respawn uses cue_spawn when legal", abs(cx - 0.05) < 1e-9 and abs(cy - 0.05) < 1e-9,
          f"respawn=({cx:.4f},{cy:.4f})")


def test_ball_slots():
    env = SimEnv(n_object_balls=3)
    env.reset(layout={0: (0.0, 0.0), 1: (0.1, 0.05), 2: (-0.1, -0.05), 3: (0.15, -0.08)})
    pos0, mask0 = env.ball_slots()
    shape_ok = pos0.shape == (8, 2) and mask0.shape == (8,) and mask0.dtype == bool
    init_ok = list(mask0) == [True] * 4 + [False] * 4 and not pos0[4:].any()
    # Pot ball 2 (nothing else moves).
    px, py = pocket_centers()[0]
    start = (px * 0.6, py * 0.6)
    env._ball_bodies[2].position = start
    d = (px - start[0], py - start[1]); n = math.hypot(*d)
    env._ball_bodies[2].velocity = (0.8 * d[0] / n, 0.8 * d[1] / n)
    for _ in range(400):
        if 2 in env.step(IDLE, render=False)["potted"]:
            break
    pos1, mask1 = env.ball_slots()
    flipped = list(mask1) == [True, True, False, True, False, False, False, False]
    pot_zero = not pos1[2].any()
    others = all(np.array_equal(pos1[i], pos0[i]) for i in (0, 1, 3))
    obs_keys_ok = isinstance(env._get_observation(render=False)["ball_positions"], list)
    try:
        SimEnv().ball_slots(); legacy_raises = False
    except RuntimeError:
        legacy_raises = True
    check("ball_slots: shape, mask flips on pot, others unchanged",
          shape_ok and init_ok and flipped and pot_zero and others and obs_keys_ok and legacy_raises,
          f"shape={shape_ok} init={init_ok} flipped={flipped} potted_pos_zero={pot_zero} "
          f"others_unchanged={others} list_obs_intact={obs_keys_ok} legacy_raises={legacy_raises}")


def test_reset_aim():
    import json
    from env.sim_env import ROBOT_LENGTH
    # (a) robot sits behind the cue along the requested line.
    worst_ang, worst_dist, n = 0.0, 0.0, 0
    for aim in (0.3, 2.0, -1.5, 3.0, -3.0, 0.0):
        for seed in range(5):
            env = SimEnv(n_object_balls=3)
            obs = env.reset(setup="behind_cue", seed=seed, aim=aim)
            x, y, h = obs["robot_pose"]
            front = (x + ROBOT_LENGTH / 2 * math.cos(h), y + ROBOT_LENGTH / 2 * math.sin(h))
            cue = obs["ball_positions"][0]
            v = (cue[0] - front[0], cue[1] - front[1])
            ang_err = abs((math.atan2(v[1], v[0]) - aim + math.pi) % (2 * math.pi) - math.pi)
            dist_err = abs(math.hypot(*v) - env.last_setup["distance"])
            worst_ang, worst_dist, n = max(worst_ang, ang_err), max(worst_dist, dist_err), n + 1
    check("aim=a: robot front -> cue lies along angle a", worst_ang < 1e-9 and worst_dist < 1e-9,
          f"{n} resets, worst angle error {worst_ang:.2e} rad, worst distance error {worst_dist:.2e} m")
    # (b) aim=None reproduces the pre-change layouts exactly (saved from the previous commit).
    with open(os.path.join(os.path.dirname(__file__), "baseline_multiball_layout.json")) as f:
        base = json.load(f)
    mism = 0
    for key, ref in base.items():
        n_obj, seed = map(int, key.split("_"))
        env = SimEnv(n_object_balls=n_obj)
        obs = env.reset(setup="behind_cue", seed=seed)
        cur = {"robot": list(obs["robot_pose"]), "balls": [list(p) for p in obs["ball_positions"]],
               "setup": env.last_setup}
        mism += cur != ref
    check("aim=None reproduces pre-change layouts exactly", mism == 0, f"{len(base)} layouts, {mism} mismatches")
    # (c) the aim only changes the aim: with the same seed the cue/distance/offset draws match.
    a = SimEnv(n_object_balls=2); b = SimEnv(n_object_balls=2)
    a.reset(setup="behind_cue", seed=7)
    b.reset(setup="behind_cue", seed=7, aim=1.0)
    check("aim override keeps distance/offset draws aligned", a.last_setup == b.last_setup or
          (a.last_setup["distance"] == b.last_setup["distance"]
           and a.last_setup["aim_offset"] == b.last_setup["aim_offset"]),
          f"{a.last_setup} vs {b.last_setup}")


if __name__ == "__main__":
    test_regression()
    test_head_on()
    test_cushion_angle()
    test_render_felt_edge()
    test_pockets_all_six()
    test_pocket_once_only()
    test_scratch()
    test_respawn_legal()
    test_ball_slots()
    test_reset_aim()
    test_no_tunneling()
    test_layout_generator()
    test_settle_terminates()
    test_speed()
    print(f"\n{sum(_results)}/{len(_results)} checks passed.")
    sys.exit(0 if all(_results) else 1)
