"""Full autonomous shot: Stage 1 aims the robot, Stage 2 picks the force, then it fires.

    python planning/full_shot_test.py --trials 100

Targets (straight-line, reachable). Each trial starts from a behind_cue scene (robot near the
cue ball, pointing anywhere within +/-45 deg of it). A random shot direction phi is drawn; the
robot is teleported to the exact firing pose for phi and a secret force f* is fired. The cue
ball's rest point becomes the target only if
  - the firing pose leaves the robot room to turn (aim_controller.firing_pose_reachable),
  - the strike connects, the cue ball rolls out along phi without touching a cushion or another
    ball (lateral deviation < 1 mm, never moving backwards, no other ball moving), and it
    travels at least MIN_TRAVEL;
otherwise another phi / f* is drawn. So "cue -> target" is the true aim line, every target is
reachable by a straight shot, and the scene the robot starts from is the unmodified one.
On this 66 x 33 cm table hard shots nearly always reach a cushion, so straight targets are
short (100 seed-0 scenes: median 6.6 cm, 90th percentile 32 cm, 20% under 3 cm). The results
are therefore also reported for targets >= 5 cm away, where "< 3 cm" is not trivially met.

Each trial is replayed from the same scene (the simulator is deterministic) under:
  stage 2 only    robot teleported to the exact firing pose (perfect aim), force from the planner
  fixed <f>       perfect aim, mid-range force (baseline)
  full loop       Stage 1 from the un-aimed start using the DECODED cue position (cue keypoints
                  of the frozen encoder) + sim robot pose, then Stage 2, then fire
  full (true cue) same, but Stage 1 sees the true cue position: separates decoding error from
                  control error (diagnostic)
Stage 2 always gets z_pre from two rest frames and the robot heading, exactly as in training.
"""
import argparse
import math
import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from models.shot_model import load_shot_model
from planning import cem
from planning.aim_controller import AimController, firing_pose, firing_pose_reachable, wrap
from planning.shot_planner import IDLE, ShotPlanner, force_grid

MIN_TRAVEL = 0.01          # m; shorter targets make the aim line ill-defined
LATERAL_TOL = 0.001        # m, max sideways deviation of a "straight" roll-out
MAX_SETTLE = 900           # steps (15 s)
STAGE1_MAX_STEPS = 900     # 15 s to get aimed
DECODE_EVERY = 6           # control steps between cue-position re-decodes (60/6 = 10 Hz)
BALL_MOVED = 0.001         # m; a ball displaced more than this before the shot was disturbed


def teleport_robot(env, pose):
    rb = env._robot_body
    rb.position = (pose[0], pose[1])
    rb.angle = pose[2]
    rb.velocity, rb.angular_velocity = (0.0, 0.0), 0.0
    env._space.reindex_shapes_for_body(rb)


def fire_and_track(env, force, line=None):
    """Fire, step until every ball stops. With line=(cue0, phi), also check the roll-out is a
    straight, unobstructed run along phi. Returns (cue rest xy, connected, straight); with a line
    check, rest is None as soon as the roll-out leaves the line (no need to finish it)."""
    balls0 = np.array(env._get_observation(render=False)["ball_positions"])
    env.step(dict(IDLE, fire=True, fire_force=force), render=False)
    connected = env.last_strike_connected
    straight = connected
    if line is not None and not connected:
        return None, connected, False
    if line is not None:
        cue0, phi = line
        u = np.array([math.cos(phi), math.sin(phi)])
        n = np.array([-u[1], u[0]])
        last = 0.0
    for _ in range(MAX_SETTLE):
        obs = env.step(IDLE, render=False)
        if line is not None and straight:
            b = np.array(obs["ball_positions"])
            d = b[0] - cue0
            along = float(d @ u)
            if abs(float(d @ n)) > LATERAL_TOL or along < last - 1e-4 or \
                    (np.linalg.norm(b[1:] - balls0[1:], axis=1) > 1e-4).any():
                return None, connected, False   # not a straight shot: no need to finish it
            last = max(last, along)
        if obs["ball_settled"]:
            break
    rest = np.array(env._get_observation(render=False)["ball_positions"][0])
    if line is not None and np.linalg.norm(rest - line[0]) < MIN_TRAVEL:
        straight = False
    return rest, connected, straight


def sample_target(env, scene, rng, f_lo, f_hi, force_tries=10, dir_tries=36):
    """A straight-line reachable target for this scene, or None.

    The force is drawn first and then directions are searched for it. (Drawing a direction and
    force together and keeping the first straight shot would favour soft shots: on this narrow
    table most hard shots reach a cushion, so targets would cluster a few cm from the cue.)"""
    random.setstate(scene)
    obs = env.reset(setup="behind_cue")
    balls = np.array(obs["ball_positions"])
    cue0 = balls[0]
    for _ in range(force_tries):
        f_star = float(rng.uniform(f_lo, f_hi))
        for _ in range(dir_tries):
            phi = float(rng.uniform(-math.pi, math.pi))
            aim_pt = cue0 + np.array([math.cos(phi), math.sin(phi)])
            if not firing_pose_reachable(cue0, aim_pt, balls[1:]):
                continue
            random.setstate(scene)
            env.reset(setup="behind_cue")
            pose = firing_pose(cue0, aim_pt)
            teleport_robot(env, pose)
            rest, connected, straight = fire_and_track(env, f_star, line=(cue0, phi))
            if straight:
                return dict(target=rest, phi=phi, pose=pose, f_star=f_star, cue0=cue0)
    return None


class Stage2:
    """Encoder + shot model + force planner, as in shot_planner.py."""

    def __init__(self, args, env, device):
        shot_path = os.path.join(args.checkpoint_dir, f"shot_model{args.suffix}.pt")
        if not os.path.exists(shot_path):
            raise SystemExit(f"{shot_path} not found: train it with experiments/train_shot_model.py "
                             "or point --checkpoint-dir at the folder that has it.")
        self.model, meta = load_shot_model(shot_path, device)
        enc_file = meta.get("encoder_file", f"encoder{args.suffix}.pt")
        self.encoder = cem.make_spatial_encoder(device)
        self.encoder.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, enc_file),
                                                map_location=device))
        self.encoder.eval()
        self.background = cem.load_background(env, args.background)
        self.kp = cem.calibrate_cue_keypoints(self.encoder, env, self.background, device,
                                              n_scenes=args.calib_scenes, seed=args.seed,
                                              near_robot_fraction=getattr(args, "calib_near_robot", 0.0))
        self.f_lo, self.f_hi = meta.get("force_range", [0.05, 1.0])
        self.planner = ShotPlanner(self.model, self.kp, device, forces=force_grid(self.f_lo, self.f_hi))
        self.device = device
        print(f"Loaded {os.path.basename(shot_path)} + frozen encoder {enc_file} "
              f"(heading input: {'yes' if self.model.use_heading else 'no'}); "
              f"{self.kp.selected.sum()} cue keypoints, readout R^2 {self.kp.readout_r2:.3f}, "
              f"held-out error {self.kp.err_far_cm:.2f} cm robot far / {self.kp.err_near_cm:.2f} cm robot near")

    def decode_cue(self, frame):
        # Same (frame, frame) pair the keypoint readout was calibrated on.
        return self.kp.decode_cue(cem.encode(self.encoder, frame, frame, self.background, self.device))

    def choose_force(self, env, target):
        """Robot at rest: two rest frames -> z_pre -> force. Returns (force, cue decoded at hand-off)."""
        f_prev = env.render()
        obs = env.step(IDLE, render=True)
        z_pre = cem.encode(self.encoder, f_prev, obs["frame"], self.background, self.device)
        heading = obs["robot_pose"][2] if self.model.use_heading else None
        force, _ = self.planner.plan(z_pre, target, heading)
        return force, self.kp.decode_cue(z_pre)


def run_stage1(env, ctl, target, cue_source, other_balls_fn=None):
    """Drive until aimed. cue_source(obs, step) -> cue xy estimate. Returns stats dict."""
    ctl.reset()
    obs = env._get_observation(render=False)
    cue_est = None
    done, n = False, 0
    for n in range(STAGE1_MAX_STEPS):
        if n % DECODE_EVERY == 0 or cue_est is None:
            cue_est = cue_source(obs)
        others = other_balls_fn(obs) if other_balls_fn else ()
        left, right, done, info = ctl.step(obs["robot_pose"], cue_est, target, others)
        if done:
            break
        obs = env.step(dict(IDLE, left_pwm=left, right_pwm=right), render=False)
    # let anything the robot nudged come to rest before the shot
    for _ in range(MAX_SETTLE):
        if env._balls_settled():
            break
        env.step(IDLE, render=False)
    return {"aligned": done, "steps": n}


def run_test(args):
    from env.sim_env import SimEnv

    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = SimEnv()
    s2 = Stage2(args, env, device)
    f_mid = round((s2.f_lo + s2.f_hi) / 2, 3)
    ctl = AimController()

    random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    names = ["stage 2 only", f"fixed {f_mid}", "full loop", "full (true cue)"]
    miss = {k: [] for k in names}
    st = {k: [] for k in ("travel", "aligned", "steps", "head_err", "cue_err", "disturbed",
                          "connected", "aligned_t", "head_err_t", "disturbed_t")}
    skipped = 0
    trial = 0
    while trial < args.trials:
        scene = random.getstate()
        random.random()  # advance so the next attempt gets a new scene
        next_scene = random.getstate()
        t = sample_target(env, scene, rng, s2.f_lo, s2.f_hi)
        if t is None:
            skipped += 1
            random.setstate(next_scene)
            continue
        target, phi = t["target"], t["phi"]
        st["travel"].append(100 * np.linalg.norm(target - t["cue0"]))

        # Perfect aim: Stage 2 only, and the fixed-force baseline
        for name in ("stage 2 only", f"fixed {f_mid}"):
            random.setstate(scene)
            env.reset(setup="behind_cue")
            teleport_robot(env, t["pose"])
            force = s2.choose_force(env, target)[0] if name == "stage 2 only" else f_mid
            rest, _, _ = fire_and_track(env, force)
            miss[name].append(100 * np.linalg.norm(rest - target))

        # Full loop: decoded cue (and, as a diagnostic, the true cue)
        for name, source, sfx in (("full loop", lambda o: s2.decode_cue(env.render()), ""),
                                  ("full (true cue)", lambda o: np.array(o["ball_positions"][0]), "_t")):
            random.setstate(scene)
            obs0 = env.reset(setup="behind_cue")
            balls0 = np.array(obs0["ball_positions"])
            others = (lambda o: np.array(o["ball_positions"][1:])) if args.avoid_balls else None
            r1 = run_stage1(env, ctl, target, source, others)
            obs = env._get_observation(render=False)
            balls = np.array(obs["ball_positions"])
            true_phi = math.atan2(*(target - balls[0])[::-1])
            force, cue_dec = s2.choose_force(env, target)
            rest, connected, _ = fire_and_track(env, force)
            miss[name].append(100 * np.linalg.norm(rest - target))
            st["aligned" + sfx].append(r1["aligned"])
            st["head_err" + sfx].append(abs(math.degrees(wrap(true_phi - obs["robot_pose"][2]))))
            st["disturbed" + sfx].append(bool((np.linalg.norm(balls - balls0, axis=1) > BALL_MOVED).any()))
            if not sfx:
                st["steps"].append(r1["steps"])
                st["cue_err"].append(100 * np.linalg.norm(cue_dec - balls[0]))
                st["connected"].append(connected)

        if trial < args.show:
            print(f"[{trial:2d}] target {st['travel'][-1]:5.1f} cm | aligned {st['aligned'][-1]!s:5} in "
                  f"{st['steps'][-1]:3d} steps, heading err {st['head_err'][-1]:.2f} deg, cue decode err "
                  f"{st['cue_err'][-1]:.2f} cm | miss: stage2 {miss['stage 2 only'][-1]:4.1f}, "
                  f"full {miss['full loop'][-1]:4.1f}, full(true cue) {miss['full (true cue)'][-1]:4.1f} cm")
        trial += 1
        random.setstate(next_scene)

    a = {k: np.array(v) for k, v in st.items()}
    print(f"\n========== {args.trials} straight-line targets (rest point of a secret force fired from "
          f"the exact firing pose) ==========")
    print(f"Scenes skipped (no straight reachable target found): {skipped}")
    print(f"Target distance from cue start: mean {a['travel'].mean():.1f} cm, median {np.median(a['travel']):.1f} cm")
    far = a["travel"] >= 5.0
    for title, mask in (("all targets", np.ones_like(far)), (f"targets >= 5 cm from the cue ({far.sum()})", far)):
        if not mask.any():
            continue
        print(f"\n{title}:")
        print(f"{'strategy':>16} {'mean miss cm':>13} {'median':>8} {'<3 cm':>7} {'<5 cm':>7}")
        for name, m in miss.items():
            m = np.array(m)[mask]
            print(f"{name:>16} {m.mean():>13.1f} {np.median(m):>8.1f} {100 * (m < 3).mean():>6.0f}% "
                  f"{100 * (m < 5).mean():>6.0f}%")
    print("\nStage 1 (decoded cue):")
    print(f"  aligned within budget  {100 * a['aligned'].mean():.0f}%   (true cue: {100 * a['aligned_t'].mean():.0f}%)")
    print(f"  steps to align         median {np.median(a['steps']):.0f} ({np.median(a['steps']) / 60:.1f} s), "
          f"max {a['steps'].max()}")
    print(f"  heading err vs true line  mean {a['head_err'].mean():.2f} deg, median {np.median(a['head_err']):.2f}, "
          f"max {a['head_err'].max():.2f}   (true cue: mean {a['head_err_t'].mean():.2f} deg)")
    print(f"  cue decode err at hand-off  mean {a['cue_err'].mean():.2f} cm, max {a['cue_err'].max():.2f} cm")
    print(f"  any ball disturbed before the shot  {100 * a['disturbed'].mean():.0f}%   "
          f"(true cue: {100 * a['disturbed_t'].mean():.0f}%)")
    print(f"  strike connected       {100 * a['connected'].mean():.0f}%")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Full autonomous shot: Stage 1 aim + Stage 2 force")
    p.add_argument("--checkpoint-dir", default=cem.CHECKPOINT_DIR)
    p.add_argument("--suffix", default="_best")
    p.add_argument("--background", default=cem.BACKGROUND_PATH)
    p.add_argument("--trials", type=int, default=100)
    p.add_argument("--show", type=int, default=10, help="print the first N trials")
    p.add_argument("--calib-scenes", type=int, default=400)
    p.add_argument("--calib-near-robot", type=float, default=0.0,
                   help="extra share of calibration scenes with the robot right behind the cue ball "
                        "(fits the cue readout where Stage 1/2 use it; 0 = original calibration)")
    p.add_argument("--avoid-balls", action="store_true",
                   help="give Stage 1 the true positions of the other balls to path around")
    p.add_argument("--seed", type=int, default=0)
    run_test(p.parse_args())
