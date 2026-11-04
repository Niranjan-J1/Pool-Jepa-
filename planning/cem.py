"""Cross-Entropy Method (CEM) planner over the learned latent world model.

The planner searches over short action sequences, imagines each one by rolling the
latent predictor forward, and keeps refining a Gaussian over sequences toward the
ones whose imagined final latent lands closest to a goal latent.

World-model interface (so a 5-model ensemble can replace the single predictor):
    model.predict(z, a) -> z_next     z: (N, D), a: (N, 4), returns (N, D)
    model.uncertainty(z, a) -> (N,)   optional; ensembles return member disagreement
`SinglePredictor` wraps one LatentPredictor; `EnsemblePredictor` wraps several and
returns their mean, plus disagreement the cost can penalize (UNCERTAINTY_WEIGHT).

Action encoding. The robot's action is (left_pwm, right_pwm, fire, fire_force), but
`fire` is a boolean and fire_force was 0 whenever the robot did not fire in training.
CEM samples all four dims as continuous Gaussians; before an action reaches the
predictor or the simulator, fire is thresholded at 0.5 and fire_force is zeroed when
not firing, so imagined actions match the training distribution.

Run the test (needs the trained spatial-softmax encoder + predictor in data/checkpoints):
    python planning/cem.py
    python planning/cem.py --mpc-steps 10 --horizon 5
    python planning/cem.py --untrained   # random weights: checks the plumbing only
    python planning/cem.py --checkpoint-dir path/to/ckpts --suffix ""   # final instead of best
"""
import argparse
import json
import math
import os
import sys
from dataclasses import dataclass

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from perception.encoder import LatentPredictor, VisualEncoder, load_predictor

ACTION_DIM = 4
# Per-dimension action bounds: left_pwm, right_pwm, fire (prob-like), fire_force
ACTION_LOW = torch.tensor([-1.0, -1.0, 0.0, 0.0])
ACTION_HIGH = torch.tensor([1.0, 1.0, 1.0, 1.0])
FIRE_THRESHOLD = 0.5


# --------------------------------------------------------------------------------------
# World models
# --------------------------------------------------------------------------------------

class SinglePredictor:
    """One LatentPredictor behind the planner's world-model interface."""

    def __init__(self, predictor: LatentPredictor):
        self.predictor = predictor.eval()

    @torch.no_grad()
    def predict(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.predictor(z, a)

    @torch.no_grad()
    def uncertainty(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return torch.zeros(z.shape[0], device=z.device)


class EnsemblePredictor:
    """Several LatentPredictors: predict() returns the member mean, uncertainty() the
    mean per-dim std across members (the disagreement signal for SQ4).

    Members run one after another here; for speed they can later be fused into a
    single batched (E, N, D) op without changing the planner.
    """

    def __init__(self, predictors: list[LatentPredictor]):
        self.predictors = [p.eval() for p in predictors]

    @torch.no_grad()
    def _all(self, z, a):
        return torch.stack([p(z, a) for p in self.predictors])  # (E, N, D)

    @torch.no_grad()
    def predict(self, z, a):
        return self._all(z, a).mean(0)

    @torch.no_grad()
    def uncertainty(self, z, a):
        return self._all(z, a).std(0).mean(-1)


# --------------------------------------------------------------------------------------
# CEM
# --------------------------------------------------------------------------------------

@dataclass
class CEMConfig:
    horizon: int = 5
    num_samples: int = 200
    num_elites: int = 20
    iterations: int = 3
    init_std: float = 0.5
    min_std: float = 0.05
    uncertainty_weight: float = 0.0  # > 0 penalizes ensemble disagreement along the rollout


def to_executable(actions: torch.Tensor) -> torch.Tensor:
    """Map raw continuous samples (..., 4) to actions the model/sim understands:
    clip to bounds, binarize fire, zero fire_force when not firing."""
    a = torch.max(torch.min(actions, ACTION_HIGH.to(actions)), ACTION_LOW.to(actions))
    fire = (a[..., 2] > FIRE_THRESHOLD).to(a.dtype)
    return torch.stack([a[..., 0], a[..., 1], fire, a[..., 3] * fire], dim=-1)


class CEMPlanner:
    """Plans action sequences by CEM through a latent world model.

    plan(z0, z_goal) -> (best first action (4,), info dict)

    Cost of a sequence = || z_H - z_goal ||_w  (+ uncertainty_weight * summed disagreement),
    where w are optional per-latent-dim weights (e.g. to ignore robot keypoints and
    score only the ones that track balls).
    """

    def __init__(self, model, config: CEMConfig = CEMConfig(), device="cpu", dim_weights=None,
                 seed: int | None = None):
        self.model = model
        self.cfg = config
        self.device = torch.device(device)
        self.dim_weights = None if dim_weights is None else torch.as_tensor(
            dim_weights, dtype=torch.float32, device=self.device)
        self.generator = torch.Generator(device=self.device)
        if seed is not None:
            self.generator.manual_seed(seed)
        self._warm_mean = None  # previous plan shifted by one step (MPC warm start)

    def reset(self):
        self._warm_mean = None

    @torch.no_grad()
    def rollout(self, z0: torch.Tensor, actions: torch.Tensor):
        """Batched imagination. z0: (D,), actions: (N, H, 4) executable.
        Returns final latents (N, D), per-step latents (N, H, D), summed uncertainty (N,)."""
        n, h, _ = actions.shape
        z = z0.unsqueeze(0).expand(n, -1)
        traj, unc = [], torch.zeros(n, device=self.device)
        for t in range(h):  # loop over time only; all N candidates step together
            a_t = actions[:, t]
            if self.cfg.uncertainty_weight > 0:
                unc = unc + self.model.uncertainty(z, a_t)
            z = self.model.predict(z, a_t)
            traj.append(z)
        return z, torch.stack(traj, 1), unc

    def cost(self, z_final, z_goal, unc):
        diff = z_final - z_goal.unsqueeze(0)
        if self.dim_weights is not None:
            diff = diff * self.dim_weights.sqrt()
        return diff.norm(dim=-1) + self.cfg.uncertainty_weight * unc

    @torch.no_grad()
    def plan(self, z0: torch.Tensor, z_goal: torch.Tensor):
        cfg, dev = self.cfg, self.device
        z0, z_goal = z0.to(dev).flatten(), z_goal.to(dev).flatten()
        low, high = ACTION_LOW.to(dev), ACTION_HIGH.to(dev)

        mean = torch.zeros(cfg.horizon, ACTION_DIM, device=dev)
        mean[:, 3] = 0.65  # mid-range strike force; fire starts at 0 (= 50/50 after noise)
        if self._warm_mean is not None:
            mean = self._warm_mean.clone()
        std = torch.full((cfg.horizon, ACTION_DIM), cfg.init_std, device=dev)

        best_cost, best_seq = math.inf, None
        history = []
        for it in range(cfg.iterations):
            noise = torch.randn(cfg.num_samples, cfg.horizon, ACTION_DIM,
                                device=dev, generator=self.generator)
            raw = torch.max(torch.min(mean + std * noise, high), low)
            if it == 0 and self._warm_mean is not None:
                raw[0] = mean  # keep the previous plan as a candidate
            actions = to_executable(raw)

            z_final, _, unc = self.rollout(z0, actions)
            costs = self.cost(z_final, z_goal, unc)

            elite_costs, elite_idx = costs.topk(cfg.num_elites, largest=False)
            elites = raw[elite_idx]
            mean = elites.mean(0)
            std = elites.std(0).clamp_min(cfg.min_std)

            if elite_costs[0].item() < best_cost:
                best_cost = elite_costs[0].item()
                best_seq = actions[elite_idx[0]].clone()
            history.append({"iter": it, "best_cost": elite_costs[0].item(),
                            "elite_mean_cost": elite_costs.mean().item()})

        # Warm start for the next call: drop the executed step, repeat the last one.
        self._warm_mean = torch.cat([mean[1:], mean[-1:]], 0)

        z_pred_final, z_traj, _ = self.rollout(z0, best_seq.unsqueeze(0))
        info = {"best_sequence": best_seq.cpu(), "best_cost": best_cost, "history": history,
                "predicted_next_latent": z_traj[0, 0].cpu(),
                "predicted_final_latent": z_pred_final[0].cpu()}
        return best_seq[0].cpu(), info


# --------------------------------------------------------------------------------------
# Encoding frames (must match experiments/train_encoder.py preprocessing)
# --------------------------------------------------------------------------------------

def frames_to_input(prev_frame: np.ndarray, cur_frame: np.ndarray, background: np.ndarray,
                    device) -> torch.Tensor:
    """Two (H, W, 3) uint8 frames -> (1, 6, H, W) background-subtracted input in [-1, 1]."""
    bg = background.astype(np.float32)
    x = np.concatenate([prev_frame.astype(np.float32) - bg,
                        cur_frame.astype(np.float32) - bg], axis=-1) / 255.0
    return torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(device)


@torch.no_grad()
def encode(encoder, prev_frame, cur_frame, background, device) -> torch.Tensor:
    return encoder(frames_to_input(prev_frame, cur_frame, background, device))[0]


# --------------------------------------------------------------------------------------
# Test: plan a cue-ball shot in SimEnv
# --------------------------------------------------------------------------------------

REPO = os.path.join(os.path.dirname(__file__), "..")
CHECKPOINT_DIR = os.path.join(REPO, "data", "checkpoints")
BACKGROUND_PATH = os.path.join(REPO, "data", "datasets", "frames_background.npy")
LATENT_DIM = 64


def make_spatial_encoder(device):
    from env.sim_env import RENDER_HEIGHT, RENDER_WIDTH
    return VisualEncoder(latent_dim=LATENT_DIM, in_channels=6, use_simnorm=False,
                         input_hw=(RENDER_HEIGHT, RENDER_WIDTH), head="spatial_softmax").to(device)


def load_models(device, untrained=False, checkpoint_dir=CHECKPOINT_DIR, suffix="_best",
                predictor_name=None):
    """Returns (state_encoder, goal_encoder, predictor).

    Joint-trained predictor (predictor{suffix}.pt, from train_encoder.py):
        state_encoder = online encoder (the predictor's INPUT space in training)
        goal_encoder  = EMA target encoder (its OUTPUT space), for goals / actual latents.
        Older checkpoints have no target encoder; the online one is used with a warning.
    Residual predictor (predictor_residual{suffix}.pt, from train_predictor.py):
        trained on one frozen encoder (named in its .json), used for both.
    predictor_name=None picks predictor_residual{suffix}.pt if it exists, else predictor{suffix}.pt.
    """
    encoder = make_spatial_encoder(device)
    predictor = LatentPredictor(latent_dim=LATENT_DIM, action_dim=ACTION_DIM, hidden_dim=128).to(device)
    if untrained:
        print("WARNING: --untrained, using random weights (checks plumbing, not planning quality)")
        return encoder.eval(), encoder.eval(), predictor.eval()

    def load(module, name):
        path = os.path.join(checkpoint_dir, name)
        try:
            module.load_state_dict(torch.load(path, map_location=device))
        except (FileNotFoundError, RuntimeError) as e:
            raise SystemExit(
                f"Could not load {path} into the spatial-softmax architecture:\n  {e}\n"
                "Train it with experiments/train_encoder.py (--head spatial_softmax), "
                "or pass --untrained to check the plumbing.") from None

    if predictor_name is None:
        res = f"predictor_residual{suffix}.pt"
        predictor_name = res if os.path.exists(os.path.join(checkpoint_dir, res)) else f"predictor{suffix}.pt"
    pred_path = os.path.join(checkpoint_dir, predictor_name)
    try:
        predictor = load_predictor(pred_path, device, LATENT_DIM, ACTION_DIM, 128)
    except (FileNotFoundError, RuntimeError) as e:
        raise SystemExit(f"Could not load predictor {pred_path}:\n  {e}") from None

    if predictor.residual:
        sidecar = pred_path.replace(".pt", ".json")
        enc_file = json.load(open(sidecar))["encoder_file"] if os.path.exists(sidecar) \
            else f"target_encoder{suffix}.pt"
        enc = make_spatial_encoder(device)
        load(enc, enc_file)
        print(f"Loaded residual predictor {predictor_name} + its frozen encoder {enc_file} "
              f"from {checkpoint_dir}")
        return enc.eval(), enc.eval(), predictor

    load(encoder, f"encoder{suffix}.pt")
    target_name = f"target_encoder{suffix}.pt"
    if os.path.exists(os.path.join(checkpoint_dir, target_name)):
        goal_encoder = make_spatial_encoder(device)
        load(goal_encoder, target_name)
        print(f"Loaded encoder, target_encoder, {predictor_name} from {checkpoint_dir}")
    else:
        goal_encoder = encoder
        print(f"WARNING: {target_name} not found (checkpoint predates target-encoder saving). "
              "Using the online encoder for goals; the predictor was trained against the EMA "
              "target, so predicted-vs-actual errors are inflated. Retrain to fix.")
    return encoder.eval(), goal_encoder.eval(), predictor.eval()


# --------------------------------------------------------------------------------------
# Cue-ball keypoints: which latent dims track the cue ball, and goals in those dims
# --------------------------------------------------------------------------------------

@dataclass
class CueKeypoints:
    """Which spatial-softmax keypoints track the cue ball (not the robot), plus
    per-keypoint affine maps cue (x, y) -> keypoint (x, y) for building goals and a
    linear readout latent -> cue (x, y) for reporting predicted positions in meters."""
    r2_cue: np.ndarray        # (K,) per-keypoint R^2 of cue position -> keypoint
    r2_robot: np.ndarray      # (K,) same for robot position
    selected: np.ndarray      # (K,) bool
    affine: np.ndarray        # (K, 3, 2): [cue_x, cue_y, 1] @ affine[k] -> keypoint k (x, y)
    readout: np.ndarray       # (2*K_sel + 1, 2): [selected dims, 1] @ readout -> cue (x, y)
    readout_r2: float         # held-out R^2 of the readout on calibration scenes

    @property
    def dim_weights(self) -> np.ndarray:
        """(2K,) cost weights: R^2_cue on both dims of each selected keypoint, 0 elsewhere."""
        w = np.where(self.selected, self.r2_cue, 0.0)
        return np.repeat(w, 2).astype(np.float32)

    def goal_latent(self, cue_xy) -> torch.Tensor:
        """Latent whose selected keypoints sit where they would for a cue ball at cue_xy.
        Unselected dims are 0 and carry zero cost weight."""
        x = np.array([cue_xy[0], cue_xy[1], 1.0])
        kp = np.einsum("i,kij->kj", x, self.affine)          # (K, 2)
        kp[~self.selected] = 0.0
        return torch.from_numpy(kp.reshape(-1).astype(np.float32))

    def decode_cue(self, z: torch.Tensor) -> np.ndarray:
        """Linear readout of the cue position (m) from a latent (D,)."""
        zs = z.detach().cpu().numpy().reshape(-1, 2)[self.selected].reshape(-1)
        return np.append(zs, 1.0) @ self.readout


def _affine_r2(X, Y):
    """Least-squares fit Y ~ [X, 1] @ A; returns (A, R^2)."""
    Xa = np.concatenate([X, np.ones((len(X), 1))], 1)
    A, *_ = np.linalg.lstsq(Xa, Y, rcond=None)
    ss_res = ((Xa @ A - Y) ** 2).sum()
    ss_tot = ((Y - Y.mean(0)) ** 2).sum() + 1e-12
    return A, 1.0 - ss_res / ss_tot


@torch.no_grad()
def calibrate_cue_keypoints(goal_encoder, env, background, device, n_scenes=400, min_r2=0.3,
                            seed=0) -> CueKeypoints:
    """Render scenes with the cue ball placed independently of the robot, encode them, and
    find keypoints whose (x, y) follow the cue ball more than the robot.

    In behind_cue episodes the cue ball starts right in front of the robot, so cue and robot
    positions are correlated; teleporting the cue ball to a uniform random spot breaks that,
    so a robot-tracking keypoint can't pass as a cue keypoint.
    """
    from env.sim_env import BALL_RADIUS, CUSHION_THICKNESS, TABLE_LENGTH, TABLE_WIDTH
    rng = np.random.default_rng(seed)
    lim = np.array([TABLE_LENGTH / 2, TABLE_WIDTH / 2]) - CUSHION_THICKNESS - BALL_RADIUS - 0.005
    Z, cue, robot = [], [], []
    while len(Z) < n_scenes:
        obs = env.reset(setup="behind_cue")
        rx, ry, th = obs["robot_pose"]
        p = rng.uniform(-lim, lim)
        if env._overlaps_robot(tuple(p), (rx, ry), th, BALL_RADIUS):
            continue
        frame = render_with_cue_at(env, tuple(p))
        Z.append(encode(goal_encoder, frame, frame, background, device).cpu().numpy())
        cue.append(p)
        robot.append((rx, ry))
    Z, cue, robot = np.array(Z), np.array(cue), np.array(robot)
    kps = Z.reshape(len(Z), -1, 2)
    n_kp = kps.shape[1]

    r2_cue, r2_robot, affine = np.zeros(n_kp), np.zeros(n_kp), np.zeros((n_kp, 3, 2))
    for k in range(n_kp):
        affine[k], r2_cue[k] = _affine_r2(cue, kps[:, k])
        _, r2_robot[k] = _affine_r2(robot, kps[:, k])
    selected = (r2_cue >= min_r2) & (r2_cue > r2_robot)

    readout, readout_r2 = np.zeros((1, 2)), float("nan")
    if selected.any():
        feats = kps[:, selected].reshape(len(Z), -1)
        n_tr = int(0.8 * len(Z))
        A, _ = _affine_r2(feats[:n_tr], cue[:n_tr])
        pred = np.concatenate([feats[n_tr:], np.ones((len(Z) - n_tr, 1))], 1) @ A
        readout_r2 = 1 - ((pred - cue[n_tr:]) ** 2).sum() / ((cue[n_tr:] - cue[:n_tr].mean(0)) ** 2).sum()
        readout, _ = _affine_r2(feats, cue)  # refit on everything for use
    return CueKeypoints(r2_cue, r2_robot, selected, affine, readout, float(readout_r2))


def load_background(env, path=BACKGROUND_PATH, n=200):
    """Training-time background if available; otherwise a median over random scenes."""
    if os.path.exists(path):
        return np.load(path)
    print(f"{path} not found; estimating background from {n} random scenes")
    frames = []
    for _ in range(n):
        env.reset(setup="behind_cue")
        frames.append(env.render())
    return np.median(np.stack(frames), axis=0).astype(np.float32)


def render_with_cue_at(env, pos):
    """Render the current scene with the cue ball moved to `pos` (goal image), then restore."""
    cue = env._ball_bodies["cue"]
    saved = cue.position
    cue.position = pos
    frame = env.render()
    cue.position = saved
    return frame


def run_test(args):
    from env.sim_env import SimEnv

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    import random
    random.seed(args.seed)

    encoder, goal_encoder, predictor = load_models(
        device, untrained=args.untrained, checkpoint_dir=args.checkpoint_dir, suffix=args.suffix,
        predictor_name=args.predictor)
    env = SimEnv()
    background = load_background(env, args.background)

    # 1. Which keypoints track the cue ball? (goal-encoder space = predictor output space)
    print(f"\nCalibrating cue keypoints on {args.calib_scenes} scenes (cue placed independently of robot)...")
    cue_kp = calibrate_cue_keypoints(goal_encoder, env, background, device,
                                     n_scenes=args.calib_scenes, seed=args.seed)
    order = np.argsort(-cue_kp.r2_cue)
    print(f"{cue_kp.selected.sum()} of {len(cue_kp.selected)} keypoints selected as cue-tracking "
          f"(R^2_cue >= 0.3 and > R^2_robot). Top 8 by R^2_cue:")
    print("  kp   R2_cue  R2_robot  selected")
    for k in order[:8]:
        print(f"  {k:2d}   {cue_kp.r2_cue[k]:6.3f}  {cue_kp.r2_robot[k]:8.3f}  {'yes' if cue_kp.selected[k] else 'no'}")
    print(f"Cue-position readout from selected keypoints: held-out R^2 = {cue_kp.readout_r2:.3f}")
    if not cue_kp.selected.any():
        raise SystemExit("No keypoint tracks the cue ball; the planner has nothing to aim at.")

    # 2. Scene: robot behind the cue ball; one no-op step gives the encoder its previous frame.
    obs = env.reset(setup="behind_cue")
    prev = obs["frame"]
    obs = env.step(dict(left_pwm=0.0, right_pwm=0.0, fire=False, fire_force=0.0))
    cur = obs["frame"]

    # 3. Goal: a cue-ball POSITION, mapped into the cue keypoints only (robot dims weight 0).
    from env.sim_env import BALL_RADIUS, CUSHION_THICKNESS, TABLE_LENGTH, TABLE_WIDTH
    x, y, theta = obs["robot_pose"]
    cue0 = np.array(obs["ball_positions"][0])
    lim = np.array([TABLE_LENGTH / 2, TABLE_WIDTH / 2]) - CUSHION_THICKNESS - BALL_RADIUS - 0.01
    target = np.clip(cue0 + args.target_dist * np.array([math.cos(theta), math.sin(theta)]), -lim, lim)
    z_goal = cue_kp.goal_latent(target).to(device)

    model = SinglePredictor(predictor)  # swap for EnsemblePredictor([...]) later
    planner = CEMPlanner(model, CEMConfig(horizon=args.horizon), device=device,
                         dim_weights=cue_kp.dim_weights, seed=args.seed)
    w = planner.dim_weights

    def goal_dist(z):  # same weighted distance the planner minimizes
        return float(((z.to(device) - z_goal) * w.sqrt()).norm())

    print(f"\nCue ball at ({cue0[0]:+.3f}, {cue0[1]:+.3f}) m -> target ({target[0]:+.3f}, {target[1]:+.3f}) m "
          f"[{100 * np.linalg.norm(target - cue0):.1f} cm away]")
    print(f"Planner: {planner.cfg.num_samples} samples x {planner.cfg.horizon} steps, "
          f"{planner.cfg.num_elites} elites, {planner.cfg.iterations} iterations, device {device}\n")

    for step in range(args.mpc_steps):
        z_t = encode(encoder, prev, cur, background, device)          # predictor input space
        action, info = planner.plan(z_t, z_goal)
        act = dict(left_pwm=float(action[0]), right_pwm=float(action[1]),
                   fire=bool(action[2] > 0.5), fire_force=float(action[3]))

        prev = cur
        obs = env.step(act)
        cur = obs["frame"]
        z_actual = encode(goal_encoder, prev, cur, background, device)  # predictor output space
        z_pred = info["predicted_next_latent"].to(device)
        cue = np.array(obs["ball_positions"][0])
        cue_pred_next = cue_kp.decode_cue(z_pred)
        cue_pred_final = cue_kp.decode_cue(info["predicted_final_latent"])

        it = info["history"]
        print(f"[step {step}] action: L={act['left_pwm']:+.2f} R={act['right_pwm']:+.2f} "
              f"fire={act['fire']!s:5} force={act['fire_force']:.2f} | "
              f"CEM best cost {it[0]['best_cost']:.3f} -> {it[-1]['best_cost']:.3f}")
        pred_err = float(((z_pred - z_actual) * w.sqrt()).norm())
        print(f"          cue-keypoint latent: predicted-vs-actual error {pred_err:.3f} "
              f"| dist to goal: predicted {goal_dist(z_pred):.3f}, actual {goal_dist(z_actual):.3f}")
        print(f"          cue ball: predicted next ({cue_pred_next[0]:+.3f}, {cue_pred_next[1]:+.3f}), "
              f"actual ({cue[0]:+.3f}, {cue[1]:+.3f}) m, {100 * np.linalg.norm(cue - target):.1f} cm from target; "
              f"plan's imagined end ({cue_pred_final[0]:+.3f}, {cue_pred_final[1]:+.3f})")

    for _ in range(args.settle_steps):
        obs = env.step(dict(left_pwm=0.0, right_pwm=0.0, fire=False, fire_force=0.0))
    cue = np.array(obs["ball_positions"][0])
    print(f"\nAfter {args.settle_steps} settle steps: cue ball at ({cue[0]:+.3f}, {cue[1]:+.3f}) m, "
          f"{100 * np.linalg.norm(cue - target):.1f} cm from target "
          f"(started {100 * np.linalg.norm(cue0 - target):.1f} cm away)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CEM planner test in SimEnv")
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--mpc-steps", type=int, default=3,
                        help="Plan-execute-replan steps (1 = plan once, execute the first action)")
    parser.add_argument("--settle-steps", type=int, default=30)
    parser.add_argument("--target-dist", type=float, default=0.10, help="m along robot heading")
    parser.add_argument("--calib-scenes", type=int, default=400)
    parser.add_argument("--checkpoint-dir", default=CHECKPOINT_DIR)
    parser.add_argument("--suffix", default="_best", help="checkpoint suffix: _best, '' (final), _epoch10, ...")
    parser.add_argument("--predictor", default=None,
                        help="predictor file in --checkpoint-dir (default: predictor_residual{suffix}.pt "
                             "if present, else predictor{suffix}.pt)")
    parser.add_argument("--background", default=BACKGROUND_PATH,
                        help="training background image (frames_background.npy next to the frame cache)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--untrained", action="store_true")
    run_test(parser.parse_args())
