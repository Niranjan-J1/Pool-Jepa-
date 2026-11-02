"""Dynamics probe: train the latent-predictor architecture on ground-truth state.

Tests whether the dynamics model can learn the simulator's physics when given
perfect state (robot pose + ball positions) instead of learned latents. This
isolates predictor capacity from encoder quality.

State (11): robot (x, y, theta) + cue/red/blue/yellow (x, y).
Action (4): left_pwm, right_pwm, fire, fire_force.

Usage:
    python experiments/train_dynamics_probe.py                  # positions only (11-d state)
    python experiments/train_dynamics_probe.py --with-velocity  # also feed s_t - s_{t-1}
"""
import argparse
import glob
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

DATASETS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "datasets")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "figures")
DATA_PATTERN = "batch_v3_*.npz"

EPOCHS = 50
BATCH_SIZE = 256
LR = 3e-4
HIDDEN = 128
HOLDOUT_FRACTION = 0.02
EARLY_STOP_PATIENCE = 5
EARLY_STOP_MIN_DELTA = 1e-6
HORIZONS = [1, 3, 5, 10]
SEED = 0
STATE_DIM = 11
ACTION_DIM = 4
THETA = 2  # index of robot heading in the state vector
BALL_NAMES = ["cue", "red", "blue", "yellow"]
MOVE_THRESHOLD = 1e-3  # m; a ball "moved" in a transition if it travelled more than this


def wrap_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def wrap_angle_t(a):
    return torch.remainder(a + np.pi, 2 * np.pi) - np.pi


def load_data():
    files = sorted(glob.glob(os.path.join(DATASETS_DIR, DATA_PATTERN)))
    if not files:
        raise FileNotFoundError(f"No {DATA_PATTERN} files in {DATASETS_DIR}")
    keys = ["robot_pose_before", "robot_pose_after", "ball_positions_before",
            "ball_positions_after", "actions_left_pwm", "actions_right_pwm",
            "actions_fire", "actions_fire_force", "episode_idx", "step_idx"]
    parts = {k: [] for k in keys}
    for f in files:
        d = np.load(f)  # npz is lazy: frames are never read
        for k in keys:
            parts[k].append(d[k])
    d = {k: np.concatenate(v) for k, v in parts.items()}

    n = len(d["episode_idx"])
    s0 = np.concatenate([d["robot_pose_before"], d["ball_positions_before"].reshape(n, -1)], 1)
    s1 = np.concatenate([d["robot_pose_after"], d["ball_positions_after"].reshape(n, -1)], 1)
    s0[:, THETA] = wrap_angle(s0[:, THETA])
    s1[:, THETA] = wrap_angle(s1[:, THETA])
    actions = np.stack([d["actions_left_pwm"], d["actions_right_pwm"],
                        d["actions_fire"].astype(np.float32), d["actions_fire_force"]], 1)

    # Sort so each episode's steps are contiguous and in order.
    order = np.lexsort((d["step_idx"], d["episode_idx"]))
    print(f"Loaded {n} transitions from {len(files)} files")
    return (s0[order].astype(np.float32), s1[order].astype(np.float32),
            actions[order].astype(np.float32), d["episode_idx"][order], d["step_idx"][order])


def state_delta(s_next, s):
    d = s_next - s
    d[..., THETA] = wrap_angle(d[..., THETA]) if isinstance(d, np.ndarray) else wrap_angle_t(d[..., THETA])
    return d


def prev_delta(s0, step_idx):
    """Finite-difference velocity s_t - s_{t-1}; zero at the first step of an episode."""
    v = np.zeros_like(s0)
    v[1:] = state_delta(s0[1:], s0[:-1])
    v[step_idx == 0] = 0.0
    return v


def report_data_stats(s0, s1, actions):
    ball_moves = np.linalg.norm((s1[:, 3:] - s0[:, 3:]).reshape(-1, 4, 2), axis=-1)
    fire = actions[:, 2] > 0.5
    any_ball = (ball_moves > MOVE_THRESHOLD).any(1)
    cue_on_fire = ball_moves[fire, 0] > MOVE_THRESHOLD
    robot_step = np.linalg.norm(s1[:, :2] - s0[:, :2], axis=1)
    print("--- Data stats ---")
    print(f"Fire transitions: {fire.sum()} ({100 * fire.mean():.1f}%)")
    print(f"Fires where the cue ball moved >1mm that step: {cue_on_fire.sum()} "
          f"({100 * cue_on_fire.mean():.1f}% of fires)")
    print(f"Transitions with any ball moving >1mm: {any_ball.sum()} ({100 * any_ball.mean():.1f}%)")
    print(f"Robot displacement per step: mean {1000 * robot_step.mean():.2f} mm, "
          f"max {1000 * robot_step.max():.2f} mm")
    print()


class DynamicsMLP(nn.Module):
    """Same shape as perception.encoder.LatentPredictor; predicts a normalized state delta."""

    def __init__(self, in_dim, out_dim, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.Mish(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.Mish(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class Model:
    """Wraps the MLP with input/output normalization; step() maps raw state -> raw next state."""

    def __init__(self, net, x_mean, x_std, d_mean, d_std):
        self.net = net
        self.x_mean, self.x_std = x_mean, x_std
        self.d_mean, self.d_std = d_mean, d_std

    def step(self, s, vel, a):
        parts = [s] + ([vel] if vel is not None else []) + [a]
        x = (torch.cat(parts, -1) - self.x_mean) / self.x_std
        d = self.net(x) * self.d_std + self.d_mean
        s_next = s + d
        s_next[..., THETA] = wrap_angle_t(s_next[..., THETA])
        return s_next


def errors(pred, true):
    """Per-sample errors in physical units: robot pos (m), heading (deg), mean ball pos (m)."""
    robot_pos = torch.linalg.norm(pred[:, :2] - true[:, :2], dim=1)
    heading = torch.rad2deg(wrap_angle_t(pred[:, THETA] - true[:, THETA]).abs())
    balls = torch.linalg.norm((pred[:, 3:] - true[:, 3:]).view(-1, 4, 2), dim=-1)
    return robot_pos, heading, balls


def train(args):
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)

    s0, s1, actions, ep_idx, step_idx = load_data()
    report_data_stats(s0, s1, actions)
    vel = prev_delta(s0, step_idx) if args.with_velocity else None

    # Split by episode so holdout rollouts never see training trajectories.
    episodes = np.unique(ep_idx)
    n_hold_ep = max(1, int(len(episodes) * HOLDOUT_FRACTION))
    hold_eps = rng.choice(episodes, n_hold_ep, replace=False)
    hold_mask = np.isin(ep_idx, hold_eps)
    tr, ho = ~hold_mask, hold_mask

    inputs = np.concatenate([s0] + ([vel] if vel is not None else []) + [actions], 1)
    deltas = state_delta(s1, s0)

    x_mean, x_std = inputs[tr].mean(0), inputs[tr].std(0) + 1e-6
    d_mean, d_std = deltas[tr].mean(0), deltas[tr].std(0) + 1e-6
    T = lambda a: torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))
    x_mean_t, x_std_t, d_mean_t, d_std_t = map(T, (x_mean, x_std, d_mean, d_std))

    X_tr, Y_tr = T((inputs[tr] - x_mean) / x_std), T((deltas[tr] - d_mean) / d_std)
    X_ho, Y_ho = T((inputs[ho] - x_mean) / x_std), T((deltas[ho] - d_mean) / d_std)

    net = DynamicsMLP(inputs.shape[1], STATE_DIM, HIDDEN)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    print(f"Mode: {'positions + velocity' if args.with_velocity else 'positions only'} "
          f"(input dim {inputs.shape[1]})")
    print(f"Params: {sum(p.numel() for p in net.parameters()):,}")
    print(f"Train: {tr.sum()} transitions, Holdout: {ho.sum()} ({n_hold_ep} episodes)\n")

    best, best_state, patience = float("inf"), None, 0
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        net.train()
        perm = torch.randperm(len(X_tr))
        total = 0.0
        for i in range(0, len(perm) - BATCH_SIZE + 1, BATCH_SIZE):
            b = perm[i:i + BATCH_SIZE]
            loss = nn.functional.mse_loss(net(X_tr[b]), Y_tr[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
        train_loss = total / (len(perm) // BATCH_SIZE * BATCH_SIZE)

        net.eval()
        with torch.no_grad():
            hold_loss = nn.functional.mse_loss(net(X_ho), Y_ho).item()
        print(f"Epoch {epoch:3d}/{EPOCHS}  train={train_loss:.5f}  holdout={hold_loss:.5f}  "
              f"({time.time() - t0:.1f}s)")

        if hold_loss < best - EARLY_STOP_MIN_DELTA:
            best, patience = hold_loss, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            patience += 1
            if patience >= EARLY_STOP_PATIENCE:
                print(f"\nEarly stopping at epoch {epoch} (best holdout {best:.5f})")
                break

    net.load_state_dict(best_state)
    net.eval()
    model = Model(net, x_mean_t, x_std_t, d_mean_t, d_std_t)

    # --- One-step holdout error in physical units ---
    with torch.no_grad():
        S0, S1, A = T(s0[ho]), T(s1[ho]), T(actions[ho])
        V = T(vel[ho]) if vel is not None else None
        rp, hd, bl = errors(model.step(S0, V, A), S1)
        rp0, hd0, bl0 = errors(S0, S1)  # baseline: predict no change
    print("\n--- One-step holdout error (model vs. no-change baseline) ---")
    print(f"Robot position: {1000 * rp.mean():.3f} mm   (baseline {1000 * rp0.mean():.3f} mm)")
    print(f"Robot heading : {hd.mean():.3f} deg  (baseline {hd0.mean():.3f} deg)")
    print(f"Ball position : {1000 * bl.mean():.4f} mm  (baseline {1000 * bl0.mean():.4f} mm)")
    for i, name in enumerate(BALL_NAMES):
        print(f"  {name:6s}: {1000 * bl[:, i].mean():.4f} mm  (baseline {1000 * bl0[:, i].mean():.4f} mm)")
    moving = (torch.linalg.norm((S1[:, 3:] - S0[:, 3:]).view(-1, 4, 2), dim=-1) > MOVE_THRESHOLD)
    if moving.any():
        print(f"Ball position, moving balls only (n={int(moving.sum())}): "
              f"{1000 * bl[moving].mean():.3f} mm  (baseline {1000 * bl0[moving].mean():.3f} mm)")

    # --- Multi-step rollout on holdout episodes ---
    rollout = rollout_eval(model, s0, s1, actions, step_idx, ho, vel is not None, T)
    plot_rollout(rollout, args.with_velocity)


def rollout_eval(model, s0, s1, actions, step_idx, ho, use_vel, T):
    H = max(HORIZONS)
    hold_rows = np.where(ho)[0]
    max_step = step_idx.max()
    # Rows are sorted by (episode, step), so row r+k is step t+k of the same episode.
    starts = hold_rows[step_idx[hold_rows] <= max_step - H + 1]

    s = T(s0[starts])
    vel = None
    if use_vel:
        vel = T(prev_delta(s0, step_idx)[starts])
    s_init = s.clone()
    out = {"model": {k: [] for k in ("robot", "heading", "ball")},
           "baseline": {k: [] for k in ("robot", "heading", "ball")}}

    with torch.no_grad():
        for h in range(1, H + 1):
            rows = starts + h - 1
            s_next = model.step(s, vel, T(actions[rows]))
            if use_vel:
                d = s_next - s
                d[:, THETA] = wrap_angle_t(d[:, THETA])
                vel = d
            s = s_next
            if h in HORIZONS:
                true = T(s1[rows])
                for key, pred in (("model", s), ("baseline", s_init)):
                    rp, hd, bl = errors(pred, true)
                    out[key]["robot"].append(rp.mean().item())
                    out[key]["heading"].append(hd.mean().item())
                    out[key]["ball"].append(bl.mean().item())

    print(f"\n--- Multi-step rollout ({len(starts)} start states from holdout episodes) ---")
    print(f"{'horizon':>8} {'robot mm':>10} {'(base)':>9} {'head deg':>9} {'(base)':>8} "
          f"{'ball mm':>9} {'(base)':>9}")
    for i, h in enumerate(HORIZONS):
        m, b = out["model"], out["baseline"]
        print(f"{h:>8} {1000 * m['robot'][i]:>10.3f} {1000 * b['robot'][i]:>9.3f} "
              f"{m['heading'][i]:>9.3f} {b['heading'][i]:>8.3f} "
              f"{1000 * m['ball'][i]:>9.4f} {1000 * b['ball'][i]:>9.4f}")
    return out


def plot_rollout(out, with_velocity):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    panels = [("robot", "Robot position error (mm)", 1000),
              ("heading", "Robot heading error (deg)", 1),
              ("ball", "Ball position error (mm)", 1000)]
    for ax, (key, title, scale) in zip(axes, panels):
        ax.plot(HORIZONS, np.array(out["model"][key]) * scale, "o-", label="MLP rollout")
        ax.plot(HORIZONS, np.array(out["baseline"][key]) * scale, "s--", color="gray",
                label="no-change baseline")
        ax.set_title(title)
        ax.set_xlabel("Rollout horizon (steps)")
        ax.set_xticks(HORIZONS)
        ax.grid(alpha=0.3)
    axes[0].legend()
    mode = "positions + velocity" if with_velocity else "positions only"
    fig.suptitle(f"Ground-truth-state dynamics probe — error vs. horizon ({mode})")
    fig.tight_layout()
    name = "dynamics_rollout_vel.png" if with_velocity else "dynamics_rollout.png"
    path = os.path.join(FIGURES_DIR, name)
    fig.savefig(path, dpi=150)
    print(f"\nSaved plot to {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-velocity", action="store_true",
                        help="Also feed the previous-step state delta (finite-difference velocity)")
    train(parser.parse_args())
