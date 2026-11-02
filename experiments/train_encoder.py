"""JEPA-style encoder + latent predictor training on two-frame inputs.

The encoder sees two stacked frames (t-1, t) so its latent can carry velocity.
The predictor maps (z_t, a_t) -> z_{t+1}, where the target z_{t+1} comes from an
EMA copy of the encoder applied to frames (t, t+1).

Memory: every distinct frame is written once to a uint8 cache file on disk
(data/datasets/frames_cache.npy) and read back memory-mapped, so RAM use stays
around one batch rather than the whole dataset.

Evaluation: holdout latent-prediction loss (early stopping), a collapse check,
and a linear probe from latents to ground-truth positions and velocities,
compared against the same probe on an untrained encoder.
"""
import glob
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from perception.encoder import LatentPredictor, VisualEncoder

DATASETS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "datasets")
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "checkpoints")
CACHE_PATH = os.path.join(DATASETS_DIR, "frames_cache.npy")
CACHE_META_PATH = os.path.join(DATASETS_DIR, "frames_cache_meta.json")
DATA_PATTERNS = ["batch_v3_*.npz", "batch_strike_*.npz"]

EPOCHS = 50
BATCH_SIZE = 256
LR = 3e-4
HOLDOUT_FRACTION = 0.02
SAVE_EVERY = 10
EARLY_STOP_PATIENCE = 5
EARLY_STOP_MIN_DELTA = 1e-7
EMA_TAU = 0.005
LATENT_DIM = 64
N_FRAMES = 2  # stacked frames per encoder input
PROBE_TRAIN_SAMPLES = 5000
SEED = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
META_KEYS = ["actions_left_pwm", "actions_right_pwm", "actions_fire", "actions_fire_force",
             "robot_pose_before", "ball_positions_before", "episode_idx", "step_idx"]


def list_files():
    files = []
    for pattern in DATA_PATTERNS:
        files += sorted(glob.glob(os.path.join(DATASETS_DIR, pattern)))
    if not files:
        raise FileNotFoundError(f"No files matching {DATA_PATTERNS} in {DATASETS_DIR}")
    return files


def file_signature(files):
    return [[os.path.basename(f), os.path.getsize(f), int(os.path.getmtime(f))] for f in files]


def load_metadata(files):
    """Load everything except frames, and build frame-slot indices.

    Each episode with n steps owns n+1 consecutive frame slots: the n frames_before
    plus the final frames_after. For transition (episode, step s):
        prev = slot s-1 (or s at the first step), cur = slot s, next = slot s+1.
    """
    parts = {k: [] for k in META_KEYS}
    ep_offset, slot_offset = 0, 0
    prev_slot, cur_slot, next_slot, global_ep = [], [], [], []
    file_slots = []  # per file: (slot index for each frames_before row, slot for each episode's last frames_after)
    for f in files:
        d = np.load(f)
        ep, step = d["episode_idx"], d["step_idx"]
        # Collection writes episodes in order; verify rather than assume.
        order = np.lexsort((step, ep))
        if not np.array_equal(order, np.arange(len(ep))):
            raise ValueError(f"{f} is not sorted by (episode, step)")
        for k in META_KEYS:
            parts[k].append(d[k])

        eps, starts, counts = np.unique(ep, return_index=True, return_counts=True)
        before_slots = np.empty(len(ep), dtype=np.int64)
        last_after_slots = np.empty(len(eps), dtype=np.int64)
        for j, (st, n) in enumerate(zip(starts, counts)):
            base = slot_offset
            s = np.arange(n)
            before_slots[st:st + n] = base + s
            last_after_slots[j] = base + n
            cur_slot.append(base + s)
            prev_slot.append(base + np.maximum(s - 1, 0))
            next_slot.append(base + s + 1)
            global_ep.append(np.full(n, ep_offset + j))
            slot_offset += n + 1
        ep_offset += len(eps)
        file_slots.append((before_slots, starts + counts - 1, last_after_slots))

    meta = {k: np.concatenate(v) for k, v in parts.items()}
    n = len(meta["step_idx"])
    meta["actions"] = np.stack([meta["actions_left_pwm"], meta["actions_right_pwm"],
                                meta["actions_fire"].astype(np.float32),
                                meta["actions_fire_force"]], 1).astype(np.float32)
    meta["state"] = np.concatenate([meta["robot_pose_before"][:, :2],
                                    meta["ball_positions_before"].reshape(n, -1)], 1)
    meta["prev"], meta["cur"], meta["next"] = (np.concatenate(prev_slot), np.concatenate(cur_slot),
                                               np.concatenate(next_slot))
    meta["episode"] = np.concatenate(global_ep)
    meta["n_slots"] = int(slot_offset)
    return meta, file_slots


def build_frame_cache(files, file_slots, n_slots):
    """Write all distinct frames to a memory-mapped uint8 .npy, one source file at a time."""
    sig = file_signature(files)
    if os.path.exists(CACHE_PATH) and os.path.exists(CACHE_META_PATH):
        with open(CACHE_META_PATH) as fh:
            if json.load(fh) == sig:
                print(f"Using frame cache {CACHE_PATH}")
                return np.load(CACHE_PATH, mmap_mode="r")

    print(f"Building frame cache ({n_slots} frames, {n_slots * 128 * 128 * 3 / 1e9:.1f} GB on disk)...")
    t0 = time.time()
    cache = np.lib.format.open_memmap(CACHE_PATH, mode="w+", dtype=np.uint8,
                                      shape=(n_slots, 128, 128, 3))
    for f, (before_slots, last_rows, last_after_slots) in zip(files, file_slots):
        d = np.load(f)
        cache[before_slots] = d["frames_before"]
        cache[last_after_slots] = d["frames_after"][last_rows]
        print(f"  {os.path.basename(f)}", flush=True)
    cache.flush()
    del cache
    with open(CACHE_META_PATH, "w") as fh:
        json.dump(sig, fh)
    print(f"Cache built in {time.time() - t0:.0f}s")
    return np.load(CACHE_PATH, mmap_mode="r")


def stack_frames(frames, idx_a, idx_b):
    """(B,) slot indices -> float tensor (B, 6, 128, 128) in [0, 1]: [frame a | frame b]."""
    order = np.argsort(idx_a)  # sorted reads are friendlier to the memory map
    inv = np.argsort(order)
    a = frames[idx_a[order]][inv]
    b = frames[idx_b[order]][inv]
    x = torch.from_numpy(np.concatenate([a, b], axis=-1))
    return x.permute(0, 3, 1, 2).float().div_(255.0)


def encode(encoder, frames, prev, cur, batch=512):
    encoder.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(cur), batch):
            out.append(encoder(stack_frames(frames, prev[i:i + batch], cur[i:i + batch]).to(DEVICE)).cpu())
    return torch.cat(out).numpy()


def check_collapse(encoder, frames, prev, cur):
    """Encode a small batch and report latent spread to detect collapse."""
    latents = encode(encoder, frames, prev[:64], cur[:64])
    encoder.train()
    mean_std = latents.std(axis=0).mean()
    pairwise = np.linalg.norm(latents[:8, None] - latents[None, :8], axis=-1)
    return mean_std, pairwise[np.triu_indices(8, k=1)].mean()


def evaluate_holdout(encoder, target_encoder, predictor, frames, meta, rows):
    """Mean latent-prediction MSE on held-out transitions."""
    encoder.eval()
    predictor.eval()
    total = 0.0
    with torch.no_grad():
        for i in range(0, len(rows), BATCH_SIZE):
            r = rows[i:i + BATCH_SIZE]
            x_t = stack_frames(frames, meta["prev"][r], meta["cur"][r]).to(DEVICE)
            x_tp1 = stack_frames(frames, meta["cur"][r], meta["next"][r]).to(DEVICE)
            act = torch.from_numpy(meta["actions"][r]).to(DEVICE)
            pred = predictor(encoder(x_t), act)
            total += nn.functional.mse_loss(pred, target_encoder(x_tp1), reduction="sum").item() / LATENT_DIM
    encoder.train()
    predictor.train()
    return total / len(rows)


def linear_probe(encoder, frames, meta, train_rows, hold_rows):
    """Ridge regression latent -> ground truth. Returns R^2 per target group on holdout."""
    state = meta["state"]                      # robot x,y + 4 balls x,y  (10)
    vel = np.zeros_like(state)                 # s_t - s_{t-1}, zero at episode start
    vel[1:] = state[1:] - state[:-1]
    vel[meta["step_idx"] == 0] = 0.0
    targets = np.concatenate([state, vel], 1)
    groups = {"robot pos": slice(0, 2), "ball pos": slice(2, 10),
              "robot vel": slice(10, 12), "ball vel": slice(12, 20)}

    def feats(rows):
        z = encode(encoder, frames, meta["prev"][rows], meta["cur"][rows])
        return np.concatenate([z, np.ones((len(z), 1), np.float32)], 1)

    Z_tr, Z_ho = feats(train_rows), feats(hold_rows)
    Y_tr, Y_ho = targets[train_rows], targets[hold_rows]
    W = np.linalg.solve(Z_tr.T @ Z_tr + 1e-3 * np.eye(Z_tr.shape[1]), Z_tr.T @ Y_tr)
    pred = Z_ho @ W
    r2 = {}
    for name, sl in groups.items():
        ss_res = ((pred[:, sl] - Y_ho[:, sl]) ** 2).sum()
        ss_tot = ((Y_ho[:, sl] - Y_tr[:, sl].mean(0)) ** 2).sum()
        r2[name] = 1 - ss_res / ss_tot
    return r2


def save_checkpoint(encoder, predictor, suffix=""):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    enc_path = os.path.join(CHECKPOINT_DIR, f"encoder{suffix}.pt")
    pred_path = os.path.join(CHECKPOINT_DIR, f"predictor{suffix}.pt")
    torch.save(encoder.state_dict(), enc_path)
    torch.save(predictor.state_dict(), pred_path)
    return enc_path, pred_path


def train():
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)

    files = list_files()
    meta, file_slots = load_metadata(files)
    frames = build_frame_cache(files, file_slots, meta["n_slots"])
    n = len(meta["cur"])
    print(f"Loaded {n} transitions from {len(files)} files ({meta['n_slots']} distinct frames)")

    # Split by episode.
    episodes = np.unique(meta["episode"])
    hold_eps = rng.choice(episodes, max(1, int(len(episodes) * HOLDOUT_FRACTION)), replace=False)
    is_hold = np.isin(meta["episode"], hold_eps)
    train_rows, hold_rows = np.where(~is_hold)[0], np.where(is_hold)[0]
    probe_rows = rng.choice(train_rows, min(PROBE_TRAIN_SAMPLES, len(train_rows)), replace=False)

    in_ch = 3 * N_FRAMES
    encoder = VisualEncoder(latent_dim=LATENT_DIM, num_simplices=8, in_channels=in_ch).to(DEVICE)
    target_encoder = VisualEncoder(latent_dim=LATENT_DIM, num_simplices=8, in_channels=in_ch).to(DEVICE)
    target_encoder.load_state_dict(encoder.state_dict())
    for p in target_encoder.parameters():
        p.requires_grad = False
    predictor = LatentPredictor(latent_dim=LATENT_DIM, action_dim=4, hidden_dim=128).to(DEVICE)

    print(f"Encoder params: {sum(p.numel() for p in encoder.parameters()):,} (input {in_ch} channels)")
    print(f"Predictor params: {sum(p.numel() for p in predictor.parameters()):,}")
    print(f"EMA tau: {EMA_TAU}  Device: {DEVICE}  Threads: {torch.get_num_threads()}")
    print(f"Train: {len(train_rows)}, Holdout: {len(hold_rows)} ({len(hold_eps)} episodes)")

    print("\nLinear probe on the untrained encoder (reference):")
    untrained_r2 = linear_probe(encoder, frames, meta, probe_rows, hold_rows)
    print("  " + "  ".join(f"{k}: R2={v:.3f}" for k, v in untrained_r2.items()))
    print()

    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(predictor.parameters()), lr=LR)
    best_loss, patience_counter = float("inf"), 0

    for epoch in range(1, EPOCHS + 1):
        encoder.train()
        predictor.train()
        epoch_loss, n_batches = 0.0, 0
        t0 = time.time()
        perm = rng.permutation(train_rows)

        for i in range(0, len(perm) - BATCH_SIZE + 1, BATCH_SIZE):
            r = perm[i:i + BATCH_SIZE]
            x_t = stack_frames(frames, meta["prev"][r], meta["cur"][r]).to(DEVICE)
            x_tp1 = stack_frames(frames, meta["cur"][r], meta["next"][r]).to(DEVICE)
            act = torch.from_numpy(meta["actions"][r]).to(DEVICE)

            z_tp1_pred = predictor(encoder(x_t), act)
            with torch.no_grad():
                z_tp1_target = target_encoder(x_tp1)
            loss = nn.functional.mse_loss(z_tp1_pred, z_tp1_target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                for p, tp in zip(encoder.parameters(), target_encoder.parameters()):
                    tp.data.lerp_(p.data, EMA_TAU)

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        elapsed = time.time() - t0
        holdout_loss = evaluate_holdout(encoder, target_encoder, predictor, frames, meta, hold_rows)
        latent_std, latent_dist = check_collapse(encoder, frames, meta["prev"][hold_rows], meta["cur"][hold_rows])
        collapsed = "COLLAPSED" if latent_dist < 0.01 else "ok"

        print(f"Epoch {epoch:3d}/{EPOCHS}  loss={avg_loss:.6f}  holdout={holdout_loss:.6f}  "
              f"latent_std={latent_std:.4f}  latent_dist={latent_dist:.4f}  "
              f"[{collapsed}]  ({elapsed:.1f}s)", flush=True)

        if epoch % SAVE_EVERY == 0:
            enc_path, _ = save_checkpoint(encoder, predictor, f"_epoch{epoch}")
            print(f"  -> Checkpoint saved: {enc_path}")

        if holdout_loss < best_loss - EARLY_STOP_MIN_DELTA:
            best_loss = holdout_loss
            patience_counter = 0
            save_checkpoint(encoder, predictor, "_best")
        else:
            patience_counter += 1

        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"\nEarly stopping at epoch {epoch} (no holdout improvement for {EARLY_STOP_PATIENCE} epochs)")
            break

    enc_path, pred_path = save_checkpoint(encoder, predictor)
    print(f"\nSaved final encoder to {enc_path}")
    print(f"Saved final predictor to {pred_path}")

    print("\n--- Linear probe: latent -> ground truth (holdout R^2, 1.0 = perfect) ---")
    best = VisualEncoder(latent_dim=LATENT_DIM, num_simplices=8, in_channels=in_ch).to(DEVICE)
    best.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, "encoder_best.pt"), map_location=DEVICE))
    trained_r2 = linear_probe(best, frames, meta, probe_rows, hold_rows)
    print(f"{'target':>10} {'trained':>9} {'untrained':>10}")
    for k in trained_r2:
        print(f"{k:>10} {trained_r2[k]:>9.3f} {untrained_r2[k]:>10.3f}")


if __name__ == "__main__":
    train()
