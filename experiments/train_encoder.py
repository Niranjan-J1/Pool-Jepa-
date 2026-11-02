import glob
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from perception.encoder import LatentPredictor, VisualEncoder

DATASETS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "datasets")
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "checkpoints")

EPOCHS = 50
BATCH_SIZE = 256
LR = 3e-4
HOLDOUT_FRACTION = 0.02
SAVE_EVERY = 10
EARLY_STOP_PATIENCE = 5
EARLY_STOP_MIN_DELTA = 1e-7
EMA_TAU = 0.005
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class TransitionDataset(Dataset):
    def __init__(self, frames_before, frames_after, actions):
        self.frames_before = frames_before
        self.frames_after = frames_after
        self.actions = actions

    def __len__(self):
        return len(self.frames_before)

    def __getitem__(self, idx):
        fb = torch.from_numpy(self.frames_before[idx]).permute(2, 0, 1).float() / 255.0
        fa = torch.from_numpy(self.frames_after[idx]).permute(2, 0, 1).float() / 255.0
        act = torch.from_numpy(self.actions[idx])
        return fb, fa, act


def load_data():
    files = sorted(glob.glob(os.path.join(DATASETS_DIR, "batch_*.npz")))
    if not files:
        raise FileNotFoundError(f"No batch files in {DATASETS_DIR}")

    all_fb, all_fa = [], []
    all_left, all_right, all_fire, all_force = [], [], [], []

    for f in files:
        d = np.load(f)
        all_fb.append(d["frames_before"])
        all_fa.append(d["frames_after"])
        all_left.append(d["actions_left_pwm"])
        all_right.append(d["actions_right_pwm"])
        all_fire.append(d["actions_fire"].astype(np.float32))
        all_force.append(d["actions_fire_force"])

    frames_before = np.concatenate(all_fb)
    frames_after = np.concatenate(all_fa)
    actions = np.stack([
        np.concatenate(all_left),
        np.concatenate(all_right),
        np.concatenate(all_fire),
        np.concatenate(all_force),
    ], axis=1).astype(np.float32)

    print(f"Loaded {len(frames_before)} transitions from {len(files)} files")
    return frames_before, frames_after, actions


def check_collapse(encoder, sample_frames, device):
    """Encode a small batch and report latent variance to detect collapse."""
    encoder.eval()
    with torch.no_grad():
        frames_t = torch.from_numpy(sample_frames[:64]).permute(0, 3, 1, 2).float() / 255.0
        latents = encoder(frames_t.to(device)).cpu().numpy()
    mean_std = latents.std(axis=0).mean()
    pairwise = np.linalg.norm(latents[:8, None] - latents[None, :8], axis=-1)
    mean_dist = pairwise[np.triu_indices(8, k=1)].mean()
    encoder.train()
    return mean_std, mean_dist


def save_checkpoint(encoder, predictor, suffix=""):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    enc_path = os.path.join(CHECKPOINT_DIR, f"encoder{suffix}.pt")
    pred_path = os.path.join(CHECKPOINT_DIR, f"predictor{suffix}.pt")
    torch.save(encoder.state_dict(), enc_path)
    torch.save(predictor.state_dict(), pred_path)
    return enc_path, pred_path


def train():
    frames_before, frames_after, actions = load_data()

    n = len(frames_before)
    n_holdout = max(1, int(n * HOLDOUT_FRACTION))
    indices = np.random.permutation(n)
    train_idx = indices[n_holdout:]
    holdout_idx = indices[:n_holdout]

    train_dataset = TransitionDataset(
        frames_before[train_idx], frames_after[train_idx], actions[train_idx]
    )
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=(DEVICE == "cuda"), drop_last=True,
    )

    encoder = VisualEncoder(latent_dim=64, num_simplices=8).to(DEVICE)
    target_encoder = VisualEncoder(latent_dim=64, num_simplices=8).to(DEVICE)
    target_encoder.load_state_dict(encoder.state_dict())
    for p in target_encoder.parameters():
        p.requires_grad = False
    predictor = LatentPredictor(latent_dim=64, action_dim=4, hidden_dim=128).to(DEVICE)

    param_count_enc = sum(p.numel() for p in encoder.parameters())
    param_count_pred = sum(p.numel() for p in predictor.parameters())
    print(f"Encoder params: {param_count_enc:,}")
    print(f"Predictor params: {param_count_pred:,}")
    print(f"EMA tau: {EMA_TAU}")
    print(f"Device: {DEVICE}")
    print(f"Train: {len(train_dataset)}, Holdout: {n_holdout}")
    print()

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(predictor.parameters()), lr=LR
    )

    best_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        encoder.train()
        predictor.train()
        epoch_loss = 0.0
        n_batches = 0
        t0 = time.time()

        for fb, fa, act in train_loader:
            fb, fa, act = fb.to(DEVICE), fa.to(DEVICE), act.to(DEVICE)

            z_t = encoder(fb)
            with torch.no_grad():
                z_tp1_target = target_encoder(fa)

            z_tp1_pred = predictor(z_t, act)
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

        latent_std, latent_dist = check_collapse(encoder, frames_before, DEVICE)
        collapsed = "COLLAPSED" if latent_dist < 0.01 else "ok"

        print(f"Epoch {epoch:3d}/{EPOCHS}  loss={avg_loss:.6f}  "
              f"latent_std={latent_std:.4f}  latent_dist={latent_dist:.4f}  "
              f"[{collapsed}]  ({elapsed:.1f}s)")

        if epoch % SAVE_EVERY == 0:
            enc_path, _ = save_checkpoint(encoder, predictor, f"_epoch{epoch}")
            print(f"  -> Checkpoint saved: {enc_path}")

        if avg_loss < best_loss - EARLY_STOP_MIN_DELTA:
            best_loss = avg_loss
            patience_counter = 0
            save_checkpoint(encoder, predictor, "_best")
        else:
            patience_counter += 1

        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"\nEarly stopping at epoch {epoch} (no improvement for {EARLY_STOP_PATIENCE} epochs)")
            break

    enc_path, pred_path = save_checkpoint(encoder, predictor)
    print(f"\nSaved final encoder to {enc_path}")
    print(f"Saved final predictor to {pred_path}")

    # --- Sanity check: latent distance vs physical distance ---
    print("\n--- Sanity Check ---")
    encoder.eval()

    holdout_fb = frames_before[holdout_idx]

    n_check = min(1000, len(holdout_fb))
    check_frames = holdout_fb[:n_check]

    frames_tensor = torch.from_numpy(check_frames).permute(0, 3, 1, 2).float() / 255.0
    with torch.no_grad():
        latents = encoder(frames_tensor.to(DEVICE)).cpu().numpy()

    n_pairs = min(5000, n_check * (n_check - 1) // 2)
    idx_i = np.random.randint(0, n_check, size=n_pairs)
    idx_j = np.random.randint(0, n_check, size=n_pairs)
    mask = idx_i != idx_j
    idx_i, idx_j = idx_i[mask], idx_j[mask]

    latent_dists = np.linalg.norm(latents[idx_i] - latents[idx_j], axis=1)

    flat_frames = check_frames.reshape(n_check, -1).astype(np.float32) / 255.0
    pixel_dists = np.linalg.norm(flat_frames[idx_i] - flat_frames[idx_j], axis=1)

    correlation = np.corrcoef(pixel_dists, latent_dists)[0, 1]
    print(f"Pairs sampled: {len(idx_i)}")
    print(f"Pixel distance  — mean: {pixel_dists.mean():.4f}, std: {pixel_dists.std():.4f}")
    print(f"Latent distance — mean: {latent_dists.mean():.4f}, std: {latent_dists.std():.4f}")
    print(f"Correlation (pixel dist vs latent dist): {correlation:.4f}")
    if correlation > 0.3:
        print("=> Good: visually different frames map to different latent embeddings")
    elif correlation > 0.1:
        print("=> Moderate: some structure captured, may improve with more training")
    else:
        print("=> Weak: encoder may need more epochs or architecture tuning")


if __name__ == "__main__":
    train()
