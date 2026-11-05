"""One-step latent prediction error of a trained encoder + predictor on held-out data.

Reproduces train_encoder.py's held-out episodes (same files, same seed) and reports:
  * prediction error  ||predictor(z_t, a_t) - z_{t+1}||   (L2 over the latent)
  * no-change baseline ||z_t - z_{t+1}||                   (what "predict nothing moves" scores)
  * scale reference   ||z_i - z_j|| for random pairs       (how far apart latents typically are)
for all transitions, transitions with ball motion > 5 mm, and static ones.

z_{t+1} is computed with BOTH the online encoder and, if saved, the EMA target encoder.
The predictor is trained against the target encoder, so the "target" row is the fair one;
the "online" row shows how much a planner that encodes goals with the online encoder loses.

Usage:
    python experiments/eval_predictor.py
    python experiments/eval_predictor.py --suffix "" --checkpoint-dir data/checkpoints
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
import train_encoder as te
from perception.encoder import load_predictor


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint-dir", default=te.CHECKPOINT_DIR)
    p.add_argument("--suffix", default="_best")
    p.add_argument("--head", default=te.ENCODER_HEAD, choices=["spatial_softmax", "linear"])
    p.add_argument("--data", nargs="+", default=te.DATA_PATTERNS,
                   help="file patterns; must match training for the holdout split to match")
    p.add_argument("--holdout-fraction", type=float, default=te.HOLDOUT_FRACTION)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--cache-dir", default=None,
                   help="where the frame cache / background live (default: data/datasets, as in training)")
    args = p.parse_args()

    te.DATA_PATTERNS, te.ENCODER_HEAD, te.HOLDOUT_FRACTION = args.data, args.head, args.holdout_fraction
    if args.cache_dir:
        te.CACHE_PATH = os.path.join(args.cache_dir, "frames_cache.npy")
        te.CACHE_META_PATH = os.path.join(args.cache_dir, "frames_cache_meta.json")
        te.BACKGROUND_PATH = os.path.join(args.cache_dir, "frames_background.npy")
    dev = te.DEVICE

    files = te.list_files()
    meta, file_slots = te.load_metadata(files)
    frames = te.build_frame_cache(files, file_slots, meta["n_slots"])
    te.BACKGROUND = torch.from_numpy(te.compute_background(frames, te.file_signature(files)))
    te.FRAME_HW = tuple(int(x) for x in frames.shape[1:3])

    # Same split as train_encoder.train(): first rng draw after seeding.
    rng = np.random.default_rng(te.SEED)
    episodes = np.unique(meta["episode"])
    hold_eps = rng.choice(episodes, max(1, int(len(episodes) * te.HOLDOUT_FRACTION)), replace=False)
    rows = np.where(np.isin(meta["episode"], hold_eps))[0]

    def load(module, name):
        module.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, name), map_location=dev))
        return module.eval()

    in_ch = 3 * te.N_FRAMES
    encoder = load(te.make_encoder(in_ch), f"encoder{args.suffix}.pt")
    predictor = load_predictor(os.path.join(args.checkpoint_dir, f"predictor{args.suffix}.pt"), dev)
    spaces = {"online": encoder}
    tpath = f"target_encoder{args.suffix}.pt"
    if os.path.exists(os.path.join(args.checkpoint_dir, tpath)):
        spaces["target"] = load(te.make_encoder(in_ch), tpath)
    else:
        print(f"NOTE: {tpath} not found (older checkpoint); only the online-encoder comparison "
              "is available, which overstates the predictor's error.")

    print(f"Held-out: {len(rows)} transitions from {len(hold_eps)} episodes "
          f"({args.checkpoint_dir}, suffix '{args.suffix}')\n")

    z_in, pred = [], []
    z_next = {k: [] for k in spaces}
    z_cur = {k: [] for k in spaces}
    with torch.no_grad():
        for i in range(0, len(rows), args.batch_size):
            r = rows[i:i + args.batch_size]
            x_t = te.stack_frames(frames, meta["prev"][r], meta["cur"][r]).to(dev)
            x_tp1 = te.stack_frames(frames, meta["cur"][r], meta["next"][r]).to(dev)
            a = torch.from_numpy(meta["actions"][r]).to(dev)
            z_t = encoder(x_t)
            z_in.append(z_t.cpu())
            pred.append(predictor(z_t, a).cpu())
            for k, enc in spaces.items():
                z_next[k].append(enc(x_tp1).cpu())
                z_cur[k].append(enc(x_t).cpu())
    pred = torch.cat(pred)
    moving = meta["ball_disp"][rows] > 5e-3
    subsets = {"all": np.ones(len(rows), bool), "balls moving >5mm": moving, "static": ~moving}

    for k in spaces:
        zn, zc = torch.cat(z_next[k]), torch.cat(z_cur[k])
        err = (pred - zn).norm(dim=1).numpy()
        base = (zc - zn).norm(dim=1).numpy()
        perm = rng.permutation(len(zc))
        scale = (zc - zc[perm]).norm(dim=1).mean().item()
        print(f"=== z_t+1 from the {k.upper()} encoder ===")
        print(f"random-pair latent distance (scale reference): {scale:.3f}")
        print(f"{'subset':>20} {'n':>6} {'pred err':>9} {'no-change':>10} {'ratio':>7}")
        for name, m in subsets.items():
            if m.any():
                e, b = err[m].mean(), base[m].mean()
                print(f"{name:>20} {m.sum():>6} {e:>9.3f} {b:>10.3f} {b / e:>6.2f}x")
        print()

    print("Reading it: 'ratio' > 1 means the predictor beats predicting no change. If the target-"
          "encoder ratio is near or below 1, the predictor is not usable for planning and needs "
          "retraining; if only the online row is bad, the planner just needs the target encoder.")


if __name__ == "__main__":
    main()
