"""Train the shot-outcome model on shot_pairs data with a frozen encoder.

    python data/collect.py --mode shot_pairs --episodes 5000      # shots_XXX.npz
    python experiments/train_shot_model.py

Each shot is encoded once (file by file, so memory stays ~1 GB): z_pre from the two frames
before the strike, z_rest from the settled frame. The model learns z_rest from (z_pre, force).

Held-out report (in physical units, via the cue-ball keypoint readout the planner uses):
  * cue rest position error (cm): model vs "cue doesn't move" vs the encoder ceiling
    (reading the TRUE z_rest; the model can't beat the encoder's own readout error)
  * how well predicted cue travel tracks true travel (correlation), and
  * force sensitivity: predicted cue travel at force 0.2 vs 0.9 from the same start.
The shot model is only useful to the planner if travel correlation is high and harder
shots are predicted to go further.

Outputs (in --checkpoint-dir): shot_model{suffix}.pt + .json (names the encoder used).
"""
import argparse
import glob
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

REPO = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, REPO)
from models.shot_model import ShotModel, save_shot_model
from planning import cem

DEFAULT_DATA_DIR = os.environ.get("POOLJEPA_DATA", os.path.join(REPO, "data", "datasets"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--pattern", default="shots_*.npz")
    p.add_argument("--checkpoint-dir", default=cem.CHECKPOINT_DIR)
    p.add_argument("--suffix", default="_best", help="encoder checkpoint suffix")
    p.add_argument("--encoder", choices=["auto", "target", "online"], default="auto",
                   help="frozen encoder (auto: target_encoder if saved, else encoder)")
    p.add_argument("--background", default=None,
                   help="the encoder's training background (default: frames_background.npy in --data-dir)")
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--holdout-fraction", type=float, default=0.1)
    p.add_argument("--calib-scenes", type=int, default=400)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


@torch.no_grad()
def encode_pairs(encoder, a, b, background, batch=64):
    """a, b: (N, H, W, 3) uint8 -> latents (N, D) of the stacked [a | b] input."""
    bg = torch.from_numpy(background.astype(np.float32))
    out = []
    for i in range(0, len(a), batch):
        x = torch.cat([torch.from_numpy(a[i:i + batch]).float() - bg,
                       torch.from_numpy(b[i:i + batch]).float() - bg], dim=-1)
        out.append(encoder(x.permute(0, 3, 1, 2).div_(255.0).to(DEVICE)).cpu())
    return torch.cat(out)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    files = sorted(glob.glob(os.path.join(args.data_dir, args.pattern)))
    if not files:
        raise SystemExit(f"No {args.pattern} in {args.data_dir}. "
                         "Collect with: python data/collect.py --mode shot_pairs --episodes 5000")

    # Frozen encoder (same choice logic as train_predictor.py).
    target_file = f"target_encoder{args.suffix}.pt"
    use_target = args.encoder == "target" or (
        args.encoder == "auto" and os.path.exists(os.path.join(args.checkpoint_dir, target_file)))
    enc_file = target_file if use_target else f"encoder{args.suffix}.pt"
    encoder = cem.make_spatial_encoder(DEVICE)
    encoder.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, enc_file), map_location=DEVICE))
    encoder.eval()

    bg_path = args.background or os.path.join(args.data_dir, "frames_background.npy")
    if os.path.exists(bg_path):
        background = np.load(bg_path)
    else:
        print(f"WARNING: {bg_path} not found; using the median of the shots' pre-strike frames. "
              "Pass --background with the encoder's training background for best results.")
        background = np.median(np.load(files[0])["frame_pre"][:300], axis=0).astype(np.float32)
    print(f"Frozen encoder: {enc_file} | {len(files)} shot files | device {DEVICE}")

    # Encode file by file.
    t0 = time.time()
    z_pre, z_rest, force, cue_pre, cue_post, connected = [], [], [], [], [], []
    for f in files:
        d = np.load(f)
        z_pre.append(encode_pairs(encoder, d["frame_pre_prev"], d["frame_pre"], background))
        z_rest.append(encode_pairs(encoder, d["frame_post"], d["frame_post"], background))
        force.append(d["fire_force"])
        cue_pre.append(d["ball_positions_pre"][:, 0])
        cue_post.append(d["ball_positions_post"][:, 0])
        connected.append(d["strike_connected"])
        print(f"  encoded {os.path.basename(f)}", flush=True)
    z_pre, z_rest = torch.cat(z_pre), torch.cat(z_rest)
    force = torch.from_numpy(np.concatenate(force)).float()
    cue_pre, cue_post = np.concatenate(cue_pre), np.concatenate(cue_post)
    connected = np.concatenate(connected)
    n = len(force)
    print(f"Encoded {n} shots in {time.time() - t0:.0f}s ({connected.mean() * 100:.1f}% connected)\n")

    perm = rng.permutation(n)
    n_ho = max(1, int(n * args.holdout_fraction))
    ho, tr = perm[:n_ho], perm[n_ho:]

    model = ShotModel(z_pre.shape[1]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    Zp, Zr, F = z_pre.to(DEVICE), z_rest.to(DEVICE), force.to(DEVICE)
    tr_t, ho_t = torch.from_numpy(tr).to(DEVICE), torch.from_numpy(ho).to(DEVICE)

    def loss_on(idx):
        return nn.functional.mse_loss(model(Zp[idx], F[idx]), Zr[idx])

    best, best_state, patience = float("inf"), None, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        p_ = tr_t[torch.randperm(len(tr_t), device=DEVICE)]
        for i in range(0, len(p_), args.batch_size):
            loss = loss_on(p_[i:i + args.batch_size])
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            hold = loss_on(ho_t).item()
        if epoch == 1 or epoch % 25 == 0:
            print(f"Epoch {epoch:4d}  holdout MSE {hold:.6f}", flush=True)
        if hold < best - 1e-8:
            best, patience = hold, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= args.patience:
                print(f"Early stopping at epoch {epoch} (best holdout MSE {best:.6f})")
                break
    model.load_state_dict(best_state)
    model.eval().cpu()

    out = os.path.join(args.checkpoint_dir, f"shot_model{args.suffix}.pt")
    save_shot_model(model, out, {"encoder_file": enc_file, "data": files, "n_shots": n,
                                 "force_range": [float(force.min()), float(force.max())]})
    print(f"Saved {out}\n")

    # ---- Evaluation in physical units via the cue-ball keypoints ----
    from env.sim_env import SimEnv
    print(f"Calibrating cue keypoints on {args.calib_scenes} scenes...")
    kp = cem.calibrate_cue_keypoints(encoder, SimEnv(), background, DEVICE,
                                     n_scenes=args.calib_scenes, seed=args.seed)
    print(f"{kp.selected.sum()} cue keypoints, readout R^2 {kp.readout_r2:.3f}\n")

    with torch.no_grad():
        zp_h, zr_h, f_h = z_pre[ho], z_rest[ho], force[ho]
        pred = model(zp_h, f_h)
        lo = model(zp_h, torch.full_like(f_h, 0.2))
        hi = model(zp_h, torch.full_like(f_h, 0.9))
    true_rest, true_start = cue_post[ho], cue_pre[ho]
    dec_pred = kp.decode_cue_batch(pred)
    dec_true_rest = kp.decode_cue_batch(zr_h)
    dec_start = kp.decode_cue_batch(zp_h)

    cm = lambda a, b: 100 * np.linalg.norm(a - b, axis=1)
    err_model = cm(dec_pred, true_rest)
    err_none = cm(true_start, true_rest)
    err_ceiling = cm(dec_true_rest, true_rest)
    travel_true = cm(true_rest, true_start)
    travel_pred = cm(dec_pred, dec_start)
    r = np.corrcoef(travel_true, travel_pred)[0, 1]
    d_lo = cm(kp.decode_cue_batch(lo), dec_start)
    d_hi = cm(kp.decode_cue_batch(hi), dec_start)

    print(f"========== Held-out shots: {len(ho)} ==========")
    print(f"Cue ball rest position error (cm), mean / median:")
    print(f"  shot model                      {err_model.mean():6.1f} / {np.median(err_model):5.1f}")
    print(f"  'cue doesn't move' baseline      {err_none.mean():6.1f} / {np.median(err_none):5.1f}")
    print(f"  encoder ceiling (true z_rest)    {err_ceiling.mean():6.1f} / {np.median(err_ceiling):5.1f}")
    print(f"  -> model is {err_none.mean() / err_model.mean():.2f}x better than no-change")
    print(f"Cue travel: true mean {travel_true.mean():.1f} cm, predicted mean {travel_pred.mean():.1f} cm, "
          f"correlation r = {r:.2f}")
    print(f"Force sensitivity (same start): predicted travel at force 0.2 = {d_lo.mean():.1f} cm, "
          f"at 0.9 = {d_hi.mean():.1f} cm "
          f"({100 * (d_hi > d_lo).mean():.0f}% of starts predict a harder shot goes further)")


if __name__ == "__main__":
    main()
