"""Train the shot-outcome model on shot_pairs data with a frozen encoder.

    python data/collect.py --mode shot_pairs --episodes 5000      # shots_XXX.npz
    python experiments/train_shot_model.py

Each shot is encoded once (file by file, so memory stays ~1 GB): z_pre from the two frames
before the strike, z_rest from the settled frame. The model learns z_rest from (z_pre, robot
heading, force); --no-heading drops the heading (the latent alone encodes it poorly). The
heading is the true pre-strike robot heading here; on hardware it comes from the marker tracker.

Train on gentle shots: above force ~0.3 the cue ball bounces off several cushions and the
outcome is unpredictable even from ground-truth state. Collect with
    python data/collect.py --mode shot_pairs --episodes 5000 --force-range 0.05 0.3

Long shots: add straight long-shot data (see data/collect.py) and train on both sets, e.g.
    python experiments/train_shot_model.py --pattern "shots_v2_*.npz" "shots_long_*.npz"
The default model is 512 wide x 5 blocks (--hidden/--depth; 256 x 3 is the original).
--cue-loss-weight W adds W x the MSE on the cue keypoints only (the dims the planner scores),
on top of the full-latent MSE; 0 (default) is the original objective.

Held-out report (in physical units, via the cue-ball keypoint readout the planner uses):
  * cue rest position error (cm): model vs "cue doesn't move" vs the encoder ceiling
    (reading the TRUE z_rest; the model can't beat the encoder's own readout error),
    for all shots and broken out by true cue travel (< 8 cm, >= 8 cm, and >= 8 cm rolled
    out straight, when the data records it), with the share of shots within 3 / 5 cm;
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
    p.add_argument("--pattern", nargs="+", default=["shots_*.npz"],
                   help="one or more globs in --data-dir (files matched by several are used once)")
    p.add_argument("--checkpoint-dir", default=cem.CHECKPOINT_DIR)
    p.add_argument("--suffix", default="_best", help="encoder checkpoint suffix")
    p.add_argument("--encoder", choices=["auto", "target", "online"], default="auto",
                   help="frozen encoder (auto: target_encoder if saved, else encoder)")
    p.add_argument("--background", default=None,
                   help="the encoder's training background (default: frames_background.npy in --data-dir)")
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay")
    p.add_argument("--hidden", type=int, default=512, help="hidden width (original model: 256)")
    p.add_argument("--depth", type=int, default=5, help="hidden blocks (original model: 3)")
    p.add_argument("--cue-loss-weight", type=float, default=0.0,
                   help="extra weight on the cue-keypoint dims' MSE (0 = original objective)")
    p.add_argument("--long-shot", type=float, default=0.08, help="m; travel threshold for the long-shot report")
    p.add_argument("--calib-near-robot", type=float, default=0.0,
                   help="fraction of keypoint-calibration scenes with the robot right behind the cue "
                        "ball (see cem.calibrate_cue_keypoints)")
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--holdout-fraction", type=float, default=0.1)
    p.add_argument("--calib-scenes", type=int, default=400)
    p.add_argument("--no-heading", action="store_true", help="latent + force only (ablation)")
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

    files = sorted({f for pat in args.pattern for f in glob.glob(os.path.join(args.data_dir, pat))})
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
    z_pre, z_rest, force, cue_pre, cue_post, connected, heading, straight = [], [], [], [], [], [], [], []
    for f in files:
        d = np.load(f)
        z_pre.append(encode_pairs(encoder, d["frame_pre_prev"], d["frame_pre"], background))
        z_rest.append(encode_pairs(encoder, d["frame_post"], d["frame_post"], background))
        force.append(d["fire_force"])
        cue_pre.append(d["ball_positions_pre"][:, 0])
        cue_post.append(d["ball_positions_post"][:, 0])
        connected.append(d["strike_connected"])
        heading.append(d["robot_pose_pre"][:, 2])
        # files collected before the straight-roll flag existed: unknown (-1)
        straight.append(d["straight"].astype(np.int8) if "straight" in d.files
                        else np.full(len(d["fire_force"]), -1, np.int8))
        print(f"  encoded {os.path.basename(f)} ({len(d['fire_force'])} shots)", flush=True)
    z_pre, z_rest = torch.cat(z_pre), torch.cat(z_rest)
    force = torch.from_numpy(np.concatenate(force)).float()
    heading = torch.from_numpy(np.concatenate(heading)).float()
    use_heading = not args.no_heading
    cue_pre, cue_post = np.concatenate(cue_pre), np.concatenate(cue_post)
    connected, straight = np.concatenate(connected), np.concatenate(straight)
    n = len(force)
    travel_all = np.linalg.norm(cue_post - cue_pre, axis=1)
    print(f"Encoded {n} shots in {time.time() - t0:.0f}s ({connected.mean() * 100:.1f}% connected; "
          f"{100 * (travel_all >= args.long_shot).mean():.0f}% travel >= {100 * args.long_shot:.0f} cm)\n")

    # Cue keypoints: needed for the evaluation, and for the cue-weighted loss if enabled.
    from env.sim_env import SimEnv
    print(f"Calibrating cue keypoints on {args.calib_scenes} scenes...")
    kp = cem.calibrate_cue_keypoints(encoder, SimEnv(), background, DEVICE, n_scenes=args.calib_scenes,
                                     seed=args.seed, near_robot_fraction=args.calib_near_robot)
    print(f"{kp.selected.sum()} cue keypoints, readout R^2 {kp.readout_r2:.3f}\n")

    perm = rng.permutation(n)
    n_ho = max(1, int(n * args.holdout_fraction))
    ho, tr = perm[:n_ho], perm[n_ho:]

    model = ShotModel(z_pre.shape[1], args.hidden, use_heading=use_heading, depth=args.depth).to(DEVICE)
    print(f"Shot model: {args.depth} x {args.hidden} ({sum(p.numel() for p in model.parameters()):,} params), "
          f"inputs latent{' + heading' if use_heading else ''} + force"
          f"{f', cue-keypoint loss weight {args.cue_loss_weight}' if args.cue_loss_weight else ''}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=10)
    Zp, Zr, F, Hd = z_pre.to(DEVICE), z_rest.to(DEVICE), force.to(DEVICE), heading.to(DEVICE)
    tr_t, ho_t = torch.from_numpy(tr).to(DEVICE), torch.from_numpy(ho).to(DEVICE)
    cue_w = torch.from_numpy(kp.dim_weights).to(DEVICE)          # R^2_cue on cue dims, 0 elsewhere
    cue_w = cue_w / cue_w.mean().clamp_min(1e-8)                 # mean 1 -> comparable scale to the MSE

    def loss_on(idx):
        err = (model(Zp[idx], F[idx], Hd[idx]) - Zr[idx]) ** 2
        loss = err.mean()
        if args.cue_loss_weight:
            loss = loss + args.cue_loss_weight * (err * cue_w).mean()
        return loss

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
        sched.step(hold)
        if epoch == 1 or epoch % 25 == 0:
            print(f"Epoch {epoch:4d}  holdout loss {hold:.6f}  lr {opt.param_groups[0]['lr']:.1e}", flush=True)
        if hold < best - 1e-8:
            best, patience = hold, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= args.patience:
                print(f"Early stopping at epoch {epoch} (best holdout loss {best:.6f})")
                break
    model.load_state_dict(best_state)
    model.eval().cpu()

    out = os.path.join(args.checkpoint_dir, f"shot_model{args.suffix}.pt")
    save_shot_model(model, out, {"encoder_file": enc_file, "data": files, "n_shots": n,
                                 "force_range": [float(force.min()), float(force.max())],
                                 "use_heading": use_heading, "hidden": args.hidden, "depth": args.depth,
                                 "cue_loss_weight": args.cue_loss_weight})
    print(f"Saved {out}\n")

    # ---- Evaluation in physical units via the cue-ball keypoints ----
    with torch.no_grad():
        zp_h, zr_h, f_h, h_h = z_pre[ho], z_rest[ho], force[ho], heading[ho]
        pred = model(zp_h, f_h, h_h)
        f_lo, f_hi = float(force.min()), float(force.max())
        lo = model(zp_h, torch.full_like(f_h, f_lo + 0.2 * (f_hi - f_lo)), h_h)
        hi = model(zp_h, torch.full_like(f_h, f_lo + 0.9 * (f_hi - f_lo)), h_h)
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

    # Broken out by true cue travel: long shots are where the planner has been missing.
    long_cm = 100 * args.long_shot
    st_h = straight[ho]
    groups = [("all", np.ones(len(ho), bool)),
              (f"travel < {long_cm:.0f} cm", travel_true < long_cm),
              (f"travel >= {long_cm:.0f} cm", travel_true >= long_cm)]
    if (st_h >= 0).any():
        groups.append((f">= {long_cm:.0f} cm, straight", (travel_true >= long_cm) & (st_h == 1)))
        groups.append((f">= {long_cm:.0f} cm, cushion/ball", (travel_true >= long_cm) & (st_h == 0)))
    print(f"\nBy true cue travel (model error; ceiling = encoder readout of the true rest frame):")
    print(f"  {'group':>26} {'n':>5} {'model mean':>11} {'median':>7} {'<3 cm':>6} {'<5 cm':>6} "
          f"{'no-move':>8} {'ceiling':>8}")
    for name, m in groups:
        if not m.any():
            continue
        e = err_model[m]
        print(f"  {name:>26} {m.sum():>5} {e.mean():>11.1f} {np.median(e):>7.1f} {100 * (e < 3).mean():>5.0f}% "
              f"{100 * (e < 5).mean():>5.0f}% {err_none[m].mean():>8.1f} {err_ceiling[m].mean():>8.1f}")
    print(f"Cue travel: true mean {travel_true.mean():.1f} cm, predicted mean {travel_pred.mean():.1f} cm, "
          f"correlation r = {r:.2f}")
    print(f"Force sensitivity (same start): predicted travel at force "
          f"{f_lo + 0.2 * (f_hi - f_lo):.2f} = {d_lo.mean():.1f} cm, "
          f"at {f_lo + 0.9 * (f_hi - f_lo):.2f} = {d_hi.mean():.1f} cm "
          f"({100 * (d_hi > d_lo).mean():.0f}% of starts predict a harder shot goes further)")


if __name__ == "__main__":
    main()
