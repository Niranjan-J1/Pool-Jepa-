"""B2: JEPA-style encoder on multiball frames (mb_frames), resumable across 2 h SLURM jobs.

Same recipe as experiments/train_encoder.py (left unchanged for the legacy checkpoints):
two stacked background-subtracted frames -> spatial-softmax keypoints; a predictor maps
(z_t, action) -> z_{t+1}, and the target z_{t+1} comes from an EMA copy of the encoder.
Loss: motion-weighted MSE, no VICReg, no decoder. Changes for multiball (B2 review gate):
  - 64 keypoints, each with a presence value: latent = 64 x (x, y, p) = 192.
  - Frames stream from the compressed .npz files (mb_data.FrameStream), no frame cache.
  - Motion weight = 0.1 + (summed displacement of on-table balls) / (mean displacement).
  - Holdout = the last mb_frames file (100 episodes), never trained on.
  - Diagnostic per epoch: distance from each on-table ball to its nearest keypoint (cm).

Resuming: all state is saved to <run-dir>/state.pt after every epoch. Before starting an
epoch the script checks the time budget; if the next epoch would not fit it exits 0 with
"RESUME NEEDED", and the next chained job picks up from state.pt. When training ends it
writes <run-dir>/DONE, and any further chained job exits immediately.

Outputs in <run-dir>: encoder_best.pt, target_encoder_best.pt, predictor_best.pt,
config.json, state.pt, DONE.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from experiments.mb_data import (DATA_DIR, FRAMES_PATTERN, FrameStream, kp_to_world,  # noqa: E402
                                 list_files, load_background, load_frames_file, stack_pair)
from perception.encoder import LatentPredictor, VisualEncoder  # noqa: E402

KEYPOINTS = 64
EPOCHS = 30
BATCH_SIZE = 256
LR = 1e-4
GRAD_CLIP = 1.0
EMA_TAU = 0.005
TARGET_WARMUP_EPOCHS = 1
EARLY_STOP_PATIENCE = 5
EARLY_STOP_MIN_DELTA = 1e-7
MOTION_WEIGHT_FLOOR = 0.1
HOLDOUT_EVAL_ROWS = 2000
DIAG_ROWS = 512
TIME_BUDGET_MIN = 105
SEED = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def make_models(frame_hw):
    enc = VisualEncoder(latent_dim=2 * KEYPOINTS, in_channels=6, use_simnorm=False,
                        input_hw=frame_hw, head="spatial_softmax", presence=True).to(DEVICE)
    tgt = VisualEncoder(latent_dim=2 * KEYPOINTS, in_channels=6, use_simnorm=False,
                        input_hw=frame_hw, head="spatial_softmax", presence=True).to(DEVICE)
    tgt.load_state_dict(enc.state_dict())
    for p in tgt.parameters():
        p.requires_grad = False
    pred = LatentPredictor(latent_dim=enc.out_dim, action_dim=4, hidden_dim=256).to(DEVICE)
    return enc, tgt, pred


def weighted_mse(pred, target, w):
    return (w * ((pred - target) ** 2).mean(1)).sum() / w.sum()


def batch_tensors(group, rows, bg, motion_mean):
    x_t = stack_pair(group["frames"], group["prev"][rows], group["cur"][rows], bg, DEVICE)
    x_tp1 = stack_pair(group["frames"], group["cur"][rows], group["next"][rows], bg, DEVICE)
    act = torch.from_numpy(group["actions"][rows]).to(DEVICE)
    w = torch.from_numpy(MOTION_WEIGHT_FLOOR + group["disp"][rows] / motion_mean).to(DEVICE)
    return x_t, x_tp1, act, w


@torch.no_grad()
def evaluate(enc, tgt, pred, hold, eval_rows, diag_rows, bg, motion_mean):
    enc.eval(); pred.eval()
    total, n = 0.0, 0
    for i in range(0, len(eval_rows), 128):
        r = eval_rows[i:i + 128]
        x_t, x_tp1, act, w = batch_tensors(hold, r, bg, motion_mean)
        total += len(r) * weighted_mse(pred(enc(x_t), act), tgt(x_tp1), w).item()
        n += len(r)
    zs = []
    for i in range(0, len(diag_rows), 128):
        r = diag_rows[i:i + 128]
        zs.append(enc(stack_pair(hold["frames"], hold["prev"][r], hold["cur"][r], bg, DEVICE)).cpu())
    z = torch.cat(zs).numpy().reshape(len(diag_rows), KEYPOINTS, 3)
    enc.train(); pred.train()

    kp = kp_to_world(z[..., :2])                                   # (N, K, 2) m
    slots, mask = hold["slots"][diag_rows], hold["mask"][diag_rows]  # (N, 8, 2), (N, 8)
    d = np.linalg.norm(slots[:, :, None, :] - kp[:, None, :, :], axis=-1).min(-1) * 100  # cm
    d = d[mask]
    pairwise = np.linalg.norm(z[:8, None].reshape(8, 1, -1) - z[None, :8].reshape(1, 8, -1), axis=-1)
    return dict(holdout=total / n, kp_med_cm=float(np.median(d)), kp_p90_cm=float(np.percentile(d, 90)),
                kp_far_pct=float(100 * (d > 3).mean()), p_mean=float(z[..., 2].mean()),
                p_on_pct=float(100 * (z[..., 2] > 0.5).mean()),
                latent_dist=float(pairwise[np.triu_indices(8, 1)].mean()))


def save_state(path, **state):
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)


def main():
    global KEYPOINTS
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run-dir", default=os.path.join(DATA_DIR, "runs", "encoder_mb"))
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--keypoints", type=int, default=KEYPOINTS)
    ap.add_argument("--time-budget-min", type=float, default=TIME_BUDGET_MIN)
    ap.add_argument("--max-files", type=int, default=0, help="smoke tests: use only this many files")
    args = ap.parse_args()
    KEYPOINTS = args.keypoints
    t_start = time.time()
    os.makedirs(args.run_dir, exist_ok=True)
    done_path, state_path = os.path.join(args.run_dir, "DONE"), os.path.join(args.run_dir, "state.pt")
    if os.path.exists(done_path):
        print(f"DONE already exists in {args.run_dir}; nothing to do.")
        return

    files = list_files(FRAMES_PATTERN)
    if args.max_files:
        files = files[:args.max_files]
    train_files, hold_file = files[:-1], files[-1]
    bg = torch.from_numpy(load_background()).to(DEVICE)
    hold = load_frames_file(hold_file)
    frame_hw = tuple(int(s) for s in hold["frames"].shape[1:3])
    motion_mean = float(max(hold["disp"].mean(), 1e-6))

    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    eval_rows = np.sort(rng.choice(len(hold["cur"]), min(HOLDOUT_EVAL_ROWS, len(hold["cur"])), replace=False))
    diag_rows = np.sort(rng.choice(len(hold["cur"]), min(DIAG_ROWS, len(hold["cur"])), replace=False))
    enc, tgt, pred = make_models(frame_hw)
    params = list(enc.parameters()) + list(pred.parameters())
    opt = torch.optim.Adam(params, lr=args.lr)
    epoch, best, patience, history = 0, float("inf"), 0, []

    if os.path.exists(state_path):
        s = torch.load(state_path, map_location=DEVICE, weights_only=False)
        enc.load_state_dict(s["encoder"]); tgt.load_state_dict(s["target"])
        pred.load_state_dict(s["predictor"]); opt.load_state_dict(s["optimizer"])
        epoch, best, patience, history = s["epoch"], s["best"], s["patience"], s["history"]
        rng.bit_generator.state = s["rng"]
        torch.set_rng_state(s["torch_rng"])
        print(f"RESUMED from {state_path} after epoch {epoch} (best holdout {best:.6f})")
    else:
        print(f"NEW RUN in {args.run_dir}")
    print(f"data {DATA_DIR}: {len(train_files)} train files, holdout {hold['name']}  frame {frame_hw}  "
          f"keypoints {KEYPOINTS} (latent {enc.out_dim})  device {DEVICE}  batch {args.batch_size}  lr {args.lr}")
    print(f"encoder params {sum(p.numel() for p in enc.parameters()):,}  "
          f"predictor params {sum(p.numel() for p in pred.parameters()):,}  motion mean {motion_mean:.4f} m", flush=True)
    if epoch == 0:
        e0 = evaluate(enc, tgt, pred, hold, eval_rows, diag_rows, bg, motion_mean)
        print("untrained: " + "  ".join(f"{k}={v:.4f}" for k, v in e0.items()), flush=True)

    last_epoch_s = None
    while epoch < args.epochs and patience < EARLY_STOP_PATIENCE:
        elapsed = time.time() - t_start
        if last_epoch_s and elapsed + 1.1 * last_epoch_s > 60 * args.time_budget_min:
            print(f"RESUME NEEDED: {elapsed / 60:.0f} min used, next epoch ~{last_epoch_s / 60:.0f} min "
                  f"would pass the {args.time_budget_min:.0f} min budget. Resubmit the same command.")
            return
        t0 = time.time()
        epoch += 1
        loss_sum, nb = 0.0, 0
        for group, rows in FrameStream(train_files, args.batch_size, rng):
            x_t, x_tp1, act, w = batch_tensors(group, rows, bg, motion_mean)
            with torch.no_grad():
                target = tgt(x_tp1)
            loss = weighted_mse(pred(enc(x_t), act), target, w)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP)
            opt.step()
            if epoch > TARGET_WARMUP_EPOCHS:
                with torch.no_grad():
                    for p, tp in zip(enc.parameters(), tgt.parameters()):
                        tp.lerp_(p, EMA_TAU)
            loss_sum += loss.item()
            nb += 1
        ev = evaluate(enc, tgt, pred, hold, eval_rows, diag_rows, bg, motion_mean)
        ev["train"] = loss_sum / max(nb, 1)
        history.append(dict(epoch=epoch, **ev))
        if ev["holdout"] < best - EARLY_STOP_MIN_DELTA:
            best, patience = ev["holdout"], 0
            for name, m in (("encoder", enc), ("target_encoder", tgt), ("predictor", pred)):
                torch.save(m.state_dict(), os.path.join(args.run_dir, f"{name}_best.pt"))
            tag = " *best"
        else:
            patience += 1
            tag = ""
        last_epoch_s = time.time() - t0
        collapsed = "COLLAPSED" if ev["latent_dist"] < 0.01 else "ok"
        print(f"EPOCH {epoch:3d}  train={ev['train']:.6f}  holdout={ev['holdout']:.6f}  "
              f"ball->nearest kp median {ev['kp_med_cm']:.2f} cm p90 {ev['kp_p90_cm']:.2f} cm "
              f">3cm {ev['kp_far_pct']:.1f}%  p_mean {ev['p_mean']:.2f} p>0.5 {ev['p_on_pct']:.0f}%  "
              f"[{collapsed}]  {nb} batches  {last_epoch_s / 60:.1f} min{tag}", flush=True)
        save_state(state_path, encoder=enc.state_dict(), target=tgt.state_dict(),
                   predictor=pred.state_dict(), optimizer=opt.state_dict(), epoch=epoch, best=best,
                   patience=patience, history=history, rng=rng.bit_generator.state,
                   torch_rng=torch.get_rng_state())

    config = dict(keypoints=KEYPOINTS, presence=True, latent_dim=2 * KEYPOINTS, out_dim=enc.out_dim,
                  in_channels=6, frame_hw=list(frame_hw), background="frames_background_mb.npy",
                  holdout_file=hold["name"], best_holdout=best, epochs_run=epoch,
                  stopped="early" if patience >= EARLY_STOP_PATIENCE else "max_epochs")
    with open(os.path.join(args.run_dir, "config.json"), "w") as fh:
        json.dump(config, fh, indent=1)
    open(done_path, "w").close()
    b = min(history, key=lambda h: h["holdout"])
    print(f"TRAINING DONE ({config['stopped']}) after {epoch} epochs. Best epoch {b['epoch']}: "
          f"holdout {b['holdout']:.6f}, ball->nearest kp median {b['kp_med_cm']:.2f} cm, "
          f"p90 {b['kp_p90_cm']:.2f} cm")


if __name__ == "__main__":
    main()
