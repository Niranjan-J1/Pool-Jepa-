"""Train a residual latent predictor on a FROZEN encoder.

    z_{t+1} = z_t + f(z_t, z_t - z_{t-1}, a_t)    (LatentPredictor(residual=True, use_velocity=True))

The latent velocity z_t - z_{t-1} is the latent version of the finite-difference velocity
that fixed the ground-truth dynamics probe (--no-velocity drops it). At the first step of
an episode there is no z_{t-1}, so velocity is zero there.

The encoder is frozen, so every transition is encoded once up front and training runs
on cached latents (fast, no image gradients). Input and output latents come from the
same encoder (the EMA target encoder when saved, else the online one), so "nothing
changes" is exactly the identity the residual starts from.

Evaluation on held-out episodes, against the no-change baseline ||z_t - z_{t+1}||:
  all / balls moving >5mm / static / STRIKES (fire=True and the cue ball moved >1mm)
plus a fire-sensitivity test on strikes: how much the prediction changes when the same
transition is re-predicted with fire switched off, relative to the actual change. If that
is ~0, the planner cannot learn that firing matters.

Every metric is reported twice: on the full latent, and on the CUE-BALL KEYPOINTS only
(the dims the planner's cost uses, found with planning.cem.calibrate_cue_keypoints and
weighted the same way), including decoded cue-ball positions in mm.

Outputs (in --checkpoint-dir):
    predictor_residual{suffix}.pt    the predictor
    predictor_residual{suffix}.json  which encoder file its latents came from

Usage:
    python experiments/train_predictor.py
    python experiments/train_predictor.py --suffix "" --epochs 300
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
import train_encoder as te
from perception.encoder import LatentPredictor, load_predictor
from planning import cem

STRIKE_CUE_MOVE = 1e-3  # m: a fire transition counts as a strike if the cue ball moved more than this


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--checkpoint-dir", default=te.CHECKPOINT_DIR)
    p.add_argument("--suffix", default="_best", help="encoder checkpoint suffix to train on")
    p.add_argument("--encoder", choices=["auto", "target", "online"], default="auto",
                   help="which frozen encoder provides latents (auto: target if saved)")
    p.add_argument("--head", default=te.ENCODER_HEAD, choices=["spatial_softmax", "linear"])
    p.add_argument("--data", nargs="+", default=te.DATA_PATTERNS)
    p.add_argument("--cache-dir", default=None,
                   help="frame cache / background location (default: data/datasets, as in training)")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--holdout-fraction", type=float, default=0.10,
                   help="episodes held out from PREDICTOR training (larger than the encoder's 2%% "
                        "so the strike subset has enough samples)")
    p.add_argument("--motion-weighting", action="store_true",
                   help="weight transitions by ball displacement, as in encoder training. Off by "
                        "default: in a local test it let the predictor hallucinate motion on static "
                        "frames (error 1.62 vs 1.01 unweighted) without helping on strikes")
    p.add_argument("--no-velocity", action="store_true",
                   help="drop the latent-velocity input (z_t - z_{t-1})")
    p.add_argument("--calib-scenes", type=int, default=400,
                   help="SimEnv scenes used to find the cue-ball keypoints for the cue-only metrics")
    p.add_argument("--seed", type=int, default=1)
    return p.parse_args()


def previous_latents(z_t, meta):
    """z_{t-1} for every transition: the previous row's z_t within the same episode
    (rows are ordered by episode, then step); z_t itself at an episode's first step."""
    z_prev = z_t.clone()
    later = np.where(meta["step_idx"] > 0)[0]
    z_prev[later] = z_t[later - 1]
    return z_prev


@torch.no_grad()
def encode_rows(encoder, frames, meta, prev_idx, cur_idx, batch=128):
    out = []
    for i in range(0, len(cur_idx), batch):
        x = te.stack_frames(frames, prev_idx[i:i + batch], cur_idx[i:i + batch]).to(te.DEVICE)
        out.append(encoder(x).cpu())
    return torch.cat(out)


def encode_transitions(encoder, frames, meta, rows=None):
    """z_t and z_{t+1} for every transition. z_{t+1} of step s is z_t of step s+1 in the same
    episode, so only each episode's last step needs an extra encode."""
    n = len(meta["cur"])
    rows = np.arange(n) if rows is None else rows
    z_t = encode_rows(encoder, frames, meta, meta["prev"][rows], meta["cur"][rows])
    z_next = torch.empty_like(z_t)
    pos = {r: i for i, r in enumerate(rows)}
    need = []
    for i, r in enumerate(rows):
        j = pos.get(r + 1)
        if j is not None and r + 1 < n and meta["episode"][r + 1] == meta["episode"][r]:
            z_next[i] = z_t[j]
        else:
            need.append(i)
    if need:
        need = np.array(need)
        z_next[need] = encode_rows(encoder, frames, meta, meta["cur"][rows[need]], meta["next"][rows[need]])
    return z_t, z_next


def subsets(meta, rows):
    moving = meta["ball_disp"][rows] > 5e-3
    cue_move = np.linalg.norm(meta["ball_positions_after"][rows, 0] - meta["ball_positions_before"][rows, 0], axis=-1)
    strike = (meta["actions"][rows, 2] > 0.5) & (cue_move > STRIKE_CUE_MOVE)
    return {"all": np.ones(len(rows), bool), "balls moving >5mm": moving, "static": ~moving,
            "STRIKES (fire & cue moved)": strike}


def _fire_report(label, z_t, z_next, pred_on, pred_off, sw=None):
    """Fire sensitivity and direction on strikes, optionally restricted/weighted by sw (D,)."""
    f = (lambda x: x) if sw is None else (lambda x: x * sw)
    sens = f(pred_on - pred_off).norm(dim=1)
    actual = f(z_next - z_t).norm(dim=1)
    cos = nn.functional.cosine_similarity(f(pred_on - z_t), f(z_next - z_t), dim=1)
    print(f"  [{label}] fire sensitivity on strikes: ||pred(fire) - pred(no fire)|| = {sens.mean():.3f} "
          f"vs actual change {actual.mean():.3f} ({100 * (sens / actual.clamp_min(1e-6)).mean():.0f}% "
          f"of the real effect); direction cosine {cos.mean():.2f} (1 = right, 0 = unrelated)")


@torch.no_grad()
def evaluate(name, predictor, z_in, z_next, actions, subs, z_prev=None, z_base=None,
             kp=None, cue_before=None, cue_after=None):
    """z_in: predictor inputs (z_prev: the matching previous latents); z_base: z_t in the same
    space as z_next (defaults to z_in), used for the no-change baseline and predicted change.
    kp: CueKeypoints for the cue-only metrics; cue_before/after: true cue positions (N, 2)."""
    z_t = z_in if z_base is None else z_base
    pred = predictor(z_in, actions, z_prev)
    err = (pred - z_next).norm(dim=1).numpy()
    base = (z_t - z_next).norm(dim=1).numpy()
    print(f"--- {name} ---")
    print("FULL LATENT")
    print(f"{'subset':>28} {'n':>6} {'pred err':>9} {'no-change':>10} {'ratio':>7}")
    for s, m in subs.items():
        if m.any():
            e, b = err[m].mean(), base[m].mean()
            print(f"{s:>28} {m.sum():>6} {e:>9.3f} {b:>10.3f} {b / e:>6.2f}x")

    m = subs["STRIKES (fire & cue moved)"]
    m_t = torch.from_numpy(m)
    if m.any():
        a_off = actions[m_t].clone()
        a_off[:, 2:] = 0.0  # same transition, no fire
        pred_off = predictor(z_in[m_t], a_off, None if z_prev is None else z_prev[m_t])
        _fire_report("full latent", z_t[m_t], z_next[m_t], pred[m_t], pred_off)

    if kp is not None:
        sw = torch.from_numpy(kp.dim_weights).sqrt()
        c_err = ((pred - z_next) * sw).norm(dim=1).numpy()
        c_base = ((z_t - z_next) * sw).norm(dim=1).numpy()
        dec_pred, dec_next, dec_t = (kp.decode_cue_batch(x) for x in (pred, z_next, z_t))
        mm_err = 1000 * np.linalg.norm(dec_pred - dec_next, axis=1)
        mm_base = 1000 * np.linalg.norm(dec_t - dec_next, axis=1)
        print(f"CUE-BALL KEYPOINTS ONLY ({kp.selected.sum()} keypoints, cost-weighted as in the planner; "
              f"mm = decoded cue position)")
        print(f"{'subset':>28} {'n':>6} {'pred err':>9} {'no-change':>10} {'ratio':>7} "
              f"{'err mm':>8} {'no-chg mm':>10}")
        for s, mm in subs.items():
            if mm.any():
                e, b = c_err[mm].mean(), c_base[mm].mean()
                print(f"{s:>28} {mm.sum():>6} {e:>9.3f} {b:>10.3f} {b / e:>6.2f}x "
                      f"{mm_err[mm].mean():>8.1f} {mm_base[mm].mean():>10.1f}")
        if m.any():
            _fire_report("cue keypoints", z_t[m_t], z_next[m_t], pred[m_t], pred_off, sw)
            on_mm = kp.decode_cue_batch(pred[m_t])
            off_mm = kp.decode_cue_batch(pred_off)
            sens_mm = 1000 * np.linalg.norm(on_mm - off_mm, axis=1).mean()
            latent_move_mm = 1000 * np.linalg.norm(dec_next[m] - dec_t[m], axis=1).mean()
            true_move_mm = 1000 * np.linalg.norm(cue_after[m] - cue_before[m], axis=1).mean() \
                if cue_before is not None else float("nan")
            print(f"  [cue, mm] firing moves the predicted cue ball {sens_mm:.1f} mm vs not firing; "
                  f"the encoded cue actually moves {latent_move_mm:.1f} mm (ground truth {true_move_mm:.1f} mm)")
    print()


def main():
    args = parse_args()
    te.DATA_PATTERNS, te.ENCODER_HEAD = args.data, args.head
    if args.cache_dir:
        te.CACHE_PATH = os.path.join(args.cache_dir, "frames_cache.npy")
        te.CACHE_META_PATH = os.path.join(args.cache_dir, "frames_cache_meta.json")
        te.BACKGROUND_PATH = os.path.join(args.cache_dir, "frames_background.npy")
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    files = te.list_files()
    meta, file_slots = te.load_metadata(files)
    frames = te.build_frame_cache(files, file_slots, meta["n_slots"])
    te.BACKGROUND = torch.from_numpy(te.compute_background(frames, te.file_signature(files)))
    te.FRAME_HW = tuple(int(x) for x in frames.shape[1:3])

    in_ch = 3 * te.N_FRAMES
    target_file = f"target_encoder{args.suffix}.pt"
    use_target = args.encoder == "target" or (
        args.encoder == "auto" and os.path.exists(os.path.join(args.checkpoint_dir, target_file)))
    enc_file = target_file if use_target else f"encoder{args.suffix}.pt"
    encoder = te.make_encoder(in_ch)
    encoder.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, enc_file), map_location=te.DEVICE))
    encoder.eval()
    print(f"Frozen encoder: {enc_file}  ({len(meta['cur'])} transitions)")

    t0 = time.time()
    z_t, z_next = encode_transitions(encoder, frames, meta)
    print(f"Encoded all transitions in {time.time() - t0:.0f}s")
    z_prev = previous_latents(z_t, meta)
    actions = torch.from_numpy(meta["actions"])
    use_velocity = not args.no_velocity

    episodes = np.unique(meta["episode"])
    hold_eps = rng.choice(episodes, max(1, int(len(episodes) * args.holdout_fraction)), replace=False)
    ho = np.isin(meta["episode"], hold_eps)
    tr_rows, ho_rows = np.where(~ho)[0], np.where(ho)[0]
    disp = meta["ball_disp"]
    w = np.ones(len(disp), np.float32) if not args.motion_weighting else \
        (te.MOTION_WEIGHT_FLOOR + disp / disp[tr_rows].mean()).astype(np.float32)
    w = torch.from_numpy(w)
    subs = subsets(meta, ho_rows)
    print(f"Train: {len(tr_rows)}  Holdout: {len(ho_rows)} ({len(hold_eps)} episodes, "
          f"{subs['STRIKES (fire & cue moved)'].sum()} strikes)\n")

    predictor = LatentPredictor(te.LATENT_DIM, 4, 128, residual=True,
                                use_velocity=use_velocity).to(te.DEVICE)
    print(f"Predictor: residual, velocity input {'ON' if use_velocity else 'OFF'}")
    opt = torch.optim.Adam(predictor.parameters(), lr=args.lr)
    Zt, Zn, Zp, A, W = (x.to(te.DEVICE) for x in (z_t, z_next, z_prev, actions, w))
    tr_t, ho_t = torch.from_numpy(tr_rows).to(te.DEVICE), torch.from_numpy(ho_rows).to(te.DEVICE)

    def loss_on(idx):
        return te.weighted_pred_loss(predictor(Zt[idx], A[idx], Zp[idx]), Zn[idx], W[idx])

    best, best_state, patience = float("inf"), None, 0
    for epoch in range(1, args.epochs + 1):
        predictor.train()
        perm = tr_t[torch.randperm(len(tr_t), device=te.DEVICE)]
        total, nb = 0.0, 0
        for i in range(0, len(perm), args.batch_size):
            loss = loss_on(perm[i:i + args.batch_size])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
            nb += 1
        predictor.eval()
        with torch.no_grad():
            hold = loss_on(ho_t).item()
        if epoch == 1 or epoch % 10 == 0:
            print(f"Epoch {epoch:4d}  train={total / nb:.6f}  holdout={hold:.6f}", flush=True)
        if hold < best - 1e-7:
            best, patience = hold, 0
            best_state = {k: v.clone() for k, v in predictor.state_dict().items()}
        else:
            patience += 1
            if patience >= args.patience:
                print(f"Early stopping at epoch {epoch} (best holdout {best:.6f})")
                break
    predictor.load_state_dict(best_state)
    predictor.eval().cpu()

    out = os.path.join(args.checkpoint_dir, f"predictor_residual{args.suffix}.pt")
    torch.save(predictor.state_dict(), out)
    with open(out.replace(".pt", ".json"), "w") as fh:
        json.dump({"encoder_file": enc_file, "residual": True, "use_velocity": use_velocity,
                   "data": args.data}, fh, indent=1)
    print(f"Saved {out}\n")

    # Cue-ball keypoints of this frozen encoder, found exactly as the planner finds them.
    from env.sim_env import SimEnv
    print(f"Calibrating cue keypoints on {args.calib_scenes} SimEnv scenes...")
    kp = cem.calibrate_cue_keypoints(encoder, SimEnv(), te.BACKGROUND.numpy(), te.DEVICE,
                                     n_scenes=args.calib_scenes, seed=args.seed)
    print(f"{kp.selected.sum()} of {len(kp.selected)} keypoints track the cue ball; "
          f"cue-position readout R^2 = {kp.readout_r2:.3f}\n")

    zt_h, zn_h, zp_h, a_h = z_t[ho_rows], z_next[ho_rows], z_prev[ho_rows], actions[ho_rows]
    cue_b = meta["ball_positions_before"][ho_rows, 0]
    cue_a = meta["ball_positions_after"][ho_rows, 0]
    print(f"========== Held-out one-step latent error (latents from {enc_file}) ==========\n")
    evaluate(f"NEW residual predictor (velocity {'ON' if use_velocity else 'OFF'})", predictor,
             zt_h, zn_h, a_h, subs, z_prev=zp_h, kp=kp, cue_before=cue_b, cue_after=cue_a)

    old_path = os.path.join(args.checkpoint_dir, f"predictor{args.suffix}.pt")
    if os.path.exists(old_path):
        old = load_predictor(old_path)
        # The old predictor was trained with ONLINE-encoder inputs; give it those.
        online = te.make_encoder(in_ch)
        online.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, f"encoder{args.suffix}.pt"),
                                          map_location=te.DEVICE))
        zt_online = encode_rows(online.eval(), frames, meta, meta["prev"][ho_rows], meta["cur"][ho_rows])
        evaluate(f"OLD predictor{args.suffix}.pt (online-encoder inputs; note: the old predictor's "
                 "training rows overlap this holdout)", old, zt_online, zn_h, a_h, subs, z_base=zt_h,
                 kp=kp, cue_before=cue_b, cue_after=cue_a)


if __name__ == "__main__":
    main()
