"""B2 probes: frozen multiball encoder latent -> ball positions, on-table mask, robot pose.

These are SUPERVISED readouts trained on sim labels (the encoder itself never saw labels).
Two probes on the same frozen features:
  - ridge: closed-form linear regression (positions, pose) and linear mask scores.
  - mlp:   192 -> 256 -> 256 -> outputs, masked MSE on on-table positions, BCE on the mask.
Probe training data: rows sampled from mb_frames train files (not the encoder's holdout file).
Evaluation sets (never used for training anything):
  holdout   the encoder's holdout mb_frames file (1-5 balls, robot driving, strikes)
  test      mb_shots test split, before the shot (prev, pre) and after it settles (post, post)
  novel     mb_shots novel split (6-7 balls, a ball count the encoder never saw), same views
The "post" view is where potted balls are missing, so it is the real test of the mask.

Done-criteria (B2): on-table ball error median <= 1.5 cm and p90 <= 3 cm, mask accuracy >= 99%
(checked on the mlp probe, test + novel). Robot pose error is reported with no target.

Usage: python experiments/probe_mb.py [--run-dir DIR] [--encoder target|online]
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from experiments.mb_data import (DATA_DIR, FRAMES_PATTERN, list_files, load_background,  # noqa: E402
                                 load_frames_file, stack_pair)
from perception.encoder import VisualEncoder  # noqa: E402

PROBE_FILES = 6
PROBE_ROWS_PER_FILE = 2000
EVAL_SHOTS = 2000          # per mb_shots split
HOLDOUT_ROWS = 4000
MLP_STEPS = 4000
MLP_LR = 1e-3
RIDGE_LAMBDA = 1e-3
SEED = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_encoder(run_dir, which):
    cfg = json.load(open(os.path.join(run_dir, "config.json")))
    enc = VisualEncoder(latent_dim=cfg["latent_dim"], in_channels=cfg["in_channels"], use_simnorm=False,
                        input_hw=tuple(cfg["frame_hw"]), head="spatial_softmax", presence=cfg["presence"])
    name = "target_encoder_best.pt" if which == "target" else "encoder_best.pt"
    enc.load_state_dict(torch.load(os.path.join(run_dir, name), map_location=DEVICE))
    return enc.to(DEVICE).eval(), cfg


@torch.no_grad()
def encode(enc, frames, idx_a, idx_b, bg):
    out = []
    for i in range(0, len(idx_a), 256):
        out.append(enc(stack_pair(frames, idx_a[i:i + 256], idx_b[i:i + 256], bg, DEVICE)).cpu())
    return torch.cat(out).numpy()


def labels(slots, mask, pose):
    """pose may be None (post-shot view has no saved pose)."""
    y = dict(slots=slots.astype(np.float32), mask=mask.astype(bool))
    if pose is not None:
        y["pose"] = np.stack([pose[:, 0], pose[:, 1], np.cos(pose[:, 2]), np.sin(pose[:, 2])], 1).astype(np.float32)
    return y


def frames_set(enc, path, rows, bg):
    d = load_frames_file(path)
    z = encode(enc, d["frames"], d["prev"][rows], d["cur"][rows], bg)
    return z, labels(d["slots"][rows], d["mask"][rows], d["pose"][rows])


def shots_sets(enc, pattern, n, bg, rng):
    """mb_shots files -> {'pre': (z, y), 'post': (z, y)} on up to n shots."""
    pre_z, post_z, pre_y, post_y = [], [], [], []
    taken = 0
    for f in list_files(pattern):
        if taken >= n:
            break
        d = np.load(f)
        k = min(n - taken, len(d["frame_pre"]))
        rows = np.sort(rng.choice(len(d["frame_pre"]), k, replace=False))
        fr = np.concatenate([d["frame_pre_prev"][rows], d["frame_pre"][rows], d["frame_post"][rows]])
        a, b, c = np.arange(k), np.arange(k, 2 * k), np.arange(2 * k, 3 * k)
        pre_z.append(encode(enc, fr, a, b, bg))
        post_z.append(encode(enc, fr, c, c, bg))  # settled: no motion, same frame twice
        pre_y.append(labels(d["slots_pre"][rows], d["mask_pre"][rows], d["robot_pose_pre"][rows]))
        post_y.append(labels(d["slots_post"][rows], d["mask_post"][rows], None))
        taken += k

    def cat(ys):
        return {key: np.concatenate([y[key] for y in ys]) for key in ys[0]}
    return {"pre": (np.concatenate(pre_z), cat(pre_y)), "post": (np.concatenate(post_z), cat(post_y))}


class MLP(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(),
                                 nn.Linear(256, 16 + 8 + 4))

    def forward(self, z):
        o = self.net(z)
        return o[:, :16].view(-1, 8, 2), o[:, 16:24], o[:, 24:]


def fit_mlp(z, y, mu, sd):
    torch.manual_seed(SEED)
    m = MLP(z.shape[1]).to(DEVICE)
    opt = torch.optim.Adam(m.parameters(), lr=MLP_LR)
    Z = torch.from_numpy((z - mu) / sd).to(DEVICE)
    S, M, P = (torch.from_numpy(y[k]).to(DEVICE) for k in ("slots", "mask", "pose"))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, MLP_STEPS)
    g = torch.Generator(device="cpu").manual_seed(SEED)
    for _ in range(MLP_STEPS):
        r = torch.randint(0, len(Z), (512,), generator=g).to(DEVICE)
        s, ml, p = m(Z[r])
        mk = M[r].float()
        pos = (((s - S[r]) ** 2).sum(-1) * mk).sum() / mk.sum().clamp(min=1)
        loss = 100 * pos + nn.functional.binary_cross_entropy_with_logits(ml, mk) + 10 * ((p - P[r]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    m.eval()

    def predict(zz):
        with torch.no_grad():
            s, ml, p = m(torch.from_numpy((zz - mu) / sd).to(DEVICE))
        return s.cpu().numpy(), ml.cpu().numpy() > 0, p.cpu().numpy()
    return predict


def fit_ridge(z, y, mu, sd):
    def feats(zz):
        zz = (zz - mu) / sd
        return np.concatenate([zz, np.ones((len(zz), 1), np.float32)], 1)
    F = feats(z)
    Y = np.concatenate([y["slots"].reshape(len(z), -1), y["mask"].astype(np.float32), y["pose"]], 1)
    W = np.linalg.solve(F.T @ F + RIDGE_LAMBDA * np.eye(F.shape[1]), F.T @ Y)

    def predict(zz):
        o = feats(zz) @ W
        return o[:, :16].reshape(-1, 8, 2), o[:, 16:24] > 0.5, o[:, 24:]
    return predict


def metrics(predict, z, y):
    s, mk, p = predict(z)
    on = y["mask"]
    err = np.linalg.norm(s - y["slots"], axis=-1)[on] * 100
    out = dict(med=np.median(err), p90=np.percentile(err, 90), mask_acc=100 * (mk == on).mean(), n=len(z))
    if "pose" in y:
        out["pose_cm"] = np.median(np.linalg.norm(p[:, :2] - y["pose"][:, :2], axis=1)) * 100
        dth = np.arctan2(p[:, 3], p[:, 2]) - np.arctan2(y["pose"][:, 3], y["pose"][:, 2])
        out["pose_deg"] = np.median(np.abs(np.degrees(np.arctan2(np.sin(dth), np.cos(dth)))))
    return out


def main():
    global MLP_STEPS
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run-dir", default=os.path.join(DATA_DIR, "runs", "encoder_mb"))
    ap.add_argument("--encoder", choices=["target", "online"], default="target")
    ap.add_argument("--mlp-steps", type=int, default=MLP_STEPS, help="lower for CPU smoke tests")
    args = ap.parse_args()
    MLP_STEPS = args.mlp_steps
    enc, cfg = load_encoder(args.run_dir, args.encoder)
    torch.manual_seed(SEED)
    untrained = VisualEncoder(latent_dim=cfg["latent_dim"], in_channels=cfg["in_channels"], use_simnorm=False,
                              input_hw=tuple(cfg["frame_hw"]), head="spatial_softmax",
                              presence=cfg["presence"]).to(DEVICE).eval()
    bg = torch.from_numpy(load_background()).to(DEVICE)
    print(f"encoder {args.run_dir} ({args.encoder}), latent {cfg['out_dim']}, data {DATA_DIR}", flush=True)

    # The untrained encoder (same architecture, random weights) is the baseline: background
    # subtraction + spatial softmax already localize objects, so "trained" must clearly beat it
    # before we credit self-supervised learning. Both see exactly the same rows (same seed).
    print(f"\n{'encoder':<10} {'set':<12} {'probe':<6} {'n':>6} {'ball med cm':>12} {'p90 cm':>8} "
          f"{'mask acc %':>11} {'pose cm':>8} {'pose deg':>9}")
    results = {}
    for label, e in (("trained", enc), ("untrained", untrained)):
        for key, r in probe_encoder(e, cfg, bg).items():
            results[(label,) + key] = r
            name, pname = key
            pose = f"{r['pose_cm']:8.2f} {r['pose_deg']:9.1f}" if "pose_cm" in r else f"{'-':>8} {'-':>9}"
            print(f"{label:<10} {name:<12} {pname:<6} {r['n']:>6} {r['med']:>12.2f} {r['p90']:>8.2f} "
                  f"{r['mask_acc']:>11.2f} {pose}", flush=True)

    print("\nTrained vs untrained (mlp probe, ball median cm / mask acc %):")
    for (label, name, pname), r in results.items():
        if label == "trained" and pname == "mlp":
            u = results[("untrained", name, pname)]
            print(f"  {name:<12} {r['med']:.2f} vs {u['med']:.2f} cm   {r['mask_acc']:.2f} vs {u['mask_acc']:.2f} %")

    gate = [r for (label, name, pname), r in results.items()
            if label == "trained" and pname == "mlp" and name.split("/")[0] in ("test", "novel")]
    ok_pos = all(r["med"] <= 1.5 and r["p90"] <= 3.0 for r in gate)
    ok_mask = all(r["mask_acc"] >= 99.0 for r in gate)
    print(f"\nB2 DONE-CRITERIA (trained, mlp, test+novel, pre+post): ball median<=1.5 & p90<=3 cm: "
          f"{'PASS' if ok_pos else 'FAIL'};  mask acc>=99%: {'PASS' if ok_mask else 'FAIL'}")
    print("Note: probes are supervised readouts trained on sim labels; the encoder is self-supervised.")


def probe_encoder(enc, cfg, bg):
    """Fit ridge + mlp probes on this encoder's features and evaluate them on every set.
    Returns {(set name, probe name): metrics}."""
    rng = np.random.default_rng(SEED)
    files = [f for f in list_files(FRAMES_PATTERN) if os.path.basename(f) != cfg["holdout_file"]]
    pick = np.linspace(0, len(files) - 1, min(PROBE_FILES, len(files))).astype(int)
    zs, ys = [], []
    for i in pick:
        n = len(np.load(files[i])["step_idx"])
        rows = np.sort(rng.choice(n, min(PROBE_ROWS_PER_FILE, n), replace=False))
        z, y = frames_set(enc, files[i], rows, bg)
        zs.append(z); ys.append(y)
    z_tr = np.concatenate(zs)
    y_tr = {k: np.concatenate([y[k] for y in ys]) for k in ys[0]}
    mu, sd = z_tr.mean(0), z_tr.std(0) + 1e-6
    print(f"probe train rows {len(z_tr)} from {len(pick)} files", flush=True)
    probes = {"ridge": fit_ridge(z_tr, y_tr, mu, sd), "mlp": fit_mlp(z_tr, y_tr, mu, sd)}

    hold_path = os.path.join(DATA_DIR, cfg["holdout_file"])
    n_hold = len(np.load(hold_path)["step_idx"])
    evals = {"holdout": frames_set(enc, hold_path, np.sort(rng.choice(n_hold, min(HOLDOUT_ROWS, n_hold), replace=False)), bg)}
    for split in ("test", "novel"):
        for view, zy in shots_sets(enc, f"mbshots_{split}_*.npz", EVAL_SHOTS, bg, rng).items():
            evals[f"{split}/{view}"] = zy
    return {(name, pname): metrics(pr, z, y) for name, (z, y) in evals.items() for pname, pr in probes.items()}


if __name__ == "__main__":
    main()
