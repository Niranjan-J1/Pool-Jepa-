"""Dataset checks.

    python data/verify.py                       # legacy: batch_*.npz in data/datasets (unchanged)
    python data/verify.py --mb "mbshots_*.npz"  # multiball shots in $POOLJEPA_DATA (or data/datasets)
    python data/verify.py --mb "mbframes_*.npz"
"""
import argparse
import glob
import os

import numpy as np

datasets_dir = os.path.join(os.path.dirname(__file__), "datasets")


def legacy():
    files = sorted(glob.glob(os.path.join(datasets_dir, "batch_*.npz")))

    print(f"{len(files)} batch files found\n")

    total_transitions = 0
    for f in files:
        d = np.load(f)
        n = d["frames_before"].shape[0]
        size_mb = os.path.getsize(f) / (1024 * 1024)
        fires = d["actions_fire"].sum()
        print(f"  {os.path.basename(f)}: {n} transitions, {fires} fires, {size_mb:.1f} MB")
        total_transitions += n

    print(f"\nTotal: {total_transitions} transitions")

    d = np.load(files[0])
    print(f"Frame dtype: {d['frames_before'].dtype}, shape: {d['frames_before'][0].shape}")
    print(f"Action dtypes: left_pwm={d['actions_left_pwm'].dtype}, fire={d['actions_fire'].dtype}")


def _slot_errors(slots, mask, n_obj):
    """Count violations of the fixed-slot contract for one (N, 8, 2) / (N, 8) pair."""
    from env.sim_env import TABLE_LENGTH, TABLE_WIDTH
    bad = 0
    bad += int((slots[~mask] != 0).any(axis=-1).sum())                     # absent slots are (0, 0)
    ids = np.arange(mask.shape[1])
    bad += int((mask & (ids[None, :] > n_obj[:, None])).sum())             # no ball beyond n
    on = slots[mask]
    bad += int(((np.abs(on[:, 0]) > TABLE_LENGTH / 2) | (np.abs(on[:, 1]) > TABLE_WIDTH / 2)).sum())
    bad += int((~mask[:, 0]).sum())                                         # cue always on table
    return bad


def mb(pattern):
    root = os.environ.get("POOLJEPA_DATA", datasets_dir)
    files = sorted(glob.glob(os.path.join(root, pattern)))
    if not files:
        raise SystemExit(f"no files match {pattern} in {root}")
    d0 = np.load(files[0])
    if "frame_post" in d0:
        return mb_shots(files)
    return mb_frames(files)


def mb_shots(files):
    cols = ["mask_pre", "mask_post", "slots_pre", "slots_post", "potted_pockets", "scratch",
            "strike_connected", "n_object_balls", "aim_kind", "settle_steps", "settled",
            "scene_seed", "frame_post"]
    data = {c: [] for c in cols}
    for f in files:
        d = np.load(f)
        for c in cols:
            data[c].append(d[c] if c != "frame_post" else np.array(d[c].shape[1:])[None])
    a = {c: np.concatenate(v) for c, v in data.items()}
    n = len(a["scratch"])
    obj_pot = (a["potted_pockets"][:, 1:] >= 0)
    # Consistency: a potted object ball was on the table before and is gone after, and only then.
    gone = a["mask_pre"] & ~a["mask_post"]
    incons = int((gone[:, 1:] != obj_pot).sum())
    incons += int((a["mask_post"] & ~a["mask_pre"]).sum())                  # nothing appears
    incons += int(((a["potted_pockets"][:, 0] >= 0) != a["scratch"]).sum())  # cue pocket <=> scratch
    incons += int((a["mask_pre"].sum(1) != a["n_object_balls"] + 1).sum())  # all balls start on table
    incons += _slot_errors(a["slots_pre"], a["mask_pre"], a["n_object_balls"])
    incons += _slot_errors(a["slots_post"], a["mask_post"], a["n_object_balls"])
    dup = n - len(np.unique(a["scene_seed"]))
    ss = a["settle_steps"]
    print(f"{len(files)} files, {n} shots, frame shape {tuple(a['frame_post'][0])}")
    print(f"  object balls per shot: " + ", ".join(
        f"{k}:{int((a['n_object_balls'] == k).sum())}" for k in np.unique(a["n_object_balls"])))
    print(f"  aim kinds ghost/ball/uniform: " + "/".join(
        str(int((a['aim_kind'] == k).sum())) for k in range(3)))
    print(f"  pot >= 1 object ball: {100 * obj_pot.any(1).mean():.1f}%   "
          f"balls potted per shot: {obj_pot.sum(1).mean():.3f}   scratch: {100 * a['scratch'].mean():.1f}%   "
          f"strike connected: {100 * a['strike_connected'].mean():.1f}%")
    print(f"  settle steps p50/p95/max: {int(np.median(ss))}/{int(np.percentile(ss, 95))}/{int(ss.max())}"
          f"   unsettled: {int((~a['settled']).sum())}")
    print(f"  INCONSISTENCIES: {incons}   duplicate scene seeds: {dup}")


def mb_frames(files):
    n = fires = conn = incons = 0
    for f in files:
        d = np.load(f)
        n += len(d["actions_fire"])
        fires += int(d["actions_fire"].sum())
        conn += int(d["strike_connected"].sum())
        incons += int((d["mask_after"] & ~d["mask_before"]).sum())
        for s, m in (("slots_before", "mask_before"), ("slots_after", "mask_after")):
            incons += _slot_errors(d[s], d[m], d["n_object_balls"].astype(int))
    print(f"{len(files)} files, {n} frame pairs, {fires} strikes ({conn} connected)")
    print(f"  INCONSISTENCIES: {incons}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mb", default=None, help="glob for multiball files, e.g. 'mbshots_*.npz'")
    args = p.parse_args()
    if args.mb:
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        mb(args.mb)
    else:
        legacy()
