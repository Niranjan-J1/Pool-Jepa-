"""Multiball data helpers shared by B2 scripts (train_encoder_mb.py, probe_mb.py).

- Files live in $POOLJEPA_DATA (else data/datasets), written by collect.py mb_* modes.
- Background: per-pixel median over frames sampled across mb_frames files, saved as
  frames_background_mb.npy (the legacy frames_background.npy is never touched). The median
  is the same method a real camera will use; check_background() compares it with an exact
  empty-table render, which only the sim can give.
- FrameStream: reads compressed mb_frames files directly (no giant cache), loading
  BUFFER_FILES files at a time in a background thread and shuffling transitions within them.
- Keypoint coordinates in [-1, 1] map to table metres with kp_to_world().

CLI:  python experiments/mb_data.py background      # build + check the multiball background
"""
import glob
import os
import queue
import sys
import threading

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from env.sim_env import (RENDER_HEIGHT, RENDER_MARGIN, RENDER_WIDTH, TABLE_LENGTH,  # noqa: E402
                         SimEnv)

DATA_DIR = os.environ.get("POOLJEPA_DATA",
                          os.path.join(os.path.dirname(__file__), "..", "data", "datasets"))
BACKGROUND_PATH = os.path.join(DATA_DIR, "frames_background_mb.npy")
FRAMES_PATTERN = "mbframes_train_*.npz"
BACKGROUND_FILES = 10        # files sampled for the median
BACKGROUND_PER_FILE = 50     # frames per sampled file (500 total)
BACKGROUND_TOL = 10          # a pixel "differs" if any channel is off by more than this (0-255)
BUFFER_FILES = 3

HALF_W = TABLE_LENGTH / 2 + RENDER_MARGIN
HALF_H = HALF_W * RENDER_HEIGHT / RENDER_WIDTH


def list_files(pattern):
    files = sorted(glob.glob(os.path.join(DATA_DIR, pattern)))
    if not files:
        raise FileNotFoundError(f"No files matching {pattern} in {DATA_DIR}")
    return files


def kp_to_world(kp):
    """(..., 2) keypoint (x, y) in [-1, 1] image coords -> (..., 2) table metres.
    Image rows grow downward, world y grows upward. Approximate to about half a feature cell
    (the softmax grid's endpoints are cell centres), fine for diagnostics; probes learn the map."""
    return np.stack([kp[..., 0] * HALF_W, -kp[..., 1] * HALF_H], -1)


# ---------------------------------------------------------------- background
def exact_empty_table():
    """Render the multiball table with no balls and the robot moved off-canvas."""
    env = SimEnv(n_object_balls=1)
    env.reset(seed=0)
    env._ball_bodies = {}
    env._robot_body.position = (100.0, 100.0)
    return env.render()


def build_background():
    files = list_files(FRAMES_PATTERN)
    pick = np.linspace(0, len(files) - 1, min(BACKGROUND_FILES, len(files))).astype(int)
    frames = []
    for i in pick:
        fb = np.load(files[i])["frames_before"]
        rows = np.linspace(0, len(fb) - 1, BACKGROUND_PER_FILE).astype(int)
        frames.append(fb[rows])
        print(f"  sampled {os.path.basename(files[i])}", flush=True)
    frames = np.concatenate(frames)
    bg = np.median(frames, axis=0).astype(np.float32)
    np.save(BACKGROUND_PATH, bg)
    print(f"Saved {BACKGROUND_PATH} from {len(frames)} frames")
    return bg


def check_background(bg):
    """Fraction of pixels where the median background differs from the exact empty table."""
    exact = exact_empty_table().astype(np.float32)
    diff = np.abs(bg - exact).max(-1)
    return float((diff > BACKGROUND_TOL).mean()), float(diff.max())


def load_background():
    if not os.path.exists(BACKGROUND_PATH):
        raise FileNotFoundError(f"{BACKGROUND_PATH} missing; run: python experiments/mb_data.py background")
    return np.load(BACKGROUND_PATH)


# ---------------------------------------------------------------- frame stream
def load_frames_file(path):
    """One mb_frames file -> dict with a frame-slot array and per-transition indices.

    Each episode of n steps owns n+1 frame slots (its n frames_before, then the last
    frames_after). Transition s uses prev = slot s-1 (s at s=0), cur = s, next = s+1."""
    d = np.load(path)
    ep, step = d["episode_idx"], d["step_idx"]
    if not np.array_equal(np.lexsort((step, ep)), np.arange(len(ep))):
        raise ValueError(f"{path} is not sorted by (episode, step)")
    fb, fa = d["frames_before"], d["frames_after"]
    eps, starts, counts = np.unique(ep, return_index=True, return_counts=True)
    frames = np.empty((len(fb) + len(eps), *fb.shape[1:]), np.uint8)
    prev, cur, nxt = (np.empty(len(fb), np.int64) for _ in range(3))
    base = 0
    for st, n in zip(starts, counts):
        frames[base:base + n] = fb[st:st + n]
        frames[base + n] = fa[st + n - 1]
        s = np.arange(n)
        cur[st:st + n] = base + s
        prev[st:st + n] = base + np.maximum(s - 1, 0)
        nxt[st:st + n] = base + s + 1
        base += n + 1
    del fa
    keep = d["mask_before"] & d["mask_after"]
    disp = (np.linalg.norm(d["slots_after"] - d["slots_before"], axis=-1) * keep).sum(-1)
    actions = np.stack([d["actions_left_pwm"], d["actions_right_pwm"],
                        d["actions_fire"].astype(np.float32), d["actions_fire_force"]], 1)
    return dict(frames=frames, prev=prev, cur=cur, next=nxt, actions=actions.astype(np.float32),
                disp=disp.astype(np.float32), slots=d["slots_before"].astype(np.float32),
                mask=d["mask_before"], pose=d["robot_pose_before"].astype(np.float32),
                name=os.path.basename(path))


def stack_pair(frames, idx_a, idx_b, background, device):
    """uint8 frames -> float (B, 6, H, W) on device: background-subtracted [a | b] in [-1, 1].
    Conversion happens on the device, so only uint8 crosses the bus."""
    a = torch.from_numpy(frames[idx_a]).to(device).float()
    b = torch.from_numpy(frames[idx_b]).to(device).float()
    x = torch.cat([a - background, b - background], dim=-1)
    return x.permute(0, 3, 1, 2).div_(255.0)


class FrameStream:
    """Iterate shuffled batches over a list of mb_frames files.

    Files are taken in a random order in groups of BUFFER_FILES; each group's transitions are
    shuffled together. The next group loads in a background thread while this one trains.
    Yields (group, rows) where rows indexes the group's concatenated transitions."""

    def __init__(self, files, batch_size, rng, buffer_files=BUFFER_FILES):
        self.files, self.batch_size, self.rng, self.buffer_files = files, batch_size, rng, buffer_files

    @staticmethod
    def _merge(parts):
        out, offset = {}, 0
        for k in ("prev", "cur", "next"):
            out[k] = []
        for p in parts:
            for k in ("prev", "cur", "next"):
                out[k].append(p[k] + offset)
            offset += len(p["frames"])
        merged = {k: np.concatenate(v) for k, v in out.items()}
        merged["frames"] = np.concatenate([p["frames"] for p in parts])
        for k in ("actions", "disp", "slots", "mask", "pose"):
            merged[k] = np.concatenate([p[k] for p in parts])
        return merged

    def __iter__(self):
        order = self.rng.permutation(len(self.files))
        groups = [order[i:i + self.buffer_files] for i in range(0, len(order), self.buffer_files)]
        q = queue.Queue(maxsize=1)

        def loader():
            for g in groups:
                q.put(self._merge([load_frames_file(self.files[i]) for i in g]))
            q.put(None)

        threading.Thread(target=loader, daemon=True).start()
        while True:
            group = q.get()
            if group is None:
                return
            perm = self.rng.permutation(len(group["cur"]))
            for i in range(0, len(perm) - self.batch_size + 1, self.batch_size):
                yield group, perm[i:i + self.batch_size]


if __name__ == "__main__":
    if sys.argv[1:] != ["background"]:
        sys.exit("usage: python experiments/mb_data.py background")
    bg = build_background()
    frac, worst = check_background(bg)
    verdict = "OK" if frac <= 0.01 else "FAIL"
    print(f"BACKGROUND {verdict}: {100 * frac:.2f}% of pixels differ from the exact empty table "
          f"by >{BACKGROUND_TOL} (target <=1%), worst {worst:.0f}")
