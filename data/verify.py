import glob
import os

import numpy as np

datasets_dir = os.path.join(os.path.dirname(__file__), "datasets")
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
