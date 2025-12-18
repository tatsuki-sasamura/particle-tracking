# %%
"""Full particle tracking analysis pipeline."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import trackpy as tp

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from particle_tracking import (
    add_time_column,
    batch_detect,
    compute_ensemble_stats,
    compute_trajectory_stats,
    compute_velocities,
    export_results,
    filter_trajectories,
    link_trajectories,
    load_nd2_file,
)

# %%
# Configuration
DATA_DIR = Path(__file__).parent.parent / "20251218test-nofixture2"
OUTPUT_DIR = Path(__file__).parent.parent / "output"
SAMPLE_FILE = DATA_DIR / "nd000.nd2"

# Detection parameters (tuned in 02_particle_detection.py)
# Note: Large diameter/separation for defocused particles (ring + center pattern)
DIAMETER = 21
MINMASS = 500
SEPARATION = 25

# Tracking parameters
SEARCH_RANGE = 15  # max displacement per frame (pixels)
MEMORY = 3  # frames a particle can disappear
MIN_TRAJ_LENGTH = 10  # minimum trajectory length

# Physical parameters
PIXEL_SIZE = 0.65  # um/pixel
FRAME_INTERVAL = 0.007  # seconds

# %%
# Load data
print(f"Loading {SAMPLE_FILE}...")
frames, metadata = load_nd2_file(SAMPLE_FILE)
print(f"Loaded {len(frames)} frames")
print(f"Pixel size: {metadata['pixel_size']} um")
print(f"Frame interval: {metadata['frame_interval'] * 1000:.2f} ms")

# %%
# Batch detection
print("\n=== Particle Detection ===")
print(f"Parameters: diameter={DIAMETER}, minmass={MINMASS}")
all_particles = batch_detect(
    frames,
    diameter=DIAMETER,
    minmass=MINMASS,
    separation=SEPARATION,
    show_progress=True,
)
print(f"Total detections: {len(all_particles)}")
print(f"Unique frames: {all_particles['frame'].nunique()}")

# %%
# Link trajectories
print("\n=== Trajectory Linking ===")
print(f"Parameters: search_range={SEARCH_RANGE}, memory={MEMORY}")
trajectories = link_trajectories(
    all_particles,
    search_range=SEARCH_RANGE,
    memory=MEMORY,
)
n_traj_raw = trajectories["particle"].nunique()
print(f"Linked {n_traj_raw} trajectories")

# %%
# Filter short trajectories
print(f"\n=== Filtering (min_length={MIN_TRAJ_LENGTH}) ===")
filtered = filter_trajectories(trajectories, min_length=MIN_TRAJ_LENGTH)
n_traj_filtered = filtered["particle"].nunique()
print(f"After filtering: {n_traj_filtered} trajectories")
print(f"Removed: {n_traj_raw - n_traj_filtered} short trajectories")

# %%
# Visualize trajectories
print("\n=== Trajectory Visualization ===")
fig, ax = plt.subplots(figsize=(16, 5))
tp.plot_traj(filtered, ax=ax)
ax.set_xlabel("X (pixels)")
ax.set_ylabel("Y (pixels)")
ax.set_title(f"Particle Trajectories (n={n_traj_filtered})")
ax.set_aspect("equal")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "figures/nd000_trajectories.png", dpi=150)
plt.show()
print("Saved: output/figures/nd000_trajectories.png")

# %%
# Overlay trajectories on first frame
fig, ax = plt.subplots(figsize=(16, 5))
ax.imshow(frames[0], cmap="gray", vmin=0, vmax=frames[0].max() * 0.3)
tp.plot_traj(filtered, ax=ax, colorby="particle", cmap="tab20")
ax.set_title(f"Trajectories overlaid on Frame 0 (n={n_traj_filtered})")
ax.axis("off")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "figures/nd000_traj_overlay.png", dpi=150)
plt.show()
print("Saved: output/figures/nd000_traj_overlay.png")

# %%
# Compute velocities
print("\n=== Velocity Analysis ===")
velocities = compute_velocities(
    filtered,
    pixel_size=PIXEL_SIZE,
    dt=FRAME_INTERVAL,
)
velocities = add_time_column(velocities, dt=FRAME_INTERVAL)

valid_speeds = velocities["speed"].dropna()
print(f"Velocity records: {len(velocities)}")
print(f"Mean speed: {valid_speeds.mean():.2f} um/s")
print(f"Median speed: {valid_speeds.median():.2f} um/s")
print(f"Max speed: {valid_speeds.max():.2f} um/s")
print(f"Std speed: {valid_speeds.std():.2f} um/s")

# %%
# Velocity histogram
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].hist(valid_speeds, bins=50, edgecolor="black", alpha=0.7)
axes[0].set_xlabel("Speed (um/s)")
axes[0].set_ylabel("Count")
axes[0].set_title("Speed Distribution")
axes[0].axvline(valid_speeds.mean(), color="red", linestyle="--", label=f"Mean: {valid_speeds.mean():.1f}")
axes[0].axvline(valid_speeds.median(), color="green", linestyle="--", label=f"Median: {valid_speeds.median():.1f}")
axes[0].legend()

# Direction histogram
directions = velocities["direction"].dropna()
axes[1].hist(directions, bins=36, edgecolor="black", alpha=0.7)
axes[1].set_xlabel("Direction (degrees)")
axes[1].set_ylabel("Count")
axes[1].set_title("Direction Distribution")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "figures/nd000_velocity_histogram.png", dpi=150)
plt.show()
print("Saved: output/figures/nd000_velocity_histogram.png")

# %%
# Trajectory statistics
print("\n=== Trajectory Statistics ===")
traj_stats = compute_trajectory_stats(velocities)
print(f"Trajectories analyzed: {len(traj_stats)}")
print(f"Mean trajectory length: {traj_stats['n_frames'].mean():.1f} frames")
print(f"Mean net displacement: {traj_stats['net_displacement'].mean():.2f} um")

# %%
# Export results
print("\n=== Exporting Results ===")
file_stem = SAMPLE_FILE.stem

# Add source file column
filtered["source_file"] = file_stem
velocities["source_file"] = file_stem

# Export
paths = export_results(
    filtered,
    velocities,
    OUTPUT_DIR / "trajectories",
    prefix=file_stem,
)
for name, path in paths.items():
    print(f"Saved: {path}")

# Export trajectory stats
traj_stats["source_file"] = file_stem
traj_stats.to_csv(OUTPUT_DIR / "trajectories" / f"{file_stem}_traj_stats.csv", index=False)
print(f"Saved: output/trajectories/{file_stem}_traj_stats.csv")

# %%
# Ensemble statistics
ensemble_stats = compute_ensemble_stats(velocities, source_file=file_stem)
print("\n=== Ensemble Statistics ===")
for key, value in ensemble_stats.items():
    if isinstance(value, float):
        print(f"  {key}: {value:.2f}")
    else:
        print(f"  {key}: {value}")

# %%
# Verify output files exist
print("\n=== Checkpoint 7 Verification ===")
expected_files = [
    OUTPUT_DIR / "trajectories" / f"{file_stem}_trajectories.csv",
    OUTPUT_DIR / "trajectories" / f"{file_stem}_velocities.csv",
    OUTPUT_DIR / "figures" / "nd000_trajectories.png",
]

all_exist = True
for f in expected_files:
    exists = f.exists()
    status = "OK" if exists else "MISSING"
    print(f"  [{status}] {f.name}")
    all_exist &= exists

if all_exist:
    print("\nCheckpoint 7 PASSED!")
else:
    print("\nCheckpoint 7 FAILED - some files missing")

# %%
print("\n=== Analysis Complete ===")
print("Next: Run 04_batch_processing.py to process all files")
