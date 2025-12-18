# %%
"""Interactive trajectory visualization with timestamp snapshots.

Adjust detection parameters and visualize trajectories at different time points.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from particle_tracking import (
    batch_detect,
    filter_trajectories,
    link_trajectories,
    load_nd2_file,
)

# %%
# =============================================================================
# CONFIGURATION - Adjust these parameters
# =============================================================================

DATA_DIR = Path(__file__).parent.parent / "20251218test-nofixture2"
OUTPUT_DIR = Path(__file__).parent.parent / "output"
SAMPLE_FILE = DATA_DIR / "nd000.nd2"

# Detection parameters - ADJUST THESE
DIAMETER = 21  # Particle diameter (must be odd, larger for defocused particles)
MINMASS = 500  # Minimum brightness
SEPARATION = 25  # Minimum separation between particles

# Tracking parameters
SEARCH_RANGE = 15  # Max displacement per frame (pixels)
MEMORY = 3  # Frames a particle can disappear
MIN_TRAJ_LENGTH = 20  # Minimum trajectory length

# Visualization parameters
TIMESTAMPS = np.linspace(0, 199, 200, dtype=int)  # Frames to visualize

# %%
# Load data
print(f"Loading {SAMPLE_FILE}...")
frames, metadata = load_nd2_file(SAMPLE_FILE)
print(f"Loaded {len(frames)} frames, shape: {frames[0].shape}")
print(f"Pixel size: {metadata['pixel_size']} um")
print(f"Frame interval: {metadata['frame_interval'] * 1000:.2f} ms")

# %%
# Detect particles
print(f"\n=== Detection ===")
print(f"Parameters: diameter={DIAMETER}, minmass={MINMASS}, separation={SEPARATION}")

all_particles = batch_detect(
    frames,
    diameter=DIAMETER,
    minmass=MINMASS,
    separation=SEPARATION,
    show_progress=True,
)
print(f"Total detections: {len(all_particles)}")
print(f"Detections per frame: {len(all_particles) / len(frames):.1f}")

# %%
# Generate detection view for all frames
print(f"\n=== Generating Detection Views ===")

for t in TIMESTAMPS:
    fig, ax = plt.subplots(figsize=(16, 5))

    # Show frame
    ax.imshow(frames[t], cmap="gray", vmin=0, vmax=frames[t].max() * 0.3, aspect="auto")

    # Get detections at this frame
    frame_particles = all_particles[all_particles["frame"] == t]

    # Plot detections
    if len(frame_particles) > 0:
        ax.scatter(
            frame_particles["x"], frame_particles["y"],
            s=100, facecolors="none", edgecolors="red", linewidths=1.5
        )

    time_ms = t * metadata["frame_interval"] * 1000
    ax.set_title(f"Frame {t} (t={time_ms:.0f}ms) - {len(frame_particles)} detections")
    ax.axis("off")

    plt.tight_layout()
    output_path = OUTPUT_DIR / f"figures/nd000_detection_frame_{t:03d}.png"
    plt.savefig(output_path, dpi=100)
    plt.show()
    print(f"Saved: {output_path}")

# %%
# Link trajectories
print(f"\n=== Linking ===")
print(f"Parameters: search_range={SEARCH_RANGE}, memory={MEMORY}")

trajectories = link_trajectories(
    all_particles,
    search_range=SEARCH_RANGE,
    memory=MEMORY,
)
n_raw = trajectories["particle"].nunique()
print(f"Linked trajectories: {n_raw}")

# %%
# Filter short trajectories
print(f"\n=== Filtering ===")
filtered = filter_trajectories(trajectories, min_length=MIN_TRAJ_LENGTH)
n_filtered = filtered["particle"].nunique()
print(f"After filtering (min_length={MIN_TRAJ_LENGTH}): {n_filtered} trajectories")
print(f"Removed: {n_raw - n_filtered} short trajectories")

# %%
# Trajectory statistics
print(f"\n=== Trajectory Statistics ===")
traj_lengths = filtered.groupby("particle").size()
print(f"Mean length: {traj_lengths.mean():.1f} frames")
print(f"Min length: {traj_lengths.min()} frames")
print(f"Max length: {traj_lengths.max()} frames")

# %%
# Visualize trajectories at different timestamps (separate files)
print(f"\n=== Generating Trajectory Visualizations ===")

for t in TIMESTAMPS:
    fig, ax = plt.subplots(figsize=(16, 5))

    # Show frame
    ax.imshow(frames[t], cmap="gray", vmin=0, vmax=frames[t].max() * 0.3, aspect="auto")

    # Get trajectories up to this frame
    traj_up_to_t = filtered[filtered["frame"] <= t]

    # Plot each trajectory
    for particle_id in traj_up_to_t["particle"].unique():
        traj = traj_up_to_t[traj_up_to_t["particle"] == particle_id]
        ax.plot(traj["x"], traj["y"], linewidth=1.5, alpha=0.8)

        # Mark current position
        current = traj[traj["frame"] == t]
        if len(current) > 0:
            ax.scatter(current["x"], current["y"], s=30, c="red", zorder=5)

    time_ms = t * metadata["frame_interval"] * 1000
    ax.set_title(f"Frame {t} (t={time_ms:.0f}ms) - {len(traj_up_to_t['particle'].unique())} trajectories")
    ax.axis("off")

    plt.tight_layout()
    output_path = OUTPUT_DIR / f"figures/nd000_traj_frame_{t:03d}.png"
    plt.savefig(output_path, dpi=100)
    plt.show()
    print(f"Saved: {output_path}")

# %%
# Single frame detailed view - useful for checking detection quality
DETAIL_FRAME = 100  # Change this to inspect different frames

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Left: Frame with current detections
frame_particles = all_particles[all_particles["frame"] == DETAIL_FRAME]
axes[0].imshow(frames[DETAIL_FRAME], cmap="gray", vmin=0, vmax=frames[DETAIL_FRAME].max() * 0.3, aspect="auto")
if len(frame_particles) > 0:
    axes[0].scatter(frame_particles["x"], frame_particles["y"],
                    s=100, facecolors="none", edgecolors="red", linewidths=1.5)
axes[0].set_title(f"Frame {DETAIL_FRAME}: {len(frame_particles)} detections")
axes[0].axis("off")

# Right: Trajectories passing through this frame
traj_at_frame = filtered[filtered["frame"] == DETAIL_FRAME]["particle"].unique()
axes[1].imshow(frames[DETAIL_FRAME], cmap="gray", vmin=0, vmax=frames[DETAIL_FRAME].max() * 0.3, aspect="auto")

for particle_id in traj_at_frame:
    traj = filtered[filtered["particle"] == particle_id]
    axes[1].plot(traj["x"], traj["y"], linewidth=2, alpha=0.8)
    current = traj[traj["frame"] == DETAIL_FRAME]
    if len(current) > 0:
        axes[1].scatter(current["x"], current["y"], s=50, c="red", zorder=5)

axes[1].set_title(f"Frame {DETAIL_FRAME}: {len(traj_at_frame)} trajectories")
axes[1].axis("off")

plt.tight_layout()
plt.show()

# %%
# Zoom into a specific region - useful for checking detection quality
ZOOM_X = 1600  # Center X
ZOOM_Y = 290   # Center Y
ZOOM_SIZE = 200  # Region size

x_start = max(0, ZOOM_X - ZOOM_SIZE // 2)
x_end = min(frames[0].shape[1], ZOOM_X + ZOOM_SIZE // 2)
y_start = max(0, ZOOM_Y - ZOOM_SIZE // 2)
y_end = min(frames[0].shape[0], ZOOM_Y + ZOOM_SIZE // 2)

fig, axes = plt.subplots(1, len(TIMESTAMPS), figsize=(4 * len(TIMESTAMPS), 4))

for ax, t in zip(axes, TIMESTAMPS):
    region = frames[t][y_start:y_end, x_start:x_end]
    ax.imshow(region, cmap="gray", vmin=0, vmax=region.max() * 0.5)

    # Get particles in this region at this frame
    traj_up_to_t = filtered[filtered["frame"] <= t]

    for particle_id in traj_up_to_t["particle"].unique():
        traj = traj_up_to_t[traj_up_to_t["particle"] == particle_id]
        # Filter to region
        traj_in_region = traj[
            (traj["x"] >= x_start) & (traj["x"] < x_end) &
            (traj["y"] >= y_start) & (traj["y"] < y_end)
        ]
        if len(traj_in_region) > 0:
            ax.plot(traj_in_region["x"] - x_start, traj_in_region["y"] - y_start,
                    linewidth=2, alpha=0.8)
            current = traj_in_region[traj_in_region["frame"] == t]
            if len(current) > 0:
                ax.scatter(current["x"] - x_start, current["y"] - y_start,
                          s=50, c="red", zorder=5)

    ax.set_title(f"Frame {t}")
    ax.axis("off")

plt.suptitle(f"Zoomed Region ({ZOOM_X}, {ZOOM_Y}) ± {ZOOM_SIZE//2}px")
plt.tight_layout()
plt.show()

# %%
print("\n=== Done ===")
print("Adjust parameters in the CONFIGURATION section and re-run cells to tune detection.")
