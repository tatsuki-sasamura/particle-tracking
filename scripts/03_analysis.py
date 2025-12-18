# %%
"""Particle tracking analysis pipeline.

This script performs:
1. Particle detection (threshold or trackpy method)
2. Trajectory linking
3. Trajectory filtering
4. Velocity analysis
5. Visualization (optional)
6. Export results

Configure parameters in config.py before running.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

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
    preprocess_frame,
)

from config import (
    DATA_DIR,
    FRAME_INTERVAL,
    GENERATE_VIDEO_FRAMES,
    MEMORY,
    METHOD,
    MIN_TRAJ_LENGTH,
    OUTPUT_DIR,
    PIXEL_SIZE,
    SEARCH_RANGE,
    VIDEO_FRAME_RANGE,
    VIS_FRAMES,
    get_detect_kwargs,
    get_output_dir,
)

# %%
# =============================================================================
# File Selection
# =============================================================================

ND_FILE_NUMBER = 0  # Change this to process different files

SAMPLE_FILE = DATA_DIR / f"nd{ND_FILE_NUMBER:03d}.nd2"
OUT_DIR = get_output_dir(__file__)

# %%
# =============================================================================
# Load Data
# =============================================================================

print(f"Loading {SAMPLE_FILE}...")
frames, metadata = load_nd2_file(SAMPLE_FILE)
file_stem = SAMPLE_FILE.stem

print(f"Loaded {len(frames)} frames, shape: {frames[0].shape}")
print(f"Pixel size: {metadata['pixel_size']} um")
print(f"Frame interval: {metadata['frame_interval'] * 1000:.2f} ms")

# Use metadata values if available
pixel_size = metadata.get("pixel_size", PIXEL_SIZE)
frame_interval = metadata.get("frame_interval", FRAME_INTERVAL)

# %%
# =============================================================================
# Particle Detection
# =============================================================================

print(f"\n=== Particle Detection (method={METHOD}) ===")
detect_kwargs = get_detect_kwargs()
print(f"Parameters: {detect_kwargs}")

all_particles = batch_detect(
    frames,
    method=METHOD,
    show_progress=True,
    **detect_kwargs,
)

print(f"Total detections: {len(all_particles)}")
print(f"Detections per frame: {len(all_particles) / len(frames):.1f}")

# %%
# =============================================================================
# Trajectory Linking
# =============================================================================

print(f"\n=== Trajectory Linking ===")
print(f"Parameters: search_range={SEARCH_RANGE}, memory={MEMORY}")

trajectories = link_trajectories(
    all_particles,
    search_range=SEARCH_RANGE,
    memory=MEMORY,
)
n_raw = trajectories["particle"].nunique()
print(f"Linked trajectories: {n_raw}")

# %%
# =============================================================================
# Trajectory Filtering
# =============================================================================

print(f"\n=== Filtering (min_length={MIN_TRAJ_LENGTH}) ===")
filtered = filter_trajectories(trajectories, min_length=MIN_TRAJ_LENGTH)
n_filtered = filtered["particle"].nunique()
print(f"After filtering: {n_filtered} trajectories")
print(f"Removed: {n_raw - n_filtered} short trajectories")

# Trajectory length statistics
traj_lengths = filtered.groupby("particle").size()
print(f"Mean length: {traj_lengths.mean():.1f} frames")
print(f"Max length: {traj_lengths.max()} frames")

# %%
# =============================================================================
# Velocity Analysis
# =============================================================================

print(f"\n=== Velocity Analysis ===")
velocities = compute_velocities(
    filtered,
    pixel_size=pixel_size,
    dt=frame_interval,
)
velocities = add_time_column(velocities, dt=frame_interval)

valid_speeds = velocities["speed"].dropna()
print(f"Mean speed: {valid_speeds.mean():.2f} um/s")
print(f"Median speed: {valid_speeds.median():.2f} um/s")
print(f"Std speed: {valid_speeds.std():.2f} um/s")

# %%
# =============================================================================
# Visualization: Detection at Selected Frames
# =============================================================================

if VIS_FRAMES:
    print(f"\n=== Generating Detection Views ===")

    for t in VIS_FRAMES:
        if t >= len(frames):
            continue

        fig, ax = plt.subplots(figsize=(20, 4))

        # Show preprocessed frame
        proc = preprocess_frame(frames[t])
        ax.imshow(proc, cmap="gray")

        # Get detections at this frame
        frame_particles = all_particles[all_particles["frame"] == t]

        if len(frame_particles) > 0:
            ax.scatter(
                frame_particles["x"], frame_particles["y"],
                s=100, facecolors="none", edgecolors="lime", linewidths=1.5
            )

        time_ms = t * frame_interval * 1000
        ax.set_title(f"Frame {t} (t={time_ms:.0f}ms) - {len(frame_particles)} detections")
        ax.axis("off")

        plt.tight_layout()
        output_path = OUT_DIR / f"{file_stem}_detection_{t:03d}.png"
        plt.savefig(output_path, dpi=100)
        plt.show()
        print(f"Saved: {output_path}")

# %%
# =============================================================================
# Visualization: Trajectories at Selected Frames
# =============================================================================

if VIS_FRAMES:
    print(f"\n=== Generating Trajectory Views ===")

    for t in VIS_FRAMES:
        if t >= len(frames):
            continue

        fig, ax = plt.subplots(figsize=(20, 4))
        ax.imshow(frames[t], cmap="gray", vmin=0, vmax=frames[t].max() * 0.3)

        # Get trajectories up to this frame
        traj_up_to_t = filtered[filtered["frame"] <= t]

        for pid in traj_up_to_t["particle"].unique():
            traj = traj_up_to_t[traj_up_to_t["particle"] == pid]
            ax.plot(traj["x"], traj["y"], linewidth=1.5, alpha=0.8)

            current = traj[traj["frame"] == t]
            if len(current) > 0:
                ax.scatter(current["x"], current["y"], s=30, c="red", zorder=5)

        time_ms = t * frame_interval * 1000
        n_traj = traj_up_to_t["particle"].nunique()
        ax.set_title(f"Frame {t} (t={time_ms:.0f}ms) - {n_traj} trajectories")
        ax.axis("off")

        plt.tight_layout()
        output_path = OUT_DIR / f"{file_stem}_traj_{t:03d}.png"
        plt.savefig(output_path, dpi=100)
        plt.show()
        print(f"Saved: {output_path}")

# %%
# =============================================================================
# Visualization: Velocity Distribution
# =============================================================================

print(f"\n=== Velocity Distribution ===")

# Speed histogram
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(valid_speeds, bins=50, edgecolor="black", alpha=0.7)
ax.axvline(valid_speeds.mean(), color="red", linestyle="--",
           label=f"Mean: {valid_speeds.mean():.1f}")
ax.axvline(valid_speeds.median(), color="green", linestyle="--",
           label=f"Median: {valid_speeds.median():.1f}")
ax.set_xlabel("Speed (um/s)")
ax.set_ylabel("Count")
ax.set_title("Speed Distribution")
ax.legend()
plt.tight_layout()
output_path = OUT_DIR / f"{file_stem}_speed_hist.png"
plt.savefig(output_path, dpi=150)
plt.show()
print(f"Saved: {output_path}")

# Direction histogram
directions = velocities["direction"].dropna()
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(directions, bins=36, edgecolor="black", alpha=0.7)
ax.set_xlabel("Direction (degrees)")
ax.set_ylabel("Count")
ax.set_title("Direction Distribution")
plt.tight_layout()
output_path = OUT_DIR / f"{file_stem}_direction_hist.png"
plt.savefig(output_path, dpi=150)
plt.show()
print(f"Saved: {output_path}")

# %%
# =============================================================================
# Video Frames (Optional)
# =============================================================================

if GENERATE_VIDEO_FRAMES:
    print(f"\n=== Generating Video Frames ===")
    print(f"Frames: {VIDEO_FRAME_RANGE.start} to {VIDEO_FRAME_RANGE.stop}")

    video_dir = OUT_DIR / f"video/{file_stem}"
    video_dir.mkdir(parents=True, exist_ok=True)

    for t in VIDEO_FRAME_RANGE:
        if t >= len(frames):
            continue

        fig, ax = plt.subplots(figsize=(20, 4))
        ax.imshow(frames[t], cmap="gray", vmin=0, vmax=frames[t].max() * 0.3)

        traj_up_to_t = filtered[filtered["frame"] <= t]

        for pid in traj_up_to_t["particle"].unique():
            traj = traj_up_to_t[traj_up_to_t["particle"] == pid]
            ax.plot(traj["x"], traj["y"], linewidth=1.5, alpha=0.8)

            current = traj[traj["frame"] == t]
            if len(current) > 0:
                ax.scatter(current["x"], current["y"], s=30, c="red", zorder=5)

        time_ms = t * frame_interval * 1000
        ax.set_title(f"t={time_ms:.0f}ms")
        ax.axis("off")

        plt.tight_layout()
        plt.savefig(video_dir / f"frame_{t:04d}.png", dpi=100)
        plt.close()

    print(f"Saved {len(list(video_dir.glob('*.png')))} frames to {video_dir}")
    print(f"To create video: ffmpeg -r 30 -i {video_dir}/frame_%04d.png -c:v libx264 output.mp4")

# %%
# =============================================================================
# Export Results
# =============================================================================

print(f"\n=== Exporting Results ===")

# Add metadata columns
filtered["source_file"] = file_stem
filtered["method"] = METHOD
velocities["source_file"] = file_stem
velocities["method"] = METHOD

# Export trajectories and velocities
paths = export_results(
    filtered,
    velocities,
    OUTPUT_DIR / "trajectories",
    prefix=file_stem,
)
for name, path in paths.items():
    print(f"Saved: {path}")

# Export trajectory statistics
traj_stats = compute_trajectory_stats(velocities)
traj_stats["source_file"] = file_stem
traj_stats["method"] = METHOD
traj_stats_path = OUTPUT_DIR / "trajectories" / f"{file_stem}_traj_stats.csv"
traj_stats.to_csv(traj_stats_path, index=False)
print(f"Saved: {traj_stats_path}")

# %%
# =============================================================================
# Summary
# =============================================================================

ensemble_stats = compute_ensemble_stats(velocities, source_file=file_stem)

print(f"\n{'='*50}")
print(f"=== Summary (method={METHOD}) ===")
print(f"{'='*50}")
print(f"File: {file_stem}")
print(f"Detection method: {METHOD}")
print(f"Parameters: {detect_kwargs}")
print(f"")
print(f"Total detections: {len(all_particles)}")
print(f"Trajectories (raw): {n_raw}")
print(f"Trajectories (filtered): {n_filtered}")
print(f"Mean trajectory length: {traj_lengths.mean():.1f} frames")
print(f"")
print(f"Mean speed: {valid_speeds.mean():.2f} um/s")
print(f"Median speed: {valid_speeds.median():.2f} um/s")
print(f"Std speed: {valid_speeds.std():.2f} um/s")
print(f"{'='*50}")
