# %%
"""Data exploration script for particle tracking."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from particle_tracking import load_nd2_file, preprocess_frame

from config import DATA_DIR, get_output_dir

# %%
# =============================================================================
# Configuration
# =============================================================================

ND_FILE_NUMBER = 0  # Change this to load different files (0 -> nd000.nd2)
SAMPLE_FRAMES = [0, 50, 100, 150, 199]  # Frames to visualize

SAMPLE_FILE = DATA_DIR / f"nd{ND_FILE_NUMBER:03d}.nd2"
OUT_DIR = get_output_dir(__file__)

# %%
# Load ND2 file
print(f"Loading {SAMPLE_FILE}...")
frames, metadata = load_nd2_file(SAMPLE_FILE)
print(f"Loaded successfully!")

# %%
# Display metadata
print("\n=== Metadata ===")
for key, value in metadata.items():
    print(f"  {key}: {value}")

# %%
# Frame statistics
print("\n=== Frame Statistics ===")
print(f"Shape: {frames.shape}")
print(f"Dtype: {frames.dtype}")
print(f"Min: {frames.min()}")
print(f"Max: {frames.max()}")
print(f"Mean: {frames.mean():.1f}")

# %%
# Visualize sample frames (raw)
print("\n=== Generating Raw Frame Images ===")

for frame_idx in SAMPLE_FRAMES:
    if frame_idx >= len(frames):
        print(f"Skipping frame {frame_idx} (out of range)")
        continue

    fig, ax = plt.subplots(figsize=(16, 4))
    ax.imshow(frames[frame_idx], cmap="gray", vmin=0, vmax=frames.max() * 0.5)
    ax.set_title(f"Frame {frame_idx} (raw)")
    ax.axis("off")

    plt.tight_layout()
    output_path = OUT_DIR / f"frame_{frame_idx:03d}_raw.png"
    plt.savefig(output_path, dpi=150)
    plt.show()
    print(f"Saved: {output_path}")

# %%
# Visualize sample frames (preprocessed)
print("\n=== Generating Preprocessed Frame Images ===")

for frame_idx in SAMPLE_FRAMES:
    if frame_idx >= len(frames):
        print(f"Skipping frame {frame_idx} (out of range)")
        continue

    processed = preprocess_frame(frames[frame_idx])

    fig, ax = plt.subplots(figsize=(16, 4))
    ax.imshow(processed, cmap="gray")
    ax.set_title(f"Frame {frame_idx} (preprocessed)")
    ax.axis("off")

    plt.tight_layout()
    output_path = OUT_DIR / f"frame_{frame_idx:03d}_preprocessed.png"
    plt.savefig(output_path, dpi=150)
    plt.show()
    print(f"Saved: {output_path}")

# %%
# Intensity histogram
print("\n=== Intensity Histogram ===")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Raw histogram
axes[0].hist(frames[0].flatten(), bins=100, log=True)
axes[0].set_xlabel("Intensity")
axes[0].set_ylabel("Count (log)")
axes[0].set_title("Raw Frame Intensity Distribution")

# Preprocessed histogram
processed = preprocess_frame(frames[0])
axes[1].hist(processed.flatten(), bins=100, log=True)
axes[1].set_xlabel("Intensity")
axes[1].set_ylabel("Count (log)")
axes[1].set_title("Preprocessed Frame Intensity Distribution")

plt.tight_layout()
output_path = OUT_DIR / "intensity_histogram.png"
plt.savefig(output_path, dpi=150)
plt.show()
print(f"Saved: {output_path}")

# %%
# Estimate particle size by looking at a zoomed region
print("\n=== Particle Zoom ===")

frame = frames[0]
threshold = np.percentile(frame, 99)
bright_points = np.where(frame > threshold)

if len(bright_points[0]) > 0:
    # Pick a bright point
    idx = len(bright_points[0]) // 2
    cy, cx = bright_points[0][idx], bright_points[1][idx]

    # Zoom into region around bright point
    margin = 30
    y_start = max(0, cy - margin)
    y_end = min(frame.shape[0], cy + margin)
    x_start = max(0, cx - margin)
    x_end = min(frame.shape[1], cx + margin)

    region = frame[y_start:y_end, x_start:x_end]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(region, cmap="gray")
    ax.set_title(f"Zoomed region around ({cx}, {cy})")
    ax.axhline(margin, color="r", linestyle="--", alpha=0.5)
    ax.axvline(margin, color="r", linestyle="--", alpha=0.5)

    output_path = OUT_DIR / "particle_zoom.png"
    plt.savefig(output_path, dpi=150)
    plt.show()
    print(f"Saved: {output_path}")
    print(f"\nBright spot found at ({cx}, {cy})")
    print("Examine the zoomed image to estimate particle diameter.")

# %%
print("\n=== Exploration Complete ===")
print(f"Output directory: {OUT_DIR}")
print("Next: Run 02_particle_detection.py to tune detection parameters")
