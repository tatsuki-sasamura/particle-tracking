# %%
"""Data exploration script for particle tracking."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from particle_tracking import load_nd2_file, preprocess_frame

# %%
# Configuration
DATA_DIR = Path(__file__).parent.parent / "20251218test-nofixture2"
SAMPLE_FILE = DATA_DIR / "nd000.nd2"

# %%
# Load sample ND2 file
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
# Visualize sample frames
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
frame_indices = [0, 50, 100, 150, 199, 0]

for idx, (ax, frame_idx) in enumerate(zip(axes.flat, frame_indices)):
    if idx < 5:
        ax.imshow(frames[frame_idx], cmap="gray", vmin=0, vmax=frames.max() * 0.5)
        ax.set_title(f"Frame {frame_idx}")
        ax.axis("off")
    else:
        # Show preprocessed version of first frame
        processed = preprocess_frame(frames[0])
        ax.imshow(processed, cmap="gray")
        ax.set_title("Frame 0 (preprocessed)")
        ax.axis("off")

plt.tight_layout()
plt.savefig(Path(__file__).parent.parent / "output/figures/sample_frames.png", dpi=150)
plt.show()
print("Saved: output/figures/sample_frames.png")

# %%
# Intensity histogram
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
plt.savefig(Path(__file__).parent.parent / "output/figures/intensity_histogram.png", dpi=150)
plt.show()
print("Saved: output/figures/intensity_histogram.png")

# %%
# Estimate particle size by looking at a zoomed region
# Find a bright spot to zoom into
frame = frames[0]
threshold = np.percentile(frame, 99)
bright_points = np.where(frame > threshold)

if len(bright_points[0]) > 0:
    # Pick a random bright point
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
    plt.savefig(
        Path(__file__).parent.parent / "output/figures/particle_zoom.png", dpi=150
    )
    plt.show()
    print("Saved: output/figures/particle_zoom.png")
    print(f"\nBright spot found at ({cx}, {cy})")
    print("Examine the zoomed image to estimate particle diameter.")

# %%
print("\n=== Exploration Complete ===")
print("Next: Run 02_particle_detection.py to tune detection parameters")
