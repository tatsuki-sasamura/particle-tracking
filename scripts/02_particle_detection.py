# %%
"""Particle detection parameter tuning script.

Use this script to find optimal detection parameters for your data.
After tuning, update config.py with the optimal parameters.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from particle_tracking import detect, load_nd2_file

from config import (
    DATA_DIR,
    DETECT_PARAMS,
    get_output_dir,
)

# %%
# =============================================================================
# Configuration
# =============================================================================

ND_FILE_NUMBER = 0  # Change this to load different files
FRAME_INDEX = 0  # Frame to use for tuning
TEST_FRAMES = [0, 50, 100, 150, 199]  # Frames for multi-frame validation

SAMPLE_FILE = DATA_DIR / f"nd{ND_FILE_NUMBER:03d}.nd2"
OUT_DIR = get_output_dir(__file__)

# %%
# =============================================================================
# Load Data
# =============================================================================

print(f"Loading {SAMPLE_FILE}...")
frames, metadata = load_nd2_file(SAMPLE_FILE)
print(f"Loaded {len(frames)} frames")

frame = frames[FRAME_INDEX]

print(f"\nFrame {FRAME_INDEX} statistics:")
print(f"  min={frame.min()}, max={frame.max()}, mean={frame.mean():.1f}")

# %%
# =============================================================================
# Visualize Frame
# =============================================================================

print("\n=== Frame Visualization ===")

fig, ax = plt.subplots(figsize=(16, 4))
ax.imshow(frame, cmap="gray", vmin=0, vmax=frame.max() * 0.3)
ax.set_title(f"Frame {FRAME_INDEX}")
ax.axis("off")
plt.tight_layout()
plt.savefig(OUT_DIR / "frame.png", dpi=150)
plt.show()
print(f"Saved: {OUT_DIR / 'frame.png'}")

# %%
# =============================================================================
# Current Parameters
# =============================================================================

print(f"\n{'='*50}")
print("Current detection parameters:")
print(f"{'='*50}")
print(f"  threshold_percentile: {DETECT_PARAMS['threshold_percentile']}")
print(f"  min_area: {DETECT_PARAMS['min_area']}")
print(f"  max_area: {DETECT_PARAMS['max_area']}")

# %%
# =============================================================================
# Threshold Visualization
# =============================================================================

print("\n=== Threshold Visualization ===")

pct = DETECT_PARAMS["threshold_percentile"]
threshold = np.percentile(frame, pct)
binary = frame > threshold

# Binary mask
fig, ax = plt.subplots(figsize=(16, 4))
ax.imshow(binary, cmap="gray")
ax.set_title(f"Binary mask (percentile={pct}, threshold={threshold:.1f})")
ax.axis("off")
plt.tight_layout()
plt.savefig(OUT_DIR / "threshold_binary.png", dpi=150)
plt.show()
print(f"Saved: {OUT_DIR / 'threshold_binary.png'}")

# Detections with current parameters
particles = detect(frame, **DETECT_PARAMS)
fig, ax = plt.subplots(figsize=(16, 4))
ax.imshow(frame, cmap="gray", vmin=0, vmax=frame.max() * 0.3)
if len(particles) > 0:
    ax.scatter(particles["x"], particles["y"],
               s=100, facecolors="none", edgecolors="lime", linewidths=1.5)
ax.set_title(f"Detections (n={len(particles)})")
ax.axis("off")
plt.tight_layout()
plt.savefig(OUT_DIR / "detections.png", dpi=150)
plt.show()
print(f"Saved: {OUT_DIR / 'detections.png'}")

# %%
# =============================================================================
# Parameter Sweep
# =============================================================================

print("\n=== Parameter Sweep ===")
print("Percentile | min_area | Detections")
print("-" * 40)

percentiles = [97, 98, 99, 99.5, 99.9]
min_areas = [10, 30, 50, 100, 200]

for pct in percentiles:
    for ma in min_areas:
        p = detect(frame,
                   threshold_percentile=pct, min_area=ma,
                   max_area=DETECT_PARAMS["max_area"])
        print(f"  {pct:6.1f}   |   {ma:4d}   |    {len(p):4d}")

        # Save each combination as separate image
        fig, ax = plt.subplots(figsize=(16, 4))
        ax.imshow(frame, cmap="gray", vmin=0, vmax=frame.max() * 0.3)
        if len(p) > 0:
            ax.scatter(p["x"], p["y"], s=50, facecolors="none",
                       edgecolors="lime", linewidths=1)
        ax.set_title(f"percentile={pct}, min_area={ma} -> n={len(p)}")
        ax.axis("off")
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"sweep_pct{pct}_area{ma}.png", dpi=100)
        plt.close()

print(f"\nSaved sweep images to {OUT_DIR}")

# %%
# =============================================================================
# Multi-frame Validation
# =============================================================================

print("\n=== Multi-frame Validation ===")

for f_idx in TEST_FRAMES:
    if f_idx >= len(frames):
        print(f"Skipping frame {f_idx} (out of range)")
        continue

    f = frames[f_idx]
    p = detect(f, **DETECT_PARAMS)

    fig, ax = plt.subplots(figsize=(16, 4))
    ax.imshow(f, cmap="gray", vmin=0, vmax=f.max() * 0.3)
    if len(p) > 0:
        ax.scatter(p["x"], p["y"], s=50, facecolors="none",
                   edgecolors="lime", linewidths=1)
    ax.set_title(f"Frame {f_idx}: {len(p)} detections")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"multiframe_{f_idx:03d}.png", dpi=100)
    plt.close()

    print(f"Frame {f_idx}: {len(p)} detections")

print(f"\nSaved multi-frame images to {OUT_DIR}")

# %%
# =============================================================================
# Summary
# =============================================================================

print("\n" + "=" * 50)
print("=== Parameter Tuning Complete ===")
print("=" * 50)
print(f"\nOutput directory: {OUT_DIR}")
print("\nRecommended next steps:")
print("1. Review the parameter sweep images")
print("2. Update DETECT_PARAMS in config.py with optimal values")
print("3. Run 03_analysis.py for full pipeline")
