# %%
"""Particle detection parameter tuning script.

Use this script to find optimal detection parameters for your data.
This implements DefocusTracker's Method 0 (boundary_threshold_2d).

After tuning, update config.py with the optimal parameters.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from particle_tracking import detect, estimate_threshold, load_nd2_file

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
# Estimate Threshold
# =============================================================================

print("\n=== Threshold Estimation ===")
print("Suggested boundary_threshold values based on percentiles:")

for pct in [95, 97, 98, 99, 99.5, 99.9]:
    thresh = estimate_threshold(frame, pct)
    print(f"  {pct:5.1f}th percentile -> threshold = {thresh:.1f}")

# %%
# =============================================================================
# Current Parameters
# =============================================================================

print(f"\n{'='*50}")
print("Current detection parameters (from config.py):")
print(f"{'='*50}")
print(f"  boundary_threshold: {DETECT_PARAMS['boundary_threshold']}")
print(f"  min_area: {DETECT_PARAMS['min_area']}")
print(f"  median_filter: {DETECT_PARAMS['median_filter']}")
print(f"  gauss_filter: {DETECT_PARAMS['gauss_filter']}")

# %%
# =============================================================================
# Threshold Visualization
# =============================================================================

print("\n=== Threshold Visualization ===")

threshold = DETECT_PARAMS["boundary_threshold"]
binary = frame > threshold

# Binary mask (before hole filling)
fig, ax = plt.subplots(figsize=(16, 4))
ax.imshow(binary, cmap="gray")
ax.set_title(f"Binary mask (threshold={threshold})")
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
# Parameter Sweep: Threshold
# =============================================================================

print("\n=== Threshold Sweep ===")
print("Threshold | Detections")
print("-" * 30)

# Generate threshold values from percentiles
thresholds = [estimate_threshold(frame, p) for p in [95, 97, 98, 99, 99.5, 99.9]]

for thresh in thresholds:
    p = detect(frame, boundary_threshold=thresh, min_area=DETECT_PARAMS["min_area"])
    print(f"  {thresh:7.1f}  |    {len(p):4d}")

    fig, ax = plt.subplots(figsize=(16, 4))
    ax.imshow(frame, cmap="gray", vmin=0, vmax=frame.max() * 0.3)
    if len(p) > 0:
        ax.scatter(p["x"], p["y"], s=50, facecolors="none",
                   edgecolors="lime", linewidths=1)
    ax.set_title(f"threshold={thresh:.0f} -> n={len(p)}")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"sweep_thresh{thresh:.0f}.png", dpi=100)
    plt.close()

print(f"\nSaved threshold sweep images to {OUT_DIR}")

# %%
# =============================================================================
# Parameter Sweep: Min Area
# =============================================================================

print("\n=== Min Area Sweep ===")
print("min_area | Detections")
print("-" * 30)

min_areas = [10, 20, 30, 50, 100, 200]

for ma in min_areas:
    p = detect(frame, boundary_threshold=DETECT_PARAMS["boundary_threshold"], min_area=ma)
    print(f"  {ma:6d}  |    {len(p):4d}")

    fig, ax = plt.subplots(figsize=(16, 4))
    ax.imshow(frame, cmap="gray", vmin=0, vmax=frame.max() * 0.3)
    if len(p) > 0:
        ax.scatter(p["x"], p["y"], s=50, facecolors="none",
                   edgecolors="lime", linewidths=1)
    ax.set_title(f"min_area={ma} -> n={len(p)}")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"sweep_area{ma}.png", dpi=100)
    plt.close()

print(f"\nSaved min_area sweep images to {OUT_DIR}")

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
print("1. Review the threshold sweep images")
print("2. Update BOUNDARY_THRESHOLD in config.py")
print("3. Review the min_area sweep images")
print("4. Update MIN_AREA in config.py if needed")
print("5. Run 03_analysis.py for full pipeline")
