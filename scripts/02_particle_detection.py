# %%
"""Particle detection parameter tuning script.

Use this script to find optimal detection parameters for your data.
Supports both threshold and trackpy detection methods.

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
    METHOD,
    THRESHOLD_PARAMS,
    TRACKPY_PARAMS,
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
# Method Selection
# =============================================================================

print(f"\n{'='*50}")
print(f"Current method: {METHOD}")
print(f"{'='*50}")

if METHOD == "threshold":
    print("\nThreshold method parameters:")
    print(f"  threshold_percentile: {THRESHOLD_PARAMS['threshold_percentile']}")
    print(f"  min_area: {THRESHOLD_PARAMS['min_area']}")
    print(f"  max_area: {THRESHOLD_PARAMS['max_area']}")
else:
    print("\nTrackpy method parameters:")
    print(f"  diameter: {TRACKPY_PARAMS['diameter']}")
    print(f"  minmass: {TRACKPY_PARAMS['minmass']}")

# %%
# =============================================================================
# Threshold Method: Parameter Sweep
# =============================================================================

if METHOD == "threshold":
    print("\n=== Threshold Method: Visualization ===")

    pct = THRESHOLD_PARAMS["threshold_percentile"]
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

    # Detections
    particles = detect(frame, method="threshold", **THRESHOLD_PARAMS)
    fig, ax = plt.subplots(figsize=(16, 4))
    ax.imshow(frame, cmap="gray", vmin=0, vmax=frame.max() * 0.3)
    if len(particles) > 0:
        ax.scatter(particles["x"], particles["y"],
                   s=100, facecolors="none", edgecolors="lime", linewidths=1.5)
    ax.set_title(f"Threshold detections (n={len(particles)})")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "threshold_detections.png", dpi=150)
    plt.show()
    print(f"Saved: {OUT_DIR / 'threshold_detections.png'}")

    # Parameter sweep
    print("\n=== Threshold Parameter Sweep ===")
    print("Percentile | min_area | Detections")
    print("-" * 40)

    percentiles = [97, 98, 99, 99.5, 99.9]
    min_areas = [10, 30, 50, 100, 200]

    for pct in percentiles:
        for ma in min_areas:
            p = detect(frame, method="threshold",
                       threshold_percentile=pct, min_area=ma,
                       max_area=THRESHOLD_PARAMS["max_area"])
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
# Trackpy Method: Parameter Sweep
# =============================================================================

if METHOD == "trackpy":
    print("\n=== Trackpy Method: Current Parameters ===")

    particles = detect(frame, method="trackpy", **TRACKPY_PARAMS)

    fig, ax = plt.subplots(figsize=(16, 4))
    ax.imshow(frame, cmap="gray", vmin=0, vmax=frame.max() * 0.3)
    if len(particles) > 0:
        ax.scatter(particles["x"], particles["y"],
                   s=100, facecolors="none", edgecolors="red", linewidths=1.5)
    ax.set_title(f"Trackpy: diameter={TRACKPY_PARAMS['diameter']}, "
                 f"minmass={TRACKPY_PARAMS['minmass']} -> n={len(particles)}")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "trackpy_detections.png", dpi=150)
    plt.show()
    print(f"Saved: {OUT_DIR / 'trackpy_detections.png'}")

    # Diameter sweep
    print("\n=== Diameter Sweep ===")
    diameters = [7, 9, 11, 13, 15, 21]
    minmass_fixed = TRACKPY_PARAMS["minmass"]

    for d in diameters:
        p = detect(frame, method="trackpy", diameter=d, minmass=minmass_fixed)
        print(f"  diameter={d:2d}: {len(p):4d} detections")

        fig, ax = plt.subplots(figsize=(16, 4))
        ax.imshow(frame, cmap="gray", vmin=0, vmax=frame.max() * 0.3)
        if len(p) > 0:
            ax.scatter(p["x"], p["y"], s=50, facecolors="none",
                       edgecolors="red", linewidths=1)
        ax.set_title(f"diameter={d}, minmass={minmass_fixed} -> n={len(p)}")
        ax.axis("off")
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"sweep_diameter{d}.png", dpi=100)
        plt.close()

    # Minmass sweep
    print("\n=== Minmass Sweep ===")
    minmasses = [100, 300, 500, 1000, 2000, 5000]
    diameter_fixed = TRACKPY_PARAMS["diameter"]

    for mm in minmasses:
        p = detect(frame, method="trackpy", diameter=diameter_fixed, minmass=mm)
        print(f"  minmass={mm:5d}: {len(p):4d} detections")

        fig, ax = plt.subplots(figsize=(16, 4))
        ax.imshow(frame, cmap="gray", vmin=0, vmax=frame.max() * 0.3)
        if len(p) > 0:
            ax.scatter(p["x"], p["y"], s=50, facecolors="none",
                       edgecolors="red", linewidths=1)
        ax.set_title(f"diameter={diameter_fixed}, minmass={mm} -> n={len(p)}")
        ax.axis("off")
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"sweep_minmass{mm}.png", dpi=100)
        plt.close()

    print(f"\nSaved sweep images to {OUT_DIR}")

# %%
# =============================================================================
# Compare Methods
# =============================================================================

print("\n=== Method Comparison ===")

# Threshold
p_threshold = detect(frame, method="threshold", **THRESHOLD_PARAMS)

fig, ax = plt.subplots(figsize=(16, 4))
ax.imshow(frame, cmap="gray", vmin=0, vmax=frame.max() * 0.3)
if len(p_threshold) > 0:
    ax.scatter(p_threshold["x"], p_threshold["y"],
               s=100, facecolors="none", edgecolors="lime", linewidths=1.5)
ax.set_title(f"Threshold method: {len(p_threshold)} detections")
ax.axis("off")
plt.tight_layout()
plt.savefig(OUT_DIR / "compare_threshold.png", dpi=150)
plt.show()
print(f"Threshold: {len(p_threshold)} detections")

# Trackpy
p_trackpy = detect(frame, method="trackpy", **TRACKPY_PARAMS)

fig, ax = plt.subplots(figsize=(16, 4))
ax.imshow(frame, cmap="gray", vmin=0, vmax=frame.max() * 0.3)
if len(p_trackpy) > 0:
    ax.scatter(p_trackpy["x"], p_trackpy["y"],
               s=100, facecolors="none", edgecolors="red", linewidths=1.5)
ax.set_title(f"Trackpy method: {len(p_trackpy)} detections")
ax.axis("off")
plt.tight_layout()
plt.savefig(OUT_DIR / "compare_trackpy.png", dpi=150)
plt.show()
print(f"Trackpy: {len(p_trackpy)} detections")

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

    # Threshold
    p_t = detect(f, method="threshold", **THRESHOLD_PARAMS)
    fig, ax = plt.subplots(figsize=(16, 4))
    ax.imshow(f, cmap="gray", vmin=0, vmax=f.max() * 0.3)
    if len(p_t) > 0:
        ax.scatter(p_t["x"], p_t["y"], s=50, facecolors="none",
                   edgecolors="lime", linewidths=1)
    ax.set_title(f"Frame {f_idx} - Threshold: {len(p_t)} detections")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"multiframe_{f_idx:03d}_threshold.png", dpi=100)
    plt.close()

    # Trackpy
    p_k = detect(f, method="trackpy", **TRACKPY_PARAMS)
    fig, ax = plt.subplots(figsize=(16, 4))
    ax.imshow(f, cmap="gray", vmin=0, vmax=f.max() * 0.3)
    if len(p_k) > 0:
        ax.scatter(p_k["x"], p_k["y"], s=50, facecolors="none",
                   edgecolors="red", linewidths=1)
    ax.set_title(f"Frame {f_idx} - Trackpy: {len(p_k)} detections")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"multiframe_{f_idx:03d}_trackpy.png", dpi=100)
    plt.close()

    print(f"Frame {f_idx}: threshold={len(p_t)}, trackpy={len(p_k)}")

print(f"\nSaved multi-frame images to {OUT_DIR}")

# %%
# =============================================================================
# Summary
# =============================================================================

print("\n" + "=" * 50)
print("=== Parameter Tuning Complete ===")
print("=" * 50)
print(f"\nCurrent METHOD in config.py: {METHOD}")
print(f"\nOutput directory: {OUT_DIR}")
print("\nRecommended next steps:")
print("1. Review the parameter sweep images")
print("2. Update config.py with optimal parameters")
print("3. Run 03_analysis.py for full pipeline")
