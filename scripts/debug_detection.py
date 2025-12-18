# %%
"""Interactive detection debugging tool.

Use this to visualize and debug particle detection on single frames.
Helps identify:
- False positives (detecting from nothing)
- Over-detection (multiple detections from single particle)
- Under-detection (missing particles)
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import trackpy as tp

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from particle_tracking import detect_particles, load_nd2_file, preprocess_frame

# %%
# Configuration
DATA_DIR = Path(__file__).parent.parent / "20251218test-nofixture2"
SAMPLE_FILE = DATA_DIR / "nd000.nd2"
FRAME_INDEX = 0  # Change this to inspect different frames

# Detection parameters to test
# Note: Large diameter/separation for defocused particles (ring + center pattern)
DIAMETER = 21
MINMASS = 500
SEPARATION = 25

# %%
# Load data
print(f"Loading {SAMPLE_FILE}...")
frames, metadata = load_nd2_file(SAMPLE_FILE)
print(f"Loaded {len(frames)} frames, shape: {frames[0].shape}")

# %%
# Get single frame
raw = frames[FRAME_INDEX]
processed = preprocess_frame(raw)

print(f"\n=== Frame {FRAME_INDEX} Statistics ===")
print(f"Raw: min={raw.min()}, max={raw.max()}, mean={raw.mean():.1f}")
print(f"Processed: min={processed.min():.2f}, max={processed.max():.2f}, mean={processed.mean():.2f}")

# %%
# Detect particles
particles = detect_particles(processed, diameter=DIAMETER, minmass=MINMASS, separation=SEPARATION, preprocess=False)
print(f"\nDetected {len(particles)} particles")

# %%
# Overview visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# Raw frame
axes[0, 0].imshow(raw, cmap="gray", vmin=0, vmax=raw.max() * 0.3)
axes[0, 0].set_title(f"Raw Frame {FRAME_INDEX}")
axes[0, 0].axis("off")

# Preprocessed frame
axes[0, 1].imshow(processed, cmap="gray")
axes[0, 1].set_title("Preprocessed")
axes[0, 1].axis("off")

# Detections on raw
axes[1, 0].imshow(raw, cmap="gray", vmin=0, vmax=raw.max() * 0.3)
if len(particles) > 0:
    axes[1, 0].scatter(particles["x"], particles["y"], s=50, facecolors="none",
                       edgecolors="red", linewidths=1)
axes[1, 0].set_title(f"Detections on Raw (n={len(particles)})")
axes[1, 0].axis("off")

# Detections on processed
axes[1, 1].imshow(processed, cmap="gray")
if len(particles) > 0:
    axes[1, 1].scatter(particles["x"], particles["y"], s=50, facecolors="none",
                       edgecolors="red", linewidths=1)
axes[1, 1].set_title(f"Detections on Processed (n={len(particles)})")
axes[1, 1].axis("off")

plt.tight_layout()
plt.savefig(Path(__file__).parent.parent / "output/figures/debug_overview.png", dpi=150)
plt.show()
print("Saved: output/figures/debug_overview.png")

# %%
# Zoom into detections - show individual particles
print("\n=== Zooming into Individual Detections ===")

if len(particles) > 0:
    # Select random detections to inspect
    n_inspect = min(12, len(particles))
    inspect_indices = np.random.choice(len(particles), n_inspect, replace=False)

    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    margin = 20  # pixels around detection

    for idx, ax in enumerate(axes.flat):
        if idx < n_inspect:
            p_idx = inspect_indices[idx]
            cx = int(particles.iloc[p_idx]["x"])
            cy = int(particles.iloc[p_idx]["y"])
            mass = particles.iloc[p_idx]["mass"]
            size = particles.iloc[p_idx]["size"]

            # Extract region
            y_start = max(0, cy - margin)
            y_end = min(raw.shape[0], cy + margin)
            x_start = max(0, cx - margin)
            x_end = min(raw.shape[1], cx + margin)

            region = processed[y_start:y_end, x_start:x_end]

            ax.imshow(region, cmap="gray")
            # Mark center
            ax.axhline(cy - y_start, color="r", linestyle="--", alpha=0.5, linewidth=0.5)
            ax.axvline(cx - x_start, color="r", linestyle="--", alpha=0.5, linewidth=0.5)
            ax.set_title(f"#{p_idx}: m={mass:.0f}, s={size:.2f}", fontsize=9)
            ax.axis("off")
        else:
            ax.axis("off")

    plt.suptitle(f"Individual Detection Inspection (margin={margin}px)", fontsize=12)
    plt.tight_layout()
    plt.savefig(Path(__file__).parent.parent / "output/figures/debug_particles.png", dpi=150)
    plt.show()
    print("Saved: output/figures/debug_particles.png")

# %%
# Mass and size distribution with thresholds
print("\n=== Particle Property Distributions ===")

if len(particles) > 0:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Mass distribution
    axes[0].hist(particles["mass"], bins=50, edgecolor="black", alpha=0.7)
    axes[0].axvline(MINMASS, color="red", linestyle="--", label=f"minmass={MINMASS}")
    axes[0].set_xlabel("Mass")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Mass Distribution")
    axes[0].legend()

    # Size distribution
    axes[1].hist(particles["size"], bins=50, edgecolor="black", alpha=0.7)
    axes[1].set_xlabel("Size (sigma)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Size Distribution")

    # Eccentricity distribution
    axes[2].hist(particles["ecc"], bins=50, edgecolor="black", alpha=0.7)
    axes[2].set_xlabel("Eccentricity")
    axes[2].set_ylabel("Count")
    axes[2].set_title("Eccentricity Distribution")

    plt.tight_layout()
    plt.savefig(Path(__file__).parent.parent / "output/figures/debug_distributions.png", dpi=150)
    plt.show()
    print("Saved: output/figures/debug_distributions.png")

    # Print statistics
    print(f"\nMass: min={particles['mass'].min():.0f}, max={particles['mass'].max():.0f}, "
          f"median={particles['mass'].median():.0f}")
    print(f"Size: min={particles['size'].min():.2f}, max={particles['size'].max():.2f}, "
          f"median={particles['size'].median():.2f}")
    print(f"Ecc: min={particles['ecc'].min():.3f}, max={particles['ecc'].max():.3f}, "
          f"median={particles['ecc'].median():.3f}")

# %%
# Check for clustered detections (potential over-detection)
print("\n=== Checking for Clustered Detections ===")

if len(particles) > 1:
    from scipy.spatial.distance import pdist, squareform

    coords = particles[["x", "y"]].values
    distances = squareform(pdist(coords))

    # Find pairs closer than diameter
    np.fill_diagonal(distances, np.inf)
    min_distances = distances.min(axis=1)

    close_pairs = (min_distances < DIAMETER).sum()
    print(f"Particles with neighbor closer than {DIAMETER}px: {close_pairs} / {len(particles)}")

    if close_pairs > 0:
        print("\n  WARNING: Some detections are very close together!")
        print("  This may indicate over-detection from single particles.")

        # Show the closest pairs
        close_mask = min_distances < DIAMETER
        close_particles = particles[close_mask]

        if len(close_particles) > 0:
            fig, axes = plt.subplots(2, 4, figsize=(16, 8))
            margin = 25

            for idx, ax in enumerate(axes.flat):
                if idx < min(8, len(close_particles)):
                    p = close_particles.iloc[idx]
                    cx, cy = int(p["x"]), int(p["y"])

                    y_start = max(0, cy - margin)
                    y_end = min(raw.shape[0], cy + margin)
                    x_start = max(0, cx - margin)
                    x_end = min(raw.shape[1], cx + margin)

                    region = processed[y_start:y_end, x_start:x_end]

                    # Find other detections in this region
                    in_region = particles[
                        (particles["x"] >= x_start) & (particles["x"] < x_end) &
                        (particles["y"] >= y_start) & (particles["y"] < y_end)
                    ]

                    ax.imshow(region, cmap="gray")
                    ax.scatter(in_region["x"] - x_start, in_region["y"] - y_start,
                              s=100, facecolors="none", edgecolors="red", linewidths=1.5)
                    ax.set_title(f"Clustered ({len(in_region)} detections)", fontsize=9)
                    ax.axis("off")
                else:
                    ax.axis("off")

            plt.suptitle("Clustered Detections (potential over-detection)", fontsize=12)
            plt.tight_layout()
            plt.savefig(Path(__file__).parent.parent / "output/figures/debug_clustered.png", dpi=150)
            plt.show()
            print("Saved: output/figures/debug_clustered.png")

# %%
# Check for weak detections (potential false positives)
print("\n=== Checking for Weak Detections (potential false positives) ===")

if len(particles) > 0:
    # Low mass particles are suspicious
    mass_threshold = particles["mass"].quantile(0.1)
    weak_particles = particles[particles["mass"] < mass_threshold]

    print(f"Particles with mass below 10th percentile ({mass_threshold:.0f}): {len(weak_particles)}")

    if len(weak_particles) > 0:
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        margin = 20

        for idx, ax in enumerate(axes.flat):
            if idx < min(8, len(weak_particles)):
                p = weak_particles.iloc[idx]
                cx, cy = int(p["x"]), int(p["y"])
                mass = p["mass"]

                y_start = max(0, cy - margin)
                y_end = min(raw.shape[0], cy + margin)
                x_start = max(0, cx - margin)
                x_end = min(raw.shape[1], cx + margin)

                region = processed[y_start:y_end, x_start:x_end]

                ax.imshow(region, cmap="gray")
                ax.axhline(cy - y_start, color="r", linestyle="--", alpha=0.5, linewidth=0.5)
                ax.axvline(cx - x_start, color="r", linestyle="--", alpha=0.5, linewidth=0.5)
                ax.set_title(f"mass={mass:.0f}", fontsize=9)
                ax.axis("off")
            else:
                ax.axis("off")

        plt.suptitle(f"Weak Detections (mass < {mass_threshold:.0f})", fontsize=12)
        plt.tight_layout()
        plt.savefig(Path(__file__).parent.parent / "output/figures/debug_weak.png", dpi=150)
        plt.show()
        print("Saved: output/figures/debug_weak.png")

# %%
# Interactive region selector
print("\n=== Region Inspection Tool ===")
print("Specify a region to inspect in detail:")

# You can modify these coordinates to inspect specific regions
INSPECT_X = 1600  # x center of region to inspect
INSPECT_Y = 290   # y center of region to inspect
INSPECT_SIZE = 200  # size of region

x_start = max(0, INSPECT_X - INSPECT_SIZE // 2)
x_end = min(raw.shape[1], INSPECT_X + INSPECT_SIZE // 2)
y_start = max(0, INSPECT_Y - INSPECT_SIZE // 2)
y_end = min(raw.shape[0], INSPECT_Y + INSPECT_SIZE // 2)

region_raw = raw[y_start:y_end, x_start:x_end]
region_proc = processed[y_start:y_end, x_start:x_end]

# Get particles in this region
if len(particles) > 0:
    in_region = particles[
        (particles["x"] >= x_start) & (particles["x"] < x_end) &
        (particles["y"] >= y_start) & (particles["y"] < y_end)
    ]
else:
    in_region = particles

fig, axes = plt.subplots(1, 2, figsize=(14, 7))

axes[0].imshow(region_raw, cmap="gray", vmin=0, vmax=raw.max() * 0.3)
if len(in_region) > 0:
    axes[0].scatter(in_region["x"] - x_start, in_region["y"] - y_start,
                   s=100, facecolors="none", edgecolors="red", linewidths=1.5)
axes[0].set_title(f"Raw Region ({x_start}:{x_end}, {y_start}:{y_end})")

axes[1].imshow(region_proc, cmap="gray")
if len(in_region) > 0:
    axes[1].scatter(in_region["x"] - x_start, in_region["y"] - y_start,
                   s=100, facecolors="none", edgecolors="red", linewidths=1.5)
    for _, p in in_region.iterrows():
        axes[1].annotate(f"{p['mass']:.0f}",
                        (p["x"] - x_start + 5, p["y"] - y_start - 5),
                        color="yellow", fontsize=8)
axes[1].set_title(f"Processed Region (n={len(in_region)} detections)")

plt.tight_layout()
plt.savefig(Path(__file__).parent.parent / "output/figures/debug_region.png", dpi=150)
plt.show()
print(f"Saved: output/figures/debug_region.png")
print(f"Found {len(in_region)} particles in this region")

# %%
# Parameter comparison
print("\n=== Parameter Comparison ===")
print("Testing different minmass values...")

minmass_values = [200, 500, 1000, 2000, 5000]
fig, axes = plt.subplots(1, len(minmass_values), figsize=(4 * len(minmass_values), 4))

for ax, mm in zip(axes, minmass_values):
    p = detect_particles(processed, diameter=DIAMETER, minmass=mm, preprocess=False)
    ax.imshow(processed, cmap="gray")
    if len(p) > 0:
        ax.scatter(p["x"], p["y"], s=10, c="red", marker="o", alpha=0.7)
    ax.set_title(f"minmass={mm}\nn={len(p)}")
    ax.axis("off")

plt.tight_layout()
plt.savefig(Path(__file__).parent.parent / "output/figures/debug_minmass_comparison.png", dpi=150)
plt.show()
print("Saved: output/figures/debug_minmass_comparison.png")

# %%
print("\n=== Debugging Complete ===")
print("\nSuggestions based on analysis:")
print("- If seeing over-detection: increase minmass or decrease diameter")
print("- If seeing false positives: increase minmass")
print("- If missing real particles: decrease minmass")
print("\nModify the DIAMETER and MINMASS values at the top and re-run to test.")
