# %%
"""Particle detection parameter tuning script."""

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

# %%
# Load sample frame
print(f"Loading {SAMPLE_FILE}...")
frames, metadata = load_nd2_file(SAMPLE_FILE)
print(f"Using frame 0 for parameter tuning")

# %%
# Preprocess frame
raw = frames[0]
processed = preprocess_frame(raw)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].imshow(raw, cmap="gray", vmin=0, vmax=raw.max() * 0.3)
axes[0].set_title("Raw")
axes[1].imshow(processed, cmap="gray")
axes[1].set_title("Preprocessed")
plt.tight_layout()
plt.show()

# %%
# Initial detection with conservative parameters
# Note: Large diameter/separation for defocused particles (ring + center pattern)
print("\n=== Initial Detection ===")
diameter = 21  # Must be odd, large for defocused particles
minmass = 500
separation = 25

particles = detect_particles(processed, diameter=diameter, minmass=minmass, separation=separation, preprocess=False)
print(f"Diameter: {diameter}, Minmass: {minmass}, Separation: {separation}")
print(f"Found {len(particles)} particles")

# %%
# Visualize detections
fig, ax = plt.subplots(figsize=(14, 5))
ax.imshow(processed, cmap="gray")
tp.annotate(particles, processed, ax=ax)
ax.set_title(f"Detected particles (diameter={diameter}, minmass={minmass}, n={len(particles)})")
plt.tight_layout()
plt.savefig(Path(__file__).parent.parent / "output/figures/detection_initial.png", dpi=150)
plt.show()
print("Saved: output/figures/detection_initial.png")

# %%
# Parameter sweep: diameter
print("\n=== Diameter Sweep ===")
diameters = [7, 9, 11, 13, 15]
minmass_fixed = 500

fig, axes = plt.subplots(1, len(diameters), figsize=(4 * len(diameters), 4))

for ax, d in zip(axes, diameters):
    particles = detect_particles(processed, diameter=d, minmass=minmass_fixed, preprocess=False)
    ax.imshow(processed, cmap="gray")
    if len(particles) > 0:
        ax.scatter(particles["x"], particles["y"], s=10, c="red", marker="o", alpha=0.7)
    ax.set_title(f"d={d}, n={len(particles)}")
    ax.axis("off")

plt.tight_layout()
plt.savefig(Path(__file__).parent.parent / "output/figures/diameter_sweep.png", dpi=150)
plt.show()
print("Saved: output/figures/diameter_sweep.png")

# %%
# Parameter sweep: minmass
print("\n=== Minmass Sweep ===")
diameter_fixed = 11
minmasses = [100, 300, 500, 1000, 2000]

fig, axes = plt.subplots(1, len(minmasses), figsize=(4 * len(minmasses), 4))

for ax, mm in zip(axes, minmasses):
    particles = detect_particles(processed, diameter=diameter_fixed, minmass=mm, preprocess=False)
    ax.imshow(processed, cmap="gray")
    if len(particles) > 0:
        ax.scatter(particles["x"], particles["y"], s=10, c="red", marker="o", alpha=0.7)
    ax.set_title(f"mm={mm}, n={len(particles)}")
    ax.axis("off")

plt.tight_layout()
plt.savefig(Path(__file__).parent.parent / "output/figures/minmass_sweep.png", dpi=150)
plt.show()
print("Saved: output/figures/minmass_sweep.png")

# %%
# Show particle properties distribution
print("\n=== Particle Properties ===")
particles = detect_particles(processed, diameter=11, minmass=500, preprocess=False)

if len(particles) > 0:
    fig, axes = plt.subplots(1, 3, figsize=(12, 3))

    axes[0].hist(particles["mass"], bins=30)
    axes[0].set_xlabel("Mass")
    axes[0].set_title("Mass Distribution")

    axes[1].hist(particles["size"], bins=30)
    axes[1].set_xlabel("Size (sigma)")
    axes[1].set_title("Size Distribution")

    axes[2].hist(particles["ecc"], bins=30)
    axes[2].set_xlabel("Eccentricity")
    axes[2].set_title("Eccentricity Distribution")

    plt.tight_layout()
    plt.savefig(
        Path(__file__).parent.parent / "output/figures/particle_properties.png", dpi=150
    )
    plt.show()
    print("Saved: output/figures/particle_properties.png")

    print(f"\nMass: min={particles['mass'].min():.0f}, max={particles['mass'].max():.0f}")
    print(f"Size: min={particles['size'].min():.2f}, max={particles['size'].max():.2f}")
    print(f"Ecc: min={particles['ecc'].min():.3f}, max={particles['ecc'].max():.3f}")

# %%
# Validate on multiple frames
print("\n=== Multi-frame Validation ===")
test_frames = [0, 50, 100, 150, 199]
diameter = 11
minmass = 500

fig, axes = plt.subplots(1, len(test_frames), figsize=(4 * len(test_frames), 4))

for ax, f_idx in zip(axes, test_frames):
    frame = frames[f_idx]
    processed = preprocess_frame(frame)
    particles = detect_particles(processed, diameter=diameter, minmass=minmass, preprocess=False)
    ax.imshow(processed, cmap="gray")
    if len(particles) > 0:
        ax.scatter(particles["x"], particles["y"], s=10, c="red", marker="o", alpha=0.7)
    ax.set_title(f"Frame {f_idx}, n={len(particles)}")
    ax.axis("off")

plt.tight_layout()
plt.savefig(Path(__file__).parent.parent / "output/figures/multiframe_detection.png", dpi=150)
plt.show()
print("Saved: output/figures/multiframe_detection.png")

# %%
print("\n=== Parameter Tuning Complete ===")
print(f"Recommended parameters:")
print(f"  diameter = {diameter}")
print(f"  minmass = {minmass}")
print("\nNext: Run 03_tracking_analysis.py for full tracking pipeline")
