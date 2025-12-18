# %%
"""Filter combined tracking data.

Equivalent to MATLAB's 3 data_filter.m.
Applies trajectory length and velocity filters to the combined dataset.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from config import OUTPUT_DIR

# %%
# =============================================================================
# Configuration
# =============================================================================

INPUT_FILE = OUTPUT_DIR / "summary" / "data_combine.csv"
OUTPUT_FILE = OUTPUT_DIR / "summary" / "data_combine_filter.csv"
PLOT_DIR = OUTPUT_DIR / "05_data_filter"

# Filter parameters
MIN_TRACK_LENGTH = 30  # Minimum number of points per trajectory
MIN_Y_DISPLACEMENT = 5.0  # Minimum cumulative Y displacement (um) for motion onset

# %%
# =============================================================================
# Load Combined Data
# =============================================================================

print(f"Loading {INPUT_FILE}...")
data = pd.read_csv(INPUT_FILE)

print(f"Loaded {len(data)} rows")
print(f"Trajectories: {data['ID'].nunique()}")

# %%
# =============================================================================
# Diagnostic: Velocity vs Frame (to detect US onset)
# =============================================================================

print(f"\n=== Diagnostic: Velocity vs Frame ===")

# Compute mean |DY| per frame to visualize US onset
valid_vel = data.dropna(subset=["DY"])
mean_dy_per_frame = valid_vel.groupby("fr")["DY"].apply(lambda x: x.abs().mean())

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(mean_dy_per_frame.index, mean_dy_per_frame.values, ".-", alpha=0.7)
ax.set_xlabel("Frame")
ax.set_ylabel("Mean |DY| (um/s)")
ax.set_title("Mean absolute DY velocity per frame (to identify US onset)")
ax.axhline(y=mean_dy_per_frame.mean(), color="r", linestyle="--", label=f"Mean: {mean_dy_per_frame.mean():.1f}")
ax.legend()
plt.tight_layout()

PLOT_DIR.mkdir(parents=True, exist_ok=True)
plot_path = PLOT_DIR / "diagnostic_velocity_vs_frame.png"
plt.savefig(plot_path, dpi=150)
plt.show()
print(f"Saved: {plot_path}")

# Y displacement per trajectory histogram
def compute_y_disp(group):
    return abs(group["Y"].iloc[-1] - group["Y"].iloc[0])

y_disp_per_traj = data.groupby("ID").apply(compute_y_disp)

fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(y_disp_per_traj, bins=50, edgecolor="black", alpha=0.7)
ax.axvline(x=MIN_Y_DISPLACEMENT, color="r", linestyle="--", label=f"MIN_Y_DISPLACEMENT: {MIN_Y_DISPLACEMENT}")
ax.set_xlabel("Net Y Displacement (um)")
ax.set_ylabel("Count (trajectories)")
ax.set_title("Y displacement per trajectory (to tune threshold)")
ax.legend()
plt.tight_layout()

plot_path = PLOT_DIR / "diagnostic_y_displacement.png"
plt.savefig(plot_path, dpi=150)
plt.show()
print(f"Saved: {plot_path}")
print(f"Trajectories with Y_disp >= {MIN_Y_DISPLACEMENT}: {(y_disp_per_traj >= MIN_Y_DISPLACEMENT).sum()} / {len(y_disp_per_traj)}")

# %%
# =============================================================================
# Filter: Remove pre-US stationary portion of each trajectory
# =============================================================================

print(f"\n=== Filter: Remove Pre-US Stationary Data ===")
print(f"Threshold: cumulative Y displacement >= {MIN_Y_DISPLACEMENT} um")

def find_motion_onset(group):
    """Find the frame where cumulative Y displacement exceeds threshold."""
    group = group.sort_values("fr")
    y_start = group["Y"].iloc[0]
    cumulative_disp = abs(group["Y"] - y_start)

    # Find first frame where displacement exceeds threshold
    onset_mask = cumulative_disp >= MIN_Y_DISPLACEMENT
    if onset_mask.any():
        onset_frame = group.loc[onset_mask, "fr"].iloc[0]
        return group[group["fr"] >= onset_frame]
    else:
        # Trajectory never exceeded threshold - remove entirely
        return group.iloc[0:0]

before_rows = len(data)
before_traj = data["ID"].nunique()

# Apply filter to each trajectory
data = data.groupby("ID", group_keys=False).apply(find_motion_onset)

print(f"Trajectories: {before_traj} -> {data['ID'].nunique()}")
print(f"Rows: {before_rows} -> {len(data)}")
print(f"Removed {before_rows - len(data)} pre-US stationary points")

# %%
# =============================================================================
# Filter by Track Length
# =============================================================================

print(f"\n=== Filter: Track Length >= {MIN_TRACK_LENGTH} ===")

# Count points per trajectory
track_lengths = data.groupby("ID").size()
short_tracks = track_lengths[track_lengths < MIN_TRACK_LENGTH].index

print(f"Trajectories before: {len(track_lengths)}")
print(f"Short trajectories (< {MIN_TRACK_LENGTH}): {len(short_tracks)}")

# Remove short trajectories
data_filtered = data[~data["ID"].isin(short_tracks)].copy()

print(f"Trajectories after: {data_filtered['ID'].nunique()}")
print(f"Rows after: {len(data_filtered)}")


# %%
# =============================================================================
# Visualization: X vs Y
# =============================================================================

print(f"\n=== Generating Plots ===")

fig, ax = plt.subplots(figsize=(10, 8))
ax.scatter(data_filtered["X"], data_filtered["Y"], s=1, alpha=0.5)
ax.set_xlabel("X (um)")
ax.set_ylabel("Y (um)")
ax.set_title(f"Filtered Positions (n={len(data_filtered)})")
ax.set_aspect("equal")
plt.tight_layout()

plot_path = PLOT_DIR / "filter_XY.png"
plt.savefig(plot_path, dpi=150)
plt.show()
print(f"Saved: {plot_path}")

# %%
# =============================================================================
# Visualization: Y vs DY
# =============================================================================

fig, ax = plt.subplots(figsize=(10, 8))
ax.scatter(data_filtered["Y"], data_filtered["DY"], s=1, alpha=0.5)
ax.set_xlabel("Y (um)")
ax.set_ylabel("DY (um/s)")
ax.set_title(f"Y vs DY (n={len(data_filtered)})")
plt.tight_layout()

plot_path = PLOT_DIR / "filter_Y_DY.png"
plt.savefig(plot_path, dpi=150)
plt.show()
print(f"Saved: {plot_path}")

# %%
# =============================================================================
# Visualization: Sample Distribution (XY heatmap)
# =============================================================================

n_x_bins, n_y_bins = 20, 20
x_edges = np.linspace(data_filtered["X"].min(), data_filtered["X"].max(), n_x_bins + 1)
y_edges = np.linspace(data_filtered["Y"].min(), data_filtered["Y"].max(), n_y_bins + 1)

counts, _, _ = np.histogram2d(data_filtered["X"], data_filtered["Y"], bins=[x_edges, y_edges])

fig, ax = plt.subplots(figsize=(12, 6))
cmap = plt.cm.viridis.copy()
cmap.set_under("lightgray")

im = ax.imshow(
    counts.T,
    origin="lower",
    aspect="auto",
    extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
    cmap=cmap,
    vmin=0.5,
    vmax=20,
)
ax.set_xlabel("X (um)")
ax.set_ylabel("Y (um)")
ax.set_title(f"Sample distribution ({n_x_bins}x{n_y_bins} bins, gray=no data)")
plt.colorbar(im, ax=ax, label="Count (capped at 20)", extend="both")
plt.tight_layout()

plot_path = PLOT_DIR / "sample_distribution_xy.png"
plt.savefig(plot_path, dpi=150)
plt.show()
print(f"Saved: {plot_path}")

empty_bins = (counts == 0).sum()
print(f"Empty bins: {empty_bins}/{counts.size} ({empty_bins/counts.size*100:.0f}%)")
print(f"Median count: {np.median(counts):.0f}")

# %%
# =============================================================================
# Save Filtered Data
# =============================================================================

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
data_filtered.to_csv(OUTPUT_FILE, index=False)
print(f"\nSaved: {OUTPUT_FILE}")

# %%
# =============================================================================
# Summary
# =============================================================================

print(f"\n=== Filter Summary ===")
print(f"Original rows: {len(data)}")
print(f"Filtered rows: {len(data_filtered)}")
print(f"Original trajectories: {data['ID'].nunique()}")
print(f"Filtered trajectories: {data_filtered['ID'].nunique()}")
print(f"Mean DY: {data_filtered['DY'].mean():.2f} um/s")
print(f"Std DY: {data_filtered['DY'].std():.2f} um/s")

print(f"\n=== Data Filter Complete ===")
