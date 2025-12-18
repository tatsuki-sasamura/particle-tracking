# %%
"""Fit acoustophoresis velocity model to tracking data.

Model: DY = A(x) * sin(2π(y - y0) / l)
- l = 375 um (known, channel width)
- y0 = unknown (node position)
- A(x) = a0 + a1*x + a2*x² (amplitude polynomial)

Uses density-based weighting to handle uneven sampling.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from config import OUTPUT_DIR

# %%
# =============================================================================
# Configuration
# =============================================================================

INPUT_FILE = OUTPUT_DIR / "summary" / "data_combine_filter.csv"
L = 375.0  # Channel width (um)

# Binning parameters
N_X_BINS = 20
N_Y_BINS = 20

# %%
# =============================================================================
# Load Data
# =============================================================================

print(f"Loading {INPUT_FILE}...")
df = pd.read_csv(INPUT_FILE)

print(f"Loaded {len(df)} points")
print(f"X range: {df['X'].min():.1f} - {df['X'].max():.1f} um")
print(f"Y range: {df['Y'].min():.1f} - {df['Y'].max():.1f} um")

# %%
# =============================================================================
# Step 1: Estimate y0 from Zero-Crossings
# =============================================================================

print(f"\n=== Step 1: Estimate y0 ===")

# Bin DY by Y and find where mean crosses zero
y_edges = np.linspace(df['Y'].min(), df['Y'].max(), 51)
y_centers = (y_edges[:-1] + y_edges[1:]) / 2

mean_dy_per_y = []
for i in range(len(y_edges) - 1):
    mask = (df['Y'] >= y_edges[i]) & (df['Y'] < y_edges[i + 1])
    if mask.sum() > 0:
        mean_dy_per_y.append(df.loc[mask, 'DY'].mean())
    else:
        mean_dy_per_y.append(np.nan)

mean_dy_per_y = np.array(mean_dy_per_y)

# Find zero crossing by interpolation
valid = ~np.isnan(mean_dy_per_y)
y_valid = y_centers[valid]
dy_valid = mean_dy_per_y[valid]

# Find sign changes
sign_changes = np.where(np.diff(np.sign(dy_valid)))[0]

if len(sign_changes) > 0:
    # Interpolate to find exact zero crossing
    i = sign_changes[0]
    y1, y2 = y_valid[i], y_valid[i + 1]
    dy1, dy2 = dy_valid[i], dy_valid[i + 1]
    y0_estimate = y1 - dy1 * (y2 - y1) / (dy2 - dy1)
else:
    # Fallback: use L/2
    y0_estimate = L / 2

print(f"y0 estimate from zero-crossing: {y0_estimate:.1f} um")
print(f"Expected (L/2): {L/2:.1f} um")

# Plot mean DY vs Y
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(y_centers, mean_dy_per_y, '.-')
ax.axhline(0, color='r', linestyle='--', alpha=0.5)
ax.axvline(y0_estimate, color='g', linestyle='--', label=f'y0 = {y0_estimate:.1f}')
ax.set_xlabel('Y (um)')
ax.set_ylabel('Mean DY (um/s)')
ax.set_title('Mean DY vs Y (for y0 estimation)')
ax.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "summary" / "fitting_y0_estimate.png", dpi=150)
plt.show()
print(f"Saved: {OUTPUT_DIR / 'summary' / 'fitting_y0_estimate.png'}")

# %%
# =============================================================================
# Step 2: Fit A for Each X Bin
# =============================================================================

print(f"\n=== Step 2: Fit A per X Bin ===")

# Create x bins
x_edges = np.linspace(df['X'].min(), df['X'].max(), N_X_BINS + 1)
x_centers = (x_edges[:-1] + x_edges[1:]) / 2

# Use estimated y0
y0_fit = y0_estimate


def model_A(y, A):
    """DY = A * sin(2π(y - y0) / L) with fixed y0"""
    return A * np.sin(2 * np.pi * (y - y0_fit) / L)


# Fit A for each x bin
A_results = []

for i in range(N_X_BINS):
    mask = (df['X'] >= x_edges[i]) & (df['X'] < x_edges[i + 1])
    subset = df[mask]

    if len(subset) < 10:
        A_results.append({
            'x': x_centers[i],
            'A': np.nan,
            'A_err': np.nan,
            'n': len(subset),
        })
        continue

    try:
        popt, pcov = curve_fit(model_A, subset['Y'].values, subset['DY'].values, p0=[-300])
        A = popt[0]
        A_err = np.sqrt(pcov[0, 0])
        A_results.append({
            'x': x_centers[i],
            'A': A,
            'A_err': A_err,
            'n': len(subset),
        })
    except Exception as e:
        A_results.append({
            'x': x_centers[i],
            'A': np.nan,
            'A_err': np.nan,
            'n': len(subset),
        })

A_df = pd.DataFrame(A_results)

print(f"X bins with valid A: {(~A_df['A'].isna()).sum()} / {N_X_BINS}")
valid_A = A_df[~A_df['A'].isna()]
print(f"A range: {valid_A['A'].min():.1f} to {valid_A['A'].max():.1f} um/s")
print(f"|A| range: {valid_A['A'].abs().min():.1f} to {valid_A['A'].abs().max():.1f} um/s")

# %%
# =============================================================================
# Diagnostic Plots
# =============================================================================

print(f"\n=== Diagnostic Plots ===")

# Plot 1: A(x) per bin
fig, ax = plt.subplots(figsize=(10, 5))
valid_mask = ~A_df['A'].isna()
ax.errorbar(A_df.loc[valid_mask, 'x'], A_df.loc[valid_mask, 'A'],
            yerr=A_df.loc[valid_mask, 'A_err'], fmt='o-', capsize=3)
ax.axhline(0, color='r', linestyle='--', alpha=0.5)
ax.set_xlabel('X (um)')
ax.set_ylabel('A (um/s)')
ax.set_title(f'Fitted amplitude A per X bin (y0 = {y0_fit:.1f} um)')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "summary" / "fitting_Ax.png", dpi=150)
plt.show()
print(f"Saved: {OUTPUT_DIR / 'summary' / 'fitting_Ax.png'}")

# Plot 2: Y vs DY with fitted curves for different X bins
fig, ax = plt.subplots(figsize=(10, 6))

# Scatter raw data (downsampled for visibility)
sample = df.sample(min(5000, len(df)))
ax.scatter(sample['Y'], sample['DY'], s=1, alpha=0.3, c='gray', label='Data')

# Plot fitted curves for a few X bins
y_plot = np.linspace(df['Y'].min(), df['Y'].max(), 200)
colors = plt.cm.viridis(np.linspace(0, 1, len(A_df)))

for idx, row in A_df.iterrows():
    if np.isnan(row['A']):
        continue
    dy_fit = row['A'] * np.sin(2 * np.pi * (y_plot - y0_fit) / L)
    ax.plot(y_plot, dy_fit, '-', color=colors[idx], alpha=0.5, linewidth=1)

ax.axhline(0, color='k', linestyle='--', alpha=0.3)
ax.axvline(y0_fit, color='r', linestyle='--', alpha=0.5, label=f'y0={y0_fit:.1f}')
ax.set_xlabel('Y (um)')
ax.set_ylabel('DY (um/s)')
ax.set_title('Y vs DY with Fitted Curves (colored by X)')
ax.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "summary" / "fitting_Y_vs_DY.png", dpi=150)
plt.show()
print(f"Saved: {OUTPUT_DIR / 'summary' / 'fitting_Y_vs_DY.png'}")

# Plot 3: Fitted vs Observed DY (2D heatmaps side by side)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

y_edges = np.linspace(df['Y'].min(), df['Y'].max(), N_Y_BINS + 1)

# Compute observed and fitted mean DY per bin
observed_dy = np.full((N_X_BINS, N_Y_BINS), np.nan)
fitted_dy = np.full((N_X_BINS, N_Y_BINS), np.nan)

for i in range(N_X_BINS):
    for j in range(N_Y_BINS):
        mask = (
            (df['X'] >= x_edges[i]) & (df['X'] < x_edges[i + 1]) &
            (df['Y'] >= y_edges[j]) & (df['Y'] < y_edges[j + 1])
        )
        if mask.sum() > 0:
            observed_dy[i, j] = df.loc[mask, 'DY'].mean()
            if not np.isnan(A_df.loc[i, 'A']):
                y_center = (y_edges[j] + y_edges[j + 1]) / 2
                fitted_dy[i, j] = A_df.loc[i, 'A'] * np.sin(2 * np.pi * (y_center - y0_fit) / L)

cmap = plt.cm.RdBu_r.copy()
cmap.set_bad('lightgray')
vmax = np.nanmax(np.abs(observed_dy))

ax = axes[0]
im = ax.imshow(observed_dy.T, origin='lower', aspect='auto',
               extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
               cmap=cmap, vmin=-vmax, vmax=vmax)
ax.set_xlabel('X (um)')
ax.set_ylabel('Y (um)')
ax.set_title('Observed Mean DY')
plt.colorbar(im, ax=ax, label='DY (um/s)')

ax = axes[1]
im = ax.imshow(fitted_dy.T, origin='lower', aspect='auto',
               extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
               cmap=cmap, vmin=-vmax, vmax=vmax)
ax.set_xlabel('X (um)')
ax.set_ylabel('Y (um)')
ax.set_title('Fitted DY')
plt.colorbar(im, ax=ax, label='DY (um/s)')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "summary" / "fitting_comparison.png", dpi=150)
plt.show()
print(f"Saved: {OUTPUT_DIR / 'summary' / 'fitting_comparison.png'}")

# %%
# =============================================================================
# Summary
# =============================================================================

print(f"\n{'='*50}")
print("=== Fitting Summary ===")
print(f"{'='*50}")
print(f"Model: DY = A(x) * sin(2π(y - y0) / {L})")
print(f"")
print(f"Fitted parameters:")
print(f"  y0 = {y0_fit:.2f} um (node position)")
print(f"  A fitted per X bin (not polynomial)")
print(f"")
print(f"A(x) statistics:")
print(f"  A range: {valid_A['A'].min():.1f} to {valid_A['A'].max():.1f} um/s")
print(f"  |A| range: {valid_A['A'].abs().min():.1f} to {valid_A['A'].abs().max():.1f} um/s")
print(f"  |A| mean: {valid_A['A'].abs().mean():.1f} um/s")
print(f"  Max |A| at X = {valid_A.loc[valid_A['A'].abs().idxmax(), 'x']:.0f} um")
print(f"{'='*50}")

# Save A(x) results to CSV
A_df.to_csv(OUTPUT_DIR / "summary" / "fitting_A_per_x.csv", index=False)
print(f"\nSaved: {OUTPUT_DIR / 'summary' / 'fitting_A_per_x.csv'}")
