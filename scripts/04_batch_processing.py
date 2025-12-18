# %%
"""Batch processing script for all ND2 files.

Processes all ND2 files in the data directory using parameters from config.py.
Detection uses DefocusTracker Method 0 (boundary_threshold_2d).
"""

import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from particle_tracking import (
    add_time_column,
    batch_detect,
    compute_ensemble_stats,
    compute_velocities,
    export_results,
    filter_trajectories,
    link_trajectories,
    list_nd2_files,
    load_nd2_file,
)

from config import (
    DATA_DIR,
    DETECT_PARAMS,
    FRAME_INTERVAL,
    MEMORY,
    MIN_TRAJ_LENGTH,
    OUTPUT_DIR,
    PIXEL_SIZE,
    SEARCH_RANGE,
)

# %%
# =============================================================================
# List Files
# =============================================================================

nd2_files = list_nd2_files(DATA_DIR)
print(f"Found {len(nd2_files)} ND2 files in {DATA_DIR}")
for f in nd2_files[:5]:
    print(f"  {f.name}")
if len(nd2_files) > 5:
    print(f"  ... and {len(nd2_files) - 5} more")

print(f"\nDetection: DefocusTracker Method 0")
print(f"Parameters: {DETECT_PARAMS}")

# %%
# =============================================================================
# Processing Function
# =============================================================================


def process_single_file(file_path: Path) -> dict:
    """Process a single ND2 file and return statistics."""
    file_stem = file_path.stem

    # Load
    frames, metadata = load_nd2_file(file_path)
    pixel_size = metadata.get("pixel_size", PIXEL_SIZE)
    frame_interval = metadata.get("frame_interval", FRAME_INTERVAL)

    # Detect
    all_particles = batch_detect(
        frames,
        show_progress=False,
        **DETECT_PARAMS,
    )

    if len(all_particles) == 0:
        return {
            "source_file": file_stem,
            "n_detections": 0,
            "n_trajectories": 0,
            "mean_speed": float("nan"),
            "std_speed": float("nan"),
            "status": "no_particles",
        }

    # Link
    trajectories = link_trajectories(
        all_particles,
        search_range=SEARCH_RANGE,
        memory=MEMORY,
    )

    # Filter
    filtered = filter_trajectories(trajectories, min_length=MIN_TRAJ_LENGTH)

    if filtered["particle"].nunique() == 0:
        return {
            "source_file": file_stem,
            "n_detections": len(all_particles),
            "n_trajectories": 0,
            "mean_speed": float("nan"),
            "std_speed": float("nan"),
            "status": "no_trajectories",
        }

    # Compute velocities
    velocities = compute_velocities(
        filtered,
        pixel_size=pixel_size,
        dt=frame_interval,
    )
    velocities = add_time_column(velocities, dt=frame_interval)

    # Add metadata columns
    filtered["source_file"] = file_stem
    velocities["source_file"] = file_stem

    # Export
    export_results(
        filtered,
        velocities,
        OUTPUT_DIR / "trajectories",
        prefix=file_stem,
    )

    # Get stats
    stats = compute_ensemble_stats(velocities, source_file=file_stem)
    stats["n_detections"] = len(all_particles)
    stats["status"] = "success"

    return stats


# %%
# =============================================================================
# Process All Files
# =============================================================================

print(f"\n=== Processing All Files ===")
all_stats = []

for file_path in tqdm(nd2_files, desc="Processing files"):
    try:
        stats = process_single_file(file_path)
        all_stats.append(stats)
    except Exception as e:
        all_stats.append({
            "source_file": file_path.stem,
            "n_detections": 0,
            "n_trajectories": 0,
            "mean_speed": float("nan"),
            "std_speed": float("nan"),
            "status": f"error: {str(e)[:50]}",
        })

# %%
# =============================================================================
# Summary
# =============================================================================

summary_df = pd.DataFrame(all_stats)

print(f"\n=== Processing Summary ===")
print(f"Detection: DefocusTracker Method 0")
print(f"Parameters: {DETECT_PARAMS}")
print(f"Total files: {len(summary_df)}")
print(f"Successful: {(summary_df['status'] == 'success').sum()}")
print(f"No particles: {(summary_df['status'] == 'no_particles').sum()}")
print(f"No trajectories: {(summary_df['status'] == 'no_trajectories').sum()}")
print(f"Errors: {summary_df['status'].str.startswith('error').sum()}")

# %%
# Save summary
summary_path = OUTPUT_DIR / "summary" / "batch_summary.csv"
summary_path.parent.mkdir(parents=True, exist_ok=True)
summary_df.to_csv(summary_path, index=False)
print(f"\nSaved: {summary_path}")

# %%
# Aggregate statistics
print(f"\n=== Aggregate Statistics ===")
successful = summary_df[summary_df["status"] == "success"]

if len(successful) > 0:
    print(f"Files with trajectories: {len(successful)}")
    print(f"Total trajectories: {successful['n_trajectories'].sum():.0f}")
    print(f"Total detections: {successful['n_detections'].sum():.0f}")
    print(f"Mean speed across files: {successful['mean_speed'].mean():.2f} um/s")
    print(f"Std of mean speeds: {successful['mean_speed'].std():.2f} um/s")
else:
    print("No successful files to analyze")

# %%
# Show summary table
print(f"\n=== Summary Table ===")
display_cols = ["source_file", "n_detections", "n_trajectories", "mean_speed", "status"]
display_cols = [c for c in display_cols if c in summary_df.columns]
print(summary_df[display_cols].to_string())

# %%
print(f"\n=== Batch Processing Complete ===")
