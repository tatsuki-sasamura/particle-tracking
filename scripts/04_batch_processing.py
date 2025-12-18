# %%
"""Batch processing script for all ND2 files."""

import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

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

# %%
# Configuration
DATA_DIR = Path(__file__).parent.parent / "20251218test-nofixture2"
OUTPUT_DIR = Path(__file__).parent.parent / "output"

# Detection parameters
# Note: Large diameter/separation for defocused particles (ring + center pattern)
DIAMETER = 21
MINMASS = 500
SEPARATION = 25

# Tracking parameters
SEARCH_RANGE = 15
MEMORY = 3
MIN_TRAJ_LENGTH = 10

# Physical parameters
PIXEL_SIZE = 0.65
FRAME_INTERVAL = 0.007

# %%
# List all ND2 files
nd2_files = list_nd2_files(DATA_DIR)
print(f"Found {len(nd2_files)} ND2 files in {DATA_DIR}")
for f in nd2_files[:5]:
    print(f"  {f.name}")
if len(nd2_files) > 5:
    print(f"  ... and {len(nd2_files) - 5} more")


# %%
def process_single_file(file_path: Path, quiet: bool = True) -> dict:
    """Process a single ND2 file and return statistics."""
    file_stem = file_path.stem

    # Load
    frames, metadata = load_nd2_file(file_path)

    # Detect
    all_particles = batch_detect(
        frames,
        diameter=DIAMETER,
        minmass=MINMASS,
        separation=SEPARATION,
        show_progress=False,
    )

    if len(all_particles) == 0:
        return {
            "source_file": file_stem,
            "n_particles": 0,
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
            "n_particles": len(all_particles),
            "n_trajectories": 0,
            "mean_speed": float("nan"),
            "std_speed": float("nan"),
            "status": "no_trajectories",
        }

    # Compute velocities
    velocities = compute_velocities(
        filtered,
        pixel_size=PIXEL_SIZE,
        dt=FRAME_INTERVAL,
    )
    velocities = add_time_column(velocities, dt=FRAME_INTERVAL)

    # Add source file
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
    stats["status"] = "success"

    return stats


# %%
# Process all files
print("\n=== Processing All Files ===")
all_stats = []

for file_path in tqdm(nd2_files, desc="Processing files"):
    try:
        stats = process_single_file(file_path, quiet=True)
        all_stats.append(stats)
    except Exception as e:
        all_stats.append({
            "source_file": file_path.stem,
            "n_particles": 0,
            "n_trajectories": 0,
            "mean_speed": float("nan"),
            "std_speed": float("nan"),
            "status": f"error: {str(e)[:50]}",
        })

# %%
# Create summary DataFrame
summary_df = pd.DataFrame(all_stats)
print("\n=== Processing Summary ===")
print(f"Total files: {len(summary_df)}")
print(f"Successful: {(summary_df['status'] == 'success').sum()}")
print(f"No particles: {(summary_df['status'] == 'no_particles').sum()}")
print(f"No trajectories: {(summary_df['status'] == 'no_trajectories').sum()}")
print(f"Errors: {summary_df['status'].str.startswith('error').sum()}")

# %%
# Save summary
summary_path = OUTPUT_DIR / "summary" / "summary_stats.csv"
summary_df.to_csv(summary_path, index=False)
print(f"\nSaved: {summary_path}")

# %%
# Display statistics
print("\n=== Aggregate Statistics ===")
successful = summary_df[summary_df["status"] == "success"]
if len(successful) > 0:
    print(f"Files with trajectories: {len(successful)}")
    print(f"Total trajectories: {successful['n_trajectories'].sum():.0f}")
    print(f"Mean speed across files: {successful['mean_speed'].mean():.2f} um/s")
    print(f"Std of mean speeds: {successful['mean_speed'].std():.2f} um/s")

# %%
# Show summary table
print("\n=== Summary Table ===")
print(summary_df.to_string())

# %%
# Checkpoint 8 verification
print("\n=== Checkpoint 8 Verification ===")
import glob

traj_files = list(Path(OUTPUT_DIR / "trajectories").glob("*_trajectories.csv"))
print(f"Trajectory files created: {len(traj_files)}")
print(f"Expected: {len(nd2_files)}")

if summary_path.exists():
    print(f"Summary file: OK")
else:
    print(f"Summary file: MISSING")

if len(traj_files) >= len(nd2_files) * 0.8:  # Allow some failures
    print("\nCheckpoint 8 PASSED!")
else:
    print("\nCheckpoint 8 PARTIAL - some files may have failed")

# %%
print("\n=== Batch Processing Complete ===")
