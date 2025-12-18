# %%
"""Batch processing script for all ND2 files.

Processes all ND2 files in the data directory using parameters from config.py.
Detection uses DefocusTracker Method 0 (boundary_threshold_2d).

Uses multiprocessing to process files in parallel.
"""

import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for multiprocessing
import matplotlib.pyplot as plt
import numpy as np
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
    VIS_FRAMES,
)

# %%
# =============================================================================
# Configuration
# =============================================================================

MAX_WORKERS = None  # Number of parallel workers (set to None for auto)
GENERATE_IMAGES = True  # Generate trajectory visualization images

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
print(f"Parallel workers: {MAX_WORKERS or 'auto'}")
print(f"Generate images: {GENERATE_IMAGES}")

# %%
# =============================================================================
# Processing Function
# =============================================================================


def generate_trajectory_images(
    frames: np.ndarray,
    all_particles: pd.DataFrame | None,
    filtered: pd.DataFrame | None,
    file_stem: str,
    output_dir: Path,
    vis_frames: list[int],
    frame_interval: float,
    status: str,
) -> None:
    """Generate trajectory visualization images."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for t in vis_frames:
        if t >= len(frames):
            continue

        fig, ax = plt.subplots(figsize=(20, 4))
        ax.imshow(frames[t], cmap="gray", vmin=0, vmax=frames[t].max() * 0.3)

        n_detections = 0
        n_traj = 0

        # Draw detections (circles) if available
        if all_particles is not None and len(all_particles) > 0:
            frame_particles = all_particles[all_particles["frame"] == t]
            n_detections = len(frame_particles)
            if n_detections > 0:
                ax.scatter(
                    frame_particles["x"], frame_particles["y"],
                    s=100, facecolors="none", edgecolors="cyan", linewidths=1, alpha=0.5
                )

        # Draw trajectories if available
        if filtered is not None and len(filtered) > 0:
            traj_up_to_t = filtered[filtered["frame"] <= t]
            n_traj = traj_up_to_t["particle"].nunique()

            for pid in traj_up_to_t["particle"].unique():
                traj = traj_up_to_t[traj_up_to_t["particle"] == pid]
                ax.plot(traj["x"], traj["y"], linewidth=1.5, alpha=0.8)

                current = traj[traj["frame"] == t]
                if len(current) > 0:
                    ax.scatter(current["x"], current["y"], s=30, c="red", zorder=5)

        time_ms = t * frame_interval * 1000
        title = f"{file_stem} - Frame {t} (t={time_ms:.0f}ms)"
        title += f" - {n_detections} detections, {n_traj} trajectories"
        if status != "success":
            title += f" [{status}]"
        ax.set_title(title)
        ax.axis("off")

        plt.tight_layout()
        plt.savefig(output_dir / f"{file_stem}_traj_{t:03d}.png", dpi=100)
        plt.close()


def process_single_file(file_path: Path) -> dict:
    """Process a single ND2 file and return statistics."""
    file_stem = file_path.stem

    try:
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

        # Initialize variables
        filtered = None
        velocities = None
        status = "success"
        n_trajectories = 0
        mean_speed = float("nan")
        std_speed = float("nan")

        if len(all_particles) == 0:
            status = "no_particles"
        else:
            # Link
            trajectories = link_trajectories(
                all_particles,
                search_range=SEARCH_RANGE,
                memory=MEMORY,
            )

            # Filter
            filtered = filter_trajectories(trajectories, min_length=MIN_TRAJ_LENGTH)
            n_trajectories = filtered["particle"].nunique()

            if n_trajectories == 0:
                status = "no_trajectories"
            else:
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

                # Export CSVs
                export_results(
                    filtered,
                    velocities,
                    OUTPUT_DIR / "trajectories",
                    prefix=file_stem,
                )

                # Get speed stats
                valid_speeds = velocities["speed"].dropna()
                if len(valid_speeds) > 0:
                    mean_speed = valid_speeds.mean()
                    std_speed = valid_speeds.std()

        # Generate trajectory images (always, regardless of status)
        if GENERATE_IMAGES:
            generate_trajectory_images(
                frames=frames,
                all_particles=all_particles if len(all_particles) > 0 else None,
                filtered=filtered,
                file_stem=file_stem,
                output_dir=OUTPUT_DIR / "04_batch_processing",
                vis_frames=VIS_FRAMES,
                frame_interval=frame_interval,
                status=status,
            )

        return {
            "source_file": file_stem,
            "n_detections": len(all_particles),
            "n_trajectories": n_trajectories,
            "mean_speed": mean_speed,
            "std_speed": std_speed,
            "status": status,
        }

    except Exception as e:
        return {
            "source_file": file_stem,
            "n_detections": 0,
            "n_trajectories": 0,
            "mean_speed": float("nan"),
            "std_speed": float("nan"),
            "status": f"error: {str(e)[:50]}",
        }


# %%
# =============================================================================
# Process All Files (Parallel)
# =============================================================================

if __name__ == "__main__":
    print(f"\n=== Processing All Files ===")
    all_stats = []

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all tasks
        futures = {executor.submit(process_single_file, f): f for f in nd2_files}

        # Collect results with progress bar
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing files"):
            stats = future.result()
            all_stats.append(stats)

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

    if GENERATE_IMAGES:
        print(f"\nImages saved to: {OUTPUT_DIR / '04_batch_processing'}")

    # %%
    print(f"\n=== Batch Processing Complete ===")
