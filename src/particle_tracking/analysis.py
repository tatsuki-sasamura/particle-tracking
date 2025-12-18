# %%
"""Velocity analysis and statistics for particle trajectories."""

from pathlib import Path

import numpy as np
import pandas as pd


# %%
def compute_velocities(
    trajectories: pd.DataFrame,
    pixel_size: float = 0.65,
    dt: float = 0.007,
) -> pd.DataFrame:
    """Compute instantaneous velocities for all trajectories.

    Parameters
    ----------
    trajectories : pd.DataFrame
        DataFrame with linked trajectories (must have 'particle', 'frame', 'x', 'y').
    pixel_size : float, optional
        Pixel size in micrometers (default: 0.65).
    dt : float, optional
        Time between frames in seconds (default: 0.007).

    Returns
    -------
    pd.DataFrame
        DataFrame with velocity data for each particle at each frame.
        Columns: particle, frame, x, y, dx, dy, vx, vy, speed, direction.
    """
    if trajectories.empty:
        return pd.DataFrame()

    results = []

    for pid, group in trajectories.groupby("particle"):
        group = group.sort_values("frame").reset_index(drop=True)

        # Compute displacements
        dx = group["x"].diff()
        dy = group["y"].diff()

        # Convert to physical units (um/s)
        vx = dx * pixel_size / dt
        vy = dy * pixel_size / dt

        # Compute speed and direction
        speed = np.sqrt(vx**2 + vy**2)
        direction = np.degrees(np.arctan2(vy, vx))

        vel_df = pd.DataFrame({
            "particle": pid,
            "frame": group["frame"],
            "x": group["x"],
            "y": group["y"],
            "x_um": group["x"] * pixel_size,
            "y_um": group["y"] * pixel_size,
            "dx": dx,
            "dy": dy,
            "vx": vx,
            "vy": vy,
            "speed": speed,
            "direction": direction,
        })

        results.append(vel_df)

    if results:
        return pd.concat(results, ignore_index=True)
    return pd.DataFrame()


# %%
def compute_trajectory_stats(trajectories: pd.DataFrame) -> pd.DataFrame:
    """Compute statistics for each trajectory.

    Parameters
    ----------
    trajectories : pd.DataFrame
        DataFrame with velocity data from compute_velocities.

    Returns
    -------
    pd.DataFrame
        Summary statistics per trajectory.
    """
    if trajectories.empty or "particle" not in trajectories.columns:
        return pd.DataFrame()

    stats = []

    for pid, group in trajectories.groupby("particle"):
        valid_speeds = group["speed"].dropna()

        stat = {
            "particle": pid,
            "n_frames": len(group),
            "start_frame": group["frame"].min(),
            "end_frame": group["frame"].max(),
            "x_start": group["x"].iloc[0],
            "y_start": group["y"].iloc[0],
            "x_end": group["x"].iloc[-1],
            "y_end": group["y"].iloc[-1],
            "mean_speed": valid_speeds.mean() if len(valid_speeds) > 0 else np.nan,
            "std_speed": valid_speeds.std() if len(valid_speeds) > 0 else np.nan,
            "max_speed": valid_speeds.max() if len(valid_speeds) > 0 else np.nan,
            "total_distance": valid_speeds.sum() * 0.007 if len(valid_speeds) > 0 else 0,
        }

        # Net displacement
        net_dx = stat["x_end"] - stat["x_start"]
        net_dy = stat["y_end"] - stat["y_start"]
        stat["net_displacement"] = np.sqrt(net_dx**2 + net_dy**2) * 0.65

        stats.append(stat)

    return pd.DataFrame(stats)


# %%
def compute_ensemble_stats(
    velocities: pd.DataFrame,
    source_file: str | None = None,
) -> dict:
    """Compute ensemble statistics across all trajectories.

    Parameters
    ----------
    velocities : pd.DataFrame
        DataFrame with velocity data from compute_velocities.
    source_file : str, optional
        Source filename to include in stats.

    Returns
    -------
    dict
        Dictionary with ensemble statistics.
    """
    valid_speeds = velocities["speed"].dropna()

    stats = {
        "source_file": source_file or "unknown",
        "n_trajectories": velocities["particle"].nunique() if "particle" in velocities else 0,
        "mean_speed": valid_speeds.mean() if len(valid_speeds) > 0 else np.nan,
        "std_speed": valid_speeds.std() if len(valid_speeds) > 0 else np.nan,
        "median_speed": valid_speeds.median() if len(valid_speeds) > 0 else np.nan,
        "max_speed": valid_speeds.max() if len(valid_speeds) > 0 else np.nan,
        "min_speed": valid_speeds.min() if len(valid_speeds) > 0 else np.nan,
    }

    return stats


# %%
def export_results(
    trajectories: pd.DataFrame,
    velocities: pd.DataFrame,
    output_dir: str | Path,
    prefix: str = "",
) -> dict[str, Path]:
    """Export trajectories and velocities to CSV files.

    Parameters
    ----------
    trajectories : pd.DataFrame
        Raw trajectory data.
    velocities : pd.DataFrame
        Velocity analysis results.
    output_dir : str or Path
        Output directory.
    prefix : str, optional
        Filename prefix (e.g., source file name).

    Returns
    -------
    dict[str, Path]
        Dictionary with paths to exported files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prefix = f"{prefix}_" if prefix else ""

    paths = {}

    # Export trajectories
    traj_path = output_dir / f"{prefix}trajectories.csv"
    trajectories.to_csv(traj_path, index=False)
    paths["trajectories"] = traj_path

    # Export velocities
    vel_path = output_dir / f"{prefix}velocities.csv"
    velocities.to_csv(vel_path, index=False)
    paths["velocities"] = vel_path

    return paths


# %%
def add_time_column(
    df: pd.DataFrame,
    dt: float = 0.007,
) -> pd.DataFrame:
    """Add time column in seconds based on frame number.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'frame' column.
    dt : float, optional
        Time between frames in seconds (default: 0.007).

    Returns
    -------
    pd.DataFrame
        DataFrame with added 'time_s' column.
    """
    if "frame" in df.columns:
        df = df.copy()
        df["time_s"] = df["frame"] * dt
    return df
