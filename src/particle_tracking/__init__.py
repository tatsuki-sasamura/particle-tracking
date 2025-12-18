"""Particle tracking analysis for microscopy images."""

__version__ = "0.1.0"

from .analysis import (
    add_time_column,
    compute_ensemble_stats,
    compute_trajectory_stats,
    compute_velocities,
    export_results,
)
from .detection import (
    batch_detect,
    detect,
    estimate_threshold,
    filter_particles,
)
from .io_utils import (
    get_frame_interval,
    get_pixel_size,
    iterate_frames,
    list_nd2_files,
    load_nd2_file,
)
from .tracking import (
    filter_trajectories,
    get_trajectory_lengths,
    link_trajectories,
    split_by_particle,
)

__all__ = [
    # io_utils
    "load_nd2_file",
    "get_pixel_size",
    "get_frame_interval",
    "iterate_frames",
    "list_nd2_files",
    # detection
    "detect",
    "batch_detect",
    "estimate_threshold",
    "filter_particles",
    # tracking
    "link_trajectories",
    "filter_trajectories",
    "get_trajectory_lengths",
    "split_by_particle",
    # analysis
    "compute_velocities",
    "compute_trajectory_stats",
    "compute_ensemble_stats",
    "export_results",
    "add_time_column",
]
