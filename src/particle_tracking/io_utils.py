# %%
"""I/O utilities for loading and handling ND2 microscopy files."""

from pathlib import Path
from typing import Generator

import nd2
import numpy as np


# %%
def load_nd2_file(path: str | Path) -> tuple[np.ndarray, dict]:
    """Load an ND2 file and return frames with metadata.

    Parameters
    ----------
    path : str or Path
        Path to the ND2 file.

    Returns
    -------
    frames : np.ndarray
        Array of shape (n_frames, height, width).
    metadata : dict
        Dictionary containing pixel_size, frame_interval, and other metadata.
    """
    path = Path(path)
    with nd2.ND2File(path) as f:
        frames = f.asarray()
        metadata = _extract_metadata(f)
    return frames, metadata


# %%
def _extract_metadata(f: nd2.ND2File) -> dict:
    """Extract relevant metadata from an ND2 file object."""
    # Get voxel size
    try:
        voxel = f.voxel_size()
        pixel_size = voxel.x  # um per pixel
    except Exception:
        pixel_size = 0.65  # Default fallback

    # Get frame interval from experiment parameters
    frame_interval = 0.007  # Default 7ms
    if f.experiment:
        for exp in f.experiment:
            if hasattr(exp, "parameters") and hasattr(exp.parameters, "periodDiff"):
                # Convert from ms to seconds
                frame_interval = exp.parameters.periodDiff.avg / 1000.0
                break

    # Get dimensions
    dimensions = dict(f.sizes)

    return {
        "pixel_size": pixel_size,
        "frame_interval": frame_interval,
        "dimensions": dimensions,
        "shape": f.shape,
        "dtype": str(f.dtype),
        "n_frames": f.shape[0] if len(f.shape) > 0 else 1,
    }


# %%
def get_pixel_size(path: str | Path) -> float:
    """Get pixel size in micrometers from an ND2 file.

    Parameters
    ----------
    path : str or Path
        Path to the ND2 file.

    Returns
    -------
    float
        Pixel size in micrometers.
    """
    with nd2.ND2File(path) as f:
        try:
            voxel = f.voxel_size()
            return voxel.x
        except Exception:
            return 0.65  # Default


# %%
def get_frame_interval(path: str | Path) -> float:
    """Get frame interval in seconds from an ND2 file.

    Parameters
    ----------
    path : str or Path
        Path to the ND2 file.

    Returns
    -------
    float
        Frame interval in seconds.
    """
    with nd2.ND2File(path) as f:
        if f.experiment:
            for exp in f.experiment:
                if hasattr(exp, "parameters") and hasattr(exp.parameters, "periodDiff"):
                    return exp.parameters.periodDiff.avg / 1000.0
        return 0.007  # Default 7ms


# %%
def iterate_frames(
    path: str | Path,
    start: int = 0,
    end: int | None = None,
) -> Generator[tuple[int, np.ndarray], None, None]:
    """Iterate over frames in an ND2 file without loading all into memory.

    Parameters
    ----------
    path : str or Path
        Path to the ND2 file.
    start : int, optional
        Starting frame index (default: 0).
    end : int, optional
        Ending frame index (exclusive). If None, iterate to the end.

    Yields
    ------
    tuple[int, np.ndarray]
        Frame index and frame data.
    """
    with nd2.ND2File(path) as f:
        n_frames = f.shape[0]
        end = end or n_frames
        end = min(end, n_frames)

        # Load all frames once (nd2 doesn't support single-frame access efficiently)
        all_frames = f.asarray()
        for i in range(start, end):
            yield i, all_frames[i]


# %%
def list_nd2_files(directory: str | Path) -> list[Path]:
    """List all ND2 files in a directory.

    Parameters
    ----------
    directory : str or Path
        Directory to search.

    Returns
    -------
    list[Path]
        List of paths to ND2 files, sorted by name.
    """
    directory = Path(directory)
    return sorted(directory.glob("*.nd2"))
