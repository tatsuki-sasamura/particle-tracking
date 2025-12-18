# %%
"""Trajectory linking for particle tracking."""

import pandas as pd
import trackpy as tp


# %%
def link_trajectories(
    features: pd.DataFrame,
    search_range: float = 15,
    memory: int = 3,
    **kwargs,
) -> pd.DataFrame:
    """Link detected particles into trajectories.

    Parameters
    ----------
    features : pd.DataFrame
        DataFrame from batch_detect with 'frame', 'x', 'y' columns.
    search_range : float, optional
        Maximum displacement per frame in pixels (default: 15).
    memory : int, optional
        Number of frames a particle can disappear and reappear (default: 3).
    **kwargs
        Additional arguments passed to trackpy.link.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with added 'particle' column for trajectory ID.
    """
    if features.empty:
        return features

    # Ensure required columns exist
    if "frame" not in features.columns:
        raise ValueError("Features must have 'frame' column")

    trajectories = tp.link(
        features,
        search_range=search_range,
        memory=memory,
        **kwargs,
    )

    return trajectories


# %%
def filter_trajectories(
    trajectories: pd.DataFrame,
    min_length: int = 10,
) -> pd.DataFrame:
    """Filter out short trajectories.

    Parameters
    ----------
    trajectories : pd.DataFrame
        DataFrame with linked trajectories.
    min_length : int, optional
        Minimum number of frames for a trajectory (default: 10).

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame containing only long trajectories.
    """
    if trajectories.empty or "particle" not in trajectories.columns:
        return trajectories

    # Use trackpy's filter_stubs for efficient filtering
    filtered = tp.filter_stubs(trajectories, threshold=min_length)

    return filtered.reset_index(drop=True)


# %%
def get_trajectory_lengths(trajectories: pd.DataFrame) -> pd.Series:
    """Get the length (number of frames) of each trajectory.

    Parameters
    ----------
    trajectories : pd.DataFrame
        DataFrame with linked trajectories.

    Returns
    -------
    pd.Series
        Series with particle ID as index and trajectory length as values.
    """
    if trajectories.empty or "particle" not in trajectories.columns:
        return pd.Series(dtype=int)

    return trajectories.groupby("particle").size()


# %%
def split_by_particle(trajectories: pd.DataFrame) -> dict[int, pd.DataFrame]:
    """Split trajectories DataFrame into individual particle trajectories.

    Parameters
    ----------
    trajectories : pd.DataFrame
        DataFrame with linked trajectories.

    Returns
    -------
    dict[int, pd.DataFrame]
        Dictionary mapping particle ID to its trajectory DataFrame.
    """
    if trajectories.empty or "particle" not in trajectories.columns:
        return {}

    return {
        pid: group.sort_values("frame").reset_index(drop=True)
        for pid, group in trajectories.groupby("particle")
    }
