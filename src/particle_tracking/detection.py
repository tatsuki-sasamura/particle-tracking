# %%
"""Particle detection using trackpy."""

import numpy as np
import pandas as pd
import trackpy as tp
from tqdm import tqdm

from .preprocessing import preprocess_frame


# %%
def detect_particles(
    frame: np.ndarray,
    diameter: int = 11,
    minmass: float | None = None,
    separation: float | None = None,
    preprocess: bool = True,
    **kwargs,
) -> pd.DataFrame:
    """Detect particles in a single frame.

    Parameters
    ----------
    frame : np.ndarray
        Input image.
    diameter : int, optional
        Expected particle diameter in pixels (must be odd, default: 11).
    minmass : float, optional
        Minimum integrated brightness for a particle.
    separation : float, optional
        Minimum separation between particles (default: diameter).
    preprocess : bool, optional
        Whether to preprocess the frame (default: True).
    **kwargs
        Additional arguments passed to trackpy.locate.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: x, y, mass, size, ecc, signal, raw_mass, etc.
    """
    if preprocess:
        frame = preprocess_frame(frame)

    # Ensure diameter is odd
    if diameter % 2 == 0:
        diameter += 1

    # Set defaults
    if separation is None:
        separation = diameter

    # Detect particles
    features = tp.locate(
        frame,
        diameter=diameter,
        minmass=minmass,
        separation=separation,
        **kwargs,
    )

    return features


# %%
def batch_detect(
    frames: np.ndarray,
    diameter: int = 11,
    minmass: float | None = None,
    separation: float | None = None,
    preprocess: bool = True,
    show_progress: bool = True,
    **kwargs,
) -> pd.DataFrame:
    """Detect particles in all frames.

    Parameters
    ----------
    frames : np.ndarray
        Array of frames with shape (n_frames, height, width).
    diameter : int, optional
        Expected particle diameter in pixels (default: 11).
    minmass : float, optional
        Minimum integrated brightness for a particle.
    separation : float, optional
        Minimum separation between particles.
    preprocess : bool, optional
        Whether to preprocess each frame (default: True).
    show_progress : bool, optional
        Whether to show progress bar (default: True).
    **kwargs
        Additional arguments passed to trackpy.locate.

    Returns
    -------
    pd.DataFrame
        DataFrame with detected particles from all frames.
        Includes 'frame' column indicating source frame.
    """
    all_features = []

    iterator = range(len(frames))
    if show_progress:
        iterator = tqdm(iterator, desc="Detecting particles")

    for i in iterator:
        frame = frames[i]
        if preprocess:
            frame = preprocess_frame(frame)

        features = tp.locate(
            frame,
            diameter=diameter if diameter % 2 == 1 else diameter + 1,
            minmass=minmass,
            separation=separation or diameter,
            **kwargs,
        )

        if len(features) > 0:
            features["frame"] = i
            all_features.append(features)

    if all_features:
        return pd.concat(all_features, ignore_index=True)
    return pd.DataFrame()


# %%
def filter_particles(
    features: pd.DataFrame,
    min_mass: float | None = None,
    max_mass: float | None = None,
    max_ecc: float | None = None,
    min_size: float | None = None,
    max_size: float | None = None,
) -> pd.DataFrame:
    """Filter detected particles based on properties.

    Parameters
    ----------
    features : pd.DataFrame
        DataFrame from detect_particles or batch_detect.
    min_mass : float, optional
        Minimum mass (integrated brightness).
    max_mass : float, optional
        Maximum mass.
    max_ecc : float, optional
        Maximum eccentricity (0 = circular).
    min_size : float, optional
        Minimum size (Gaussian sigma).
    max_size : float, optional
        Maximum size.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame.
    """
    mask = pd.Series(True, index=features.index)

    if min_mass is not None and "mass" in features.columns:
        mask &= features["mass"] >= min_mass
    if max_mass is not None and "mass" in features.columns:
        mask &= features["mass"] <= max_mass
    if max_ecc is not None and "ecc" in features.columns:
        mask &= features["ecc"] <= max_ecc
    if min_size is not None and "size" in features.columns:
        mask &= features["size"] >= min_size
    if max_size is not None and "size" in features.columns:
        mask &= features["size"] <= max_size

    return features[mask].reset_index(drop=True)
