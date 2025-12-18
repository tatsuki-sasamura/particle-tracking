# %%
"""Particle detection using trackpy and threshold-based methods."""

import numpy as np
import pandas as pd
import trackpy as tp
from scipy import ndimage
from skimage import measure
from tqdm import tqdm


# %%
# =============================================================================
# Method 0: Threshold-based detection (like DefocusTracker)
# =============================================================================


def _detect_threshold(
    frame: np.ndarray,
    threshold: float | None = None,
    threshold_percentile: float = 99.0,
    min_area: int = 10,
    max_area: int = 10000,
) -> pd.DataFrame:
    """Detect particles using intensity threshold and connected components.

    This is similar to DefocusTracker's Method 0 - simple threshold-based
    detection that works well for defocused particle patterns.

    Parameters
    ----------
    frame : np.ndarray
        Input image.
    threshold : float, optional
        Absolute intensity threshold. If None, uses threshold_percentile.
    threshold_percentile : float, optional
        Percentile of image intensity to use as threshold (default: 99.0).
    min_area : int, optional
        Minimum particle area in pixels (default: 10).
    max_area : int, optional
        Maximum particle area in pixels (default: 10000).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: x, y, mass, area, bbox, etc.
    """

    # Determine threshold
    if threshold is None:
        threshold = np.percentile(frame, threshold_percentile)

    # Binary threshold
    binary = frame > threshold

    # Label connected components
    labeled, num_features = ndimage.label(binary)

    if num_features == 0:
        return pd.DataFrame(columns=["x", "y", "mass", "area"])

    # Get region properties
    regions = measure.regionprops(labeled, intensity_image=frame)

    # Extract particle data
    particles = []
    for region in regions:
        area = region.area

        # Filter by area
        if area < min_area or area > max_area:
            continue

        # Centroid (y, x in regionprops)
        cy, cx = region.centroid
        # Weighted centroid (intensity-weighted)
        cy_w, cx_w = region.centroid_weighted

        particles.append({
            "x": cx_w,  # Use intensity-weighted centroid
            "y": cy_w,
            "mass": region.intensity_mean * area,  # Total intensity
            "area": area,
            "intensity_mean": region.intensity_mean,
            "intensity_max": region.intensity_max,
            "eccentricity": region.eccentricity,
            "bbox_x": region.bbox[1],
            "bbox_y": region.bbox[0],
            "bbox_w": region.bbox[3] - region.bbox[1],
            "bbox_h": region.bbox[2] - region.bbox[0],
        })

    if particles:
        return pd.DataFrame(particles)
    return pd.DataFrame(columns=["x", "y", "mass", "area"])


def _batch_threshold(
    frames: np.ndarray,
    threshold: float | None = None,
    threshold_percentile: float = 99.0,
    min_area: int = 10,
    max_area: int = 10000,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Detect particles in all frames using threshold method.

    Parameters
    ----------
    frames : np.ndarray
        Array of frames with shape (n_frames, height, width).
    threshold : float, optional
        Absolute intensity threshold. If None, uses threshold_percentile.
    threshold_percentile : float, optional
        Percentile of image intensity to use as threshold (default: 99.0).
    min_area : int, optional
        Minimum particle area in pixels (default: 10).
    max_area : int, optional
        Maximum particle area in pixels (default: 10000).
    show_progress : bool, optional
        Whether to show progress bar (default: True).

    Returns
    -------
    pd.DataFrame
        DataFrame with detected particles from all frames.
    """
    all_features = []

    iterator = range(len(frames))
    if show_progress:
        iterator = tqdm(iterator, desc="Detecting particles (threshold)")

    for i in iterator:
        features = _detect_threshold(
            frames[i],
            threshold=threshold,
            threshold_percentile=threshold_percentile,
            min_area=min_area,
            max_area=max_area,
        )

        if len(features) > 0:
            features["frame"] = i
            all_features.append(features)

    if all_features:
        return pd.concat(all_features, ignore_index=True)
    return pd.DataFrame()


# %%
# =============================================================================
# Trackpy blob detection (Crocker-Grier algorithm)
# =============================================================================


def _detect_trackpy(
    frame: np.ndarray,
    diameter: int = 11,
    minmass: float | None = None,
    separation: float | None = None,
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
    **kwargs
        Additional arguments passed to trackpy.locate.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: x, y, mass, size, ecc, signal, raw_mass, etc.
    """
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
def _batch_trackpy(
    frames: np.ndarray,
    diameter: int = 11,
    minmass: float | None = None,
    separation: float | None = None,
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
# =============================================================================
# Unified interface
# =============================================================================


def detect(
    frame: np.ndarray,
    method: str = "threshold",
    **kwargs,
) -> pd.DataFrame:
    """Detect particles in a single frame using specified method.

    Parameters
    ----------
    frame : np.ndarray
        Input image.
    method : str, optional
        Detection method: "threshold" (DefocusTracker Method 0) or "trackpy".
        Default: "threshold".
    **kwargs
        Method-specific parameters:
        - threshold: threshold_percentile, min_area, max_area
        - trackpy: diameter, minmass, separation

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: x, y, mass, frame, and method-specific columns.
    """
    if method == "threshold":
        return _detect_threshold(frame, **kwargs)
    elif method == "trackpy":
        return _detect_trackpy(frame, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'threshold' or 'trackpy'.")


def batch_detect(
    frames: np.ndarray,
    method: str = "threshold",
    show_progress: bool = True,
    **kwargs,
) -> pd.DataFrame:
    """Detect particles in all frames using specified method.

    Parameters
    ----------
    frames : np.ndarray
        Array of frames with shape (n_frames, height, width).
    method : str, optional
        Detection method: "threshold" (DefocusTracker Method 0) or "trackpy".
        Default: "threshold".
    show_progress : bool, optional
        Whether to show progress bar (default: True).
    **kwargs
        Method-specific parameters:
        - threshold: threshold_percentile, min_area, max_area
        - trackpy: diameter, minmass, separation

    Returns
    -------
    pd.DataFrame
        DataFrame with detected particles from all frames.
    """
    if method == "threshold":
        return _batch_threshold(frames, show_progress=show_progress, **kwargs)
    elif method == "trackpy":
        return _batch_trackpy(frames, show_progress=show_progress, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'threshold' or 'trackpy'.")


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
        DataFrame from detect or batch_detect.
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
