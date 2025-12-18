# %%
"""Particle detection using threshold-based method (DefocusTracker Method 0)."""

import numpy as np
import pandas as pd
from scipy import ndimage
from skimage import measure
from tqdm import tqdm


# %%
def detect(
    frame: np.ndarray,
    threshold: float | None = None,
    threshold_percentile: float = 99.0,
    min_area: int = 10,
    max_area: int = 10000,
) -> pd.DataFrame:
    """Detect particles using intensity threshold and connected components.

    This implements DefocusTracker's Method 0 - simple threshold-based
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


# %%
def batch_detect(
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
        iterator = tqdm(iterator, desc="Detecting particles")

    for i in iterator:
        features = detect(
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
def filter_particles(
    features: pd.DataFrame,
    min_mass: float | None = None,
    max_mass: float | None = None,
    max_ecc: float | None = None,
    min_area: float | None = None,
    max_area: float | None = None,
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
    min_area : float, optional
        Minimum area in pixels.
    max_area : float, optional
        Maximum area in pixels.

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
    if max_ecc is not None and "eccentricity" in features.columns:
        mask &= features["eccentricity"] <= max_ecc
    if min_area is not None and "area" in features.columns:
        mask &= features["area"] >= min_area
    if max_area is not None and "area" in features.columns:
        mask &= features["area"] <= max_area

    return features[mask].reset_index(drop=True)
