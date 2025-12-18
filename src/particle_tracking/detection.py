# %%
"""Particle detection using threshold-based method.

This implements DefocusTracker's Method 0 (boundary_threshold_2d) algorithm
for 2D particle detection. The implementation follows the original MATLAB
code from: https://gitlab.com/defocustracking/defocustracker-matlab

Reference:
    R. Barnkob and M. Rossi, DefocusTracker: A modular toolbox for
    defocusing-based, single-camera, 3D particle tracking.
    Journal of Open Research Software, 9(1), 22 (2021).
"""

import numpy as np
import pandas as pd
from scipy import ndimage
from skimage import measure
from tqdm import tqdm


# %%
def detect(
    frame: np.ndarray,
    boundary_threshold: float,
    min_area: int = 30,
    median_filter: int = 1,
    gauss_filter: int = 1,
) -> pd.DataFrame:
    """Detect particles using intensity threshold and connected components.

    This implements DefocusTracker's Method 0 (boundary_threshold_2d).
    Algorithm:
    1. Optional preprocessing (median filter, Gaussian filter)
    2. Binary threshold: pixels > boundary_threshold
    3. Fill holes in binary mask
    4. Label connected components (8-connectivity)
    5. Filter by minimum area
    6. Extract geometric centroid of each region

    Parameters
    ----------
    frame : np.ndarray
        Input image.
    boundary_threshold : float
        Absolute intensity threshold. Pixels above this value are considered
        part of a particle.
    min_area : int, optional
        Minimum particle area in pixels (default: 30).
    median_filter : int, optional
        Median filter size. Set to 1 to disable (default: 1).
    gauss_filter : int, optional
        Gaussian filter size. Set to 1 to disable (default: 1).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: x, y, area.
    """
    # Preprocessing (same as DefocusTracker)
    img = frame.copy()

    # Convert to grayscale if needed
    if img.ndim == 3:
        img = np.mean(img, axis=2)

    # Median filter
    if median_filter > 1:
        img = ndimage.median_filter(img, size=median_filter)

    # Gaussian filter
    if gauss_filter > 1:
        img = ndimage.gaussian_filter(img, sigma=gauss_filter)

    # Binary threshold
    binary = img > boundary_threshold

    # Fill holes (same as MATLAB's imfill)
    binary = ndimage.binary_fill_holes(binary)

    # Label connected components (8-connectivity, same as bwboundaries with 8)
    # Using structure for 8-connectivity
    structure = ndimage.generate_binary_structure(2, 2)  # 8-connectivity
    labeled, num_features = ndimage.label(binary, structure=structure)

    if num_features == 0:
        return pd.DataFrame(columns=["x", "y", "area"])

    # Get region properties (geometric centroid, same as DefocusTracker)
    regions = measure.regionprops(labeled)

    # Extract particle data
    particles = []
    for region in regions:
        area = region.area

        # Filter by min area (same as DefocusTracker)
        if area < min_area:
            continue

        # Geometric centroid (y, x in regionprops) - same as DefocusTracker
        cy, cx = region.centroid

        particles.append({
            "x": cx,
            "y": cy,
            "area": area,
        })

    if particles:
        return pd.DataFrame(particles)
    return pd.DataFrame(columns=["x", "y", "area"])


# %%
def batch_detect(
    frames: np.ndarray,
    boundary_threshold: float,
    min_area: int = 30,
    median_filter: int = 1,
    gauss_filter: int = 1,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Detect particles in all frames using threshold method.

    Parameters
    ----------
    frames : np.ndarray
        Array of frames with shape (n_frames, height, width).
    boundary_threshold : float
        Absolute intensity threshold.
    min_area : int, optional
        Minimum particle area in pixels (default: 30).
    median_filter : int, optional
        Median filter size. Set to 1 to disable (default: 1).
    gauss_filter : int, optional
        Gaussian filter size. Set to 1 to disable (default: 1).
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
            boundary_threshold=boundary_threshold,
            min_area=min_area,
            median_filter=median_filter,
            gauss_filter=gauss_filter,
        )

        if len(features) > 0:
            features["frame"] = i
            all_features.append(features)

    if all_features:
        return pd.concat(all_features, ignore_index=True)
    return pd.DataFrame()


# %%
def estimate_threshold(frame: np.ndarray, percentile: float = 99.0) -> float:
    """Estimate a good boundary threshold from image percentile.

    Helper function to estimate boundary_threshold from a sample frame.
    DefocusTracker requires an absolute threshold value, this helps
    find a reasonable starting point.

    Parameters
    ----------
    frame : np.ndarray
        Sample image.
    percentile : float, optional
        Percentile to use (default: 99.0).

    Returns
    -------
    float
        Estimated threshold value.
    """
    return float(np.percentile(frame, percentile))


# %%
def filter_particles(
    features: pd.DataFrame,
    min_area: float | None = None,
    max_area: float | None = None,
) -> pd.DataFrame:
    """Filter detected particles based on properties.

    Parameters
    ----------
    features : pd.DataFrame
        DataFrame from detect or batch_detect.
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

    if min_area is not None and "area" in features.columns:
        mask &= features["area"] >= min_area
    if max_area is not None and "area" in features.columns:
        mask &= features["area"] <= max_area

    return features[mask].reset_index(drop=True)
