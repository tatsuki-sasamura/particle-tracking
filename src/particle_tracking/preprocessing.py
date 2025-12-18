# %%
"""Image preprocessing utilities for particle tracking."""

import numpy as np
from scipy import ndimage


# %%
def subtract_background(frame: np.ndarray, percentile: float = 10) -> np.ndarray:
    """Subtract background using percentile-based estimation.

    Parameters
    ----------
    frame : np.ndarray
        Input image.
    percentile : float, optional
        Percentile value to estimate background (default: 10).

    Returns
    -------
    np.ndarray
        Background-subtracted image (clipped to non-negative values).
    """
    background = np.percentile(frame, percentile)
    result = frame.astype(np.float32) - background
    return np.clip(result, 0, None)


# %%
def bandpass_filter(
    frame: np.ndarray,
    low_sigma: float = 1.0,
    high_sigma: float | None = None,
) -> np.ndarray:
    """Apply bandpass filter to enhance particles.

    Subtracts a heavily smoothed image (removes large-scale variations)
    and applies Gaussian blur (removes pixel noise).

    Parameters
    ----------
    frame : np.ndarray
        Input image.
    low_sigma : float, optional
        Sigma for noise smoothing (default: 1.0).
    high_sigma : float, optional
        Sigma for background removal. If None, uses 10 * low_sigma.

    Returns
    -------
    np.ndarray
        Bandpass-filtered image.
    """
    if high_sigma is None:
        high_sigma = 10 * low_sigma

    frame_float = frame.astype(np.float32)

    # Remove large-scale background
    background = ndimage.gaussian_filter(frame_float, sigma=high_sigma)
    filtered = frame_float - background

    # Smooth noise
    if low_sigma > 0:
        filtered = ndimage.gaussian_filter(filtered, sigma=low_sigma)

    return np.clip(filtered, 0, None)


# %%
def normalize_intensity(frame: np.ndarray) -> np.ndarray:
    """Normalize image intensity to 0-1 range.

    Parameters
    ----------
    frame : np.ndarray
        Input image.

    Returns
    -------
    np.ndarray
        Normalized image with values in [0, 1].
    """
    frame_float = frame.astype(np.float32)
    vmin, vmax = frame_float.min(), frame_float.max()
    if vmax > vmin:
        return (frame_float - vmin) / (vmax - vmin)
    return np.zeros_like(frame_float)


# %%
def preprocess_frame(
    frame: np.ndarray,
    subtract_bg: bool = True,
    bg_percentile: float = 10,
    bandpass: bool = True,
    low_sigma: float = 1.0,
    high_sigma: float | None = None,
    normalize: bool = False,
) -> np.ndarray:
    """Apply preprocessing pipeline to a single frame.

    Parameters
    ----------
    frame : np.ndarray
        Input image.
    subtract_bg : bool, optional
        Whether to subtract background (default: True).
    bg_percentile : float, optional
        Percentile for background estimation (default: 10).
    bandpass : bool, optional
        Whether to apply bandpass filter (default: True).
    low_sigma : float, optional
        Sigma for noise smoothing (default: 1.0).
    high_sigma : float, optional
        Sigma for background removal in bandpass.
    normalize : bool, optional
        Whether to normalize to 0-1 range (default: False).

    Returns
    -------
    np.ndarray
        Preprocessed image.
    """
    result = frame.astype(np.float32)

    if subtract_bg:
        result = subtract_background(result, percentile=bg_percentile)

    if bandpass:
        result = bandpass_filter(result, low_sigma=low_sigma, high_sigma=high_sigma)

    if normalize:
        result = normalize_intensity(result)

    return result
