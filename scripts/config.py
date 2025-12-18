# %%
"""Shared configuration for particle tracking scripts.

Detection parameters follow DefocusTracker's Method 0 (boundary_threshold_2d).
Reference: https://gitlab.com/defocustracking/defocustracker-matlab
"""

from pathlib import Path

# =============================================================================
# Paths
# =============================================================================

ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "20251218test-nofixture2"
OUTPUT_DIR = ROOT_DIR / "output"


def get_output_dir(script_file: str) -> Path:
    """Get output directory for a script (creates subfolder based on script name)."""
    script_name = Path(script_file).stem
    out_dir = OUTPUT_DIR / script_name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir

# =============================================================================
# Detection Parameters (DefocusTracker Method 0)
# =============================================================================


# boundary_threshold: Absolute intensity threshold
# - Pixels above this value are considered part of a particle
# - Must be set based on your data (use 02_particle_detection.py to tune)
# - Use estimate_threshold(frame, percentile) to get a starting value
BOUNDARY_THRESHOLD = 4000  # Adjust based on your data

# min_area: Minimum particle area in pixels
MIN_AREA = 30

# Preprocessing filters (set to 1 to disable)
MEDIAN_FILTER = 1  # Median filter size
GAUSS_FILTER = 1   # Gaussian filter size

# Convenience dict for passing to detect/batch_detect
DETECT_PARAMS = {
    "boundary_threshold": BOUNDARY_THRESHOLD,
    "min_area": MIN_AREA,
    "median_filter": MEDIAN_FILTER,
    "gauss_filter": GAUSS_FILTER,
}

# =============================================================================
# Tracking Parameters
# =============================================================================

SEARCH_RANGE = 15  # Max displacement per frame (pixels)
MEMORY = 3  # Frames a particle can disappear
MIN_TRAJ_LENGTH = 10  # Minimum trajectory length to keep

# =============================================================================
# Physical Parameters
# =============================================================================

PIXEL_SIZE = 0.65  # um/pixel (will be overwritten by metadata if available)
# seconds (will be overwritten by metadata if available)
FRAME_INTERVAL = 0.007

# =============================================================================
# Visualization
# =============================================================================

# Frames to visualize (None = no visualization)
VIS_FRAMES = [0, 50, 100, 150, 199]

# Generate frame-by-frame images for video
GENERATE_VIDEO_FRAMES = False
VIDEO_FRAME_RANGE = range(0, 200)  # Which frames to export
