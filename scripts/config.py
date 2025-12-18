# %%
"""Shared configuration for particle tracking scripts."""

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
# Detection Method
# =============================================================================

# Choose detection method: "threshold" or "trackpy"
METHOD = "threshold"

# Threshold method parameters (DefocusTracker Method 0)
# Good for defocused particles with ring patterns
THRESHOLD_PARAMS = {
    "threshold_percentile": 99.5,  # Higher = fewer detections
    "min_area": 50,  # Minimum particle area in pixels
    "max_area": 5000,  # Maximum particle area in pixels
}

# Trackpy method parameters (Crocker-Grier blob detection)
# Good for Gaussian-like particle profiles
TRACKPY_PARAMS = {
    "diameter": 11,  # Expected particle diameter (must be odd)
    "minmass": 1000,  # Minimum integrated brightness
    "separation": None,  # Minimum separation (default: diameter)
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
FRAME_INTERVAL = 0.007  # seconds (will be overwritten by metadata if available)

# =============================================================================
# Visualization
# =============================================================================

# Frames to visualize (None = no visualization)
VIS_FRAMES = [0, 50, 100, 150, 199]

# Generate frame-by-frame images for video
GENERATE_VIDEO_FRAMES = False
VIDEO_FRAME_RANGE = range(0, 200)  # Which frames to export

# =============================================================================
# Helper function
# =============================================================================


def get_detect_kwargs() -> dict:
    """Get detection parameters for current method."""
    if METHOD == "threshold":
        return THRESHOLD_PARAMS.copy()
    else:
        return TRACKPY_PARAMS.copy()
