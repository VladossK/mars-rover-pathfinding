"""Project-wide constants and defaults."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TIFF_FILE = PROJECT_ROOT / "Mars_HRSC_MOLA_BlendDEM_Global_200mp_v2_Test2.tif"
RESULTS_DIR = PROJECT_ROOT / "results"

PIXEL_SIZE = 200.0
NUM_PAIRS = 200

START_ANGLE = 1
ANGLE_STEP = 1
MAX_ANGLE = 89

PAIRS_SEED = 42
REACH_SEED = 7777  # single fixed seed for the one reachability start used across all angles

HEAP_CAPACITY_MULT = 5

# Cost surface coefficients (shared across hybrid + hfm backends).
# Default = gentle: k_slope=1 means cost grows linearly with tan(slope),
# k_rough=0 disables TRI (Mars terrain TRI distribution is bimodal -
# enabling it pushes Theta* into long detours through cost-cheap valleys).
# Enable TRI via --k-rough 0.3 .. 1.0 if needed for paper comparison.
COST_K_SLOPE = 1.0
COST_K_ROUGH = 0.0
COST_TRI_NORMALIZE = "mean"   # "mean" or "max"

# Hard cap on path length as a safety net: paths whose 2D length exceeds
# (MAX_PATH_DETOUR_FACTOR * straight_line) are flagged as detour failures
# and dropped from CSV stats. Set to None to disable.
MAX_PATH_DETOUR_FACTOR = 5.0

# Default backend if --backend is not specified.
DEFAULT_BACKEND = "astar"

# Subset of pairs used by the (slow) hfm backend to keep wall time tractable.
HFM_SUBSET_SIZE = 30


def get_backend_versions() -> dict:
    """Library versions embedded in CSV output for paper reproducibility."""
    out = {}
    try:
        import numpy; out["numpy"] = numpy.__version__
    except Exception: pass
    try:
        import scipy; out["scipy"] = scipy.__version__
    except Exception: pass
    try:
        import numba; out["numba"] = numba.__version__
    except Exception: pass
    try:
        import skfmm; out["scikit-fmm"] = skfmm.__version__
    except Exception: pass
    try:
        import agd; out["agd"] = getattr(agd, "__version__", "?")
    except Exception: pass
    return out
