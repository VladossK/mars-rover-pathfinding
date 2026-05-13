"""Cost surfaces shared across all backends.

Provides:
- Pixel-wise slope (max over 8 neighbors)
- TRI (Terrain Ruggedness Index)
- Static cost map: cost = (1 + k_slope * tan(slope)) * (1 + k_rough * tri_norm)
- 3D step distance helper
- NoData smooth fill via nearest-neighbor extrapolation
"""
from __future__ import annotations
import numpy as np

_DR8 = np.array([1, -1, 0, 0, 1, -1, 1, -1], dtype=np.int32)
_DC8 = np.array([0, 0, 1, -1, 1, 1, -1, -1], dtype=np.int32)
_DD8 = np.array([1.0, 1.0, 1.0, 1.0, np.sqrt(2.0), np.sqrt(2.0), np.sqrt(2.0), np.sqrt(2.0)],
                dtype=np.float64)


def fill_nodata_nearest(elev: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """Replace invalid cells with nearest valid neighbor (smooth boundary for FMM/HFM).

    Returns elev_filled with the same shape and float32 dtype. Valid cells unchanged.
    """
    from scipy.ndimage import distance_transform_edt
    if valid_mask.all():
        return elev.astype(np.float32, copy=True)
    _, indices = distance_transform_edt(~valid_mask, return_indices=True)
    return elev[tuple(indices)].astype(np.float32, copy=False)


def pixel_max_slope(elev: np.ndarray, pixel_size: float) -> np.ndarray:
    """Max 8-neighbor slope at every cell, in degrees.

    slope[r,c] = max over 8 neighbors of arctan(|dz| / h_dist).
    Boundary cells use only available neighbors.
    Returns float32 array of shape elev.shape.
    """
    H, W = elev.shape
    e = elev.astype(np.float64, copy=False)
    slope = np.zeros((H, W), dtype=np.float64)
    for k in range(8):
        dr = int(_DR8[k])
        dc = int(_DC8[k])
        h_dist = float(_DD8[k]) * pixel_size
        shifted = np.roll(e, shift=(-dr, -dc), axis=(0, 1))
        dz = np.abs(shifted - e)
        s = np.degrees(np.arctan2(dz, h_dist))
        if dr == 1:
            s[-1, :] = 0
        elif dr == -1:
            s[0, :] = 0
        if dc == 1:
            s[:, -1] = 0
        elif dc == -1:
            s[:, 0] = 0
        slope = np.maximum(slope, s)
    return slope.astype(np.float32)


def terrain_ruggedness_index(elev: np.ndarray) -> np.ndarray:
    """TRI = sqrt(mean((elev[3x3] - elev[r,c])^2)), 3x3 window.

    Riley et al. 1999. Boundary cells use available neighbors (mean over <8).
    Returns float32 array.
    """
    H, W = elev.shape
    e = elev.astype(np.float64, copy=False)
    sum_sq = np.zeros((H, W), dtype=np.float64)
    count = np.zeros((H, W), dtype=np.float64)
    for k in range(8):
        dr = int(_DR8[k])
        dc = int(_DC8[k])
        shifted = np.roll(e, shift=(-dr, -dc), axis=(0, 1))
        valid = np.ones((H, W), dtype=bool)
        if dr == 1: valid[-1, :] = False
        elif dr == -1: valid[0, :] = False
        if dc == 1: valid[:, -1] = False
        elif dc == -1: valid[:, 0] = False
        diff = shifted - e
        sum_sq += np.where(valid, diff * diff, 0.0)
        count += valid.astype(np.float64)
    count = np.where(count > 0, count, 1.0)
    tri = np.sqrt(sum_sq / count)
    return tri.astype(np.float32)


def build_cost_map(
    elev: np.ndarray,
    valid_mask: np.ndarray,
    pixel_size: float,
    k_slope: float = 2.0,
    k_rough: float = 1.0,
    tri_normalize: str = "mean",  # "mean" or "max"
) -> dict:
    """Build the per-cell cost map shared by all backends.

    cost = (1 + k_slope * tan(slope_rad)) * (1 + k_rough * tri_norm)

    Returns dict with arrays:
        slope_deg  — float32[H, W] max 8-neighbor slope (degrees)
        tri        — float32[H, W] TRI (meters)
        tri_norm   — float32[H, W] normalized to ~[0, 1]
        cost       — float32[H, W] >=1, +inf at invalid cells
    Invalid cells get +inf in cost.
    """
    elev_filled = fill_nodata_nearest(elev, valid_mask)
    slope_deg = pixel_max_slope(elev_filled, pixel_size)
    tri = terrain_ruggedness_index(elev_filled)

    if tri_normalize == "max":
        denom = float(tri[valid_mask].max()) if valid_mask.any() else 1.0
    else:
        denom = float(tri[valid_mask].mean()) if valid_mask.any() else 1.0
    if denom <= 0.0:
        denom = 1.0
    tri_norm = (tri / denom).astype(np.float32)

    slope_rad = np.radians(slope_deg.astype(np.float64))
    tan_slope = np.tan(slope_rad).astype(np.float32)
    cost = (1.0 + k_slope * tan_slope) * (1.0 + k_rough * tri_norm)
    cost = cost.astype(np.float32)
    cost[~valid_mask] = np.float32(np.inf)

    return {
        "elev_filled": elev_filled,
        "slope_deg": slope_deg,
        "tri": tri,
        "tri_norm": tri_norm,
        "cost": cost,
    }


def step_cost_3d(
    elev: np.ndarray,
    r0: int, c0: int, r1: int, c1: int,
    cost_map: np.ndarray,
    pixel_size: float,
) -> float:
    """3D step distance scaled by mean of endpoint cost.

    cost(edge a->b) = sqrt(h_dist^2 + dz^2) * 0.5*(cost[a] + cost[b])
    """
    dr = r1 - r0
    dc = c1 - c0
    h_dist = float(np.hypot(dr, dc)) * pixel_size
    dz = float(elev[r1, c1]) - float(elev[r0, c0])
    geom = float(np.sqrt(h_dist * h_dist + dz * dz))
    return geom * 0.5 * (float(cost_map[r0, c0]) + float(cost_map[r1, c1]))


def path_length_3d_meters(path_idx: np.ndarray, elev: np.ndarray, W: int,
                           pixel_size: float) -> float:
    """3D-length of a discrete path (sum of true 3D step distances)."""
    if path_idx.size < 2:
        return 0.0
    rows = (path_idx // W).astype(np.int64)
    cols = (path_idx - rows * W).astype(np.int64)
    dr = np.diff(rows)
    dc = np.diff(cols)
    h2 = (dr * dr + dc * dc).astype(np.float64) * (pixel_size * pixel_size)
    z = elev[rows, cols].astype(np.float64)
    dz = np.diff(z)
    return float(np.sum(np.sqrt(h2 + dz * dz)))


def path_slope_stats(path_idx: np.ndarray, elev: np.ndarray, W: int,
                      pixel_size: float) -> dict:
    """Per-edge slope statistics along a path."""
    if path_idx.size < 2:
        return {"max_deg": 0.0, "mean_deg": 0.0, "climb_m": 0.0, "descent_m": 0.0}
    rows = (path_idx // W).astype(np.int64)
    cols = (path_idx - rows * W).astype(np.int64)
    dr = np.diff(rows)
    dc = np.diff(cols)
    h = np.hypot(dr, dc).astype(np.float64) * pixel_size
    z = elev[rows, cols].astype(np.float64)
    dz = np.diff(z)
    h_safe = np.where(h > 0, h, 1.0)
    slope = np.degrees(np.arctan2(np.abs(dz), h_safe))
    climb = float(np.sum(dz[dz > 0]))
    descent = float(-np.sum(dz[dz < 0]))
    return {
        "max_deg": float(np.max(slope)) if slope.size else 0.0,
        "mean_deg": float(np.mean(slope)) if slope.size else 0.0,
        "climb_m": climb,
        "descent_m": descent,
    }
