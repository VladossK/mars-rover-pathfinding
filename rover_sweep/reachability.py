"""Reachability solvers: scipy Dijkstra (Phase 1) + scikit-fmm eikonal (Phase 2+)."""
from __future__ import annotations
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra


def full_dijkstra(csr: csr_matrix, start_idx: int) -> np.ndarray:
    """Run Dijkstra from a single source over the whole graph (Phase 1 backend).

    Returns: float64[N] distances in meters. Unreachable cells = +inf.
    """
    dist = dijkstra(csr, directed=False, indices=start_idx, return_predecessors=False)
    return np.asarray(dist, dtype=np.float64)


def fmm_reachability(
    passable_mask: np.ndarray,
    cost_map: np.ndarray,
    start_rc: tuple[int, int],
    pixel_size: float,
    order: int = 2,
) -> np.ndarray:
    """Solve eikonal equation for cost-weighted travel time from start_rc.

    Uses scikit-fmm 2nd-order accurate FMM. Speed field = 1/cost in passable cells,
    0 in blocked cells (NoData + slope-violating after masking).

    passable_mask: bool[H, W] — cells the rover can stand on
    cost_map:      float32[H, W] — per-cell cost multiplier (>= 1); +inf in blocked
    start_rc:      (row, col)
    pixel_size:    meters per pixel
    order:         FMM accuracy (1 or 2)

    Returns: float64[H, W] travel-time field. Unreachable cells = +inf.
    """
    import skfmm

    H, W = passable_mask.shape
    sr, sc = int(start_rc[0]), int(start_rc[1])

    phi = np.ma.MaskedArray(
        np.ones((H, W), dtype=np.float64),
        mask=~passable_mask,
    )
    phi[sr, sc] = 0.0

    cost_safe = np.where(np.isfinite(cost_map) & (cost_map > 0), cost_map, np.float32(1.0))
    speed_arr = (1.0 / cost_safe.astype(np.float64))
    speed = np.ma.MaskedArray(speed_arr, mask=~passable_mask)

    try:
        t = skfmm.travel_time(phi, speed=speed, dx=float(pixel_size), order=int(order))
    except Exception:
        t = skfmm.travel_time(phi, speed=speed, dx=float(pixel_size), order=1)

    if isinstance(t, np.ma.MaskedArray):
        dist = t.filled(np.inf).astype(np.float64)
    else:
        dist = np.asarray(t, dtype=np.float64)

    dist = np.where(passable_mask, dist, np.inf)
    return dist


def reachability_stats(
    dist: np.ndarray,
    valid_flat: np.ndarray,
    pixel_size: float = 200.0,
    finite_threshold: float | None = None,
) -> dict:
    """Compute reachability stats from a Dijkstra distance vector.

    Returns dict with:
        n_total, n_valid, n_nodata,
        reachable_pixels, unreachable_pixels (slope-blocked, excludes nodata),
        reachable_pct, unreachable_pct (both expressed over valid cells),
        nodata_pct (over total cells),
        area_total_km2, area_reachable_km2, area_unreachable_km2, area_nodata_km2
    """
    N = int(dist.size)
    n_valid = int(valid_flat.sum())
    n_nodata = N - n_valid

    if finite_threshold is None:
        finite_mask = np.isfinite(dist)
    else:
        finite_mask = np.isfinite(dist) & (dist < float(finite_threshold))
    reachable_mask = finite_mask & valid_flat
    unreachable_mask = (~finite_mask) & valid_flat

    reachable = int(reachable_mask.sum())
    unreachable = int(unreachable_mask.sum())

    px_area_km2 = (pixel_size * pixel_size) / 1_000_000.0

    return {
        "n_total": N,
        "n_valid": n_valid,
        "n_nodata": n_nodata,
        "reachable_pixels": reachable,
        "unreachable_pixels": unreachable,
        "reachable_pct": (reachable / n_valid) if n_valid else 0.0,
        "unreachable_pct": (unreachable / n_valid) if n_valid else 0.0,
        "nodata_pct": (n_nodata / N) if N else 0.0,
        "area_total_km2": N * px_area_km2,
        "area_reachable_km2": reachable * px_area_km2,
        "area_unreachable_km2": unreachable * px_area_km2,
        "area_nodata_km2": n_nodata * px_area_km2,
    }
