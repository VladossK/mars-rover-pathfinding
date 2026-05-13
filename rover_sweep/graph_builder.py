"""Vectorized slope mask + sparse CSR graph construction."""
from __future__ import annotations
import numpy as np
from scipy.sparse import csr_matrix

_SQRT2 = float(np.sqrt(2.0))
DIRECTIONS: list[tuple[int, int, float]] = [
    ( 1,  0, 1.0),     (-1,  0, 1.0),
    ( 0,  1, 1.0),     ( 0, -1, 1.0),
    ( 1,  1, _SQRT2),  (-1,  1, _SQRT2),
    ( 1, -1, _SQRT2),  (-1, -1, _SQRT2),
]


def build_edge_mask(
    elev: np.ndarray,
    valid_mask: np.ndarray,
    max_slope_deg: float,
    pixel_size: float,
) -> np.ndarray:
    """Vectorized 8-direction slope mask.

    Returns: bool[H, W, 8] — edge_mask[r, c, k] is True iff the step from (r, c)
    to neighbor k is traversable (within slope limit and both endpoints valid).
    """
    H, W = elev.shape
    mask = np.zeros((H, W, 8), dtype=bool)

    elev64 = elev.astype(np.float64, copy=False)

    for k, (dr, dc, d_pix) in enumerate(DIRECTIONS):
        h_dist = d_pix * pixel_size
        shifted_elev = np.roll(elev64, shift=(-dr, -dc), axis=(0, 1))
        shifted_valid = np.roll(valid_mask, shift=(-dr, -dc), axis=(0, 1))

        dz = np.abs(shifted_elev - elev64)
        slope_deg = np.degrees(np.arctan2(dz, h_dist))
        ok = (slope_deg <= max_slope_deg) & valid_mask & shifted_valid

        if dr == 1:
            ok[-1, :] = False
        elif dr == -1:
            ok[0, :] = False
        if dc == 1:
            ok[:, -1] = False
        elif dc == -1:
            ok[:, 0] = False

        mask[..., k] = ok

    return mask


def edge_mask_to_csr(edge_mask: np.ndarray, pixel_size: float) -> csr_matrix:
    """Convert (H,W,8) edge mask to a CSR adjacency matrix with edge weights in meters."""
    H, W, _ = edge_mask.shape
    N = H * W

    assert N * 8 < 2**31, f"Grid too large for int32 indices: N*8={N*8}"

    rows_list = []
    cols_list = []
    data_list = []

    idx_grid = np.arange(N, dtype=np.int32).reshape(H, W)

    for k, (dr, dc, d_pix) in enumerate(DIRECTIONS):
        ok = edge_mask[..., k]
        if not ok.any():
            continue
        src = idx_grid[ok]
        shifted_idx = np.roll(idx_grid, shift=(-dr, -dc), axis=(0, 1))
        dst = shifted_idx[ok]
        w = np.full(src.shape, d_pix * pixel_size, dtype=np.float64)
        rows_list.append(src)
        cols_list.append(dst)
        data_list.append(w)

    if not rows_list:
        return csr_matrix((N, N), dtype=np.float64)

    rows = np.concatenate(rows_list)
    cols = np.concatenate(cols_list)
    data = np.concatenate(data_list)

    return csr_matrix((data, (rows, cols)), shape=(N, N), dtype=np.float64)
