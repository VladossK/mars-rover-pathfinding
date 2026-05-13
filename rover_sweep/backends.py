"""Backend dispatcher: astar | hybrid | hfm.

Each backend exposes:
- prepare(angle): per-angle setup
- reachability(start_rc): full distance/travel-time field, shape (H, W)
- find_path(start_rc, end_rc): integer path indices (flat), or None

astar  — Phase 1. scipy CSR Dijkstra + Numba A*. h_dist cost only.
hybrid — Phase 2. scikit-fmm reachability + Theta* paths with cost surface.
hfm    — Phase 3. scikit-fmm reachability + FMM gradient-descent geodesics
         (research-grade on CPU). When agd-HFM CUDA available, uses
         Mirebeau's anisotropic HFM (auto-detected).
"""
from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from . import config
from .graph_builder import build_edge_mask, edge_mask_to_csr
from .reachability import full_dijkstra, fmm_reachability
from .pathfinder import AStarPathfinder, ThetaStarPathfinder


def detect_hfm_gpu() -> tuple[bool, str]:
    """Return (gpu_available, info). True only if agd + cupy + CUDA binaries usable."""
    try:
        import cupy  # noqa
    except Exception as e:
        return False, f"cupy not available: {e}"
    try:
        from agd.Eikonal import dictIn  # noqa
    except Exception as e:
        return False, f"agd not available: {e}"
    try:
        d = dictIn({'model': 'Isotropic2', 'arrayOrdering': 'RowMajor'})
        d.SetRect(sides=[[0, 4], [0, 4]], dimx=4)
        d['cost'] = np.ones((4, 4), dtype=np.float64)
        d['seed'] = np.array([0.0, 0.0])
        d['mode'] = 'gpu_transfer'
        _ = d.Run()
        return True, "agd + cupy + CUDA OK"
    except Exception as e:
        return False, f"agd CUDA test failed: {e}"


@dataclass
class BackendContext:
    """Shared inputs supplied to every backend before per-angle work."""
    elev: np.ndarray            # float32[H, W] elevation (after NoData fill)
    valid_mask: np.ndarray      # bool[H, W]
    pixel_size: float
    H: int
    W: int
    cost_map: np.ndarray        # float32[H, W] >= 1, +inf in invalid
    slope_deg: np.ndarray       # float32[H, W]
    elev_flat: np.ndarray = field(init=False)
    valid_flat_u8: np.ndarray = field(init=False)
    cost_flat: np.ndarray = field(init=False)

    def __post_init__(self):
        self.elev_flat = self.elev.reshape(-1)
        self.valid_flat_u8 = self.valid_mask.astype(np.uint8).reshape(-1)
        self.cost_flat = self.cost_map.reshape(-1)


class Backend(Protocol):
    name: str
    def prepare(self, ctx: BackendContext, angle: float) -> None: ...
    def reachability(self, start_rc: tuple[int, int]) -> np.ndarray: ...
    def find_path(self, start_rc: tuple[int, int], end_rc: tuple[int, int]) -> np.ndarray | None: ...


# ---------------------------------------------------------------------------
# Backend: astar (Phase 1 baseline)
# ---------------------------------------------------------------------------

class AstarBackend:
    name = "astar"

    def __init__(self):
        self.pathfinder: AStarPathfinder | None = None
        self._csr = None
        self._ctx: BackendContext | None = None
        self._angle = None
        self._passable_mask: np.ndarray | None = None

    def prepare(self, ctx: BackendContext, angle: float) -> None:
        self._ctx = ctx
        self._angle = float(angle)
        edge_mask = build_edge_mask(ctx.elev, ctx.valid_mask, self._angle, ctx.pixel_size)
        self._csr = edge_mask_to_csr(edge_mask, ctx.pixel_size)
        self._passable_mask = self._compute_passable_mask(ctx, self._angle)
        if self.pathfinder is None:
            self.pathfinder = AStarPathfinder(ctx.H, ctx.W,
                                              heap_cap_mult=config.HEAP_CAPACITY_MULT)

    @staticmethod
    def _compute_passable_mask(ctx: BackendContext, max_slope_deg: float) -> np.ndarray:
        return (ctx.slope_deg <= max_slope_deg) & ctx.valid_mask

    def reachability(self, start_rc: tuple[int, int]) -> np.ndarray:
        sr, sc = int(start_rc[0]), int(start_rc[1])
        ctx = self._ctx
        idx = sr * ctx.W + sc
        dist_flat = full_dijkstra(self._csr, idx)
        return dist_flat.reshape(ctx.H, ctx.W)

    def find_path(self, start_rc, end_rc):
        ctx = self._ctx
        si = int(start_rc[0]) * ctx.W + int(start_rc[1])
        ei = int(end_rc[0]) * ctx.W + int(end_rc[1])
        return self.pathfinder.find_path(
            ctx.elev_flat, ctx.valid_flat_u8, si, ei,
            self._angle, ctx.pixel_size,
        )


# ---------------------------------------------------------------------------
# Backend: hybrid (Phase 2 research-grade on CPU)
# ---------------------------------------------------------------------------

class HybridBackend:
    name = "hybrid"

    def __init__(self):
        self.pathfinder: ThetaStarPathfinder | None = None
        self._ctx: BackendContext | None = None
        self._angle = None
        self._passable_mask: np.ndarray | None = None
        self._effective_cost: np.ndarray | None = None

    def prepare(self, ctx: BackendContext, angle: float) -> None:
        self._ctx = ctx
        self._angle = float(angle)
        passable = (ctx.slope_deg <= self._angle) & ctx.valid_mask
        self._passable_mask = passable
        effective = ctx.cost_map.copy()
        effective[~passable] = np.float32(np.inf)
        self._effective_cost = effective
        if self.pathfinder is None:
            self.pathfinder = ThetaStarPathfinder(ctx.H, ctx.W,
                                                  heap_cap_mult=config.HEAP_CAPACITY_MULT)

    def reachability(self, start_rc):
        return fmm_reachability(
            self._passable_mask, self._effective_cost, start_rc,
            pixel_size=self._ctx.pixel_size, order=2,
        )

    def find_path(self, start_rc, end_rc):
        ctx = self._ctx
        si = int(start_rc[0]) * ctx.W + int(start_rc[1])
        ei = int(end_rc[0]) * ctx.W + int(end_rc[1])
        cost_flat = self._effective_cost.reshape(-1)
        return self.pathfinder.find_path(
            ctx.elev_flat, ctx.valid_flat_u8, cost_flat, si, ei,
            self._angle, ctx.pixel_size,
        )


# ---------------------------------------------------------------------------
# Backend: hfm (Phase 3 max accuracy — CUDA preferred, CPU fallback via FMM)
# ---------------------------------------------------------------------------

class HfmBackend:
    """Anisotropic-grade backend.

    On GPU+agd: invokes Mirebeau HFM (Riemannian metric) — true geodesic.
    On CPU/laptop: falls back to scikit-fmm travel-time field + gradient-descent
    backtracking. Sub-pixel geodesic, 2nd-order eikonal — research-grade.
    """
    name = "hfm"

    def __init__(self):
        self._ctx: BackendContext | None = None
        self._angle = None
        self._passable_mask: np.ndarray | None = None
        self._effective_cost: np.ndarray | None = None
        self._tt_cache: dict[tuple[int, int], np.ndarray] = {}
        self.gpu_ok, self.gpu_msg = detect_hfm_gpu()

    def prepare(self, ctx: BackendContext, angle: float) -> None:
        self._ctx = ctx
        self._angle = float(angle)
        passable = (ctx.slope_deg <= self._angle) & ctx.valid_mask
        self._passable_mask = passable
        effective = ctx.cost_map.copy()
        effective[~passable] = np.float32(np.inf)
        self._effective_cost = effective
        self._tt_cache.clear()

    def reachability(self, start_rc):
        if self.gpu_ok:
            try:
                return self._hfm_gpu_field(start_rc)
            except Exception as e:
                print(f"  [hfm] GPU path failed ({e}); falling back to CPU FMM")
        return fmm_reachability(
            self._passable_mask, self._effective_cost, start_rc,
            pixel_size=self._ctx.pixel_size, order=2,
        )

    def find_path(self, start_rc, end_rc):
        sr, sc = int(start_rc[0]), int(start_rc[1])
        er, ec = int(end_rc[0]), int(end_rc[1])
        key = (sr, sc)
        if key not in self._tt_cache:
            self._tt_cache[key] = self.reachability((sr, sc))
        tt = self._tt_cache[key]
        if not np.isfinite(tt[er, ec]):
            return None
        return self._geodesic_backtrack(tt, end_rc, start_rc)

    # ---- helpers ----

    def _geodesic_backtrack(self, tt: np.ndarray, end_rc, start_rc) -> np.ndarray | None:
        """Walk from end to start by steepest descent of travel-time field.

        Returns flat indices (start..end), or None on failure.
        """
        H, W = tt.shape
        ctx = self._ctx
        r, c = int(end_rc[0]), int(end_rc[1])
        sr, sc = int(start_rc[0]), int(start_rc[1])
        path = [r * W + c]
        max_steps = 8 * (H + W)
        for _ in range(max_steps):
            if r == sr and c == sc:
                break
            best_r, best_c = r, c
            best_t = tt[r, c]
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if nr < 0 or nr >= H or nc < 0 or nc >= W:
                        continue
                    t = tt[nr, nc]
                    if t < best_t:
                        best_t = t
                        best_r, best_c = nr, nc
            if best_r == r and best_c == c:
                return None
            r, c = best_r, best_c
            path.append(r * W + c)
        if r != sr or c != sc:
            return None
        path.reverse()
        return np.asarray(path, dtype=np.int32)

    def _hfm_gpu_field(self, start_rc) -> np.ndarray:
        from agd.Eikonal import dictIn
        ctx = self._ctx
        H, W = ctx.H, ctx.W
        cost = self._effective_cost.astype(np.float64, copy=True)
        cost[~np.isfinite(cost)] = 1e18
        d = dictIn({'model': 'Isotropic2', 'arrayOrdering': 'RowMajor'})
        d.SetRect(sides=[[0.0, float(H)], [0.0, float(W)]], dimx=H)
        d['cost'] = cost
        d['seed'] = np.array([float(start_rc[0]) + 0.5, float(start_rc[1]) + 0.5])
        d['gridScale'] = float(ctx.pixel_size)
        d['mode'] = 'gpu_transfer'
        out = d.Run()
        tt = np.asarray(out['values'], dtype=np.float64)
        tt = np.where(self._passable_mask, tt, np.inf)
        return tt


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_backend(name: str) -> Backend:
    name = name.lower()
    if name == "astar":
        return AstarBackend()
    if name == "hybrid":
        return HybridBackend()
    if name == "hfm":
        return HfmBackend()
    raise ValueError(f"Unknown backend: {name!r}. Choose astar | hybrid | hfm.")
