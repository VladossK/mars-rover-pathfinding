"""Tests for graph_builder + reachability."""
import numpy as np
import pytest

from rover_sweep.graph_builder import build_edge_mask, edge_mask_to_csr, DIRECTIONS
from rover_sweep.reachability import full_dijkstra, reachability_stats


def test_flat_grid_all_edges():
    elev = np.zeros((5, 5), dtype=np.float32)
    valid = np.ones((5, 5), dtype=bool)
    mask = build_edge_mask(elev, valid, max_slope_deg=10.0, pixel_size=200.0)
    inner = mask[1:-1, 1:-1]
    assert inner.all(), "Flat terrain → every inner edge passable"


def test_wall_blocks_edges():
    elev = np.zeros((3, 3), dtype=np.float32)
    elev[1, 1] = 1000.0
    valid = np.ones((3, 3), dtype=bool)
    mask = build_edge_mask(elev, valid, max_slope_deg=10.0, pixel_size=200.0)
    for k, (dr, dc, _) in enumerate(DIRECTIONS):
        nr, nc = 0 + dr, 0 + dc
        if (nr, nc) == (1, 1):
            assert mask[0, 0, k] == False, "Edge into wall must be blocked"


def test_csr_dijkstra_reachability():
    elev = np.zeros((4, 4), dtype=np.float32)
    valid = np.ones((4, 4), dtype=bool)
    mask = build_edge_mask(elev, valid, max_slope_deg=10.0, pixel_size=200.0)
    csr = edge_mask_to_csr(mask, pixel_size=200.0)
    dist = full_dijkstra(csr, start_idx=0)
    assert np.isfinite(dist).all()
    assert dist[0] == 0.0
    assert dist[1] == pytest.approx(200.0)
    assert dist[5] == pytest.approx(200.0 * np.sqrt(2))


def test_isolated_region_unreachable():
    elev = np.zeros((3, 5), dtype=np.float32)
    elev[:, 2] = 5000.0
    valid = np.ones((3, 5), dtype=bool)
    mask = build_edge_mask(elev, valid, max_slope_deg=10.0, pixel_size=200.0)
    csr = edge_mask_to_csr(mask, pixel_size=200.0)
    dist = full_dijkstra(csr, start_idx=0)
    assert np.isinf(dist[4])
    stats = reachability_stats(dist, valid.reshape(-1), pixel_size=200.0)
    assert stats["unreachable_pixels"] > 0
    assert stats["area_unreachable_km2"] > 0.0
    assert stats["reachable_pct"] + stats["unreachable_pct"] == pytest.approx(1.0)
