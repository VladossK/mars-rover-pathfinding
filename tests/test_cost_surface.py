"""Tests for cost_surface utilities."""
import numpy as np
import pytest

from rover_sweep.cost_surface import (
    pixel_max_slope, terrain_ruggedness_index, build_cost_map,
    fill_nodata_nearest, path_length_3d_meters, path_slope_stats,
)


def test_flat_terrain_zero_slope_zero_tri():
    elev = np.zeros((10, 10), dtype=np.float32)
    valid = np.ones((10, 10), dtype=bool)
    out = build_cost_map(elev, valid, pixel_size=200.0)
    assert out["slope_deg"].max() == pytest.approx(0.0)
    assert out["tri"].max() == pytest.approx(0.0)
    assert out["cost"][valid].max() == pytest.approx(1.0)


def test_slope_monotonicity():
    elev = np.zeros((3, 3), dtype=np.float32)
    elev[1, 1] = 100.0
    valid = np.ones((3, 3), dtype=bool)
    slope = pixel_max_slope(elev, pixel_size=200.0)
    assert slope[1, 1] > 0
    expected = np.degrees(np.arctan2(100.0, 200.0))
    assert slope[0, 1] == pytest.approx(expected, abs=0.5)


def test_tri_nonzero_on_wall():
    elev = np.zeros((5, 5), dtype=np.float32)
    elev[:, 2] = 100.0
    tri = terrain_ruggedness_index(elev)
    assert tri[2, 1] > 0
    assert tri[0, 0] == pytest.approx(0.0)


def test_cost_blocks_invalid_cells():
    elev = np.zeros((5, 5), dtype=np.float32)
    valid = np.ones((5, 5), dtype=bool)
    valid[2, 2] = False
    out = build_cost_map(elev, valid, pixel_size=200.0)
    assert np.isinf(out["cost"][2, 2])


def test_fill_nodata_nearest_smooths():
    elev = np.array([[10.0, 10.0, 10.0],
                     [10.0, 99.0, 10.0],
                     [10.0, 10.0, 10.0]], dtype=np.float32)
    valid = np.ones((3, 3), dtype=bool)
    valid[1, 1] = False
    filled = fill_nodata_nearest(elev, valid)
    assert filled[1, 1] == pytest.approx(10.0)


def test_3d_path_length_includes_z():
    elev = np.zeros((1, 3), dtype=np.float32)
    elev[0, 2] = 300.0
    path = np.array([0, 1, 2], dtype=np.int32)
    length = path_length_3d_meters(path, elev, W=3, pixel_size=200.0)
    expected = np.sqrt(200**2 + 0) + np.sqrt(200**2 + 300**2)
    assert length == pytest.approx(expected, rel=1e-6)


def test_path_slope_stats():
    elev = np.zeros((1, 3), dtype=np.float32)
    elev[0, 1] = 100.0
    elev[0, 2] = 50.0
    path = np.array([0, 1, 2], dtype=np.int32)
    stats = path_slope_stats(path, elev, W=3, pixel_size=200.0)
    assert stats["climb_m"] == pytest.approx(100.0)
    assert stats["descent_m"] == pytest.approx(50.0)
    assert stats["max_deg"] > 0
