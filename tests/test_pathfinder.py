"""Tests for Numba A* pathfinder."""
import numpy as np
import pytest

from rover_sweep.pathfinder import AStarPathfinder, path_length_meters, warmup


@pytest.fixture(scope="module", autouse=True)
def _warm():
    warmup()


def test_straight_path_on_flat():
    H, W = 10, 10
    elev = np.zeros(H * W, dtype=np.float32)
    valid = np.ones(H * W, dtype=np.uint8)
    pf = AStarPathfinder(H, W, heap_cap_mult=2)
    path = pf.find_path(elev, valid, 0, W - 1, 89.0, 200.0)
    assert path is not None
    assert path[0] == 0 and path[-1] == W - 1
    length = path_length_meters(path, W, 200.0)
    assert length == pytest.approx(200.0 * (W - 1))


def test_diagonal_path_on_flat():
    H, W = 8, 8
    elev = np.zeros(H * W, dtype=np.float32)
    valid = np.ones(H * W, dtype=np.uint8)
    pf = AStarPathfinder(H, W, heap_cap_mult=2)
    start = 0
    end = (H - 1) * W + (W - 1)
    path = pf.find_path(elev, valid, start, end, 89.0, 200.0)
    assert path is not None
    expected = 200.0 * (H - 1) * np.sqrt(2)
    assert path_length_meters(path, W, 200.0) == pytest.approx(expected, rel=1e-6)


def test_wall_blocks_path():
    H, W = 5, 5
    elev = np.zeros((H, W), dtype=np.float32)
    elev[:, 2] = 5000.0
    elev_flat = elev.reshape(-1)
    valid = np.ones(H * W, dtype=np.uint8)
    pf = AStarPathfinder(H, W, heap_cap_mult=2)
    path = pf.find_path(elev_flat, valid, 0, 4, max_slope_deg=10.0, pixel_size=200.0)
    assert path is None


def test_path_goes_around_low_wall():
    H, W = 7, 7
    elev = np.zeros((H, W), dtype=np.float32)
    elev[3, 0:5] = 1000.0
    elev_flat = elev.reshape(-1)
    valid = np.ones(H * W, dtype=np.uint8)
    pf = AStarPathfinder(H, W, heap_cap_mult=2)
    path = pf.find_path(elev_flat, valid, 0, (H - 1) * W + (W - 1), 10.0, 200.0)
    assert path is not None
    rows = path // W
    assert (rows < 3).any() or (rows > 3).any()
