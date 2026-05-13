"""Tests for Theta* (any-angle A*)."""
import numpy as np
import pytest

from rover_sweep.pathfinder import (
    AStarPathfinder, ThetaStarPathfinder, path_length_meters, warmup,
)
from rover_sweep.cost_surface import path_length_3d_meters


@pytest.fixture(scope="module", autouse=True)
def _warm():
    warmup()


def test_theta_corner_path_shorter_than_astar():
    H, W = 20, 20
    elev = np.zeros(H * W, dtype=np.float32)
    valid = np.ones(H * W, dtype=np.uint8)
    cost = np.ones(H * W, dtype=np.float32)

    start = 0
    end = (H - 1) * W + (W - 1)

    pf_a = AStarPathfinder(H, W, heap_cap_mult=2)
    pf_t = ThetaStarPathfinder(H, W, heap_cap_mult=2)

    p_a = pf_a.find_path(elev, valid, start, end, 89.0, 200.0)
    p_t = pf_t.find_path(elev, valid, cost, start, end, 89.0, 200.0)

    assert p_a is not None and p_t is not None
    la = path_length_meters(p_a, W, 200.0)
    lt = path_length_meters(p_t, W, 200.0)
    assert lt <= la + 1e-6


def test_theta_obstacle_avoidance():
    H, W = 15, 15
    elev = np.zeros((H, W), dtype=np.float32)
    elev[7, 3:12] = 5000.0
    elev_flat = elev.reshape(-1)
    valid = np.ones(H * W, dtype=np.uint8)
    cost = np.ones(H * W, dtype=np.float32)

    pf = ThetaStarPathfinder(H, W, heap_cap_mult=2)
    start = 0
    end = (H - 1) * W + (W - 1)
    path = pf.find_path(elev_flat, valid, cost, start, end, 10.0, 200.0)
    assert path is not None
    rows = path // W
    assert int(rows.min()) >= 0 and int(rows.max()) <= H - 1


def test_theta_wall_blocks():
    H, W = 5, 5
    elev = np.zeros((H, W), dtype=np.float32)
    elev[:, 2] = 5000.0
    elev_flat = elev.reshape(-1)
    valid = np.ones(H * W, dtype=np.uint8)
    cost = np.ones(H * W, dtype=np.float32)

    pf = ThetaStarPathfinder(H, W, heap_cap_mult=2)
    path = pf.find_path(elev_flat, valid, cost, 0, 4, 10.0, 200.0)
    assert path is None


def test_theta_generation_counter_reuse():
    """Two consecutive find_path calls share buffers — second call must not be polluted."""
    H, W = 10, 10
    elev = np.zeros(H * W, dtype=np.float32)
    valid = np.ones(H * W, dtype=np.uint8)
    cost = np.ones(H * W, dtype=np.float32)

    pf = ThetaStarPathfinder(H, W, heap_cap_mult=2)
    p1 = pf.find_path(elev, valid, cost, 0, H * W - 1, 89.0, 200.0)
    p2 = pf.find_path(elev, valid, cost, 0, H * W - 1, 89.0, 200.0)
    assert p1 is not None and p2 is not None
    assert path_length_meters(p1, W, 200.0) == pytest.approx(path_length_meters(p2, W, 200.0))
