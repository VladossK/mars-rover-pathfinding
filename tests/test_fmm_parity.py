"""Tests for scikit-fmm reachability backend."""
import numpy as np
import pytest

from rover_sweep.reachability import fmm_reachability


def test_fmm_flat_isotropic_grows_with_distance():
    H, W = 40, 40
    passable = np.ones((H, W), dtype=bool)
    cost = np.ones((H, W), dtype=np.float32)
    dist = fmm_reachability(passable, cost, start_rc=(0, 0), pixel_size=1.0)
    assert dist[0, 0] == pytest.approx(0.0, abs=1e-6)
    assert dist[0, 10] == pytest.approx(10.0, rel=0.1)
    assert dist[10, 10] == pytest.approx(np.sqrt(200.0), rel=0.1)


def test_fmm_blocks_isolated_region():
    H, W = 10, 25
    passable = np.ones((H, W), dtype=bool)
    passable[:, 12] = False
    cost = np.ones((H, W), dtype=np.float32)
    dist = fmm_reachability(passable, cost, start_rc=(0, 0), pixel_size=1.0)
    assert np.isinf(dist[0, 20])


def test_fmm_cost_scales_distance():
    H, W = 30, 30
    passable = np.ones((H, W), dtype=bool)
    cost_lo = np.ones((H, W), dtype=np.float32)
    cost_hi = np.ones((H, W), dtype=np.float32) * 5.0
    d_lo = fmm_reachability(passable, cost_lo, start_rc=(0, 0), pixel_size=1.0)
    d_hi = fmm_reachability(passable, cost_hi, start_rc=(0, 0), pixel_size=1.0)
    assert d_hi[10, 10] == pytest.approx(5.0 * d_lo[10, 10], rel=0.15)
