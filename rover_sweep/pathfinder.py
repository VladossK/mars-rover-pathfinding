"""Numba A* pathfinder on implicit grid with custom binary heap."""
from __future__ import annotations
import math
import numpy as np
from numba import njit

SQRT2 = math.sqrt(2.0)

_DR = np.array([ 1, -1,  0,  0,  1, -1,  1, -1], dtype=np.int32)
_DC = np.array([ 0,  0,  1, -1,  1,  1, -1, -1], dtype=np.int32)
_DD = np.array([1.0, 1.0, 1.0, 1.0, SQRT2, SQRT2, SQRT2, SQRT2], dtype=np.float64)


@njit(cache=True, fastmath=False, boundscheck=False)
def _heap_push(heap_f, heap_n, size, f, node):
    i = size
    heap_f[i] = f
    heap_n[i] = node
    size += 1
    while i > 0:
        parent = (i - 1) >> 1
        if heap_f[parent] > heap_f[i]:
            tf = heap_f[parent]; tn = heap_n[parent]
            heap_f[parent] = heap_f[i]; heap_n[parent] = heap_n[i]
            heap_f[i] = tf; heap_n[i] = tn
            i = parent
        else:
            break
    return size


@njit(cache=True, fastmath=False, boundscheck=False)
def _heap_pop(heap_f, heap_n, size):
    top_f = heap_f[0]
    top_n = heap_n[0]
    size -= 1
    heap_f[0] = heap_f[size]
    heap_n[0] = heap_n[size]
    i = 0
    while True:
        l = 2 * i + 1
        r = l + 1
        smallest = i
        if l < size and heap_f[l] < heap_f[smallest]:
            smallest = l
        if r < size and heap_f[r] < heap_f[smallest]:
            smallest = r
        if smallest == i:
            break
        tf = heap_f[i]; tn = heap_n[i]
        heap_f[i] = heap_f[smallest]; heap_n[i] = heap_n[smallest]
        heap_f[smallest] = tf; heap_n[smallest] = tn
        i = smallest
    return top_f, top_n, size


@njit(cache=True, fastmath=False, boundscheck=False)
def astar_core(
    elev_flat,      # float32[N]
    valid_flat,     # uint8[N]   (1 = valid, 0 = nodata)
    H, W,
    start_idx,
    end_idx,
    max_slope_deg,
    pixel_size,
    heap_f,         # float64[cap]
    heap_n,         # int32[cap]
    g,              # float64[N]
    prev,           # int32[N]
    dr_arr,         # int32[8]
    dc_arr,         # int32[8]
    dd_arr,         # float64[8]
):
    """A* on implicit grid. Returns (status, path_len_cells).

    status: 1 = success (path written to prev), 0 = unreachable.
    Path reconstruction is done outside via prev[].
    """
    N = H * W
    INF = 1e308
    for i in range(N):
        g[i] = INF
        prev[i] = -1

    if valid_flat[start_idx] == 0 or valid_flat[end_idx] == 0:
        return 0, 0

    end_r = end_idx // W
    end_c = end_idx - end_r * W

    g[start_idx] = 0.0
    sr = start_idx // W
    sc = start_idx - sr * W
    h0 = math.sqrt((sr - end_r) * (sr - end_r) + (sc - end_c) * (sc - end_c)) * pixel_size

    heap_size = 0
    heap_size = _heap_push(heap_f, heap_n, heap_size, h0, np.int32(start_idx))

    cap = heap_f.size

    while heap_size > 0:
        f_curr, curr, heap_size = _heap_pop(heap_f, heap_n, heap_size)

        if curr == end_idx:
            return 1, 0

        g_curr = g[curr]
        if f_curr > g_curr + math.sqrt(
            ((curr // W) - end_r) * ((curr // W) - end_r) +
            ((curr - (curr // W) * W) - end_c) * ((curr - (curr // W) * W) - end_c)
        ) * pixel_size + 1e-6:
            continue

        cr = curr // W
        cc = curr - cr * W
        elev_c = elev_flat[curr]

        for k in range(8):
            nr = cr + dr_arr[k]
            nc = cc + dc_arr[k]
            if nr < 0 or nr >= H or nc < 0 or nc >= W:
                continue
            nidx = nr * W + nc
            if valid_flat[nidx] == 0:
                continue

            h_dist = dd_arr[k] * pixel_size
            dz = elev_flat[nidx] - elev_c
            if dz < 0.0:
                dz = -dz
            slope = math.degrees(math.atan2(dz, h_dist))
            if slope > max_slope_deg:
                continue

            new_g = g_curr + h_dist
            if new_g < g[nidx]:
                g[nidx] = new_g
                prev[nidx] = np.int32(curr)
                h = math.sqrt((nr - end_r) * (nr - end_r) + (nc - end_c) * (nc - end_c)) * pixel_size
                f = new_g + h
                if heap_size >= cap:
                    return -1, 0
                heap_size = _heap_push(heap_f, heap_n, heap_size, f, np.int32(nidx))

    return 0, 0


@njit(cache=True, boundscheck=False)
def reconstruct_path(prev, start_idx, end_idx, out_buf):
    """Walk prev[] from end to start, write reverse into out_buf. Returns path length."""
    n = 0
    node = end_idx
    cap = out_buf.size
    while node != -1 and n < cap:
        out_buf[n] = node
        if node == start_idx:
            n += 1
            break
        node = prev[node]
        n += 1
    if n == 0 or out_buf[n - 1] != start_idx:
        return 0
    for i in range(n // 2):
        tmp = out_buf[i]
        out_buf[i] = out_buf[n - 1 - i]
        out_buf[n - 1 - i] = tmp
    return n


class AStarPathfinder:
    """Reusable A* worker. Allocates buffers once, calls Numba core per pair."""

    def __init__(self, H, W, heap_cap_mult: int = 5, max_path_cells: int | None = None):
        self.H = int(H)
        self.W = int(W)
        N = self.H * self.W
        self.N = N
        cap = max(1024, heap_cap_mult * N)
        self.heap_f = np.empty(cap, dtype=np.float64)
        self.heap_n = np.empty(cap, dtype=np.int32)
        self.g = np.empty(N, dtype=np.float64)
        self.prev = np.empty(N, dtype=np.int32)
        self.path_buf = np.empty(max_path_cells or (4 * (self.H + self.W)), dtype=np.int32)

    def find_path(
        self,
        elev_flat: np.ndarray,
        valid_flat: np.ndarray,
        start_idx: int,
        end_idx: int,
        max_slope_deg: float,
        pixel_size: float,
    ) -> np.ndarray | None:
        status, _ = astar_core(
            elev_flat, valid_flat, self.H, self.W,
            np.int32(start_idx), np.int32(end_idx),
            float(max_slope_deg), float(pixel_size),
            self.heap_f, self.heap_n, self.g, self.prev,
            _DR, _DC, _DD,
        )
        if status != 1:
            return None
        n = reconstruct_path(self.prev, np.int32(start_idx), np.int32(end_idx), self.path_buf)
        if n == 0:
            return None
        return self.path_buf[:n].copy()


def path_length_meters(path_idx: np.ndarray, W: int, pixel_size: float) -> float:
    """Ground distance of a path in meters."""
    if path_idx.size < 2:
        return 0.0
    rows = (path_idx // W).astype(np.float64)
    cols = (path_idx - (path_idx // W) * W).astype(np.float64)
    dr = np.diff(rows)
    dc = np.diff(cols)
    return float(np.sum(np.hypot(dr, dc)) * pixel_size)


def warmup():
    """Pre-compile Numba kernels with a tiny grid."""
    H, W = 8, 8
    elev = np.zeros(H * W, dtype=np.float32)
    valid = np.ones(H * W, dtype=np.uint8)
    pf = AStarPathfinder(H, W, heap_cap_mult=2)
    pf.find_path(elev, valid, 0, H * W - 1, 89.0, 200.0)
    cost = np.ones(H * W, dtype=np.float32)
    pf_t = ThetaStarPathfinder(H, W, heap_cap_mult=2)
    pf_t.find_path(elev, valid, cost, 0, H * W - 1, 89.0, 200.0)


# ===========================================================================
# Lazy Theta* — any-angle A* with line-of-sight relaxation
# ===========================================================================

@njit(cache=True, fastmath=False, boundscheck=False)
def _line_of_sight_clear(
    elev_flat, valid_flat, H, W,
    r0, c0, r1, c1,
    max_slope_deg, pixel_size,
):
    """Walk Bresenham from (r0,c0) to (r1,c1). Returns 1 if LoS clear, 0 if blocked.

    Blocked if any cell along the line is invalid OR any step exceeds max_slope.
    """
    dr = r1 - r0
    dc = c1 - c0
    abs_dr = dr if dr >= 0 else -dr
    abs_dc = dc if dc >= 0 else -dc
    step_r = 1 if dr > 0 else (-1 if dr < 0 else 0)
    step_c = 1 if dc > 0 else (-1 if dc < 0 else 0)

    r = r0
    c = c0
    if valid_flat[r0 * W + c0] == 0 or valid_flat[r1 * W + c1] == 0:
        return 0

    if abs_dr >= abs_dc:
        err = abs_dr // 2
        n_steps = abs_dr
        for _ in range(n_steps):
            err -= abs_dc
            if err < 0:
                nc = c + step_c
                err += abs_dr
            else:
                nc = c
            nr = r + step_r

            if nr < 0 or nr >= H or nc < 0 or nc >= W:
                return 0
            if valid_flat[nr * W + nc] == 0:
                return 0
            sr = nr if nr == r else r
            sc = nc if nc == c else c
            dz = elev_flat[nr * W + nc] - elev_flat[r * W + c]
            if dz < 0.0:
                dz = -dz
            d_pix = math.sqrt((nr - r) * (nr - r) + (nc - c) * (nc - c))
            h = d_pix * pixel_size
            if h <= 0.0:
                return 0
            slope = math.degrees(math.atan2(dz, h))
            if slope > max_slope_deg:
                return 0
            r = nr
            c = nc
    else:
        err = abs_dc // 2
        n_steps = abs_dc
        for _ in range(n_steps):
            err -= abs_dr
            if err < 0:
                nr = r + step_r
                err += abs_dc
            else:
                nr = r
            nc = c + step_c

            if nr < 0 or nr >= H or nc < 0 or nc >= W:
                return 0
            if valid_flat[nr * W + nc] == 0:
                return 0
            dz = elev_flat[nr * W + nc] - elev_flat[r * W + c]
            if dz < 0.0:
                dz = -dz
            d_pix = math.sqrt((nr - r) * (nr - r) + (nc - c) * (nc - c))
            h = d_pix * pixel_size
            if h <= 0.0:
                return 0
            slope = math.degrees(math.atan2(dz, h))
            if slope > max_slope_deg:
                return 0
            r = nr
            c = nc
    return 1


@njit(cache=True, fastmath=False, boundscheck=False)
def theta_star_core(
    elev_flat,       # float32[N]
    valid_flat,      # uint8[N]
    cost_flat,       # float32[N]
    H, W,
    start_idx,
    end_idx,
    max_slope_deg,
    pixel_size,
    heap_f,
    heap_n,
    g,
    prev,
    gen,
    cur_gen,
    dr_arr, dc_arr, dd_arr,
):
    """Theta* on implicit grid with cost surface modulation.

    cost_flat: per-cell scalar cost multiplier (>= 1).
    Edge cost a->b = 3D_geom(a,b) * 0.5 * (cost[a] + cost[b]).

    Returns (status, end_idx_for_recon).
    status: 1 = success, 0 = unreachable, -1 = heap overflow.
    """
    if valid_flat[start_idx] == 0 or valid_flat[end_idx] == 0:
        return 0

    end_r = end_idx // W
    end_c = end_idx - end_r * W
    sr = start_idx // W
    sc = start_idx - sr * W

    g[start_idx] = 0.0
    prev[start_idx] = np.int32(start_idx)
    gen[start_idx] = cur_gen

    h0 = math.sqrt((sr - end_r) * (sr - end_r) + (sc - end_c) * (sc - end_c)) * pixel_size

    heap_size = _heap_push(heap_f, heap_n, 0, h0, np.int32(start_idx))
    cap = heap_f.size

    while heap_size > 0:
        f_curr, curr, heap_size = _heap_pop(heap_f, heap_n, heap_size)

        if curr == end_idx:
            return 1

        if gen[curr] != cur_gen:
            continue
        g_curr = g[curr]
        cr = curr // W
        cc = curr - cr * W
        h_curr = math.sqrt((cr - end_r) * (cr - end_r) + (cc - end_c) * (cc - end_c)) * pixel_size
        if f_curr > g_curr + h_curr + 1e-6:
            continue

        par = prev[curr]
        pr = par // W
        pc = par - pr * W

        for k in range(8):
            nr = cr + dr_arr[k]
            nc = cc + dc_arr[k]
            if nr < 0 or nr >= H or nc < 0 or nc >= W:
                continue
            nidx = nr * W + nc
            if valid_flat[nidx] == 0:
                continue

            h_dist = dd_arr[k] * pixel_size
            dz = elev_flat[nidx] - elev_flat[curr]
            if dz < 0.0:
                dz = -dz
            slope = math.degrees(math.atan2(dz, h_dist))
            if slope > max_slope_deg:
                continue

            par_g = g[par] if gen[par] == cur_gen else 1e308
            par_visible = 0
            if par != curr:
                par_visible = _line_of_sight_clear(
                    elev_flat, valid_flat, H, W, pr, pc, nr, nc,
                    max_slope_deg, pixel_size,
                )

            if par_visible == 1:
                d_pr_nr = math.sqrt((pr - nr) * (pr - nr) + (pc - nc) * (pc - nc)) * pixel_size
                dz_pn = elev_flat[nidx] - elev_flat[par]
                geom = math.sqrt(d_pr_nr * d_pr_nr + dz_pn * dz_pn)
                edge = geom * 0.5 * (cost_flat[par] + cost_flat[nidx])
                new_g = par_g + edge
                cand_prev = par
            else:
                dz_cn = elev_flat[nidx] - elev_flat[curr]
                geom = math.sqrt(h_dist * h_dist + dz_cn * dz_cn)
                edge = geom * 0.5 * (cost_flat[curr] + cost_flat[nidx])
                new_g = g_curr + edge
                cand_prev = curr

            if gen[nidx] != cur_gen or new_g < g[nidx]:
                g[nidx] = new_g
                prev[nidx] = np.int32(cand_prev)
                gen[nidx] = cur_gen
                h = math.sqrt((nr - end_r) * (nr - end_r) + (nc - end_c) * (nc - end_c)) * pixel_size
                f = new_g + h
                if heap_size >= cap:
                    return -1
                heap_size = _heap_push(heap_f, heap_n, heap_size, f, np.int32(nidx))

    return 0


class ThetaStarPathfinder:
    """Theta* (any-angle A*) with cost surface modulation and generation-counter heap reset."""

    def __init__(self, H, W, heap_cap_mult: int = 5, max_path_cells: int | None = None):
        self.H = int(H)
        self.W = int(W)
        N = self.H * self.W
        self.N = N
        cap = max(1024, heap_cap_mult * N)
        self.heap_f = np.empty(cap, dtype=np.float64)
        self.heap_n = np.empty(cap, dtype=np.int32)
        self.g = np.empty(N, dtype=np.float64)
        self.prev = np.empty(N, dtype=np.int32)
        self.gen = np.zeros(N, dtype=np.int32)
        self.cur_gen = np.int32(0)
        self.path_buf = np.empty(max_path_cells or (4 * (self.H + self.W)), dtype=np.int32)

    def _next_gen(self):
        self.cur_gen += 1
        if int(self.cur_gen) > 2_000_000_000:
            self.gen[:] = 0
            self.cur_gen = np.int32(1)

    def find_path(
        self,
        elev_flat: np.ndarray,
        valid_flat: np.ndarray,
        cost_flat: np.ndarray,
        start_idx: int,
        end_idx: int,
        max_slope_deg: float,
        pixel_size: float,
    ) -> np.ndarray | None:
        self._next_gen()
        status = theta_star_core(
            elev_flat, valid_flat, cost_flat, self.H, self.W,
            np.int32(start_idx), np.int32(end_idx),
            float(max_slope_deg), float(pixel_size),
            self.heap_f, self.heap_n, self.g, self.prev,
            self.gen, np.int32(self.cur_gen),
            _DR, _DC, _DD,
        )
        if status != 1:
            return None
        n = reconstruct_path(self.prev, np.int32(start_idx), np.int32(end_idx), self.path_buf)
        if n == 0:
            return None
        return self.path_buf[:n].copy()
