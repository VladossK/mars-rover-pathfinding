"""Angle-sweep orchestrator with multi-backend dispatcher."""
from __future__ import annotations
import os
import sys
import time

if sys.platform == "win32":
    os.system("")
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
from tqdm import tqdm

from . import config
from . import reporting
from .pathfinder import path_length_meters, warmup
from .reachability import reachability_stats
from .cost_surface import (
    build_cost_map, path_length_3d_meters, path_slope_stats,
)
from .backends import BackendContext, make_backend


@dataclass
class AngleResult:
    angle: int
    backend: str
    n_success: int
    n_fail: int
    vec_2d_m: list = field(default_factory=list)
    vec_3d_m: list = field(default_factory=list)
    path_len_m: list = field(default_factory=list)
    path_len_3d_m: list = field(default_factory=list)
    path_max_slope_deg: list = field(default_factory=list)
    path_mean_slope_deg: list = field(default_factory=list)
    climb_m: list = field(default_factory=list)
    descent_m: list = field(default_factory=list)
    times_ms: list = field(default_factory=list)
    reach: dict = field(default_factory=dict)
    full_reach_time_s: float = 0.0


def generate_pairs(valid_mask: np.ndarray, n_pairs: int, seed: int) -> np.ndarray:
    H, W = valid_mask.shape
    rng = np.random.default_rng(seed)
    out = np.empty((n_pairs, 4), dtype=np.int32)
    n = 0
    safety = 0
    while n < n_pairs and safety < n_pairs * 100:
        sr = int(rng.integers(0, H))
        sc = int(rng.integers(0, W))
        er = int(rng.integers(0, H))
        ec = int(rng.integers(0, W))
        if valid_mask[sr, sc] and valid_mask[er, ec] and (sr != er or sc != ec):
            out[n] = (sr, sc, er, ec)
            n += 1
        safety += 1
    if n < n_pairs:
        raise RuntimeError(f"Only generated {n}/{n_pairs} valid pairs")
    return out


def pick_reach_start(valid_mask: np.ndarray, seed: int) -> tuple[int, int]:
    H, W = valid_mask.shape
    rng = np.random.default_rng(seed)
    valid_idx = np.flatnonzero(valid_mask.ravel())
    pick = int(rng.choice(valid_idx))
    return divmod(pick, W)


def run_angle(
    ctx: BackendContext,
    backend,
    pairs: np.ndarray,
    angle: int,
    reach_start_rc: tuple[int, int],
    out_dir: Path,
    pair_subset: np.ndarray | None = None,
    verbose: bool = False,
) -> AngleResult:
    H, W = ctx.H, ctx.W

    print(f"\n[angle={angle:02d}] backend={backend.name} prepare ...", flush=True)
    t_prep = time.perf_counter()
    backend.prepare(ctx, float(angle))
    print(f"[angle={angle:02d}]   prepare done in {time.perf_counter()-t_prep:.2f}s", flush=True)

    print(f"[angle={angle:02d}] reachability from {reach_start_rc} ...", flush=True)
    t0 = time.perf_counter()
    dist_2d = backend.reachability(reach_start_rc)
    full_reach_time = time.perf_counter() - t0
    print(f"[angle={angle:02d}]   reachability done in {full_reach_time:.2f}s", flush=True)

    valid_flat_bool = ctx.valid_mask.reshape(-1)
    reach = reachability_stats(dist_2d.reshape(-1), valid_flat_bool, ctx.pixel_size)
    print(f"[angle={angle:02d}]   reach%={100*reach['reachable_pct']:.3f}  "
          f"unreach={reach['unreachable_pixels']:,} px = {reach['area_unreachable_km2']:.1f} km^2",
          flush=True)

    result = AngleResult(
        angle=angle, backend=backend.name,
        n_success=0, n_fail=0,
        reach=reach, full_reach_time_s=full_reach_time,
    )

    angle_rows = []
    sample_path_rc = None

    if pair_subset is None:
        pair_indices = list(range(pairs.shape[0]))
    else:
        pair_indices = pair_subset.tolist()

    n_pairs_total = len(pair_indices)
    print(f"[angle={angle:02d}] pathfinding {n_pairs_total} pairs ...", flush=True)
    inner = tqdm(total=n_pairs_total, desc=f"  paths[{backend.name},{angle}d]",
                 unit="path", leave=False, mininterval=0.5, dynamic_ncols=True)

    for i in pair_indices:
        sr, sc, er, ec = int(pairs[i, 0]), int(pairs[i, 1]), int(pairs[i, 2]), int(pairs[i, 3])

        t_pair = time.perf_counter()
        path = backend.find_path((sr, sc), (er, ec))
        dt_ms = (time.perf_counter() - t_pair) * 1000.0

        if path is None:
            result.n_fail += 1
            row = {
                "iteration": i + 1, "backend": backend.name,
                "start_row": sr, "start_col": sc,
                "end_row": er, "end_col": ec,
                "success": 0,
                "vec_2d_m": float(np.hypot(sr - er, sc - ec) * ctx.pixel_size),
                "vec_3d_m": "", "path_len_m": "", "path_len_3d_m": "",
                "path_max_slope_deg": "", "path_mean_slope_deg": "",
                "climb_m": "", "descent_m": "",
                "time_ms": round(dt_ms, 3),
                "detour_factor": "",
                "fail_reason": "NO_PATH",
            }
        else:
            length_m = path_length_meters(path, W, ctx.pixel_size)
            length_3d = path_length_3d_meters(path, ctx.elev, W, ctx.pixel_size)
            sl = path_slope_stats(path, ctx.elev, W, ctx.pixel_size)
            dz = float(ctx.elev[er, ec]) - float(ctx.elev[sr, sc])
            vec2d = float(np.hypot(sr - er, sc - ec) * ctx.pixel_size)
            vec3d = float(np.sqrt(vec2d * vec2d + dz * dz))

            detour_cap = config.MAX_PATH_DETOUR_FACTOR
            detour_factor = (length_m / vec2d) if vec2d > 0 else 0.0
            is_detour_fail = (detour_cap is not None) and (detour_factor > detour_cap)

            if is_detour_fail:
                result.n_fail += 1
                row = {
                    "iteration": i + 1, "backend": backend.name,
                    "start_row": sr, "start_col": sc,
                    "end_row": er, "end_col": ec,
                    "success": 0,
                    "vec_2d_m": round(vec2d, 3),
                    "vec_3d_m": "",
                    "path_len_m": round(length_m, 3),
                    "path_len_3d_m": "", "path_max_slope_deg": "",
                    "path_mean_slope_deg": "", "climb_m": "", "descent_m": "",
                    "time_ms": round(dt_ms, 3),
                    "detour_factor": round(detour_factor, 2),
                    "fail_reason": "DETOUR_RUNAWAY",
                }
            else:
                result.n_success += 1
                result.vec_2d_m.append(vec2d)
                result.vec_3d_m.append(vec3d)
                result.path_len_m.append(length_m)
                result.path_len_3d_m.append(length_3d)
                result.path_max_slope_deg.append(sl["max_deg"])
                result.path_mean_slope_deg.append(sl["mean_deg"])
                result.climb_m.append(sl["climb_m"])
                result.descent_m.append(sl["descent_m"])
                result.times_ms.append(dt_ms)

                row = {
                    "iteration": i + 1, "backend": backend.name,
                    "start_row": sr, "start_col": sc,
                    "end_row": er, "end_col": ec,
                    "success": 1,
                    "vec_2d_m": round(vec2d, 3),
                    "vec_3d_m": round(vec3d, 3),
                    "path_len_m": round(length_m, 3),
                    "path_len_3d_m": round(length_3d, 3),
                    "path_max_slope_deg": round(sl["max_deg"], 3),
                    "path_mean_slope_deg": round(sl["mean_deg"], 3),
                    "climb_m": round(sl["climb_m"], 3),
                    "descent_m": round(sl["descent_m"], 3),
                    "time_ms": round(dt_ms, 3),
                    "detour_factor": round(detour_factor, 3),
                    "fail_reason": "",
                }
            if sample_path_rc is None and path.size >= 2:
                rows = (path // W).astype(np.int32)
                cols = (path - (path // W) * W).astype(np.int32)
                sample_path_rc = np.stack([rows, cols], axis=1)
        angle_rows.append(row)
        n_done = result.n_success + result.n_fail
        mean_t = (np.mean(result.times_ms) if result.times_ms else dt_ms)
        inner.set_postfix({
            "ok": result.n_success,
            "fail": result.n_fail,
            "last_ms": f"{dt_ms:.0f}",
            "mean_ms": f"{mean_t:.0f}",
        })
        inner.update(1)
        if verbose:
            ok = "OK " if path is not None else "FAIL"
            tqdm.write(f"    pair {i+1:>3}/{n_pairs_total}  {ok}  "
                       f"({sr},{sc})->({er},{ec})  vec={int(np.hypot(sr-er,sc-ec)*ctx.pixel_size/1000)}km  "
                       f"t={dt_ms:.0f}ms")
    inner.close()
    print(f"[angle={angle:02d}]   paths done: {result.n_success} ok / {result.n_fail} fail  "
          f"mean_t={float(np.mean(result.times_ms)) if result.times_ms else 0:.0f}ms",
          flush=True)

    angle_dir = out_dir / f"angle_{angle:02d}"
    print(f"[angle={angle:02d}] writing CSV + plot to {angle_dir} ...", flush=True)
    t_io = time.perf_counter()
    reporting.write_angle_csv(angle_rows, angle_dir / "paths.csv")
    reporting.plot_reachability(
        ctx.elev, dist_2d, ctx.valid_mask, reach_start_rc, sample_path_rc, angle,
        angle_dir / "reachability.png", ctx.pixel_size,
    )
    print(f"[angle={angle:02d}]   IO done in {time.perf_counter()-t_io:.2f}s", flush=True)

    return result


def run_sweep(
    elev: np.ndarray,
    valid_mask: np.ndarray,
    pixel_size: float,
    out_dir: Path,
    backend_name: str = "astar",
    n_pairs: int = config.NUM_PAIRS,
    start_angle: int = config.START_ANGLE,
    angle_step: int = config.ANGLE_STEP,
    max_angle: int = config.MAX_ANGLE,
    pairs_seed: int = config.PAIRS_SEED,
    reach_seed: int = config.REACH_SEED,
    k_slope: float = config.COST_K_SLOPE,
    k_rough: float = config.COST_K_ROUGH,
    hfm_subset_size: int = config.HFM_SUBSET_SIZE,
    verbose: bool = False,
) -> list[AngleResult]:
    H, W = elev.shape
    out_dir.mkdir(parents=True, exist_ok=True)

    t_warm = time.perf_counter()
    print("[init] Pre-warming Numba kernels ...", flush=True)
    warmup()
    print(f"[init]   warmup done in {time.perf_counter()-t_warm:.2f}s", flush=True)

    t_cost = time.perf_counter()
    print("[init] Building shared cost surfaces (slope, TRI, cost map) ...", flush=True)
    surf = build_cost_map(
        elev, valid_mask, pixel_size,
        k_slope=k_slope, k_rough=k_rough,
        tri_normalize=config.COST_TRI_NORMALIZE,
    )
    print(f"[init]   cost surface built in {time.perf_counter()-t_cost:.2f}s  "
          f"(slope.max={float(surf['slope_deg'].max()):.1f}°  "
          f"TRI.max={float(surf['tri'].max()):.1f}m)", flush=True)
    ctx = BackendContext(
        elev=surf["elev_filled"], valid_mask=valid_mask, pixel_size=pixel_size,
        H=H, W=W,
        cost_map=surf["cost"], slope_deg=surf["slope_deg"],
    )

    print(f"Generating {n_pairs} random pairs (seed={pairs_seed}) ...")
    pairs = generate_pairs(valid_mask, n_pairs, pairs_seed)
    reach_start_rc = pick_reach_start(valid_mask, reach_seed)
    print(f"Fixed reachability start: {reach_start_rc}")

    pair_subset = None
    if backend_name == "hfm" and pairs.shape[0] > hfm_subset_size:
        rng_sub = np.random.default_rng(pairs_seed + 1)
        pair_subset = rng_sub.choice(pairs.shape[0], size=hfm_subset_size, replace=False)
        pair_subset.sort()
        print(f"  [hfm] using subset of {hfm_subset_size} pairs (indices saved in summary)")

    backend = make_backend(backend_name)
    if backend_name == "hfm":
        gpu_ok = getattr(backend, "gpu_ok", False)
        gpu_msg = getattr(backend, "gpu_msg", "")
        if gpu_ok:
            print(f"\033[32m  [hfm] GPU detected: {gpu_msg}\033[0m")
        else:
            print(
                "\033[1;31m"
                "==========================================================\n"
                "  WARNING: GPU NOT DETECTED -- falling back to CPU\n"
                "  Reason: " + str(gpu_msg) + "\n"
                "  CPU FMM on 4M-cell grid is SLOW.\n"
                "  Estimated wall time:\n"
                "    * 1 angle, 30 pairs (subset)  ~ 5-10 min\n"
                "    * 1 angle, 200 pairs          ~ 50-150 min\n"
                "    * 20-angle full sweep, 200    ~ 17-50 hours\n"
                "  Reduce load: keep --hfm-subset-size small (default 30),\n"
                "    or pick a few angles via --start-angle / --max-angle.\n"
                "  For full max-accuracy runs, use a CUDA GPU machine\n"
                "    (pip install cupy-cuda12x agd).\n"
                "==========================================================\n"
                "\033[0m"
            )

    results: list[AngleResult] = []
    summary_rows: list[dict] = []

    angle = start_angle
    n_steps = max(1, (max_angle - start_angle) // max(1, angle_step) + 1)
    pbar = tqdm(total=n_steps, desc=f"sweep[{backend_name}]", unit="angle",
                position=0, leave=True, dynamic_ncols=True)
    while angle <= max_angle:
        t_angle = time.perf_counter()
        res = run_angle(ctx, backend, pairs, angle, reach_start_rc, out_dir,
                         pair_subset, verbose=verbose)
        elapsed = time.perf_counter() - t_angle
        results.append(res)
        print(f"[angle={angle:02d}] TOTAL: {elapsed:.1f}s\n", flush=True)

        def _mean(xs): return float(np.mean(xs)) if xs else 0.0
        r = res.reach
        summary_rows.append({
            "angle": angle, "backend": backend.name,
            "n_success": res.n_success, "n_fail": res.n_fail,
            "mean_vec_2d_m": round(_mean(res.vec_2d_m), 2),
            "mean_vec_3d_m": round(_mean(res.vec_3d_m), 2),
            "mean_path_len_m": round(_mean(res.path_len_m), 2),
            "mean_path_len_3d_m": round(_mean(res.path_len_3d_m), 2),
            "mean_path_max_slope_deg": round(_mean(res.path_max_slope_deg), 3),
            "mean_path_mean_slope_deg": round(_mean(res.path_mean_slope_deg), 3),
            "mean_climb_m": round(_mean(res.climb_m), 2),
            "mean_descent_m": round(_mean(res.descent_m), 2),
            "mean_time_ms": round(_mean(res.times_ms), 2),
            "reachable_pixels": r["reachable_pixels"],
            "unreachable_pixels": r["unreachable_pixels"],
            "reachable_pct": round(r["reachable_pct"], 6),
            "unreachable_pct": round(r["unreachable_pct"], 6),
            "nodata_pct": round(r["nodata_pct"], 6),
            "area_total_km2": round(r["area_total_km2"], 3),
            "area_reachable_km2": round(r["area_reachable_km2"], 3),
            "area_unreachable_km2": round(r["area_unreachable_km2"], 3),
            "area_nodata_km2": round(r["area_nodata_km2"], 3),
            "full_reach_time_s": round(res.full_reach_time_s, 3),
        })

        reporting.write_summary_csv(summary_rows, out_dir / "summary.csv")
        reporting.plot_unreachable_vs_angle(summary_rows, out_dir / "unreachable_vs_angle.png")

        pbar.set_postfix({
            "reach%": f"{100*r['reachable_pct']:.2f}",
            "fail": res.n_fail,
            "t": f"{elapsed:.1f}s",
        })
        pbar.update(1)

        if r["unreachable_pct"] == 0.0:
            print(f"\nAll cells reachable at angle={angle}°. Stopping.")
            break
        angle += angle_step
    pbar.close()
    return results
