# Mars Rover Slope-Sweep Reachability

Pathfinding and reachability analysis on Mars DEMs across a range of slope-angle thresholds. Built for university research — three pluggable backends (`astar` / `hybrid` / `hfm`) on a shared cost surface, with reproducible RNG seeds, per-pair CSV output, and publication-ready plots.

---

## Quick start

```bash
# Default sweep (1° → 89°, stops when reachable% = 100)
python run_sweep.py

# One angle, fast baseline
python run_sweep.py --backend astar --start-angle 15 --max-angle 15

# Research-grade backend on a single angle
python run_sweep.py --backend hybrid --start-angle 12 --max-angle 12

# Compare all three backends on one or more angles
python -m rover_sweep.compare --angles 5,10,15 --n-pairs 50
```

---

## Install

Tested on Python 3.11 / Windows + WSL / 16 GB RAM.

```bash
conda install conda-forge::gdal
pip install numpy scipy numba matplotlib tqdm scikit-fmm agd pytest
```

Optional (GPU `hfm` backend):
```bash
pip install cupy-cuda12x   # or cupy-cuda11x depending on toolkit
```

---

## What it computes

For each slope angle θ ∈ [start, max] (default 1°…89°, step 1°):

1. **200 random (start, end) pairs** — same seed every angle, reproducible.
2. **Pathfinding** between every pair on terrain with slope > θ blocked.
3. **One full reachability map** from a fixed seed-chosen start point.
4. Stops early when reachable% reaches 100.

Outputs per angle:
- `paths.csv` — per-pair stats (length, time, slope along path, climb / descent).
- `reachability.png` — topography + sample path | reachability heatmap.

Outputs per sweep:
- `summary.csv` — one row per angle (reachable%, area km², means, fail counts).
- `unreachable_vs_angle.png` — reachable / unreachable curves + area km² curve.

---

## File structure

```
Rover/
├── Mars_HRSC_MOLA_BlendDEM_Global_200mp_v2_Test2.tif   # input DEM (2000×2000)
├── run_sweep.py                                         # CLI launcher
├── rover_sweep/                                         # main package
│   ├── config.py             # constants, RNG seeds, cost coefficients
│   ├── dem_loader.py         # GDAL loader + elevation offset detection
│   ├── cost_surface.py       # slope, TRI, NoData fill, 3D path utilities
│   ├── graph_builder.py      # vectorized edge mask + CSR construction (astar)
│   ├── pathfinder.py         # Numba A* + Theta* + custom binary heap
│   ├── reachability.py       # scipy Dijkstra + scikit-fmm wrapper
│   ├── backends.py           # Backend dispatcher: astar | hybrid | hfm
│   ├── sweep.py              # Angle loop + per-pair pipeline
│   ├── reporting.py          # CSV writers + matplotlib plots
│   ├── compare.py            # Multi-backend comparison runner
│   └── cli.py                # argparse entry
├── tests/                    # pytest suite (22 tests)
├── results/                  # output (default --out)
├── reachability_map.py       # original prototype (untouched, for reference)
└── rover_pathfinder.py       # original prototype (untouched)
```

---

## Backends

| Backend | Reachability | Pathfinding | Cost model | Speed | Accuracy |
|---|---|---|---|---|---|
| `astar` | scipy CSR Dijkstra, 8-conn | Numba A* 8-conn | `cost = h_dist`, slope hard-block | ★★★★★ (~170 ms/path) | baseline |
| `hybrid` | scikit-fmm 2nd-order eikonal | Numba Theta* (any-angle) | 3D dist × (1 + k·tan θ) × (1 + r·TRI) | ★★ (slow, 20–90 s/path on 4M grid) | research-grade |
| `hfm` | scikit-fmm + cost (CPU) / agd HFM (GPU) | FMM gradient backtrack / agd geodesic | same as `hybrid`, isotropic Riemannian | ★★★ GPU / ★ CPU | max accuracy |

GPU auto-detection in `hfm`: `cupy` + `agd.Eikonal` + CUDA test run. Falls back to CPU FMM with **red warning** if any check fails.

### Algorithm references

- A* — Hart, Nilsson, Raphael 1968.
- Dijkstra — Dijkstra 1959.
- Theta* (any-angle A*) — Daniel, Nash, Koenig, Felner, *JAIR* 2010.
- Fast Marching Method — Sethian, *PNAS* 1996.
- Anisotropic FMM — Mirebeau, *SIAM J. Numer. Anal.* 2014 / 2019.
- Terrain Ruggedness Index — Riley, DeGloria, Elliot 1999.
- Hiking cost model — Tobler 1993.

---

## CLI flags

```
python run_sweep.py [options]
```

| Flag | Default | Description |
|---|---|---|
| `--backend` | `astar` | `astar` / `hybrid` / `hfm` |
| `--tif PATH` | bundled file | Input GeoTIFF DEM |
| `--out DIR` | `results` | Output directory |
| `--start-angle N` | 1 | First angle (°) |
| `--max-angle N` | 89 | Last angle (hard cap; sweep stops earlier if reach% = 100) |
| `--angle-step N` | 1 | Step between angles |
| `--n-pairs N` | 200 | Random (start, end) pairs per angle |
| `--seed N` | 42 | RNG seed for pairs (reproducible) |
| `--pixel-size M` | auto | Override pixel size in meters |
| `--elev-offset V` | `auto` | Elevation offset (`auto` / `none` / number). UInt16 DEM → 32768 |
| `--k-slope F` | 1.0 | Slope penalty coefficient (cost surface) |
| `--k-rough F` | 0.0 | Roughness (TRI) coefficient — disabled by default on Mars terrain |
| `--hfm-subset-size N` | 30 | Number of pairs used by `hfm` per angle (HFM is expensive) |
| `--verbose` `-v` | off | Print one line per pair |

### Examples

```bash
# Adaptive sweep, default backend (stops when reach=100%)
python run_sweep.py

# Single angle, hybrid backend, 50 pairs
python run_sweep.py --backend hybrid --start-angle 10 --max-angle 10 --n-pairs 50

# Custom angle range, step 2°
python run_sweep.py --start-angle 3 --max-angle 21 --angle-step 2

# Enable TRI roughness penalty for ablation study
python run_sweep.py --backend hybrid --start-angle 12 --max-angle 12 --k-rough 0.5

# HFM at 15°, on GPU machine (auto-detects CUDA)
python run_sweep.py --backend hfm --start-angle 15 --max-angle 15

# HFM subset of 20 pairs (laptop CPU mode)
python run_sweep.py --backend hfm --start-angle 15 --max-angle 15 --hfm-subset-size 20

# Multi-backend comparison
python -m rover_sweep.compare --angles 5,10,15 --n-pairs 50 --out results_compare
```

---

## CSV schema

### Per-angle `paths.csv`

| Column | Meaning |
|---|---|
| `iteration` | Pair index (1…N) |
| `backend` | `astar` / `hybrid` / `hfm` |
| `start_row`, `start_col` | Start cell in DEM grid |
| `end_row`, `end_col` | End cell in DEM grid |
| `success` | 1 = path found, 0 = failure |
| `vec_2d_m` | Straight-line ground distance (meters) |
| `vec_3d_m` | Straight-line distance including Δz |
| `path_len_m` | 2D length of actual path |
| `path_len_3d_m` | 3D length along terrain |
| `path_max_slope_deg` | Max per-edge slope along path |
| `path_mean_slope_deg` | Mean per-edge slope |
| `climb_m`, `descent_m` | Total ascending / descending vertical |
| `time_ms` | Compute time for this pair |
| `detour_factor` | `path_len_m / vec_2d_m` |
| `fail_reason` | `""` / `NO_PATH` / `DETOUR_RUNAWAY` |

### `summary.csv`

| Column | Meaning |
|---|---|
| `angle` | Slope threshold (°) |
| `backend` | Backend used |
| `n_success`, `n_fail` | Pair-success counts |
| `mean_vec_2d_m`, `mean_vec_3d_m` | Mean straight-line distances |
| `mean_path_len_m`, `mean_path_len_3d_m` | Mean path lengths |
| `mean_path_max_slope_deg`, `mean_path_mean_slope_deg` | Per-path slope stats |
| `mean_climb_m`, `mean_descent_m` | Mean vertical work |
| `mean_time_ms` | Mean compute time per path |
| `reachable_pixels`, `unreachable_pixels` | From reachability map |
| `reachable_pct`, `unreachable_pct` | Fractions of valid cells |
| `nodata_pct` | Fraction of NoData cells (constant across sweep) |
| `area_total_km2`, `area_reachable_km2`, `area_unreachable_km2` | Areas |
| `full_reach_time_s` | Time to compute the reachability map |

---

## Cost surface (used by `hybrid` and `hfm`)

```
cost[r, c] = (1 + k_slope · tan(slope_rad)) · (1 + k_rough · TRI_norm)
```

- `slope_rad` — max 8-neighbor slope at cell (radians).
- `TRI_norm` — Terrain Ruggedness Index normalized by mean over valid cells.
- Invalid cells (NoData / slope > θ) → cost = +∞.

Defaults (`config.py`): `k_slope = 1.0`, `k_rough = 0.0`. TRI disabled by default because on rough Mars terrain it pushes Theta* into long detours.

For a paper ablation: run with `--k-rough 0.5` and compare against `--k-rough 0.0`.

---

## Reproducibility

- All RNG paths derived from two integer seeds in `config.py`:
  - `PAIRS_SEED = 42` — controls the 200 (start, end) pairs (same across all angles, all backends).
  - `REACH_SEED = 7777` — controls the single fixed reachability map start point.
- `config.get_backend_versions()` records library versions for embedding in paper appendix.
- Same seed → identical CSV rows.

---

## Logging

The runner prints staged progress so you always know which step is running:

```
[init] Pre-warming Numba kernels ...
[init]   warmup done in 4.2s
[init] Building shared cost surfaces (slope, TRI, cost map) ...
[init]   cost surface built in 0.8s  (slope.max=51.0°  TRI.max=184.4m)
Generating 200 random pairs (seed=42) ...
Fixed reachability start: (1000, 976)

[angle=12] backend=hybrid prepare ...
[angle=12]   prepare done in 0.3s
[angle=12] reachability from (1000, 976) ...
[angle=12]   reachability done in 2.7s
[angle=12]   reach%=68.438  unreach=1,262,470 px = 50498.8 km^2
[angle=12] pathfinding 200 pairs ...
  paths[hybrid,12d]:  47%|██████  | 94/200 [12:30<14:00, t=8400ms, ok=68, fail=26, mean_ms=7800]
```

`-v` / `--verbose` prints one line per pair.

---

## Tests

```bash
python -m pytest tests/ -x
```

22 tests covering:
- Cost surface (slope, TRI, NoData fill, 3D path length, slope stats).
- Graph builder + scipy Dijkstra.
- A* pathfinder (path correctness, wall avoidance, generation counter heap).
- Theta* (path ≤ A* length, LoS check, obstacle avoidance, buffer reuse).
- FMM reachability (analytical parity, cost scaling).

---

## Performance budget

Per-angle wall time on 4 M-cell DEM (2000×2000), 8-core CPU laptop:

| Stage | astar | hybrid | hfm (CPU) | hfm (CUDA) |
|---|---|---|---|---|
| Cost surfaces build | once, ~3 s | once, ~3 s | once, ~3 s | once, ~3 s |
| Edge mask + CSR | 1–2 s | — | — | — |
| Reachability (1×) | 3–15 s | 2–5 s | 8–20 s | 0.5–1 s |
| 200 paths | 30–90 s | 30–250 min | 50–150 min | 4–8 min |
| Plot + CSV | ~3 s | ~3 s | ~3 s | ~3 s |
| **Per angle** | **~45–120 s** | **30 min – 4 h** | **50–150 min** | **5–10 min** |

`hfm` on laptop CPU is for spot checks (subset of 20–30 pairs). For full 200-pair sweeps at `hfm`, run on a CUDA machine.

---

## Known limitations

1. **`hybrid` Theta* on dense cost surfaces** — LoS check is per-relax. On 4 M-cell grids over high-relief terrain, paths take 20–90 s/path. ~40–50 % of pairs hit `DETOUR_RUNAWAY` (capped at 5× straight-line) when `k_rough` > 0. Workaround: keep `--k-rough 0.0`, or implement Lazy Theta*.
2. **Reachability semantics differ between backends**:
   - `astar` uses edge-based slope check (cell passable if at least one neighbor edge ≤ θ).
   - `hybrid` / `hfm` use cell-based slope check (cell passable only if max-neighbor slope ≤ θ).
   - Cell-based is stricter — `hybrid`/`hfm` typically report lower reach%. Both are valid; document in Methods.
3. **GeoTransform** may be missing in cropped TIFFs — falls back to `config.PIXEL_SIZE = 200.0`.
4. **agd CPU mode** requires precompiled HFM binaries (`FileHFM_binary_dir.txt`). Without them, only CUDA path works. Without CUDA, `hfm` backend falls back to scikit-fmm + gradient backtracking — still 2nd-order accurate, but not Riemannian.

---

## Citation

If this code or its results are used in a publication, cite:

- Sethian, J. A. (1996). A fast marching level set method for monotonically advancing fronts. *PNAS*, 93(4), 1591–1595.
- Daniel, K., Nash, A., Koenig, S., Felner, A. (2010). Theta*: Any-angle path planning on grids. *JAIR*, 39, 533–579.
- Mirebeau, J.-M. (2019). Riemannian fast-marching on Cartesian grids using Voronoi's first reduction. *SIAM J. Numer. Anal.*, 57(6), 2608–2655.
- Riley, S. J., DeGloria, S. D., Elliot, R. (1999). A terrain ruggedness index that quantifies topographic heterogeneity. *Intermountain Journal of Sciences*, 5(1–4), 23–27.
- Tobler, W. (1993). Three Presentations on Geographical Analysis and Modeling. NCGIA Technical Report 93-1.

---

## License

Research code, no warranty. Use at your own risk.
