"""CLI entry point for the slope sweep."""
from __future__ import annotations
import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")


def main():
    p = argparse.ArgumentParser(description="Mars rover slope-sweep analysis.")
    p.add_argument("--tif", type=str, default=None, help="Path to GeoTIFF DEM.")
    p.add_argument("--out", type=str, default=None, help="Output directory (default: ./results).")
    p.add_argument("--n-pairs", type=int, default=None, help="Number of random pairs per angle.")
    p.add_argument("--start-angle", type=int, default=None)
    p.add_argument("--angle-step", type=int, default=None)
    p.add_argument("--max-angle", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--pixel-size", type=float, default=None,
                   help="Override pixel size (meters). Default: from GeoTransform or config.PIXEL_SIZE.")
    p.add_argument("--backend", choices=["astar", "hybrid", "hfm"], default=None,
                   help="Solver stack. astar = Phase 1 baseline. hybrid = scikit-fmm + Theta*. "
                        "hfm = anisotropic FMM (GPU when available, CPU fallback).")
    p.add_argument("--k-slope", type=float, default=None, help="Cost slope weight.")
    p.add_argument("--k-rough", type=float, default=None, help="Cost roughness weight.")
    p.add_argument("--hfm-subset-size", type=int, default=None,
                   help="Number of pairs evaluated by --backend hfm per angle.")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="Print one line per pair (start->end, time, success).")
    p.add_argument("--elev-offset", default="auto",
                   help="Elevation offset subtracted from DEM. "
                        "'auto' (default) detects UInt16=32768 / GDAL band offset. "
                        "'none' for raw values. Or a number, e.g. 32768.")
    args = p.parse_args()

    from . import config
    from .dem_loader import load_dem
    from .sweep import run_sweep

    tif = Path(args.tif) if args.tif else config.TIFF_FILE
    out_dir = Path(args.out) if args.out else config.RESULTS_DIR

    print(f"Loading DEM: {tif}")
    off_arg = args.elev_offset
    if off_arg == "none":
        off_arg = None
    elif off_arg == "auto":
        pass
    else:
        try:
            off_arg = float(off_arg)
        except ValueError:
            off_arg = "auto"
    elev, valid_mask, pixel_size = load_dem(tif, elev_offset=off_arg)
    if args.pixel_size is not None:
        pixel_size = float(args.pixel_size)
    if pixel_size <= 0.0:
        pixel_size = float(config.PIXEL_SIZE)
        print(f"  GeoTransform missing or invalid — falling back to config.PIXEL_SIZE={pixel_size:.1f} m")
    H, W = elev.shape
    n_valid = int(valid_mask.sum())
    print(f"  shape={H}x{W}  valid={n_valid:,}/{H*W:,}  pixel_size={pixel_size:.1f} m")
    e_min = float(elev[valid_mask].min())
    e_max = float(elev[valid_mask].max())
    print(f"  elev (m above areoid): min={e_min:.1f}  max={e_max:.1f}  relief={e_max-e_min:.1f}  "
          f"(offset={args.elev_offset})")

    results = run_sweep(
        elev=elev,
        valid_mask=valid_mask,
        pixel_size=pixel_size,
        out_dir=out_dir,
        backend_name=args.backend if args.backend is not None else config.DEFAULT_BACKEND,
        n_pairs=args.n_pairs if args.n_pairs is not None else config.NUM_PAIRS,
        start_angle=args.start_angle if args.start_angle is not None else config.START_ANGLE,
        angle_step=args.angle_step if args.angle_step is not None else config.ANGLE_STEP,
        max_angle=args.max_angle if args.max_angle is not None else config.MAX_ANGLE,
        pairs_seed=args.seed if args.seed is not None else config.PAIRS_SEED,
        k_slope=args.k_slope if args.k_slope is not None else config.COST_K_SLOPE,
        k_rough=args.k_rough if args.k_rough is not None else config.COST_K_ROUGH,
        hfm_subset_size=args.hfm_subset_size if args.hfm_subset_size is not None else config.HFM_SUBSET_SIZE,
        verbose=args.verbose,
    )

    print(f"\nDone. {len(results)} angles processed. Output: {out_dir}")


if __name__ == "__main__":
    main()
