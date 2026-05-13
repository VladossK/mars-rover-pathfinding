"""Run all three backends (astar / hybrid / hfm) on the same angles
and produce a side-by-side comparison plot + CSV.

Usage:
    python -m rover_sweep.compare --angles 5,10,15 --n-pairs 50 --out results_compare

Each backend writes its own subdirectory; a top-level comparison.png is rendered.
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

from . import config, reporting
from .dem_loader import load_dem
from .sweep import run_sweep


def main():
    p = argparse.ArgumentParser(description="Run astar | hybrid | hfm side-by-side.")
    p.add_argument("--tif", type=str, default=None)
    p.add_argument("--out", type=str, default="results_compare")
    p.add_argument("--n-pairs", type=int, default=50)
    p.add_argument("--angles", type=str, default="5,10,15",
                   help="Comma-separated list of angles (deg)")
    p.add_argument("--backends", type=str, default="astar,hybrid,hfm")
    p.add_argument("--hfm-subset-size", type=int, default=30)
    p.add_argument("--pixel-size", type=float, default=None)
    args = p.parse_args()

    tif = Path(args.tif) if args.tif else config.TIFF_FILE
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"Loading DEM: {tif}")
    elev, valid_mask, pixel_size = load_dem(tif)
    if args.pixel_size is not None:
        pixel_size = float(args.pixel_size)
    if pixel_size <= 0.0:
        pixel_size = float(config.PIXEL_SIZE)
    H, W = elev.shape
    print(f"  shape={H}x{W} pixel_size={pixel_size:.1f} m")

    angles = [int(a.strip()) for a in args.angles.split(",")]
    backends = [b.strip() for b in args.backends.split(",")]

    sorted_angles = sorted(set(angles))
    per_backend = {}
    for name in backends:
        backend_dir = out_root / name
        print(f"\n=== Backend: {name} -> {backend_dir} ===")
        rows_accum = []
        for ang in sorted_angles:
            angle_dir = backend_dir / f"_a{ang:02d}"
            run_sweep(
                elev=elev, valid_mask=valid_mask, pixel_size=pixel_size,
                out_dir=angle_dir,
                backend_name=name,
                n_pairs=args.n_pairs,
                start_angle=ang, max_angle=ang, angle_step=1,
                hfm_subset_size=args.hfm_subset_size,
            )
            sub = angle_dir / "summary.csv"
            if sub.exists():
                with open(sub, "r", encoding="utf-8") as f:
                    for row in csv.DictReader(f):
                        for k, v in list(row.items()):
                            if k == "backend":
                                continue
                            try:
                                row[k] = float(v)
                            except (ValueError, TypeError):
                                pass
                        row["angle"] = int(row["angle"])
                        rows_accum.append(row)
        if rows_accum:
            from . import reporting as _rep
            _rep.write_summary_csv(rows_accum, backend_dir / "summary.csv")
        per_backend[name] = rows_accum

    if per_backend:
        reporting.plot_backend_comparison(per_backend, out_root / "comparison.png")
        print(f"\nComparison plot: {out_root / 'comparison.png'}")

    print(f"\nDone. Backends: {list(per_backend.keys())}")


if __name__ == "__main__":
    main()
