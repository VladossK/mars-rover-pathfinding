"""CSV writers + PNG plots."""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np


def write_angle_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def write_summary_csv(summary_rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not summary_rows:
        return
    fieldnames = list(summary_rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(summary_rows)


def plot_reachability(elev, dist_2d, valid_mask, start_rc, sample_path_rc, angle, out_path, pixel_size):
    """Two-panel PNG: topography + sample path | reachability heatmap (km)."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    ax = axes[0]
    elev_for_plot = np.where(valid_mask, elev, np.nan)
    vmin = float(np.nanmin(elev_for_plot))
    vmax = float(np.nanmax(elev_for_plot))
    im_elev = ax.imshow(elev_for_plot, cmap="terrain", origin="upper", vmin=vmin, vmax=vmax)
    cbar_e = fig.colorbar(im_elev, ax=ax, fraction=0.046, pad=0.04)
    cbar_e.set_label("Elevation (m)", fontsize=10)
    ax.set_title(f"Topography + sample path  (angle={angle}°)")
    ax.set_xlabel("Column"); ax.set_ylabel("Row")
    if sample_path_rc is not None and len(sample_path_rc) >= 2:
        rs = sample_path_rc[:, 0]
        cs = sample_path_rc[:, 1]
        ax.plot(cs, rs, color="red", linewidth=1.2, label="sample path (1 of 200)")
        ax.scatter(cs[0], rs[0], c="lime", s=70, zorder=6,
                    label=f"path start ({rs[0]},{cs[0]})")
        ax.scatter(cs[-1], rs[-1], c="magenta", s=70, zorder=6,
                    label=f"path end ({rs[-1]},{cs[-1]})")
        ax.legend(loc="upper right", fontsize=8)

    ax2 = axes[1]
    dist_km = dist_2d / 1000.0
    masked_invalid = np.ma.masked_where(~valid_mask, dist_km)
    cmap = plt.cm.plasma.copy()
    cmap.set_bad(color="black")
    masked_unreach = np.ma.masked_where(np.isinf(masked_invalid), masked_invalid)
    im = ax2.imshow(masked_unreach, cmap=cmap, origin="upper")
    fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04,
                  label="Cost-weighted distance from reach-start (km)")
    ax2.set_title(f"Reachability field from ({start_rc[0]},{start_rc[1]})  (angle={angle}°)")
    ax2.set_xlabel("Column"); ax2.set_ylabel("Row")
    ax2.scatter(start_rc[1], start_rc[0], c="lime", s=120, zorder=6,
                 marker="*", edgecolor="black", linewidth=1,
                 label=f"reach-map start ({start_rc[0]},{start_rc[1]})")
    ax2.legend(loc="upper right", fontsize=8)

    plt.suptitle(f"Mars Rover  |  pixel={pixel_size:.0f} m  |  max_slope={angle}°", y=1.01)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_backend_comparison(per_backend_summaries: dict[str, list[dict]], out_path: Path) -> None:
    """One PNG comparing multiple backends.

    per_backend_summaries: {"astar": [summary_rows...], "hybrid": [...], "hfm": [...]}
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    colors = {"astar": "steelblue", "hybrid": "seagreen", "hfm": "crimson"}

    ax = axes[0, 0]
    for name, rows in per_backend_summaries.items():
        if not rows: continue
        angles = [r["angle"] for r in rows]
        reach_pct = [100.0 * r["reachable_pct"] for r in rows]
        ax.plot(angles, reach_pct, "o-", color=colors.get(name, "black"), label=name)
    ax.set_xlabel("Max slope (deg)")
    ax.set_ylabel("Reachable %")
    ax.set_title("Reachability % by backend")
    ax.grid(True, alpha=0.3); ax.legend()

    ax = axes[0, 1]
    for name, rows in per_backend_summaries.items():
        if not rows: continue
        angles = [r["angle"] for r in rows]
        mean_len = [r["mean_path_len_m"] / 1000.0 for r in rows]
        ax.plot(angles, mean_len, "o-", color=colors.get(name, "black"), label=name)
    ax.set_xlabel("Max slope (deg)")
    ax.set_ylabel("Mean path length (km, 2D)")
    ax.set_title("Mean path length")
    ax.grid(True, alpha=0.3); ax.legend()

    ax = axes[1, 0]
    for name, rows in per_backend_summaries.items():
        if not rows: continue
        angles = [r["angle"] for r in rows]
        mean_t = [r["mean_time_ms"] for r in rows]
        ax.plot(angles, mean_t, "o-", color=colors.get(name, "black"), label=name)
    ax.set_xlabel("Max slope (deg)")
    ax.set_ylabel("Mean time per path (ms)")
    ax.set_title("Compute cost per path")
    ax.grid(True, alpha=0.3); ax.legend(); ax.set_yscale("log")

    ax = axes[1, 1]
    base = per_backend_summaries.get("astar", [])
    base_map = {r["angle"]: r for r in base}
    for name, rows in per_backend_summaries.items():
        if name == "astar" or not rows: continue
        angles = []
        ratios = []
        for r in rows:
            if r["angle"] in base_map and base_map[r["angle"]]["mean_path_len_m"] > 0:
                angles.append(r["angle"])
                ratios.append(r["mean_path_len_m"] / base_map[r["angle"]]["mean_path_len_m"])
        if angles:
            ax.plot(angles, ratios, "o-", color=colors.get(name, "black"),
                    label=f"{name}/astar")
    ax.axhline(1.0, color="gray", linestyle=":", alpha=0.7)
    ax.set_xlabel("Max slope (deg)")
    ax.set_ylabel("path_len ratio vs astar")
    ax.set_title("Length ratio (< 1.0 = shorter than astar baseline)")
    ax.grid(True, alpha=0.3); ax.legend()

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_unreachable_vs_angle(summary_rows: list[dict], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    angles = [r["angle"] for r in summary_rows]
    reach_pct = [100.0 * r["reachable_pct"] for r in summary_rows]
    unreach_pct = [100.0 * r["unreachable_pct"] for r in summary_rows]
    unreach_km2 = [r["area_unreachable_km2"] for r in summary_rows]
    fail = [r["n_fail"] for r in summary_rows]

    fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(16, 6))

    ax1.plot(angles, reach_pct, "o-", color="seagreen", label="Reachable %")
    ax1.plot(angles, unreach_pct, "o-", color="crimson", label="Unreachable %")
    ax1.set_xlabel("Max slope (deg)")
    ax1.set_ylabel("% of valid cells")
    ax1.set_ylim(0, 100)
    ax1.legend(loc="center right")
    ax1.grid(True, alpha=0.3)
    ax1.set_title("Reachability vs slope threshold")

    ax2 = ax1.twinx()
    ax2.plot(angles, fail, "s--", color="steelblue", alpha=0.6, label="Failed paths (of 200)")
    ax2.set_ylabel("Failed pair count", color="steelblue")
    ax2.tick_params(axis="y", labelcolor="steelblue")

    ax3.plot(angles, unreach_km2, "D-", color="darkred")
    ax3.set_xlabel("Max slope (deg)")
    ax3.set_ylabel("Unreachable area (km²)")
    ax3.set_title("Impassable area vs slope")
    ax3.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=130)
    plt.close(fig)
