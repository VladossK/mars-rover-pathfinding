from osgeo import gdal
import numpy as np
import matplotlib.pyplot as plt
import heapq

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TIFF_FILE  = "Mars_HRSC_MOLA_BlendDEM_Global_200mp_v2_Test2.tif"
PIXEL_SIZE = 200.0   # meters per pixel
MAX_SLOPE  = 15 # degrees — rover cannot climb steeper terrain

# Start point for the reachability map (row, col).
# (10, 10) = top-left corner. Change to any point you want.
START = (10, 10)

# End point — rover will extract the shortest path from START to END.
# Set to None to skip path drawing.
END = (1750, 1500)

# ---------------------------------------------------------------------------
# DEM loading
# ---------------------------------------------------------------------------

def load_dem(file_path: str) -> np.ndarray:
    dataset = gdal.Open(file_path)
    if dataset is None:
        raise FileNotFoundError(f"Could not open: {file_path}")
    elevation = dataset.ReadAsArray().astype(np.float32)
    if elevation.ndim == 3:
        elevation = elevation[0]
    return elevation

# ---------------------------------------------------------------------------
# Dijkstra — full reachability map from a single start point
# ---------------------------------------------------------------------------

_SQRT2 = np.sqrt(2.0)
_DIRECTIONS = [
    ( 1,  0, 1.0),     (-1,  0, 1.0),
    ( 0,  1, 1.0),     ( 0, -1, 1.0),
    ( 1,  1, _SQRT2),  (-1,  1, _SQRT2),
    ( 1, -1, _SQRT2),  (-1, -1, _SQRT2),
]

def dijkstra_full(
    elevation: np.ndarray,
    start: tuple[int, int],
    max_slope: float = MAX_SLOPE,
) -> tuple[np.ndarray, list[list]]:
    """
    Run Dijkstra from `start` over the entire DEM.

    Returns:
        dist     — 2D array of shortest distances in meters from start to every cell
                   (np.inf = unreachable)
        previous — 2D array of predecessor (row, col) tuples for path reconstruction
    """
    rows, cols = elevation.shape

    dist = np.full((rows, cols), np.inf, dtype=np.float64)
    dist[start[0], start[1]] = 0.0

    previous: list[list] = [[None] * cols for _ in range(rows)]

    # (distance, node)
    pq = [(0.0, start)]

    visited = 0
    total   = rows * cols

    while pq:
        d, curr = heapq.heappop(pq)

        if d > dist[curr[0], curr[1]]:
            continue  # stale entry

        visited += 1
        if visited % 500_000 == 0:
            print(f"  Progress: {visited:,} / {total:,} cells  "
                  f"({100*visited/total:.1f}%)  "
                  f"current dist = {d/1000:.0f} km")

        for dr, dc, pixel_dist in _DIRECTIONS:
            nr, nc = curr[0] + dr, curr[1] + dc

            if not (0 <= nr < rows and 0 <= nc < cols):
                continue

            h_dist   = pixel_dist * PIXEL_SIZE
            alt_diff = float(elevation[nr, nc]) - float(elevation[curr[0], curr[1]])
            slope    = float(np.degrees(np.arctan(np.abs(alt_diff) / h_dist)))

            if slope > max_slope:
                continue  # hard constraint

            new_d = d + h_dist
            if new_d < dist[nr, nc]:
                dist[nr, nc]     = new_d
                previous[nr][nc] = curr
                heapq.heappush(pq, (new_d, (nr, nc)))

    return dist, previous

# ---------------------------------------------------------------------------
# Path reconstruction
# ---------------------------------------------------------------------------

def reconstruct_path(
    previous: list[list],
    start: tuple[int, int],
    end: tuple[int, int],
) -> list[tuple[int, int]] | None:
    """Trace back from end to start using the predecessor array."""
    path = []
    node = end
    while node is not None:
        path.append(node)
        if node == start:
            break
        node = previous[node[0]][node[1]]
    else:
        return None  # start not reached → unreachable

    return path[::-1]


def path_length_meters(path: list[tuple]) -> float:
    arr = np.array(path, dtype=np.float64)
    diffs = np.diff(arr, axis=0)
    return float(np.sum(np.hypot(diffs[:, 0], diffs[:, 1])) * PIXEL_SIZE)

# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_reachability(
    elevation:  np.ndarray,
    dist:       np.ndarray,
    start:      tuple[int, int],
    end:        tuple[int, int] | None = None,
    path:       list[tuple] | None     = None,
) -> None:
    """
    Two-panel figure:
      Left  — topographical map with the rover path overlaid
      Right — reachability heatmap (distance from start in km)
    """
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # ── Left panel: topography + path ───────────────────────────────────────
    ax = axes[0]
    ax.imshow(elevation, cmap="terrain", origin="upper")
    ax.set_title("Topographical Map + Rover Path", fontsize=13)
    ax.set_xlabel("Column (pixels)")
    ax.set_ylabel("Row (pixels)")

    # Start marker
    ax.scatter(start[1], start[0], c="lime",   s=100, zorder=6, label="Start")

    if end is not None:
        reachable = np.isfinite(dist[end[0], end[1]])
        ax.scatter(end[1], end[0],
                   c="red" if reachable else "gray",
                   s=100, zorder=6,
                   label= "End" if reachable else "End (unreachable)")

    if path:
        pr = [p[0] for p in path]
        pc = [p[1] for p in path]
        ax.plot(pc, pr, color="red", linewidth=1.5, label="Shortest path")

    ax.legend(loc="upper right", fontsize=9)

    # ── Right panel: reachability heatmap ───────────────────────────────────
    ax2 = axes[1]

    dist_km = dist / 1000.0  # convert to km for readability

    # Mask unreachable cells (show as black)
    masked = np.ma.masked_where(np.isinf(dist_km), dist_km)
    cmap = plt.cm.plasma.copy()
    cmap.set_bad(color="black")

    im = ax2.imshow(masked, cmap=cmap, origin="upper")
    cbar = fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label("Distance from start (km)", fontsize=10)

    ax2.set_title(f"Reachability Map  (MAX_SLOPE={MAX_SLOPE}°)", fontsize=13)
    ax2.set_xlabel("Column (pixels)")
    ax2.set_ylabel("Row (pixels)")

    # Mark start and end on heatmap too
    ax2.scatter(start[1], start[0], c="lime", s=100, zorder=6, label="Start")
    if end is not None and np.isfinite(dist[end[0], end[1]]):
        ax2.scatter(end[1], end[0], c="red", s=100, zorder=6, label="End")
        ax2.legend(loc="upper right", fontsize=9)

    plt.suptitle(
        f"Mars Rover — Start {start}  |  "
        f"Pixel size {PIXEL_SIZE:.0f} m  |  Max slope {MAX_SLOPE}°",
        fontsize=12, y=1.01
    )
    plt.tight_layout()
    plt.show(block=True)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # 1. Load DEM
    print(f"Loading DEM: {TIFF_FILE}")
    elevation = load_dem(TIFF_FILE)
    rows, cols = elevation.shape
    print(f"Shape: {rows}×{cols}  |  "
          f"Elevation min={elevation.min():.0f}  max={elevation.max():.0f}")

    # 2. Full Dijkstra from START — fills the entire distance map
    print(f"\nRunning Dijkstra from {START} with MAX_SLOPE={MAX_SLOPE}° …")
    dist, previous = dijkstra_full(elevation, START)

    reachable = np.isfinite(dist).sum()
    print(f"\nReachable cells: {reachable:,} / {rows*cols:,} "
          f"({100*reachable/(rows*cols):.1f}%)")

    # 3. Extract shortest path to END (if defined)
    path = None
    if END is not None:
        if np.isfinite(dist[END[0], END[1]]):
            path = reconstruct_path(previous, START, END)
            if path:
                length_km = path_length_meters(path) / 1000
                print(f"Shortest path {START} → {END}: {length_km:.1f} km  "
                      f"({len(path)} waypoints)")
        else:
            print(f"END {END} is UNREACHABLE from {START} at MAX_SLOPE={MAX_SLOPE}°")

    # 4. Plot reachability map + path
    plot_reachability(elevation, dist, START, END, path)

    print("Done")
