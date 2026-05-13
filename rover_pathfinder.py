from osgeo import gdal
import numpy as np
import matplotlib.pyplot as plt
import heapq
import csv
from random import randrange
from multiprocessing import Pool, cpu_count

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TIFF_FILE   = "Mars_HRSC_MOLA_BlendDEM_Global_200mp_v2_Test2.tif"
OUTPUT_CSV  = "Path_Data.csv"

PIXEL_SIZE     = 200.0   # meters per pixel
MAX_SLOPE      = 15.0    # degrees — hard limit: rover cannot traverse steeper terrain
NUM_ITERATIONS = 200     # number of random start/end pairs to compute

# ---------------------------------------------------------------------------
# DEM loading
# ---------------------------------------------------------------------------

def load_dem(file_path: str) -> np.ndarray:
    """Load a GeoTIFF DEM and return a float32 NumPy array."""
    dataset = gdal.Open(file_path)
    if dataset is None:
        raise FileNotFoundError(f"Could not open: {file_path}")
    elevation = dataset.ReadAsArray().astype(np.float32)
    if elevation.ndim == 3:
        # Multi-band: take the first band (or mean if needed)
        elevation = elevation[0]
    return elevation

# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_dem(elevation: np.ndarray, path: list[tuple] | None = None) -> None:
    """Plot the DEM as a topographical map, optionally overlaying a path."""
    plt.figure(figsize=(10, 8))
    plt.imshow(elevation, cmap="terrain", origin="upper")
    plt.colorbar(label="Elevation (m)")
    plt.title("Topographical Map")
    plt.xlabel("Column (pixels)")
    plt.ylabel("Row (pixels)")

    if path:
        rows = [p[0] for p in path]
        cols = [p[1] for p in path]
        plt.plot(cols, rows, color="red", linewidth=1.5, label="Rover path")
        plt.scatter([cols[0], cols[-1]], [rows[0], rows[-1]],
                    c=["lime", "red"], s=80, zorder=5,
                    label="Start / End")
        plt.legend()

    plt.tight_layout()

# ---------------------------------------------------------------------------
# A* pathfinding
# ---------------------------------------------------------------------------

# Pre-compute 8-direction vectors with pixel distances
_SQRT2 = np.sqrt(2.0)
_DIRECTIONS: list[tuple[int, int, float]] = [
    ( 1,  0, 1.0),    ( -1,  0, 1.0),
    ( 0,  1, 1.0),    (  0, -1, 1.0),
    ( 1,  1, _SQRT2), ( -1,  1, _SQRT2),
    ( 1, -1, _SQRT2), ( -1, -1, _SQRT2),
]


def astar(
    elevation: np.ndarray,
    start: tuple[int, int],
    end:   tuple[int, int],
    max_slope: float = MAX_SLOPE,
) -> list[tuple[int, int]] | None:
    """
    A* shortest path on a DEM grid.

    Goal: minimise total horizontal distance travelled.

    Step cost:
        h_dist — actual horizontal distance in meters (200 m straight, 283 m diagonal)

    Constraint:
        slope > max_slope → cell is impassable (hard block, not a penalty)

    Heuristic:
        Euclidean distance in meters — admissible because it never overestimates
        the true shortest traversable path.

    Returns an ordered list of (row, col) tuples, or None if unreachable.
    """
    rows, cols = elevation.shape

    def heuristic(a: tuple, b: tuple) -> float:
        return np.hypot(a[0] - b[0], a[1] - b[1]) * PIXEL_SIZE

    # g_score array — cost from start to each cell
    g = np.full((rows, cols), np.inf, dtype=np.float64)
    g[start[0], start[1]] = 0.0

    # Previous-node array for path reconstruction
    previous: list[list[tuple | None]] = [[None] * cols for _ in range(rows)]

    # Priority queue: (f_score, g_score, node)
    pq: list[tuple[float, float, tuple]] = [
        (heuristic(start, end), 0.0, start)
    ]

    while pq:
        f_curr, g_curr, curr = heapq.heappop(pq)

        # Goal reached — reconstruct path
        if curr == end:
            path: list[tuple] = []
            node: tuple | None = curr
            while node is not None:
                path.append(node)
                node = previous[node[0]][node[1]]
            return path[::-1]

        # Skip stale queue entries
        if g_curr > g[curr[0], curr[1]]:
            continue

        for dr, dc, pixel_dist in _DIRECTIONS:
            nr, nc = curr[0] + dr, curr[1] + dc

            if not (0 <= nr < rows and 0 <= nc < cols):
                continue

            h_dist   = pixel_dist * PIXEL_SIZE
            alt_diff = float(elevation[nr, nc]) - float(elevation[curr[0], curr[1]])
            slope    = float(np.degrees(np.arctan(np.abs(alt_diff) / h_dist)))

            if slope > max_slope:
                continue  # Too steep — hard constraint

            # Cost = actual ground distance only.
            # Slope is a hard block, not a penalty — this gives the true shortest path.
            step_cost = h_dist

            new_g = g_curr + step_cost
            if new_g < g[nr, nc]:
                g[nr, nc]       = new_g
                previous[nr][nc] = curr
                new_f = new_g + heuristic((nr, nc), end)
                heapq.heappush(pq, (new_f, new_g, (nr, nc)))

    return None  # No traversable path exists


def path_length_meters(path: list[tuple]) -> float:
    """Ground distance of a path in meters (Euclidean, pixel-scaled)."""
    arr = np.array(path, dtype=np.float64)
    diffs = np.diff(arr, axis=0)
    return float(np.sum(np.hypot(diffs[:, 0], diffs[:, 1])) * PIXEL_SIZE)

# ---------------------------------------------------------------------------
# Single iteration worker (used by multiprocessing)
# ---------------------------------------------------------------------------

def _run_iteration(args: tuple) -> tuple | None:
    elevation, idx, max_slope = args
    rows, cols = elevation.shape

    start = (randrange(1, rows - 1), randrange(1, cols - 1))
    end   = (randrange(1, rows - 1), randrange(1, cols - 1))

    path = astar(elevation, start, end, max_slope)

    if path is None:
        print(f"[{idx+1:>3}] No path found  {start} -> {end}")
        return None

    length = path_length_meters(path)
    print(f"[{idx+1:>3}] {start} -> {end}  |  {length:,.0f} m")
    return (idx + 1, start, end, length)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # 1. Load DEM
    print(f"Loading DEM: {TIFF_FILE}")
    elevation = load_dem(TIFF_FILE)
    print(f"DEM shape: {elevation.shape}  |  "
          f"min={elevation.min():.0f} m  max={elevation.max():.0f} m")

    # 2. Run pathfinding in parallel
    print(f"\nRunning {NUM_ITERATIONS} iterations on {cpu_count()} CPU cores …\n")
    worker_args = [
        (elevation, i, MAX_SLOPE)
        for i in range(NUM_ITERATIONS)
    ]

    with Pool(processes=cpu_count()) as pool:
        results = pool.map(_run_iteration, worker_args)

    # 4. Save results to CSV
    valid = [r for r in results if r is not None]
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Iteration", "Start Point", "End Point", "Path Length (m)"])
        writer.writerows(valid)

    print(f"\nSaved {len(valid)}/{NUM_ITERATIONS} paths to '{OUTPUT_CSV}'")

    # 5. Show the map with a corner-to-corner path (top-left -> bottom-right)
    rows_n, cols_n = elevation.shape
    corner_start = (10, 10)
    corner_end   = (rows_n - 10, cols_n - 10)
    print(f"\nPlotting corner-to-corner path: {corner_start} -> {corner_end}")
    example_path = astar(elevation, corner_start, corner_end, MAX_SLOPE)
    if example_path:
        length = path_length_meters(example_path)
        print(f"Path length: {length:,.0f} m")
        plot_dem(elevation, path=example_path)
    else:
        print("No path found between corners — plotting map without path")
        plot_dem(elevation)

    plt.show(block=True)
    print("Done")
