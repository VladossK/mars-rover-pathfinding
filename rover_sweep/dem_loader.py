"""GDAL DEM loader. Returns elevation + validity mask."""
from __future__ import annotations
import numpy as np
from osgeo import gdal


def detect_elev_offset(dtype_name: str, gdal_offset: float | None) -> float:
    """Pick a sensible elevation offset.

    GDAL band offset wins if present.
    Otherwise: Mars HRSC/MOLA Blend DEM stored as UInt16 is signed int16 + 32768
    (standard unsigned encoding). Subtract 32768 -> real meters above areoid.
    """
    if gdal_offset is not None:
        return float(gdal_offset)
    if dtype_name == "UInt16":
        return 32768.0
    return 0.0


def load_dem(
    path: str | "Path",
    elev_offset: float | str | None = "auto",
) -> tuple[np.ndarray, np.ndarray, float]:
    """Load a GeoTIFF DEM.

    Args:
        elev_offset:
            "auto"  -> detect from GDAL metadata or DataType (default).
            float   -> subtract this value from every cell.
            None    -> no shift (raw DEM values).

    Returns:
        elev:        float32[H, W] elevation in meters above Mars areoid (after offset)
        valid_mask:  bool[H, W]    True where the cell is valid
        pixel_size:  float         meters per pixel
    """
    ds = gdal.Open(str(path))
    if ds is None:
        raise FileNotFoundError(f"Could not open: {path}")

    band = ds.GetRasterBand(1)
    nodata = band.GetNoDataValue()
    gdal_offset = band.GetOffset()
    dtype_name = gdal.GetDataTypeName(band.DataType)

    arr = ds.ReadAsArray()
    if arr.ndim == 3:
        arr = arr[0]
    elev = arr.astype(np.float32, copy=False)

    valid_mask = np.isfinite(elev)
    if nodata is not None:
        valid_mask &= elev != np.float32(nodata)

    if elev_offset == "auto":
        off = detect_elev_offset(dtype_name, gdal_offset)
    elif elev_offset is None:
        off = 0.0
    else:
        off = float(elev_offset)

    if off != 0.0:
        elev = (elev - np.float32(off))

    elev = np.where(valid_mask, elev, np.float32(0.0))

    gt = ds.GetGeoTransform()
    pixel_size = float(abs(gt[1])) if gt is not None else 0.0
    if pixel_size < 10.0 or pixel_size > 1e6:
        pixel_size = 0.0

    return elev, valid_mask, pixel_size
