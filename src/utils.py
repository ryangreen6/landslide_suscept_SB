"""
utils.py
────────
Reusable geospatial utility functions shared across all pipeline scripts.

Covers:
  - Directory initialisation
  - Raster I/O (read / write)
  - Reprojection, clipping, resampling, alignment
  - Raster normalisation and reclassification
  - Vector-to-raster conversion
  - Logging setup
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Optional, Union

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.mask import mask as rasterio_mask
from rasterio.transform import from_bounds
from rasterio.warp import (
    calculate_default_transform,
    reproject as rasterio_reproject,
    transform_bounds,
)
import geopandas as gpd
from shapely.geometry import mapping

from src import config


# ── Logging ───────────────────────────────────────────────────────────────────

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a logger with a consistent console handler.

    Args:
        name: Logger name (typically ``__name__`` of the calling module).
        level: Logging level (default ``logging.INFO``).

    Returns:
        Configured :class:`logging.Logger` instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter(
            "%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


# ── Directory Helpers ─────────────────────────────────────────────────────────

def ensure_dirs() -> None:
    """Create all project output directories if they don't already exist."""
    for d in (
        config.RAW_DIR,
        config.PROCESSED_DIR,
        config.OUTPUTS_DIR,
        config.FIGURES_DIR,
    ):
        d.mkdir(parents=True, exist_ok=True)


# ── Raster I/O ────────────────────────────────────────────────────────────────

def read_raster(path: Union[str, Path]) -> tuple[np.ndarray, dict]:
    """Read the first band of a raster into a masked float32 array.

    NoData values are replaced with ``np.nan``.

    Args:
        path: Path to the GeoTIFF file.

    Returns:
        Tuple of (2-D float32 numpy array, rasterio profile dict).
    """
    path = Path(path)
    with rasterio.open(path) as src:
        profile = src.profile.copy()
        data = src.read(1, out_dtype=np.float32)
        nodata = src.nodata
        if nodata is not None:
            data = np.where(data == nodata, np.nan, data)
    return data, profile


def write_raster(
    arr: np.ndarray,
    profile: dict,
    path: Union[str, Path],
    nodata: float = config.NODATA,
) -> None:
    """Write a 2-D float32 array to a GeoTIFF.

    ``np.nan`` values are written as ``nodata``.

    Args:
        arr: 2-D numpy array to write.
        profile: Rasterio profile dict (crs, transform, etc.).
        path: Output file path.
        nodata: Value to use for missing data cells.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    out = np.where(np.isnan(arr), nodata, arr).astype(np.float32)
    profile.update(
        dtype=rasterio.float32,
        count=1,
        nodata=nodata,
        compress="lzw",
        tiled=True,
        blockxsize=256,
        blockysize=256,
    )
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(out, 1)


def get_reference_profile(path: Union[str, Path]) -> dict:
    """Return the rasterio profile for an existing raster (used as alignment target).

    Args:
        path: Path to reference raster.

    Returns:
        Rasterio profile dict.
    """
    with rasterio.open(path) as src:
        return src.profile.copy()


# ── Reprojection ──────────────────────────────────────────────────────────────

def reproject_raster(
    src_path: Union[str, Path],
    dst_path: Union[str, Path],
    dst_crs: str = config.CRS_ANALYSIS,
    resampling: Resampling = Resampling.bilinear,
    target_res: Optional[float] = None,
) -> None:
    """Reproject a raster to a new CRS (and optionally a target resolution).

    Args:
        src_path: Input raster path.
        dst_path: Output raster path.
        dst_crs: Target CRS string (e.g. ``"EPSG:26911"``).
        resampling: Rasterio resampling algorithm.
        target_res: If provided, override the default calculated resolution (metres).
    """
    src_path = Path(src_path)
    dst_path = Path(dst_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(src_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds
        )
        if target_res is not None:
            from rasterio.transform import from_origin
            bounds = transform_bounds(src.crs, dst_crs, *src.bounds)
            w, s, e, n = bounds
            width  = max(1, int((e - w) / target_res))
            height = max(1, int((n - s) / target_res))
            transform = from_bounds(w, s, e, n, width, height)

        # Choose a nodata value compatible with the source dtype
        import numpy as np
        dtype = src.dtypes[0]
        if np.dtype(dtype).kind in ('u', 'i'):
            # integer dtype: use 0 for unsigned, keep -9999 clipped for signed
            nodata_val = 0 if np.dtype(dtype).kind == 'u' else max(
                config.NODATA, np.iinfo(np.dtype(dtype)).min
            )
        else:
            nodata_val = config.NODATA

        profile = src.profile.copy()
        profile.update(
            crs=dst_crs,
            transform=transform,
            width=width,
            height=height,
            nodata=nodata_val,
            compress="lzw",
            tiled=True,
            blockxsize=256,
            blockysize=256,
        )

        with rasterio.open(dst_path, "w", **profile) as dst:
            for i in range(1, src.count + 1):
                rasterio_reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=resampling,
                )


# ── Clipping ──────────────────────────────────────────────────────────────────

def clip_raster_to_shape(
    src_path: Union[str, Path],
    shapes,                # GeoDataFrame or list of shapely geometries
    dst_path: Union[str, Path],
    all_touched: bool = False,
) -> None:
    """Clip a raster to the extent and mask of a vector geometry.

    Args:
        src_path: Input raster path.
        shapes: GeoDataFrame or list of Shapely geometries (same CRS as raster).
        dst_path: Output clipped raster path.
        all_touched: If True, include all cells touching the geometry.
    """
    src_path = Path(src_path)
    dst_path = Path(dst_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    if hasattr(shapes, "geometry"):
        geoms = [mapping(g) for g in shapes.geometry]
    elif hasattr(shapes, "__iter__"):
        geoms = [mapping(g) if hasattr(g, "__geo_interface__") else g for g in shapes]
    else:
        geoms = [mapping(shapes)]

    with rasterio.open(src_path) as src:
        out_arr, out_transform = rasterio_mask(
            src, geoms, crop=True, all_touched=all_touched, nodata=config.NODATA
        )
        profile = src.profile.copy()
        profile.update(
            transform=out_transform,
            width=out_arr.shape[2],
            height=out_arr.shape[1],
            nodata=config.NODATA,
            compress="lzw",
            tiled=True,
            blockxsize=256,
            blockysize=256,
        )
        with rasterio.open(dst_path, "w", **profile) as dst:
            dst.write(out_arr)


# ── Resampling / Alignment ────────────────────────────────────────────────────

def align_raster_to_reference(
    src_path: Union[str, Path],
    ref_path: Union[str, Path],
    dst_path: Union[str, Path],
    resampling: Resampling = Resampling.bilinear,
) -> None:
    """Resample and snap a raster to exactly match a reference grid.

    Reprojects, resamples, and snaps the source raster so it shares the same
    CRS, transform, width, and height as the reference raster.

    Args:
        src_path: Input raster path.
        ref_path: Reference raster whose grid defines the output.
        dst_path: Output aligned raster path.
        resampling: Rasterio resampling algorithm.
    """
    src_path = Path(src_path)
    ref_path = Path(ref_path)
    dst_path = Path(dst_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(ref_path) as ref:
        ref_profile = ref.profile.copy()

    with rasterio.open(src_path) as src:
        profile = ref_profile.copy()
        profile.update(
            count=src.count,
            dtype=rasterio.float32,
            nodata=config.NODATA,
            compress="lzw",
            tiled=True,
            blockxsize=256,
            blockysize=256,
        )
        with rasterio.open(dst_path, "w", **profile) as dst:
            for i in range(1, src.count + 1):
                rasterio_reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=ref_profile["transform"],
                    dst_crs=ref_profile["crs"],
                    resampling=resampling,
                )


# ── Raster Normalisation ──────────────────────────────────────────────────────

def normalize_to_01(
    arr: np.ndarray,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> np.ndarray:
    """Min-max normalise an array to the [0, 1] range.

    NaN cells are preserved.  If ``vmin``/``vmax`` are not provided, they are
    computed from the finite values of the array.

    Args:
        arr: Input 2-D numpy array (may contain NaN).
        vmin: Override minimum value for normalisation.
        vmax: Override maximum value for normalisation.

    Returns:
        Normalised float32 array in [0, 1] with NaN preserved.
    """
    finite = arr[np.isfinite(arr)]
    if vmin is None:
        vmin = float(finite.min()) if len(finite) > 0 else 0.0
    if vmax is None:
        vmax = float(finite.max()) if len(finite) > 0 else 1.0
    if vmax == vmin:
        return np.where(np.isfinite(arr), 0.5, np.nan).astype(np.float32)
    normed = (arr - vmin) / (vmax - vmin)
    return np.clip(normed, 0.0, 1.0).astype(np.float32)


def normalize_risk_score(arr: np.ndarray, max_score: float = 5.0) -> np.ndarray:
    """Normalise a risk-score raster (1–5 integer classes) to [0, 1].

    Args:
        arr: 2-D risk score array (values 1–max_score).
        max_score: Maximum possible risk score (default 5).

    Returns:
        Normalised float32 array where 1→0.0 and max_score→1.0.
    """
    return normalize_to_01(arr, vmin=1.0, vmax=max_score)


# ── Reclassification ──────────────────────────────────────────────────────────

def reclassify_by_breaks(
    arr: np.ndarray,
    breaks: list[tuple],
) -> np.ndarray:
    """Reclassify a continuous array using (min, max, value) break tuples.

    Args:
        arr: Input 2-D float array.
        breaks: List of ``(lower_bound, upper_bound_or_None, output_value)``
                tuples.  ``upper_bound=None`` means no upper bound.
                Lower bound is inclusive; upper bound is exclusive.

    Returns:
        Reclassified float32 array.  Cells not matching any break → NaN.
    """
    result = np.full(arr.shape, np.nan, dtype=np.float32)
    for low, high, val in breaks:
        if high is None:
            mask = arr >= low
        else:
            mask = (arr >= low) & (arr < high)
        result = np.where(mask & np.isfinite(arr), val, result)
    return result


def reclassify_fixed(
    arr: np.ndarray,
    breaks: list[float],
) -> np.ndarray:
    """Classify a continuous array using fixed break values.

    Args:
        arr: Input 2-D float array (may contain NaN/NODATA).
        breaks: Monotonically increasing list of N+1 boundary values defining N classes.

    Returns:
        Classified float32 array with values 1..N and NODATA where arr is invalid.
    """
    n_classes = len(breaks) - 1
    classified = np.full(arr.shape, config.NODATA, dtype=np.float32)
    for i in range(n_classes):
        low, high = breaks[i], breaks[i + 1]
        if i == 0:
            mask = (arr >= low) & (arr <= high)
        else:
            mask = (arr > low) & (arr <= high)
        classified = np.where(mask & np.isfinite(arr), float(i + 1), classified)
    return classified


# ── Rasterisation ─────────────────────────────────────────────────────────────

def rasterize_vector(
    gdf: gpd.GeoDataFrame,
    ref_path: Union[str, Path],
    burn_field: Optional[str] = None,
    burn_value: float = 1.0,
    fill: float = 0.0,
    all_touched: bool = False,
    dtype: str = "float32",
) -> np.ndarray:
    """Rasterize a GeoDataFrame onto a reference raster grid.

    Args:
        gdf: Input GeoDataFrame (must be in same CRS as reference raster).
        ref_path: Path to reference raster that defines the output grid.
        burn_field: Column name whose values are burned into the raster.
                    If None, ``burn_value`` is used for all features.
        burn_value: Constant burn value when ``burn_field`` is None.
        fill: Background value for cells not covered by any feature.
        all_touched: Burn all cells that the geometry touches.
        dtype: Output array dtype string.

    Returns:
        2-D numpy array rasterized onto the reference grid.
    """
    with rasterio.open(ref_path) as ref:
        transform = ref.transform
        out_shape = (ref.height, ref.width)

    if burn_field is not None:
        shapes_vals = [
            (geom, val) for geom, val in zip(gdf.geometry, gdf[burn_field])
        ]
    else:
        shapes_vals = [(geom, burn_value) for geom in gdf.geometry]

    arr = rasterize(
        shapes_vals,
        out_shape=out_shape,
        transform=transform,
        fill=fill,
        all_touched=all_touched,
        dtype=dtype,
    )
    return arr.astype(np.float32)


# ── Distance Raster ───────────────────────────────────────────────────────────

def euclidean_distance_raster(
    gdf: gpd.GeoDataFrame,
    ref_path: Union[str, Path],
    max_dist: Optional[float] = None,
) -> np.ndarray:
    """Compute a Euclidean distance-to-features raster.

    Args:
        gdf: GeoDataFrame of features to measure distance from.
        ref_path: Reference raster defining the output grid and CRS.
        max_dist: If provided, distances beyond this are capped at this value.

    Returns:
        2-D float32 array of distances in the raster's units (metres for UTM).
    """
    from scipy.ndimage import distance_transform_edt

    with rasterio.open(ref_path) as ref:
        pixel_size = ref.res[0]   # assume square pixels

    # Burn features to binary mask
    presence = rasterize_vector(gdf, ref_path, fill=0.0, all_touched=True)
    binary = (presence > 0).astype(bool)

    # distance_transform_edt gives distance in pixels; multiply by pixel size
    dist = distance_transform_edt(~binary).astype(np.float32) * pixel_size

    if max_dist is not None:
        dist = np.clip(dist, 0, max_dist)
    return dist


# ── Raster Stack Helpers ──────────────────────────────────────────────────────

def stack_rasters(paths: list[Union[str, Path]]) -> np.ndarray:
    """Read multiple aligned single-band rasters into a 3-D array (bands, H, W).

    All rasters must share the same grid (shape and transform).

    Args:
        paths: List of raster file paths.

    Returns:
        3-D float32 array of shape ``(n_bands, height, width)``.

    Raises:
        ValueError: If rasters do not share the same shape.
    """
    arrays = []
    ref_shape = None
    for p in paths:
        arr, _ = read_raster(p)
        if ref_shape is None:
            ref_shape = arr.shape
        elif arr.shape != ref_shape:
            raise ValueError(
                f"Shape mismatch: {p} has shape {arr.shape}, expected {ref_shape}"
            )
        arrays.append(arr)
    return np.stack(arrays, axis=0)


def raster_to_samples(
    stack: np.ndarray,
    labels: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    """Flatten a (bands, H, W) raster stack into a 2-D feature matrix.

    Rows with any NaN feature are dropped.

    Args:
        stack: 3-D array of shape ``(n_features, H, W)``.
        labels: Optional 2-D label array of shape ``(H, W)``.

    Returns:
        Tuple of:
          - ``X``: 2-D float32 array of shape ``(valid_pixels, n_features)``
          - ``y``: 1-D label array or None
          - ``valid_mask``: 2-D boolean mask of which pixels were valid
    """
    n_features, H, W = stack.shape
    X_flat = stack.reshape(n_features, -1).T         # (H*W, n_features)
    valid = np.all(np.isfinite(X_flat), axis=1)      # rows with no NaN

    X = X_flat[valid].astype(np.float32)
    valid_mask = valid.reshape(H, W)

    if labels is not None:
        y_flat = labels.flatten()
        y = y_flat[valid]
    else:
        y = None

    return X, y, valid_mask


def samples_to_raster(
    predictions: np.ndarray,
    valid_mask: np.ndarray,
    fill: float = np.nan,
) -> np.ndarray:
    """Map flat prediction array back onto a 2-D raster using a valid mask.

    Args:
        predictions: 1-D array of predicted values for valid pixels.
        valid_mask: 2-D boolean mask identifying valid pixel positions.
        fill: Value to use for invalid (NaN) pixels.

    Returns:
        2-D float32 array with predictions in place and ``fill`` elsewhere.
    """
    H, W = valid_mask.shape
    result = np.full(H * W, fill, dtype=np.float32)
    result[valid_mask.flatten()] = predictions
    return result.reshape(H, W)
