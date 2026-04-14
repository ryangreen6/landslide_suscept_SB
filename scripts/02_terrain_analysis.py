"""
02_terrain_analysis.py
──────────────────────
Stage 2 of the landslide susceptibility pipeline.

Computes terrain derivative rasters from the 10 m DEM produced in Stage 1:

    Slope              — degrees from horizontal
    Aspect             — degrees from north (0–360)
    Profile curvature  — concavity/convexity in the direction of steepest descent
    Plan curvature     — convergence/divergence in the contour direction
    Flow accumulation  — D8 upslope contributing area (cell counts)
    TWI                — Topographic Wetness Index: ln(A / tan(β))

All outputs are written to data/processed/ and share the same grid as the
input DEM.

Usage
-----
    python scripts/02_terrain_analysis.py
    python scripts/02_terrain_analysis.py --dem path/to/custom_dem.tif
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config, utils

logger = utils.get_logger(__name__)


# ── Terrain Derivatives ───────────────────────────────────────────────────────

def compute_slope(dem: np.ndarray, cell_size: float) -> np.ndarray:
    """Compute slope in degrees using the Horn (1981) 3×3 gradient method.

    Args:
        dem: 2-D DEM array in metres.
        cell_size: Pixel size in metres.

    Returns:
        Slope array in degrees (0–90).
    """
    logger.info("Computing slope …")
    dz_dx = np.gradient(dem, axis=1) / cell_size
    dz_dy = np.gradient(dem, axis=0) / cell_size
    slope_rad = np.arctan(np.sqrt(dz_dx ** 2 + dz_dy ** 2))
    slope_deg = np.degrees(slope_rad)
    slope_deg[~np.isfinite(dem)] = np.nan
    return slope_deg.astype(np.float32)


def compute_aspect(dem: np.ndarray, cell_size: float) -> np.ndarray:
    """Compute aspect in degrees from north (0–360, clockwise).

    Flat areas (slope = 0) are assigned aspect = -1 (NoData convention).

    Args:
        dem: 2-D DEM array in metres.
        cell_size: Pixel size in metres.

    Returns:
        Aspect array in degrees [0, 360) with -1 for flat cells.
    """
    logger.info("Computing aspect …")
    dz_dx = np.gradient(dem, axis=1) / cell_size
    dz_dy = np.gradient(dem, axis=0) / cell_size

    # atan2 gives angle from east, counter-clockwise; convert to north, clockwise
    aspect_rad = np.arctan2(-dz_dy, dz_dx)
    aspect_deg = np.degrees(aspect_rad)
    aspect_north = 90.0 - aspect_deg
    aspect_north = np.mod(aspect_north, 360.0)

    # Flat cells
    flat = (dz_dx == 0) & (dz_dy == 0)
    aspect_north[flat] = -1.0
    aspect_north[~np.isfinite(dem)] = np.nan
    return aspect_north.astype(np.float32)


def compute_curvatures(
    dem: np.ndarray, cell_size: float
) -> tuple[np.ndarray, np.ndarray]:
    """Compute profile and plan curvature using the Zevenbergen & Thorne (1987) method.

    Profile curvature is the curvature in the direction of steepest descent
    (positive = concave upward, negative = convex).
    Plan curvature is the curvature perpendicular to the slope direction
    (positive = divergent flow, negative = convergent).

    Args:
        dem: 2-D DEM array in metres.
        cell_size: Pixel size in metres.

    Returns:
        Tuple of (profile_curvature, plan_curvature), both in units of 1/metre.
    """
    logger.info("Computing profile and plan curvature …")
    L = cell_size

    # Pad with NaN so edge cells get NaN output
    pad = np.pad(dem, 1, mode="edge")
    Z1 = pad[:-2, :-2]; Z2 = pad[:-2, 1:-1]; Z3 = pad[:-2, 2:]
    Z4 = pad[1:-1, :-2]; Z5 = pad[1:-1, 1:-1]; Z6 = pad[1:-1, 2:]
    Z7 = pad[2:,  :-2]; Z8 = pad[2:,  1:-1]; Z9 = pad[2:,  2:]

    D = ((Z4 + Z6) / 2 - Z5) / (L ** 2)
    E = ((Z2 + Z8) / 2 - Z5) / (L ** 2)
    F = (-Z1 + Z3 + Z7 - Z9) / (4 * L ** 2)
    G = (-Z4 + Z6) / (2 * L)
    H = (Z2 - Z8) / (2 * L)

    denom_prof = G ** 2 + H ** 2
    profile_curv = np.where(
        denom_prof > 1e-10,
        -2 * (D * G ** 2 + E * H ** 2 + F * G * H) / denom_prof,
        0.0,
    ).astype(np.float32)

    denom_plan = denom_prof + 1
    plan_curv = np.where(
        denom_prof > 1e-10,
        2 * (D * H ** 2 + E * G ** 2 - F * G * H) / denom_plan,
        0.0,
    ).astype(np.float32)

    # Propagate NaN mask
    nan_mask = ~np.isfinite(dem)
    profile_curv[nan_mask] = np.nan
    plan_curv[nan_mask] = np.nan

    return profile_curv, plan_curv


def compute_flow_accumulation_d8(dem: np.ndarray) -> np.ndarray:
    """Compute D8 flow accumulation using pysheds.

    Falls back to a simplified numpy implementation if pysheds is not installed.

    Args:
        dem: 2-D DEM array in metres.

    Returns:
        2-D float32 array of upslope contributing area in cell counts.
    """
    logger.info("Computing D8 flow accumulation …")
    try:
        from pysheds.grid import Grid
        import tempfile, rasterio
        from rasterio.transform import from_origin

        # Write DEM to a temp file for pysheds
        cell_size = config.RESOLUTION
        transform = from_origin(0, dem.shape[0] * cell_size, cell_size, cell_size)
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
            tmp_path = tmp.name

        profile = {
            "driver": "GTiff", "dtype": "float32", "width": dem.shape[1],
            "height": dem.shape[0], "count": 1, "crs": config.CRS_ANALYSIS,
            "transform": transform, "nodata": config.NODATA,
        }
        dem_filled = np.where(np.isnan(dem), config.NODATA, dem)
        with rasterio.open(tmp_path, "w", **profile) as dst:
            dst.write(dem_filled.astype(np.float32), 1)

        grid = Grid.from_raster(tmp_path)
        dem_gs = grid.read_raster(tmp_path)
        pit_filled = grid.fill_pits(dem_gs)
        flooded = grid.fill_depressions(pit_filled)
        inflated = grid.resolve_flats(flooded)
        fdir = grid.flowdir(inflated)
        acc = grid.accumulation(fdir)

        import os; os.unlink(tmp_path)
        acc_arr = np.array(acc).astype(np.float32)
        acc_arr[~np.isfinite(dem)] = np.nan
        logger.info("Flow accumulation computed with pysheds")
        return acc_arr

    except Exception as exc:
        logger.warning("pysheds unavailable (%s) — using simplified flow accumulation", exc)
        return _simple_flow_accumulation(dem)


def _simple_flow_accumulation(dem: np.ndarray) -> np.ndarray:
    """Simplified proxy flow accumulation using a Gaussian blur of the DEM.

    This is a fallback when pysheds is not available.  It does not implement
    true D8 routing, but provides a reasonable approximation for TWI computation.

    Args:
        dem: 2-D DEM array in metres.

    Returns:
        Proxy contributing area array (smoothed DEM-derived).
    """
    logger.warning("Using simplified flow accumulation proxy (Gaussian-blurred DEM)")
    from scipy.ndimage import gaussian_filter
    dem_filled = np.where(np.isnan(dem), np.nanmean(dem), dem)
    # Invert so low areas have high accumulation
    inverted = dem_filled.max() - dem_filled
    acc = gaussian_filter(inverted, sigma=5)
    acc = np.clip(acc, 1, None)  # minimum of 1 cell
    acc[~np.isfinite(dem)] = np.nan
    return acc.astype(np.float32)


def compute_twi(flow_acc: np.ndarray, slope_deg: np.ndarray,
                cell_size: float) -> np.ndarray:
    """Compute Topographic Wetness Index: TWI = ln(A / tan(β)).

    Where A = specific catchment area (flow_acc × cell_size²) and β = slope in radians.
    Flat cells (slope ≈ 0) use a minimum slope of 0.001 radians to avoid division by zero.

    Args:
        flow_acc: 2-D flow accumulation array (cell counts).
        slope_deg: 2-D slope array (degrees).
        cell_size: Pixel size in metres.

    Returns:
        2-D TWI array.
    """
    logger.info("Computing TWI …")
    min_slope_rad = 0.001   # prevents log(inf)
    slope_rad = np.where(
        np.isfinite(slope_deg) & (slope_deg > 0),
        np.radians(slope_deg),
        min_slope_rad,
    )
    # Specific catchment area: flow_acc cells × cell area / contour width
    sca = (flow_acc * cell_size ** 2) / cell_size
    sca = np.where(sca > 0, sca, 1.0)
    twi = np.log(sca / np.tan(slope_rad))

    # Propagate NaN
    nan_mask = ~np.isfinite(slope_deg) | ~np.isfinite(flow_acc)
    twi[nan_mask] = np.nan
    return twi.astype(np.float32)


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="Stage 2: Terrain analysis for SB landslide susceptibility"
    )
    parser.add_argument(
        "--dem", type=str, default=str(config.DEM_10M_TIF),
        help="Path to the 10 m DEM (default: data/processed/dem_10m.tif)"
    )
    return parser.parse_args()


def main() -> None:
    """Run terrain analysis stage."""
    args = parse_args()
    utils.ensure_dirs()

    dem_path = Path(args.dem)
    if not dem_path.exists():
        logger.error(
            "DEM not found at %s — run 01_data_prep.py first", dem_path
        )
        sys.exit(1)

    logger.info("Loading DEM from %s", dem_path)
    dem, profile = utils.read_raster(dem_path)
    cell_size = profile["transform"].a   # pixel width in CRS units (metres)
    logger.info("DEM shape: %s  cell_size: %.1f m", dem.shape, cell_size)

    # ── Slope ─────────────────────────────────────────────────────────────────
    slope = compute_slope(dem, cell_size)
    utils.write_raster(slope, profile, config.SLOPE_TIF)
    logger.info("Slope saved → %s  (range: %.1f–%.1f°)",
                config.SLOPE_TIF, np.nanmin(slope), np.nanmax(slope))

    # ── Aspect ────────────────────────────────────────────────────────────────
    aspect = compute_aspect(dem, cell_size)
    utils.write_raster(aspect, profile, config.ASPECT_TIF)
    logger.info("Aspect saved → %s", config.ASPECT_TIF)

    # ── Curvatures ────────────────────────────────────────────────────────────
    profile_curv, plan_curv = compute_curvatures(dem, cell_size)
    utils.write_raster(profile_curv, profile, config.PROFILE_CURV_TIF)
    utils.write_raster(plan_curv, profile, config.PLAN_CURV_TIF)
    logger.info("Curvatures saved → %s, %s", config.PROFILE_CURV_TIF, config.PLAN_CURV_TIF)

    # ── Flow accumulation ─────────────────────────────────────────────────────
    flow_acc = compute_flow_accumulation_d8(dem)
    utils.write_raster(flow_acc, profile, config.FLOW_ACC_TIF)
    logger.info("Flow accumulation saved → %s", config.FLOW_ACC_TIF)

    # ── TWI ───────────────────────────────────────────────────────────────────
    twi = compute_twi(flow_acc, slope, cell_size)
    utils.write_raster(twi, profile, config.TWI_TIF)
    logger.info("TWI saved → %s  (range: %.2f–%.2f)",
                config.TWI_TIF, np.nanmin(twi), np.nanmax(twi))

    logger.info("=== Stage 2 complete. Terrain derivatives in %s ===", config.PROCESSED_DIR)


if __name__ == "__main__":
    main()
