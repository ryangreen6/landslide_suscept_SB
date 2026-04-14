"""
01_data_prep.py
───────────────
Stage 1 of the landslide susceptibility pipeline.

Responsibilities
----------------
- Mosaic all 1-metre 3DEP DEM tiles in data/raw/dem_tiles/ into a single VRT/TIF
- Reproject every raw input dataset to the analysis CRS (EPSG:26911)
- Resample the DEM to the target 10 m grid
- Clip all rasters and vectors to the Santa Barbara County boundary + 500 m buffer
- Write all processed outputs to data/processed/

All subsequent scripts read from data/processed/ and expect the outputs of this
script to be present before they run.

Usage
-----
    python scripts/01_data_prep.py
    python scripts/01_data_prep.py --res 10 --buffer 500
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.merge import merge as rio_merge
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.warp import transform_bounds
import fiona

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config, utils

logger = utils.get_logger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def mosaic_dem_tiles(tiles_dir: Path, output_path: Path) -> None:
    """Mosaic all GeoTIFF tiles in ``tiles_dir`` into a single raster.

    Args:
        tiles_dir: Directory containing 1-metre DEM tiles (``*.tif``).
        output_path: Path for the mosaicked output raster.

    Raises:
        FileNotFoundError: If no ``.tif`` files are found in ``tiles_dir``.
    """
    tiles = sorted(tiles_dir.glob("*.tif"))
    if not tiles:
        raise FileNotFoundError(f"No .tif files found in {tiles_dir}")

    logger.info("Mosaicking %d DEM tiles from %s …", len(tiles), tiles_dir)
    srcs = [rasterio.open(t) for t in tiles]
    mosaic, transform = rio_merge(srcs)
    profile = srcs[0].profile.copy()
    profile.update(
        driver="GTiff",
        height=mosaic.shape[1],
        width=mosaic.shape[2],
        transform=transform,
        compress="lzw",
        tiled=True,
        blockxsize=256,
        blockysize=256,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(mosaic)
    for s in srcs:
        s.close()
    logger.info("Mosaicked DEM saved → %s", output_path)


def prep_county_boundary(resolution: int, buffer_m: int) -> gpd.GeoDataFrame:
    """Load, reproject, and buffer the county boundary shapefile.

    Args:
        resolution: Target raster resolution in metres (used for logging).
        buffer_m: Buffer distance in metres for edge-effect mitigation.

    Returns:
        Buffered GeoDataFrame in the analysis CRS.

    Raises:
        FileNotFoundError: If the county boundary shapefile is not found.
    """
    shp = config.COUNTY_BOUNDARY_SHP
    if not shp.exists():
        raise FileNotFoundError(
            f"County boundary shapefile not found: {shp}\n"
            "Download from https://data.ca.gov/dataset/ca-geographic-boundaries"
        )
    logger.info("Loading county boundary from %s", shp)
    gdf = gpd.read_file(shp)
    gdf = gdf.to_crs(config.CRS_ANALYSIS)
    buffered = gdf.copy()
    buffered["geometry"] = gdf.geometry.buffer(buffer_m)
    logger.info("County boundary reprojected to %s, buffered by %d m", config.CRS_ANALYSIS, buffer_m)

    # Save the unbuffered UTM boundary for later use
    out_shp = config.COUNTY_UTM_SHP
    out_shp.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(out_shp)
    logger.info("UTM county boundary saved → %s", out_shp)

    return buffered


def prep_dem(mosaic_path: Path, county_buffered: gpd.GeoDataFrame,
             resolution: int) -> None:
    """Reproject, resample, and clip the mosaicked DEM.

    Workflow: raw mosaic → reproject to UTM → resample to ``resolution`` m
              → clip to buffered county boundary.

    Args:
        mosaic_path: Path to raw 1 m mosaicked DEM.
        county_buffered: Buffered county boundary GeoDataFrame (UTM CRS).
        resolution: Target pixel size in metres.
    """
    logger.info("Preparing DEM: reproject + resample to %d m …", resolution)

    tmp_reprojected = config.PROCESSED_DIR / "_dem_utm_tmp.tif"
    utils.reproject_raster(
        mosaic_path, tmp_reprojected,
        dst_crs=config.CRS_ANALYSIS,
        resampling=Resampling.bilinear,
        target_res=resolution,
    )

    logger.info("Clipping DEM to county boundary + buffer …")
    utils.clip_raster_to_shape(tmp_reprojected, county_buffered, config.DEM_10M_TIF)
    tmp_reprojected.unlink(missing_ok=True)

    logger.info("DEM (10 m, UTM) saved → %s", config.DEM_10M_TIF)


def prep_vector_clip_reproject(
    src_shp: Path,
    dst_shp: Path,
    county_buffered: gpd.GeoDataFrame,
    label: str,
) -> gpd.GeoDataFrame:
    """Reproject a vector shapefile to analysis CRS and clip to AOI.

    Args:
        src_shp: Input shapefile path.
        dst_shp: Output shapefile path.
        county_buffered: Buffered county boundary (UTM CRS) for clipping.
        label: Human-readable dataset name for logging.

    Returns:
        Clipped GeoDataFrame in the analysis CRS.  Returns empty GeoDataFrame
        if the source file does not exist (with a warning).
    """
    if not src_shp.exists():
        logger.warning("%s not found at %s — skipping", label, src_shp)
        return gpd.GeoDataFrame()

    logger.info("Processing %s …", label)
    gdf = gpd.read_file(src_shp)
    gdf = gdf.to_crs(config.CRS_ANALYSIS)
    aoi = county_buffered.union_all() if hasattr(county_buffered, "union_all") \
          else county_buffered.geometry.unary_union
    gdf = gdf[gdf.intersects(aoi)].copy()
    gdf = gpd.clip(gdf, county_buffered)
    dst_shp.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(dst_shp)
    logger.info("  %s: %d features clipped → %s", label, len(gdf), dst_shp)
    return gdf


def prep_raster_clip_reproject(
    src_path: Path,
    dst_path: Path,
    county_buffered: gpd.GeoDataFrame,
    label: str,
    resampling: Resampling = Resampling.bilinear,
) -> None:
    """Reproject a raster to analysis CRS, snap to 10 m grid, and clip to AOI.

    Args:
        src_path: Input raster path.
        dst_path: Output raster path.
        county_buffered: Buffered county boundary GeoDataFrame.
        label: Human-readable dataset name for logging.
        resampling: Rasterio resampling method.
    """
    if not src_path.exists():
        logger.warning("%s not found at %s — skipping", label, src_path)
        return

    logger.info("Processing raster %s …", label)
    tmp = config.PROCESSED_DIR / f"_tmp_{dst_path.stem}.tif"

    utils.reproject_raster(
        src_path, tmp,
        dst_crs=config.CRS_ANALYSIS,
        resampling=resampling,
        target_res=config.RESOLUTION,
    )
    utils.align_raster_to_reference(tmp, config.DEM_10M_TIF, dst_path, resampling=resampling)
    utils.clip_raster_to_shape(dst_path, county_buffered, dst_path)
    tmp.unlink(missing_ok=True)
    logger.info("  %s saved → %s", label, dst_path)


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="Stage 1: Data download preparation for SB landslide susceptibility"
    )
    parser.add_argument(
        "--res", type=int, default=config.RESOLUTION,
        help=f"Target raster resolution in metres (default: {config.RESOLUTION})"
    )
    parser.add_argument(
        "--buffer", type=int, default=config.BUFFER_M,
        help=f"County boundary buffer in metres (default: {config.BUFFER_M})"
    )
    parser.add_argument(
        "--skip-dem", action="store_true",
        help="Skip DEM mosaicking and reprojection (if already done)"
    )
    return parser.parse_args()


def main() -> None:
    """Run the data preparation stage."""
    args = parse_args()
    utils.ensure_dirs()

    # ── County boundary ───────────────────────────────────────────────────────
    county_buffered = prep_county_boundary(args.res, args.buffer)

    # ── DEM ───────────────────────────────────────────────────────────────────
    if not args.skip_dem:
        if not config.DEM_MOSAIC_TIF.exists():
            if config.DEM_TILES_DIR.exists():
                mosaic_dem_tiles(config.DEM_TILES_DIR, config.DEM_MOSAIC_TIF)
            else:
                logger.error(
                    "DEM tiles not found at %s\n"
                    "Download 1m 3DEP tiles from:\n"
                    "  https://apps.nationalmap.gov/downloader/\n"
                    "Place tiles in data/raw/dem_tiles/ and re-run.",
                    config.DEM_TILES_DIR,
                )
                sys.exit(1)
        prep_dem(config.DEM_MOSAIC_TIF, county_buffered, args.res)
    else:
        if not config.DEM_10M_TIF.exists():
            logger.error("--skip-dem specified but %s not found", config.DEM_10M_TIF)
            sys.exit(1)
        logger.info("Skipping DEM prep (--skip-dem flag set)")

    # ── Geology ───────────────────────────────────────────────────────────────
    prep_vector_clip_reproject(
        config.GEOLOGY_SHP,
        config.PROCESSED_DIR / "geology_utm.shp",
        county_buffered,
        "California geology",
    )

    # ── Landslide inventory ───────────────────────────────────────────────────
    prep_vector_clip_reproject(
        config.LANDSLIDE_INVENTORY_SHP,
        config.PROCESSED_DIR / "landslide_inventory_utm.shp",
        county_buffered,
        "Landslide inventory",
    )

    # ── Fault lines ───────────────────────────────────────────────────────────
    # Use 10 km buffer for fault lines to capture regional influence
    county_10km = county_buffered.copy()
    county_10km["geometry"] = county_10km.geometry.buffer(10_000 - args.buffer)
    prep_vector_clip_reproject(
        config.FAULT_LINES_SHP,
        config.PROCESSED_DIR / "faults_utm.shp",
        county_10km,
        "Quaternary faults",
    )

    # ── Fire perimeters ───────────────────────────────────────────────────────
    # Use 5 km extra buffer for fire perimeters
    county_5km = county_buffered.copy()
    county_5km["geometry"] = county_5km.geometry.buffer(5_000)
    prep_vector_clip_reproject(
        config.FIRE_PERIMETERS_SHP,
        config.PROCESSED_DIR / "fire_perimeters_utm.shp",
        county_5km,
        "CAL FIRE fire perimeters",
    )

    # ── NLCD land cover ───────────────────────────────────────────────────────
    prep_raster_clip_reproject(
        config.NLCD_TIF,
        config.PROCESSED_DIR / "nlcd_utm.tif",
        county_buffered,
        "NLCD 2021",
        resampling=Resampling.nearest,   # categorical data
    )

    # ── PRISM precipitation ───────────────────────────────────────────────────
    prep_raster_clip_reproject(
        config.PRISM_PRECIP_TIF,
        config.PROCESSED_DIR / "prism_precip_utm.tif",
        county_buffered,
        "PRISM 30-year precipitation",
        resampling=Resampling.bilinear,
    )

    # ── Montecito debris flow polygon (validation) ────────────────────────────
    if config.MONTECITO_DEBRIS_SHP.exists():
        prep_vector_clip_reproject(
            config.MONTECITO_DEBRIS_SHP,
            config.PROCESSED_DIR / "montecito_debris_utm.shp",
            county_buffered,
            "Montecito 2018 debris flow",
        )
    else:
        logger.warning(
            "Montecito debris flow shapefile not found at %s.\n"
            "  Download from CGS or USGS publications.\n"
            "  Validation step in 04_modeling.py will be skipped.",
            config.MONTECITO_DEBRIS_SHP,
        )

    logger.info("=== Stage 1 complete. All processed data in %s ===", config.PROCESSED_DIR)


if __name__ == "__main__":
    main()
