"""
01b_slope_units.py
──────────────────
Delineates slope units from the 10m DEM using WhiteboxTools.

Method: fill depressions → D8 flow direction → flow accumulation →
stream network (threshold = 5,000 cells = 0.5 km²) → sub-basins.
Each sub-basin is one slope unit bounded by a ridgeline and a stream link.

Output: data/processed/slope_units.gpkg

Usage
-----
    python scripts/01b_slope_units.py
"""

import sys
import tempfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import shapes as rio_shapes
from shapely.geometry import shape

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config, utils

logger = utils.get_logger(__name__)

MIN_AREA_M2    = 25_000
STREAM_THRESH  = 5_000    # upstream cells; at 10m: 0.5 km²


def main() -> None:
    utils.ensure_dirs()

    if not config.DEM_10M_TIF.exists():
        logger.error("DEM not found — run 01_data_prep.py first")
        sys.exit(1)

    import whitebox_workflows as wbw

    logger.info("=== Slope Unit Delineation ===")

    wbe = wbw.WbEnvironment()
    wbe.verbose = False

    logger.info("Reading DEM ...")
    dem = wbe.read_raster(str(config.DEM_10M_TIF))

    logger.info("Filling depressions ...")
    filled = wbe.fill_depressions_wang_and_liu(dem)

    logger.info("Computing D8 flow direction ...")
    d8 = wbe.d8_pointer(filled)

    logger.info("Computing flow accumulation ...")
    flow_acc = wbe.d8_flow_accum(d8, input_is_pointer=True, out_type="cells")

    logger.info("Extracting stream network (threshold = %d cells) ...", STREAM_THRESH)
    streams = wbe.extract_streams(flow_acc, threshold=float(STREAM_THRESH))

    logger.info("Delineating sub-basins ...")
    subbasins = wbe.subbasins(d8, streams)

    tmp_path = Path(tempfile.mktemp(suffix=".tif", dir=config.PROCESSED_DIR))
    wbe.write_raster(subbasins, str(tmp_path), compress=True)

    logger.info("Converting raster to polygons ...")
    with rasterio.open(tmp_path) as src:
        basin_arr = src.read(1).astype(np.int32)
        transform = src.transform
        crs = src.crs
        nodata_val = src.nodata

    mask = basin_arr > 0
    if nodata_val is not None:
        mask = mask & (basin_arr != int(nodata_val))

    polys, ids = [], []
    for geom, val in rio_shapes(basin_arr, mask=mask.astype(np.uint8), transform=transform):
        polys.append(shape(geom))
        ids.append(int(val))

    tmp_path.unlink(missing_ok=True)

    slope_units = gpd.GeoDataFrame({"basin_id": ids, "geometry": polys}, crs=crs)
    logger.info("  %d raw polygons", len(slope_units))

    slope_units = slope_units[slope_units.geometry.area >= MIN_AREA_M2].reset_index(drop=True)
    logger.info("  %d slope units after area filter (>= %d m²)", len(slope_units), MIN_AREA_M2)

    slope_units.to_file(str(config.SLOPE_UNITS_GPKG), driver="GPKG")
    logger.info("Slope units saved → %s", config.SLOPE_UNITS_GPKG)
    logger.info("=== Stage 1b complete ===")


if __name__ == "__main__":
    main()
