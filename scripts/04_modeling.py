"""
04_modeling.py
──────────────
Stage 4 of the landslide susceptibility pipeline.

Weighted Linear Combination (WLC) Model
    No training inventory required. Each normalised factor raster is multiplied
    by its literature-derived weight (see config.WLC_WEIGHTS) and summed to
    produce a continuous susceptibility index (0–1), then classified into 5
    classes using Jenks Natural Breaks.

Validation against the January 9, 2018 Montecito debris flow polygon is
performed if the shapefile is available.

Outputs
-------
    data/outputs/susceptibility_wlc_probability.tif
    data/outputs/susceptibility_wlc_classified.tif
    data/outputs/model_metrics.json
    data/outputs/montecito_validation.csv

Usage
-----
    python scripts/04_modeling.py
"""

import json
import sys
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config, utils

logger = utils.get_logger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)


# ── Weighted Linear Combination ───────────────────────────────────────────────

def run_wlc_model(
    factor_paths: list[Path],
    feature_names: list[str],
    weights: dict,
    profile: dict,
) -> tuple[np.ndarray, np.ndarray, list]:
    """Compute a weighted linear combination susceptibility index.

    Args:
        factor_paths: Ordered list of normalised factor raster paths.
        feature_names: Names matching the order of factor_paths.
        weights: Dict mapping feature name to weight (should sum to 1.0).
        profile: Rasterio profile for output rasters.

    Returns:
        Tuple of (wlc_prob array 0-1, wlc_classified array 1-5, jenks_breaks).
    """
    logger.info("Computing WLC susceptibility index …")
    logger.info("  Weights: %s", {k: round(v, 2) for k, v in weights.items()})

    wlc = None
    valid_count = None
    weight_used = 0.0
    for path, name in zip(factor_paths, feature_names):
        w = weights.get(name, 0.0)
        if w == 0.0 or not path.exists():
            continue
        arr, _ = utils.read_raster(path)
        arr = np.where(np.isfinite(arr), arr, np.nan)
        has_data = np.isfinite(arr).astype(np.uint8)
        layer = w * arr
        wlc = layer if wlc is None else np.nansum(np.stack([wlc, layer]), axis=0)
        valid_count = has_data if valid_count is None else valid_count + has_data
        weight_used += w

    if wlc is None:
        raise ValueError("No valid factor rasters found for WLC.")

    wlc = np.where(valid_count > 0, wlc, np.nan)

    if abs(weight_used - 1.0) > 0.01:
        logger.warning("  WLC weights sum to %.3f — normalising", weight_used)
        wlc = wlc / weight_used

    wlc = np.where(np.isfinite(wlc), np.clip(wlc, 0.0, 1.0), np.nan).astype(np.float32)

    if config.COUNTY_UTM_SHP.exists():
        import tempfile, os
        from rasterio.mask import mask as rio_mask
        from shapely.geometry import mapping as geo_mapping
        county = gpd.read_file(config.COUNTY_UTM_SHP)
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
            tmp_path = tmp.name
        utils.write_raster(wlc, profile.copy(), tmp_path)
        with rasterio.open(tmp_path) as src:
            geoms = [geo_mapping(g) for g in county.geometry]
            masked, _ = rio_mask(src, geoms, crop=False, nodata=np.nan, filled=True)
        wlc = masked[0].astype(np.float32)
        os.unlink(tmp_path)
        logger.info("  WLC masked to county boundary")

    if config.DEM_10M_TIF.exists():
        dem, _ = utils.read_raster(config.DEM_10M_TIF)
        land = (dem > 0.5) & np.isfinite(dem)
        wlc = np.where(land, wlc, np.nan)
        logger.info("  WLC masked to land (DEM > 0.5)")

    wlc_classified = utils.reclassify_fixed(wlc, config.WLC_BREAKS)
    logger.info("  WLC fixed breaks: %s", config.WLC_BREAKS)
    return wlc, wlc_classified, config.WLC_BREAKS


# ── Montecito Validation ──────────────────────────────────────────────────────

def validate_montecito(classified: np.ndarray, profile: dict) -> dict:
    """Assess WLC output within the 2018 Montecito debris flow extent.

    Args:
        classified: 5-class WLC susceptibility array.
        profile: Rasterio profile for the susceptibility raster.

    Returns:
        Dict mapping class labels to % area within the debris flow polygon.
    """
    debris_shp = config.PROCESSED_DIR / "montecito_debris_utm.shp"
    if not debris_shp.exists():
        logger.warning("Montecito debris flow shapefile not found — skipping validation")
        return {}

    logger.info("Running Montecito 2018 validation …")
    debris = gpd.read_file(debris_shp)
    from shapely.geometry import mapping as geo_mapping
    from rasterio.mask import mask as rio_mask
    import tempfile, os, rasterio as rio

    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
        tmp_path = tmp.name
    utils.write_raster(classified, profile.copy(), tmp_path)

    with rio.open(tmp_path) as src:
        geoms = [geo_mapping(g) for g in debris.geometry]
        masked, _ = rio_mask(src, geoms, crop=True, nodata=config.NODATA)
        clipped = masked[0].astype(np.float32)
        clipped[clipped == config.NODATA] = np.nan

    valid_cells = clipped[np.isfinite(clipped)]
    total = len(valid_cells)
    class_pct = {}
    for cls in range(1, 6):
        count = np.sum(valid_cells == cls)
        class_pct[config.SUSCEPTIBILITY_LABELS[cls]] = round(
            float(100.0 * count / total) if total > 0 else 0.0, 2
        )

    high_pct = class_pct.get("High", 0) + class_pct.get("Very High", 0)
    logger.info("  WLC — High+Very High within debris flow: %.1f%%  (target: majority)", high_pct)

    os.unlink(tmp_path)

    result = {"wlc": class_pct}
    df = pd.DataFrame(result)
    df.index.name = "susceptibility_class"
    df.to_csv(config.MONTECITO_VALIDATION_CSV)
    logger.info("Montecito validation saved → %s", config.MONTECITO_VALIDATION_CSV)
    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    """Run the WLC modeling stage."""
    utils.ensure_dirs()

    factor_paths = [
        config.NORM_SLOPE_TIF,
        config.NORM_CURVATURE_TIF,
        config.NORM_TWI_TIF,
        config.NORM_LITHOLOGY_TIF,
        config.NORM_LANDCOVER_TIF,
        config.NORM_FAULT_TIF,
        config.NORM_PRECIP_TIF,
        config.NORM_NDVI_TIF,
        config.NORM_SOIL_TIF,
        config.NORM_ROAD_TIF,
    ]
    feature_names = config.FEATURE_COLS

    existing = [p for p in factor_paths if p.exists()]
    if not existing:
        logger.error("No normalised factor rasters found — run 03_factor_layers.py first")
        sys.exit(1)

    with rasterio.open(existing[0]) as src:
        profile = src.profile.copy()

    metrics = {}

    logger.info("=== Weighted Linear Combination Model ===")
    wlc_prob, wlc_classified, wlc_breaks = run_wlc_model(
        factor_paths, feature_names, config.WLC_WEIGHTS, profile
    )
    utils.write_raster(wlc_prob, profile.copy(), config.SUSCEPTIBILITY_WLC_PROB_TIF)
    logger.info("WLC probability map → %s", config.SUSCEPTIBILITY_WLC_PROB_TIF)
    utils.write_raster(wlc_classified, profile.copy(), config.SUSCEPTIBILITY_WLC_TIF)
    logger.info("WLC classified map → %s", config.SUSCEPTIBILITY_WLC_TIF)
    metrics["wlc_jenks_breaks"] = wlc_breaks
    metrics["wlc_weights"] = config.WLC_WEIGHTS

    logger.info("=== Montecito Validation ===")
    val_results = validate_montecito(wlc_classified, profile)
    if val_results:
        metrics["montecito_validation"] = val_results

    with open(config.MODEL_METRICS_JSON, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    logger.info("Model metrics saved → %s", config.MODEL_METRICS_JSON)

    logger.info("=== Stage 4 complete ===")


if __name__ == "__main__":
    main()
