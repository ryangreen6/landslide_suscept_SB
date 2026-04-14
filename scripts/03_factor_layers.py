"""
03_factor_layers.py
───────────────────
Stage 3 of the landslide susceptibility pipeline.

Prepares all non-terrain factor rasters and normalises every factor (including
terrain layers from Stage 2) to a common [0, 1] scale for input to the models.

Factor layers produced
----------------------
1. Lithology risk      — USGS geology rasterized and reclassified (1–5)
2. Land cover risk     — NLCD 2021 reclassified (1–5)
3. Fault distance risk — Euclidean distance to faults, reclassified (1–5)
4. Precipitation       — PRISM 30-year normal, normalized (0–1)
5. Fire history risk   — CAL FIRE perimeters age-weighted (1–5)

Plus normalised versions of terrain layers (slope, combined curvature, TWI)
from Stage 2.

All outputs written to data/processed/ as float32 GeoTIFFs.

Usage
-----
    python scripts/03_factor_layers.py
    python scripts/03_factor_layers.py --ref-date 2018-01-09
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.enums import Resampling

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config, utils

logger = utils.get_logger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _check_required(paths: list[Path], label: str) -> bool:
    """Log an error if any required path does not exist.

    Args:
        paths: List of paths that must exist.
        label: Human-readable dataset name for the error message.

    Returns:
        True if all paths exist; False otherwise.
    """
    missing = [p for p in paths if not p.exists()]
    if missing:
        logger.error("%s: required file(s) not found: %s", label, missing)
        return False
    return True


# ── Factor: Lithology ─────────────────────────────────────────────────────────

def build_lithology_risk(ref_path: Path) -> None:
    """Rasterize geology polygons and reclassify to a risk score (1–5).

    Unit names are matched against ``config.LITHOLOGY_RISK_KEYWORDS`` using
    case-insensitive substring matching.  Unmatched units receive
    ``config.LITHOLOGY_DEFAULT_RISK``.

    Args:
        ref_path: Reference raster for grid definition.
    """
    geo_shp = config.PROCESSED_DIR / "geology_utm.shp"
    if not _check_required([geo_shp], "Geology shapefile"):
        logger.warning("Lithology layer skipped — using default risk score 3 everywhere")
        dem, profile = utils.read_raster(ref_path)
        default = np.where(np.isfinite(dem), float(config.LITHOLOGY_DEFAULT_RISK), np.nan)
        utils.write_raster(default.astype(np.float32), profile, config.LITHOLOGY_RISK_TIF)
        return

    logger.info("Building lithology risk layer …")
    gdf = gpd.read_file(geo_shp)

    # Identify the rock-type field (different USGS products use different names)
    name_cols = [c for c in gdf.columns if any(
        k in c.lower() for k in ["rocktype", "unit_name", "lith", "rock", "unitname", "generalize"]
    )]
    name_col = name_cols[0] if name_cols else None
    logger.info("  Using geology field: %s", name_col)

    # Assign risk scores
    scores = []
    for _, row in gdf.iterrows():
        unit = str(row.get(name_col, "")).lower() if name_col else ""
        score = config.LITHOLOGY_DEFAULT_RISK
        # Check from highest to lowest so more specific matches win
        for risk, keywords in sorted(config.LITHOLOGY_RISK_KEYWORDS.items(), reverse=True):
            if any(kw in unit for kw in keywords):
                score = risk
                break
        scores.append(score)

    gdf = gdf.copy()
    gdf["risk_score"] = scores

    risk_arr = utils.rasterize_vector(
        gdf, ref_path, burn_field="risk_score",
        fill=config.LITHOLOGY_DEFAULT_RISK, all_touched=False,
    )
    # Mask outside county
    dem, profile = utils.read_raster(ref_path)
    risk_arr = np.where(np.isfinite(dem), risk_arr, np.nan)
    utils.write_raster(risk_arr, profile, config.LITHOLOGY_RISK_TIF)
    logger.info("  Lithology risk saved → %s", config.LITHOLOGY_RISK_TIF)


# ── Factor: Land Cover ────────────────────────────────────────────────────────

def build_landcover_risk(ref_path: Path) -> None:
    """Reclassify NLCD 2021 land cover codes to a risk score (1–5).

    Codes not in ``config.NLCD_RISK`` receive the median risk score (3).
    Codes mapped to ``None`` are treated as NoData (excluded from analysis).

    Args:
        ref_path: Reference raster for alignment verification.
    """
    nlcd_path = config.PROCESSED_DIR / "nlcd_utm.tif"
    if not _check_required([nlcd_path], "NLCD"):
        logger.warning("Land cover layer skipped — using default risk score 3")
        dem, profile = utils.read_raster(ref_path)
        default = np.where(np.isfinite(dem), 3.0, np.nan)
        utils.write_raster(default.astype(np.float32), profile, config.LANDCOVER_RISK_TIF)
        return

    logger.info("Building land cover risk layer …")
    nlcd_arr, profile = utils.read_raster(nlcd_path)
    risk_arr = np.full(nlcd_arr.shape, np.nan, dtype=np.float32)

    for code, score in config.NLCD_RISK.items():
        if score is None:
            continue  # NoData classes stay NaN
        mask = np.isclose(nlcd_arr, code)
        risk_arr[mask] = float(score)

    # Unknown codes → default risk 3
    unknown = np.isfinite(nlcd_arr) & np.isnan(risk_arr)
    risk_arr[unknown] = 3.0

    utils.write_raster(risk_arr, profile, config.LANDCOVER_RISK_TIF)
    logger.info("  Land cover risk saved → %s", config.LANDCOVER_RISK_TIF)


# ── Factor: Fault Distance ────────────────────────────────────────────────────

def build_fault_distance_risk(ref_path: Path) -> None:
    """Compute Euclidean distance to faults and reclassify to risk scores.

    Distance breaks from ``config.FAULT_DISTANCE_BREAKS``:
        0–100 m → 5, 100–500 m → 4, 500–1000 m → 3,
        1000–2000 m → 2, >2000 m → 1.

    Args:
        ref_path: Reference raster for grid definition.
    """
    faults_shp = config.PROCESSED_DIR / "faults_utm.shp"
    if not _check_required([faults_shp], "Fault lines"):
        logger.warning("Fault distance layer skipped — assigning risk 1 everywhere")
        dem, profile = utils.read_raster(ref_path)
        default = np.where(np.isfinite(dem), 1.0, np.nan)
        utils.write_raster(default.astype(np.float32), profile, config.FAULT_DIST_RISK_TIF)
        return

    logger.info("Building fault distance risk layer …")
    faults = gpd.read_file(faults_shp)
    dist_arr = utils.euclidean_distance_raster(faults, ref_path)

    risk_arr = utils.reclassify_by_breaks(dist_arr, config.FAULT_DISTANCE_BREAKS)
    dem, profile = utils.read_raster(ref_path)
    risk_arr = np.where(np.isfinite(dem), risk_arr, np.nan)

    utils.write_raster(risk_arr, profile, config.FAULT_DIST_RISK_TIF)
    logger.info("  Fault distance risk saved → %s", config.FAULT_DIST_RISK_TIF)


# ── Factor: Precipitation ─────────────────────────────────────────────────────

def build_precipitation_layer(ref_path: Path) -> None:
    """Normalise PRISM 30-year mean annual precipitation to [0, 1].

    Higher precipitation → higher value → higher susceptibility contribution.

    Args:
        ref_path: Reference raster for alignment.
    """
    prism_path = config.PROCESSED_DIR / "prism_precip_utm.tif"
    if not _check_required([prism_path], "PRISM precipitation"):
        logger.warning("Precipitation layer skipped — uniform 0.5 assigned")
        dem, profile = utils.read_raster(ref_path)
        default = np.where(np.isfinite(dem), 0.5, np.nan)
        utils.write_raster(default.astype(np.float32), profile, config.PRECIP_NORM_TIF)
        return

    logger.info("Building precipitation layer …")
    precip, profile = utils.read_raster(prism_path)
    normed = utils.normalize_to_01(precip)
    utils.write_raster(normed, profile, config.PRECIP_NORM_TIF)
    logger.info("  Precipitation (normalised) saved → %s", config.PRECIP_NORM_TIF)


# ── Factor: Fire History ──────────────────────────────────────────────────────

def build_fire_history_risk(ref_path: Path, reference_date: datetime) -> None:
    """Build a post-fire vulnerability raster from CAL FIRE perimeters.

    Each burned area is scored based on years elapsed since the fire relative
    to ``reference_date`` (Jan 9, 2018 — Montecito debris flow):

        < 1 yr  → 5  (vegetation gone, hydrophobic soil)
        1–3 yr  → 4  (partial recovery)
        3–5 yr  → 3  (moderate recovery)
        5–10 yr → 2  (substantial recovery)
        > 10 yr → 1  (baseline)

    Where multiple fires overlap, the highest risk score is retained.
    The Thomas Fire (2017) is explicitly logged when found in the dataset.

    Args:
        ref_path: Reference raster for grid definition.
        reference_date: Date used to compute years since each fire.
    """
    fire_shp = config.PROCESSED_DIR / "fire_perimeters_utm.shp"
    dem, profile = utils.read_raster(ref_path)

    if not fire_shp.exists():
        logger.warning("Fire perimeters not found — assigning unburned risk (1) everywhere")
        default = np.where(np.isfinite(dem), float(config.FIRE_UNBURNED_RISK), np.nan)
        utils.write_raster(default.astype(np.float32), profile, config.FIRE_RISK_TIF)
        return

    logger.info("Building fire history risk layer (reference date: %s) …",
                reference_date.strftime("%Y-%m-%d"))
    fires = gpd.read_file(fire_shp)

    # Start with unburned baseline risk everywhere
    risk_canvas = np.where(np.isfinite(dem), float(config.FIRE_UNBURNED_RISK), np.nan)

    # Identify date column(s)
    date_cols = [c for c in fires.columns if any(
        k in c.lower() for k in ["year_", "alarm_date", "cont_date", "fire_year", "year"]
    )]
    year_col = date_cols[0] if date_cols else None
    logger.info("  Fire date column: %s", year_col)

    thomas_found = False
    processed = 0

    for _, row in fires.iterrows():
        # Determine fire year
        fire_year = None
        if year_col:
            try:
                val = str(row[year_col])
                fire_year = int(val[:4]) if len(val) >= 4 else None
            except (ValueError, TypeError):
                pass

        if fire_year is None:
            continue

        years_ago = (reference_date - datetime(fire_year, 7, 1)).days / 365.25

        # Determine risk score
        risk_score = config.FIRE_UNBURNED_RISK
        for low, high, score in config.FIRE_RISK_BREAKS:
            if high is None:
                if years_ago >= low:
                    risk_score = score
                    break
            elif low <= years_ago < high:
                risk_score = score
                break

        # Log Thomas Fire specifically
        fire_name = str(row.get("FIRE_NAME", row.get("fire_name", ""))).upper()
        if config.THOMAS_FIRE_NAME in fire_name:
            logger.info(
                "  THOMAS FIRE identified: year=%d, years_ago=%.2f, risk_score=%d",
                fire_year, years_ago, risk_score,
            )
            thomas_found = True

        # Burn this fire's risk onto the canvas (take maximum)
        fire_gdf = gpd.GeoDataFrame([row], crs=fires.crs)
        burned = utils.rasterize_vector(fire_gdf, ref_path, fill=0.0, all_touched=False)
        risk_canvas = np.where(
            (burned > 0) & np.isfinite(risk_canvas),
            np.maximum(risk_canvas, float(risk_score)),
            risk_canvas,
        )
        processed += 1

    if not thomas_found:
        logger.warning(
            "Thomas Fire not found in the dataset. Ensure the CAL FIRE perimeter "
            "shapefile covers 2017 and contains a fire named 'THOMAS'."
        )

    logger.info("  Processed %d fire perimeters", processed)
    utils.write_raster(risk_canvas.astype(np.float32), profile, config.FIRE_RISK_TIF)
    logger.info("  Fire history risk saved → %s", config.FIRE_RISK_TIF)


# ── Normalise All Layers ──────────────────────────────────────────────────────

def normalise_all_layers() -> None:
    """Normalise every factor layer to [0, 1] and write as norm_*.tif files.

    Risk-score layers (1–5) are normalised with vmin=1, vmax=5.
    Continuous layers (slope, curvature, TWI, precip) are normalised by their
    actual data range.
    """
    logger.info("Normalising all factor layers to [0, 1] …")

    # (source_tif, output_tif, use_risk_scale)
    layers = [
        (config.SLOPE_TIF,          config.NORM_SLOPE_TIF,     False),
        (config.LITHOLOGY_RISK_TIF, config.NORM_LITHOLOGY_TIF, True),
        (config.LANDCOVER_RISK_TIF, config.NORM_LANDCOVER_TIF, True),
        (config.FAULT_DIST_RISK_TIF, config.NORM_FAULT_TIF,    True),
        (config.PRECIP_NORM_TIF,    config.NORM_PRECIP_TIF,    False),  # already 0-1
        (config.FIRE_RISK_TIF,      config.NORM_FIRE_TIF,      True),
        (config.TWI_TIF,            config.NORM_TWI_TIF,       False),
    ]

    for src_path, dst_path, is_risk in layers:
        if not src_path.exists():
            logger.warning("  Skipping %s — source not found", src_path.name)
            continue
        arr, profile = utils.read_raster(src_path)
        if is_risk:
            normed = utils.normalize_risk_score(arr, max_score=5.0)
        else:
            normed = utils.normalize_to_01(arr)
        utils.write_raster(normed, profile, dst_path)
        logger.info("  %s → %s", src_path.name, dst_path.name)

    # Combined curvature: average of normalised profile and plan curvature
    if config.PROFILE_CURV_TIF.exists() and config.PLAN_CURV_TIF.exists():
        prof, profile = utils.read_raster(config.PROFILE_CURV_TIF)
        plan, _       = utils.read_raster(config.PLAN_CURV_TIF)
        # Flip sign: positive profile curvature (concave) = higher susceptibility
        combined = (-prof + -plan) / 2.0   # concave hollows → high values
        normed = utils.normalize_to_01(combined)
        utils.write_raster(normed, profile, config.NORM_CURVATURE_TIF)
        logger.info("  Combined curvature normalised → %s", config.NORM_CURVATURE_TIF)

    logger.info("Normalisation complete.")


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="Stage 3: Factor layers for SB landslide susceptibility"
    )
    parser.add_argument(
        "--ref-date", type=str, default=config.FIRE_REFERENCE_DATE,
        help=f"Reference date for fire age calculation (YYYY-MM-DD, default: {config.FIRE_REFERENCE_DATE})"
    )
    parser.add_argument(
        "--ref-raster", type=str, default=str(config.DEM_10M_TIF),
        help="Reference raster for grid alignment (default: DEM 10m)"
    )
    return parser.parse_args()


def main() -> None:
    """Run factor layer preparation stage."""
    args = parse_args()
    utils.ensure_dirs()

    ref_path = Path(args.ref_raster)
    if not ref_path.exists():
        logger.error("Reference raster not found: %s — run 01_data_prep.py first", ref_path)
        sys.exit(1)

    reference_date = datetime.strptime(args.ref_date, "%Y-%m-%d")

    build_lithology_risk(ref_path)
    build_landcover_risk(ref_path)
    build_fault_distance_risk(ref_path)
    build_precipitation_layer(ref_path)
    build_fire_history_risk(ref_path, reference_date)
    normalise_all_layers()

    # Verify all normalised layers exist
    expected = [
        config.NORM_SLOPE_TIF, config.NORM_CURVATURE_TIF, config.NORM_TWI_TIF,
        config.NORM_LITHOLOGY_TIF, config.NORM_LANDCOVER_TIF,
        config.NORM_FAULT_TIF, config.NORM_PRECIP_TIF, config.NORM_FIRE_TIF,
    ]
    missing = [p for p in expected if not p.exists()]
    if missing:
        logger.warning("Some normalised layers are missing: %s", [p.name for p in missing])
    else:
        logger.info("All %d normalised factor layers ready.", len(expected))

    logger.info("=== Stage 3 complete. Factor layers in %s ===", config.PROCESSED_DIR)


if __name__ == "__main__":
    main()
