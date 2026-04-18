"""
03_factor_layers.py
───────────────────
Stage 3 of the landslide susceptibility pipeline.

Prepares all non-terrain factor rasters and normalises every factor (including
terrain layers from Stage 2) to a common [0, 1] scale for input to WLC.

Factor layers produced
----------------------
1. Lithology risk      — USGS geology rasterized and reclassified (1–5)
2. Land cover risk     — GAP/LANDFIRE 2011 from Planetary Computer (1–5)
3. Fault distance risk — Distance to faults, slip-rate/sense enhanced (1–5)
4. Precipitation       — PRISM 30-year normal, normalized (0–1)
5. NDVI                — Sentinel-2 L2A median composite, inverted (0–1)
6. Soil erodibility    — SSURGO K factor + hydgrp via SDA API (1–5)

All outputs written to data/processed/ as float32 GeoTIFFs.

Usage
-----
    python scripts/03_factor_layers.py
"""

import sys
from io import StringIO
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config, utils

logger = utils.get_logger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _check_required(paths: list[Path], label: str) -> bool:
    missing = [p for p in paths if not p.exists()]
    if missing:
        logger.error("%s: required file(s) not found: %s", label, missing)
        return False
    return True


# ── Factor: Lithology ─────────────────────────────────────────────────────────

def build_lithology_risk(ref_path: Path) -> None:
    geo_shp = config.PROCESSED_DIR / "geology_utm.shp"
    if not _check_required([geo_shp], "Geology shapefile"):
        logger.warning("Lithology layer skipped — using default risk score 3 everywhere")
        dem, profile = utils.read_raster(ref_path)
        default = np.where(np.isfinite(dem), float(config.LITHOLOGY_DEFAULT_RISK), np.nan)
        utils.write_raster(default.astype(np.float32), profile, config.LITHOLOGY_RISK_TIF)
        return

    logger.info("Building lithology risk layer …")
    gdf = gpd.read_file(geo_shp)

    name_cols = [c for c in gdf.columns if any(
        k in c.lower() for k in ["rocktype", "unit_name", "lith", "rock", "unitname", "generalize"]
    )]
    name_col = name_cols[0] if name_cols else None
    logger.info("  Using SGMC geology field: %s", name_col)

    mac_cache = {}
    mac_cache_path = config.PROCESSED_DIR / "geology_macrostrat_cache.json"
    if mac_cache_path.exists():
        import json as _json
        with open(mac_cache_path) as _f:
            mac_cache = _json.load(_f)
        logger.info("  Macrostrat cache loaded (%d entries)", len(mac_cache))

    def _score_unit(text: str) -> int:
        text = text.lower()
        for risk, keywords in sorted(config.LITHOLOGY_RISK_KEYWORDS.items(), reverse=True):
            if any(kw in text for kw in keywords):
                return risk
        return config.LITHOLOGY_DEFAULT_RISK

    scores = []
    mac_used = 0
    for idx, (_, row) in enumerate(gdf.iterrows()):
        sgmc_unit = str(row.get(name_col, "")).lower() if name_col else ""
        info = mac_cache.get(str(idx))
        mac_name = (info or {}).get("name", "")
        mac_mismatch = (
            mac_name.lower() == "landslide deposit"
            and "landslide" not in sgmc_unit
        )
        if info and mac_name and not mac_mismatch:
            unit = (mac_name + " " + (info.get("lith") or "")).lower()
            mac_used += 1
        else:
            unit = sgmc_unit
        scores.append(_score_unit(unit))

    logger.info("  Macrostrat used for %d/%d polygons; SGMC fallback for %d",
                mac_used, len(gdf), len(gdf) - mac_used)

    gdf = gdf.copy()
    gdf["risk_score"] = scores

    risk_arr = utils.rasterize_vector(
        gdf, ref_path, burn_field="risk_score",
        fill=config.LITHOLOGY_DEFAULT_RISK, all_touched=False,
    )
    dem, profile = utils.read_raster(ref_path)
    risk_arr = np.where(np.isfinite(dem), risk_arr, np.nan)
    utils.write_raster(risk_arr, profile, config.LITHOLOGY_RISK_TIF)
    logger.info("  Lithology risk saved → %s", config.LITHOLOGY_RISK_TIF)


# ── Factor: Land Cover ────────────────────────────────────────────────────────

def build_landcover_risk(ref_path: Path) -> None:
    import rioxarray
    import pystac_client
    import planetary_computer
    from rasterio.warp import reproject as _reproject
    from rasterio.transform import from_bounds as _from_bounds
    import rasterio as _rio

    logger.info("Building land cover risk layer …")

    if not config.GAP_TIF.exists():
        logger.info("  Fetching GAP/LANDFIRE land cover from Planetary Computer …")
        sb_bbox = [-120.706, 34.26, -119.005, 35.127]
        catalog = pystac_client.Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1",
            modifier=planetary_computer.sign_inplace,
        )
        items = catalog.search(collections=["gap"], bbox=sb_bbox).item_collection()
        lc = rioxarray.open_rasterio(items[0].assets["data"].href).squeeze()

        import geopandas as _gpd
        from shapely.geometry import box as _box
        sb = _gpd.GeoDataFrame(
            {"geometry": [_box(*sb_bbox)]}, crs="EPSG:4326"
        ).to_crs(lc.rio.crs)
        lc_clip = lc.rio.clip_box(*sb.total_bounds).rio.clip(sb.geometry)
        lc_clip.rio.write_nodata(0, inplace=True)

        with _rio.open(ref_path) as ref:
            dst_crs = ref.crs
            dst_transform = ref.transform
            dst_height = ref.height
            dst_width = ref.width

        src_arr = lc_clip.values.astype(np.float32)
        dst_arr = np.zeros((dst_height, dst_width), dtype=np.float32)

        src_h, src_w = src_arr.shape
        west, south, east, north = lc_clip.rio.bounds()
        src_transform = _from_bounds(west, south, east, north, src_w, src_h)
        src_crs = lc_clip.rio.crs

        _reproject(
            source=src_arr,
            destination=dst_arr,
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.nearest,
            src_nodata=0,
            dst_nodata=0,
        )

        with _rio.open(ref_path) as ref:
            out_profile = ref.profile.copy()
        out_profile.update(dtype="float32", nodata=0)
        config.GAP_TIF.parent.mkdir(parents=True, exist_ok=True)
        with _rio.open(config.GAP_TIF, "w", **out_profile) as dst:
            dst.write(dst_arr, 1)
        logger.info("  GAP raster saved → %s", config.GAP_TIF)
    else:
        logger.info("  Using cached GAP raster → %s", config.GAP_TIF)

    gap_arr, profile = utils.read_raster(config.GAP_TIF)
    risk_arr = np.full(gap_arr.shape, np.nan, dtype=np.float32)

    for code, score in config.GAP_RISK.items():
        if score is None:
            continue
        risk_arr[np.isclose(gap_arr, code)] = float(score)

    risk_arr[np.isfinite(gap_arr) & (gap_arr > 0) & np.isnan(risk_arr)] = 3.0

    utils.write_raster(risk_arr, profile, config.LANDCOVER_RISK_TIF)
    logger.info("  Land cover risk saved → %s", config.LANDCOVER_RISK_TIF)


# ── Factor: Fault Distance (slip-rate/sense enhanced) ─────────────────────────

def build_fault_distance_risk(ref_path: Path) -> None:
    from scipy.ndimage import distance_transform_edt

    faults_shp = config.PROCESSED_DIR / "faults_utm.shp"
    if not _check_required([faults_shp], "Fault lines"):
        logger.warning("Fault distance layer skipped — assigning risk 1 everywhere")
        dem, profile = utils.read_raster(ref_path)
        default = np.where(np.isfinite(dem), 1.0, np.nan)
        utils.write_raster(default.astype(np.float32), profile, config.FAULT_DIST_RISK_TIF)
        return

    logger.info("Building fault distance risk layer (slip-enhanced) …")
    faults = gpd.read_file(faults_shp)

    dist_arr = utils.euclidean_distance_raster(faults, ref_path)
    risk_arr = utils.reclassify_by_breaks(dist_arr, config.FAULT_DISTANCE_BREAKS)

    def _rate_mult(val):
        v = str(val).lower() if pd.notna(val) else ""
        if "greater than 5" in v: return 1.30
        if "between 1.0 and 5" in v: return 1.15
        if "less than 0.2" in v: return 0.85
        return 1.00

    def _sense_mult(val):
        v = str(val).lower() if pd.notna(val) else ""
        if "reverse" in v or "thrust" in v: return 1.15
        return 1.00

    faults = faults.copy()
    has_rate = "slip_rate" in faults.columns
    has_sense = "slip_sense" in faults.columns

    faults["_mult"] = [
        _rate_mult(row.get("slip_rate") if has_rate else None) *
        _sense_mult(row.get("slip_sense") if has_sense else None)
        for _, row in faults.iterrows()
    ]

    faults_sorted = faults.sort_values("_mult", ascending=True)
    mult_raster = utils.rasterize_vector(
        faults_sorted, ref_path, burn_field="_mult", fill=-1.0, all_touched=True,
    )

    no_fault = mult_raster < 0
    _, idx = distance_transform_edt(no_fault, return_indices=True)
    mult_arr = mult_raster[idx[0], idx[1]]
    mult_arr = np.where(no_fault, mult_arr, mult_raster)
    mult_arr = np.where(mult_arr < 0, 1.0, mult_arr)

    risk_arr = np.clip(risk_arr * mult_arr, 1.0, 5.0)
    dem, profile = utils.read_raster(ref_path)
    risk_arr = np.where(np.isfinite(dem), risk_arr, np.nan)

    utils.write_raster(risk_arr.astype(np.float32), profile, config.FAULT_DIST_RISK_TIF)
    logger.info("  Fault distance risk saved → %s", config.FAULT_DIST_RISK_TIF)


# ── Factor: Precipitation (NOAA Atlas 14 — 100-yr / 24-hr AMS) ───────────────

def build_precipitation_layer(ref_path: Path) -> None:
    if not config.ATLAS14_ASC.exists():
        logger.warning("Atlas 14 ASC not found — falling back to PRISM")
        prism_path = config.PROCESSED_DIR / "prism_precip_utm.tif"
        if not prism_path.exists():
            logger.warning("PRISM also missing — uniform 0.5 assigned")
            dem, profile = utils.read_raster(ref_path)
            default = np.where(np.isfinite(dem), 0.5, np.nan)
            utils.write_raster(default.astype(np.float32), profile, config.PRECIP_NORM_TIF)
            return
        precip, profile = utils.read_raster(prism_path)
        normed = utils.normalize_to_01(precip)
        utils.write_raster(normed, profile, config.PRECIP_NORM_TIF)
        logger.info("  PRISM precipitation (normalised) saved → %s", config.PRECIP_NORM_TIF)
        return

    logger.info("Building precipitation layer (NOAA Atlas 14 — 100-yr/24-hr AMS) …")

    if not config.ATLAS14_UTM_TIF.exists():
        logger.info("  Reprojecting Atlas 14 to UTM 11N …")
        utils.align_raster_to_reference(
            config.ATLAS14_ASC, ref_path, config.ATLAS14_UTM_TIF,
            resampling=Resampling.bilinear,
        )
        logger.info("  Atlas 14 UTM raster cached → %s", config.ATLAS14_UTM_TIF)
    else:
        logger.info("  Using cached Atlas 14 UTM raster → %s", config.ATLAS14_UTM_TIF)

    precip, profile = utils.read_raster(config.ATLAS14_UTM_TIF)
    precip = np.where(precip == -999, np.nan, precip)
    normed = utils.normalize_to_01(precip)
    utils.write_raster(normed, profile, config.PRECIP_NORM_TIF)
    logger.info("  Atlas 14 precipitation (normalised) saved → %s", config.PRECIP_NORM_TIF)


# ── Factor: NDVI (Sentinel-2 L2A median composite) ───────────────────────────

def build_ndvi_layer(ref_path: Path) -> None:
    import pystac_client
    import planetary_computer
    from rasterio.warp import reproject as _reproject, transform_bounds
    from rasterio.windows import from_bounds as _win_from_bounds, Window as _Window
    from rasterio.transform import from_bounds as _tfb, array_bounds as _arr_bounds

    if config.NDVI_TIF.exists():
        logger.info("  Using cached NDVI raster → %s", config.NDVI_TIF)
        return

    logger.info("Building NDVI layer from Sentinel-2 L2A (Planetary Computer) …")

    with rasterio.open(ref_path) as src:
        dst_crs = src.crs
        dst_transform = src.transform
        dst_height = src.height
        dst_width = src.width
        ref_bounds = src.bounds
        bounds_wgs84 = transform_bounds(src.crs, "EPSG:4326", *ref_bounds)

    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )

    items = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=list(bounds_wgs84),
        datetime="2022-07-01/2024-09-30",
        query={"eo:cloud_cover": {"lt": 15}},
        max_items=15,
    ).item_collection()

    logger.info("  Found %d Sentinel-2 scenes", len(items))
    if not items:
        logger.warning("  No Sentinel-2 scenes found — skipping NDVI layer")
        return

    ndvi_sum = np.zeros((dst_height, dst_width), dtype=np.float32)
    ndvi_count = np.zeros((dst_height, dst_width), dtype=np.uint8)
    DOWNSAMPLE = 3

    for item in items:
        try:
            b04_url = item.assets["B04"].href
            b08_url = item.assets["B08"].href

            with rasterio.open(b04_url) as b04_src:
                scene_crs = b04_src.crs
                lb, bb, rb, tb = transform_bounds(dst_crs, scene_crs, *ref_bounds)
                win = _win_from_bounds(lb, bb, rb, tb, transform=b04_src.transform)
                win = win.intersection(_Window(0, 0, b04_src.width, b04_src.height))
                if win.width < 10 or win.height < 10:
                    continue
                out_h = max(10, int(win.height / DOWNSAMPLE))
                out_w = max(10, int(win.width / DOWNSAMPLE))
                b04_data = b04_src.read(
                    1, window=win, out_shape=(out_h, out_w),
                    resampling=Resampling.average,
                ).astype("float32") * 1e-4
                win_bounds = _arr_bounds(
                    int(round(win.height)), int(round(win.width)),
                    b04_src.window_transform(win),
                )
                src_transform = _tfb(*win_bounds, out_w, out_h)

            with rasterio.open(b08_url) as b08_src:
                win8 = _win_from_bounds(lb, bb, rb, tb, transform=b08_src.transform)
                win8 = win8.intersection(_Window(0, 0, b08_src.width, b08_src.height))
                b08_data = b08_src.read(
                    1, window=win8, out_shape=(out_h, out_w),
                    resampling=Resampling.average,
                ).astype("float32") * 1e-4

            with np.errstate(invalid="ignore", divide="ignore"):
                denom = b08_data + b04_data
                ndvi_scene = np.where(denom > 0, (b08_data - b04_data) / denom, np.nan).astype("float32")

            dst_arr = np.full((dst_height, dst_width), np.nan, dtype=np.float32)
            _reproject(
                source=ndvi_scene,
                destination=dst_arr,
                src_transform=src_transform,
                src_crs=scene_crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=Resampling.bilinear,
                src_nodata=np.nan,
                dst_nodata=np.nan,
            )

            valid = np.isfinite(dst_arr)
            ndvi_sum[valid] += dst_arr[valid]
            ndvi_count[valid] += 1
            del dst_arr, ndvi_scene, b04_data, b08_data

        except Exception as exc:
            logger.warning("  Scene failed: %s", exc)
            continue

    good = ndvi_count > 0
    if not good.any():
        logger.warning("  No valid NDVI scenes processed — skipping NDVI layer")
        return

    ndvi_mean = np.where(good, ndvi_sum / ndvi_count.astype(np.float32), np.nan).astype(np.float32)
    logger.info("  NDVI mean from %d scenes (max coverage %d)", len(items), int(ndvi_count.max()))

    with rasterio.open(ref_path) as ref:
        out_profile = ref.profile.copy()
    out_profile.update(dtype="float32", nodata=-9999.0)
    config.NDVI_TIF.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(config.NDVI_TIF, "w", **out_profile) as dst:
        arr_out = np.where(np.isnan(ndvi_mean), -9999.0, ndvi_mean)
        dst.write(arr_out, 1)
    logger.info("  NDVI mean saved → %s", config.NDVI_TIF)


# ── Factor: Soil Erodibility (SSURGO via USDA SDA API) ───────────────────────

def build_soil_risk(ref_path: Path) -> None:
    logger.info("Building soil erodibility risk layer (SSURGO) …")
    dem, profile = utils.read_raster(ref_path)

    soil_vec = config.PROCESSED_DIR / "ssurgo_soil_utm.gpkg"
    GSSURGO_GDB = config.GSSURGO_GDB

    if not soil_vec.exists():
        if GSSURGO_GDB.exists():
            logger.info("  Reading gSSURGO_CA.gdb …")
            polys = gpd.read_file(GSSURGO_GDB, layer="MUPOLYGON")
            polys.columns = [c.lower() for c in polys.columns]

            comp = gpd.read_file(GSSURGO_GDB, layer="component")
            comp.columns = [c.lower() for c in comp.columns]
            comp = comp[comp["majcompflag"] == "Yes"][["mukey", "cokey", "hydgrp"]]

            horiz = gpd.read_file(GSSURGO_GDB, layer="chorizon")
            horiz.columns = [c.lower() for c in horiz.columns]
            kf = (
                horiz[["cokey", "kffact"]]
                .dropna(subset=["kffact"])
                .astype({"kffact": float})
                .groupby("cokey", as_index=False)["kffact"].mean()
            )

            attr = comp.merge(kf, on="cokey", how="left")
            attr = attr.groupby("mukey", as_index=False).agg(
                hydgrp=("hydgrp", "first"),
                kffact=("kffact", "mean"),
            )

            with rasterio.open(ref_path) as src:
                from rasterio.warp import transform_bounds
                bounds_wgs84 = transform_bounds(src.crs, "EPSG:4326", *src.bounds)

            from shapely.geometry import box
            aoi = gpd.GeoDataFrame(geometry=[box(*bounds_wgs84)], crs="EPSG:4326")
            polys = polys.to_crs("EPSG:4326")
            polys = gpd.clip(polys, aoi)
            polys = polys.merge(attr, on="mukey", how="left")
            polys = polys.to_crs(config.CRS_ANALYSIS)
            polys.to_file(soil_vec, driver="GPKG")
            logger.info("  SSURGO: %d mapunit polygons clipped to AOI", len(polys))
        else:
            logger.warning("  gSSURGO_CA.gdb not found — assigning default risk 3")
            default = np.where(np.isfinite(dem), 3.0, np.nan)
            utils.write_raster(default.astype(np.float32), profile, config.SOIL_RISK_TIF)
            return
    else:
        logger.info("  Using cached SSURGO data → %s", soil_vec)

    gdf = gpd.read_file(soil_vec)
    gdf.columns = [c.lower() for c in gdf.columns]

    def _kffact_score(k):
        if pd.isna(k): return 3.0
        k = float(k)
        if k < 0.20: return 1.0
        if k < 0.32: return 2.0
        if k < 0.43: return 3.0
        if k < 0.55: return 4.0
        return 5.0

    _hydgrp_map = {
        "A": 1.0, "A/D": 1.5, "B": 2.0, "B/D": 2.5,
        "C": 3.5, "C/D": 4.0, "D": 5.0,
    }

    def _hydgrp_score(hg):
        if pd.isna(hg): return 3.0
        return _hydgrp_map.get(str(hg).strip(), 3.0)

    kf_col = "kffact" if "kffact" in gdf.columns else None
    hg_col = "hydgrp" if "hydgrp" in gdf.columns else None

    gdf["_kf"] = gdf[kf_col].apply(_kffact_score) if kf_col else 3.0
    gdf["_hg"] = gdf[hg_col].apply(_hydgrp_score) if hg_col else 3.0
    gdf["risk_score"] = (gdf["_kf"] * 0.7 + gdf["_hg"] * 0.3).clip(1.0, 5.0)

    risk_arr = utils.rasterize_vector(gdf, ref_path, burn_field="risk_score", fill=3.0)
    risk_arr = np.where(np.isfinite(dem), risk_arr, np.nan)
    utils.write_raster(risk_arr.astype(np.float32), profile, config.SOIL_RISK_TIF)
    logger.info("  Soil erodibility risk saved → %s", config.SOIL_RISK_TIF)


# ── Normalise All Layers ──────────────────────────────────────────────────────

def normalise_all_layers() -> None:
    logger.info("Normalising all factor layers to [0, 1] …")

    layers = [
        (config.SLOPE_TIF,          config.NORM_SLOPE_TIF,     False),
        (config.LITHOLOGY_RISK_TIF, config.NORM_LITHOLOGY_TIF, True),
        (config.LANDCOVER_RISK_TIF, config.NORM_LANDCOVER_TIF, True),
        (config.FAULT_DIST_RISK_TIF, config.NORM_FAULT_TIF,    True),
        (config.PRECIP_NORM_TIF,    config.NORM_PRECIP_TIF,    False),
        (config.SOIL_RISK_TIF,      config.NORM_SOIL_TIF,      True),
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

    if config.PROFILE_CURV_TIF.exists() and config.PLAN_CURV_TIF.exists():
        prof, profile = utils.read_raster(config.PROFILE_CURV_TIF)
        plan, _       = utils.read_raster(config.PLAN_CURV_TIF)
        combined = (-prof + -plan) / 2.0
        normed = utils.normalize_to_01(combined)
        utils.write_raster(normed, profile, config.NORM_CURVATURE_TIF)
        logger.info("  Combined curvature normalised → %s", config.NORM_CURVATURE_TIF)

    if config.NDVI_TIF.exists():
        ndvi, profile = utils.read_raster(config.NDVI_TIF)
        ndvi_norm = utils.normalize_to_01(ndvi)
        ndvi_inv = 1.0 - ndvi_norm
        ndvi_inv = np.where(np.isfinite(ndvi), ndvi_inv, np.nan).astype(np.float32)
        utils.write_raster(ndvi_inv, profile, config.NORM_NDVI_TIF)
        logger.info("  NDVI (inverted) normalised → %s", config.NORM_NDVI_TIF)
    else:
        logger.warning("  Skipping ndvi — source not found")

    logger.info("Normalisation complete.")


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(
        description="Stage 3: Factor layers for SB landslide susceptibility"
    )
    parser.add_argument(
        "--ref-raster", type=str, default=str(config.DEM_10M_TIF),
        help="Reference raster for grid alignment (default: DEM 10m)"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    utils.ensure_dirs()

    ref_path = Path(args.ref_raster)
    if not ref_path.exists():
        logger.error("Reference raster not found: %s — run 01_data_prep.py first", ref_path)
        sys.exit(1)

    build_lithology_risk(ref_path)
    build_landcover_risk(ref_path)
    build_fault_distance_risk(ref_path)
    build_precipitation_layer(ref_path)
    build_ndvi_layer(ref_path)
    build_soil_risk(ref_path)
    normalise_all_layers()

    expected = [
        config.NORM_SLOPE_TIF, config.NORM_CURVATURE_TIF, config.NORM_TWI_TIF,
        config.NORM_LITHOLOGY_TIF, config.NORM_LANDCOVER_TIF,
        config.NORM_FAULT_TIF, config.NORM_PRECIP_TIF,
        config.NORM_NDVI_TIF, config.NORM_SOIL_TIF,
    ]
    missing = [p for p in expected if not p.exists()]
    if missing:
        logger.warning("Some normalised layers are missing: %s", [p.name for p in missing])
    else:
        logger.info("All %d normalised factor layers ready.", len(expected))

    logger.info("=== Stage 3 complete. Factor layers in %s ===", config.PROCESSED_DIR)


if __name__ == "__main__":
    main()
