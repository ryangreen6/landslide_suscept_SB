"""
04_modeling.py
──────────────
Stage 4 of the landslide susceptibility pipeline.

Logistic Regression Model
    Presence points: high-confidence USGS NLI polygon centroids (Confidence >= 3).
    Pseudo-absences: random county-wide sample (5:1 ratio).
    TWI excluded — collinear with slope (TWI = ln(upslope_area / tan(slope))).
    Fault distance excluded — trigger factor, not inherent terrain property.
    NaN factors imputed with per-column median.
    Classification: percentile breaks on the LR probability distribution
    (bottom 30% / 30-50 / 50-70 / 70-85 / top 15%).
    Performance: spatial block CV AUC + 20% random hold-out hit rate.
    External validation: USGS 2023 Santa Ynez Mountains inventory (not used in training).

Outputs
-------
    data/outputs/susceptibility_lr_probability.tif
    data/outputs/susceptibility_lr_classified.tif
    data/outputs/lr_coefficients.csv
    data/outputs/lr_validation.csv
    data/outputs/model_metrics.json

Usage
-----
    python scripts/04_modeling.py
"""

from __future__ import annotations

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


def run_logistic_regression(
    factor_paths: list,
    feature_names: list,
    profile: dict,
) -> dict:
    if not config.USGS_LS_GPKG.exists():
        logger.warning("NLI GeoPackage not found — skipping logistic regression")
        return {}

    from rasterio.transform import rowcol
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    import shapely

    logger.info("Running logistic regression model …")

    transform = profile["transform"]
    height, width = profile["height"], profile["width"]

    # ── Load factor rasters (TWI excluded — collinear with slope) ─────────────
    factor_arrs = {}
    valid_names = []
    for path, name in zip(factor_paths, feature_names):
        if name == "twi":
            continue
        if path.exists():
            arr, _ = utils.read_raster(path)
            factor_arrs[name] = arr.astype(np.float32)
            valid_names.append(name)

    if not valid_names:
        logger.warning("No factor rasters available for logistic regression")
        return {}

    # ── Presence points: NLI high-confidence centroids ────────────────────────
    nli = gpd.read_file(config.USGS_LS_GPKG).to_crs(config.CRS_ANALYSIS)
    nli_hq = nli[nli["Confidence"] >= 3].copy()
    centroids = nli_hq.geometry.centroid
    pres_x, pres_y = centroids.x.values, centroids.y.values
    logger.info("  %d NLI presence points (Confidence >= 3)", len(pres_x))

    rows_p, cols_p = rowcol(transform, pres_x, pres_y)
    rows_p, cols_p = np.asarray(rows_p), np.asarray(cols_p)
    ib_p = (rows_p >= 0) & (rows_p < height) & (cols_p >= 0) & (cols_p < width)
    rows_p, cols_p = rows_p[ib_p], cols_p[ib_p]
    pres_x, pres_y = pres_x[ib_p], pres_y[ib_p]

    X_pres_raw = np.column_stack([factor_arrs[n][rows_p, cols_p] for n in valid_names])
    has_data_p = np.any(np.isfinite(X_pres_raw), axis=1)
    X_pres_raw = X_pres_raw[has_data_p]
    rows_p, cols_p = rows_p[has_data_p], cols_p[has_data_p]
    pres_x, pres_y = pres_x[has_data_p], pres_y[has_data_p]
    n_pres = len(X_pres_raw)
    logger.info("  %d presence points", n_pres)

    # ── Pseudo-absence points (county-wide, 200m exclusion buffer) ────────────
    from scipy.ndimage import distance_transform_edt as _edt

    county = gpd.read_file(config.COUNTY_UTM_SHP)
    county_geom = county.geometry.union_all()
    bounds = county_geom.bounds

    # Exclusion zone: 200m buffer around all known landslide locations
    ls_raster = np.zeros((height, width), dtype=np.uint8)
    ls_raster[rows_p, cols_p] = 1
    if config.SANTA_YNEZ_2023_CSV.exists():
        sy_excl = pd.read_csv(config.SANTA_YNEZ_2023_CSV)
        sy_excl = sy_excl[sy_excl["HillslopeSetting"] == "Unmodified"]
        sy_gdf = gpd.GeoDataFrame(
            sy_excl, geometry=gpd.points_from_xy(sy_excl.Longitude, sy_excl.Latitude), crs="EPSG:4326"
        ).to_crs(config.CRS_ANALYSIS)
        sy_r, sy_c = rowcol(transform, sy_gdf.geometry.x.values, sy_gdf.geometry.y.values)
        sy_r, sy_c = np.asarray(sy_r), np.asarray(sy_c)
        ib_sy = (sy_r >= 0) & (sy_r < height) & (sy_c >= 0) & (sy_c < width)
        ls_raster[sy_r[ib_sy], sy_c[ib_sy]] = 1
    excl_mask = _edt(1 - ls_raster) < 20  # 200m at 10m resolution

    rng = np.random.default_rng(config.RANDOM_SEED)
    n_pseudo = n_pres * 5
    cand_x = rng.uniform(bounds[0], bounds[2], n_pseudo * 8)
    cand_y = rng.uniform(bounds[1], bounds[3], n_pseudo * 8)
    in_county = shapely.contains_xy(county_geom, cand_x, cand_y)
    cand_x, cand_y = cand_x[in_county], cand_y[in_county]

    rows_cand, cols_cand = rowcol(transform, cand_x, cand_y)
    rows_cand, cols_cand = np.asarray(rows_cand), np.asarray(cols_cand)
    ib = (rows_cand >= 0) & (rows_cand < height) & (cols_cand >= 0) & (cols_cand < width)
    rows_cand, cols_cand = rows_cand[ib], cols_cand[ib]
    cand_x, cand_y = cand_x[ib], cand_y[ib]

    eligible = ~excl_mask[rows_cand, cols_cand]
    rows_a = rows_cand[eligible][:n_pseudo]
    cols_a = cols_cand[eligible][:n_pseudo]
    abs_x  = cand_x[eligible][:n_pseudo]
    abs_y  = cand_y[eligible][:n_pseudo]

    X_abs_raw = np.column_stack([factor_arrs[n][rows_a, cols_a] for n in valid_names])
    has_data_a = np.any(np.isfinite(X_abs_raw), axis=1)
    X_abs_raw = X_abs_raw[has_data_a]
    abs_x, abs_y = abs_x[has_data_a], abs_y[has_data_a]
    n_abs = len(X_abs_raw)
    logger.info("  %d pseudo-absence points", n_abs)

    # ── Impute NaN with per-column median (fit on combined training data) ──────
    imputer = SimpleImputer(strategy="median")
    X_combined = imputer.fit_transform(np.vstack([X_pres_raw, X_abs_raw]))
    X_pres = X_combined[:n_pres]
    X_abs = X_combined[n_pres:]

    X = np.vstack([X_pres, X_abs])
    y = np.concatenate([np.ones(n_pres), np.zeros(n_abs)])
    all_x = np.concatenate([pres_x, abs_x])
    all_y = np.concatenate([pres_y, abs_y])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── Spatial block CV (5x5 grid, leave-one-block-out) ──────────────────────
    n_blocks = 5
    x_bin = np.digitize(all_x, np.linspace(bounds[0], bounds[2], n_blocks + 1)[1:-1])
    y_bin = np.digitize(all_y, np.linspace(bounds[1], bounds[3], n_blocks + 1)[1:-1])
    block_ids = x_bin * n_blocks + y_bin

    cv_aucs = []
    for block in np.unique(block_ids):
        test = block_ids == block
        train = ~test
        if len(np.unique(y[test])) < 2 or train.sum() < 20:
            continue
        clf_cv = LogisticRegression(
            C=1.0, solver="lbfgs", max_iter=1000, random_state=config.RANDOM_SEED
        )
        clf_cv.fit(X_scaled[train], y[train])
        cv_aucs.append(float(roc_auc_score(y[test], clf_cv.predict_proba(X_scaled[test])[:, 1])))

    if cv_aucs:
        cv_auc_mean = round(float(np.mean(cv_aucs)), 4)
        cv_auc_std = round(float(np.std(cv_aucs)), 4)
    else:
        logger.warning("  No valid spatial CV blocks — CV AUC unavailable")
        cv_auc_mean, cv_auc_std = float("nan"), float("nan")
    logger.info("  Spatial CV AUC: %.3f +/- %.3f (%d blocks)", cv_auc_mean, cv_auc_std, len(cv_aucs))

    # ── Final model on all data ────────────────────────────────────────────────
    clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=config.RANDOM_SEED)
    clf.fit(X_scaled, y)

    # ── Apply to full raster ───────────────────────────────────────────────────
    logger.info("  Applying LR to full raster …")
    from rasterio.features import rasterize as _rasterize
    county_gdf = gpd.read_file(config.COUNTY_UTM_SHP)
    county_union = county_gdf.geometry.union_all()
    county_raster = _rasterize(
        [(county_union, 1)],
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype=np.uint8,
    ).astype(bool)
    dem, _ = utils.read_raster(config.DEM_10M_TIF)
    land_mask = county_raster & np.isfinite(dem) & (dem > 0)

    flat_raw = np.column_stack([factor_arrs[n][land_mask] for n in valid_names])
    flat_imputed = imputer.transform(flat_raw)
    prob_flat = clf.predict_proba(scaler.transform(flat_imputed))[:, 1].astype(np.float32)

    lr_prob = np.full((height, width), np.nan, dtype=np.float32)
    lr_prob[land_mask] = prob_flat

    # ── Percentile breaks on LR probability distribution ──────────────────────
    valid_prob = lr_prob[np.isfinite(lr_prob)].flatten()
    p30, p50, p70, p85 = np.percentile(valid_prob, [30, 50, 70, 85])
    lr_breaks = [0.0, float(p30), float(p50), float(p70), float(p85), 1.0]
    logger.info("  LR percentile breaks: %s", [round(b, 4) for b in lr_breaks])

    lr_classified = utils.reclassify_fixed(lr_prob, lr_breaks)

    utils.write_raster(lr_prob, profile.copy(), config.SUSCEPTIBILITY_LR_PROB_TIF)
    utils.write_raster(lr_classified, profile.copy(), config.SUSCEPTIBILITY_LR_TIF)
    logger.info("LR probability map → %s", config.SUSCEPTIBILITY_LR_PROB_TIF)
    logger.info("LR classified map → %s", config.SUSCEPTIBILITY_LR_TIF)

    # ── Factor importance ──────────────────────────────────────────────────────
    coef = clf.coef_[0]
    norm_coef = np.abs(coef) / np.abs(coef).sum()
    coef_df = pd.DataFrame({
        "factor":               valid_names,
        "lr_coefficient":       coef.round(4).tolist(),
        "lr_normalized_weight": norm_coef.round(4).tolist(),
    }).sort_values("lr_normalized_weight", ascending=False)
    coef_df.to_csv(config.LR_COEFFICIENTS_CSV, index=False)
    logger.info("LR coefficients → %s", config.LR_COEFFICIENTS_CSV)
    for _, row in coef_df.iterrows():
        logger.info(
            "  %-18s  raw %+.3f  norm %.3f",
            row["factor"], row["lr_coefficient"], row["lr_normalized_weight"],
        )

    # ── 20% random hold-out hit rate (unbiased) ───────────────────────────────
    rng_ho = np.random.default_rng(config.RANDOM_SEED + 1)
    test_idx = rng_ho.choice(n_pres, size=int(n_pres * 0.2), replace=False)
    is_test_p = np.zeros(n_pres, dtype=bool)
    is_test_p[test_idx] = True
    test_rows_p, test_cols_p = rows_p[is_test_p], cols_p[is_test_p]
    logger.info("  Hit-rate hold-out: %d of %d NLI presences (random 20%%)", int(is_test_p.sum()), n_pres)

    cls_vals = lr_classified[test_rows_p, test_cols_p]
    cls_valid = cls_vals[(cls_vals >= 1) & (cls_vals <= 5)]
    n_cls = len(cls_valid)
    hit_rate = {}
    for c in range(1, 6):
        pct = round(float(100.0 * np.sum(cls_valid == c) / n_cls) if n_cls else 0.0, 2)
        hit_rate[config.SUSCEPTIBILITY_LABELS[c]] = pct
    high_pct = hit_rate.get("High", 0.0) + hit_rate.get("Very High", 0.0)
    logger.info("  High+Very High at held-out NLI centroids: %.1f%%", high_pct)

    pd.DataFrame({"lr": hit_rate}).rename_axis("susceptibility_class").to_csv(
        config.LR_VALIDATION_CSV
    )
    logger.info("LR validation saved → %s", config.LR_VALIDATION_CSV)

    # ── External validation: Santa Ynez 2023 (not used in training) ───────────
    sy_ext = {}
    if config.SANTA_YNEZ_2023_CSV.exists():
        sy = pd.read_csv(config.SANTA_YNEZ_2023_CSV)
        sy = sy[sy["HillslopeSetting"] == "Unmodified"].copy()
        sy_gdf = gpd.GeoDataFrame(
            sy, geometry=gpd.points_from_xy(sy.Longitude, sy.Latitude), crs="EPSG:4326"
        ).to_crs(config.CRS_ANALYSIS)
        sy_ext_x = sy_gdf.geometry.x.values
        sy_ext_y = sy_gdf.geometry.y.values
        rows_sy, cols_sy = rowcol(transform, sy_ext_x, sy_ext_y)
        rows_sy, cols_sy = np.asarray(rows_sy), np.asarray(cols_sy)
        ib_sy = (rows_sy >= 0) & (rows_sy < height) & (cols_sy >= 0) & (cols_sy < width)
        rows_sy, cols_sy = rows_sy[ib_sy], cols_sy[ib_sy]
        cls_sy = lr_classified[rows_sy, cols_sy]
        cls_sy_valid = cls_sy[(cls_sy >= 1) & (cls_sy <= 5)]
        n_sy = len(cls_sy_valid)
        sy_hit = {}
        for c in range(1, 6):
            pct = round(float(100.0 * np.sum(cls_sy_valid == c) / n_sy) if n_sy else 0.0, 2)
            sy_hit[config.SUSCEPTIBILITY_LABELS[c]] = pct
        sy_high_pct = sy_hit.get("High", 0.0) + sy_hit.get("Very High", 0.0)
        logger.info("  Santa Ynez 2023 external validation: %d points", n_sy)
        logger.info("  High+Very High at Santa Ynez 2023 points: %.1f%%", sy_high_pct)
        sy_ext = {
            "n_points": n_sy,
            "high_plus_very_high_pct": round(sy_high_pct, 2),
            "hit_rate_by_class": sy_hit,
        }

    return {
        "n_presences": n_pres,
        "n_pseudo_absences": n_abs,
        "cv_auc_mean": cv_auc_mean,
        "cv_auc_std": cv_auc_std,
        "n_cv_blocks": len(cv_aucs),
        "lr_breaks": lr_breaks,
        "high_plus_very_high_pct": round(high_pct, 2),
        "hit_rate_by_class": hit_rate,
        "coefficients": coef_df.to_dict(orient="records"),
        "santa_ynez_2023_external_validation": sy_ext,
    }


def main() -> None:
    utils.ensure_dirs()

    factor_paths = [
        config.NORM_SLOPE_TIF,
        config.NORM_CURVATURE_TIF,
        config.NORM_TWI_TIF,
        config.NORM_LITHOLOGY_TIF,
        config.NORM_LANDCOVER_TIF,
        config.NORM_PRECIP_TIF,
        config.NORM_NDVI_TIF,
        config.NORM_SOIL_TIF,
        config.NORM_BURN_TIF,
    ]
    feature_names = config.FEATURE_COLS

    existing = [p for p in factor_paths if p.exists()]
    if not existing:
        logger.error("No normalised factor rasters found — run 03_factor_layers.py first")
        sys.exit(1)

    with rasterio.open(existing[0]) as src:
        profile = src.profile.copy()

    logger.info("=== Logistic Regression Model ===")
    lr_results = run_logistic_regression(factor_paths, feature_names, profile)

    metrics = {}
    if lr_results:
        metrics["logistic_regression"] = lr_results

    with open(config.MODEL_METRICS_JSON, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    logger.info("Model metrics saved → %s", config.MODEL_METRICS_JSON)
    logger.info("=== Stage 4 complete ===")


if __name__ == "__main__":
    main()
