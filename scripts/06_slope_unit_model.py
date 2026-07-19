"""
06_slope_unit_model.py
──────────────────────
Logistic regression model at the slope unit level.

Each slope unit is characterized by the mean of all 8 factor rasters
(zonal statistics). Units containing NLI presence points (Confidence >= 3)
are labeled positive. Units containing no recorded landslide of any
confidence are labeled negative. Negatives are subsampled to a 5:1 ratio.

Output: data/outputs/slope_units_classified.geojson

Usage
-----
    python scripts/06_slope_unit_model.py
"""

import sys
from pathlib import Path

import geopandas as gpd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config, utils

logger = utils.get_logger(__name__)

FACTOR_PATHS = [
    (config.NORM_SLOPE_TIF,     "slope"),
    (config.NORM_CURVATURE_TIF, "curvature"),
    (config.NORM_LITHOLOGY_TIF, "lithology"),
    (config.NORM_LANDCOVER_TIF, "landcover"),
    (config.NORM_PRECIP_TIF,    "rainfall"),
    (config.NORM_NDVI_TIF,      "ndvi"),
    (config.NORM_SOIL_TIF,      "soil"),
    (config.NORM_BURN_TIF,      "burn_severity"),
]


def main() -> None:
    utils.ensure_dirs()

    for req, label in [(config.SLOPE_UNITS_GPKG, "slope units"), (config.USGS_LS_GPKG, "NLI GeoPackage")]:
        if not req.exists():
            logger.error("%s not found", label)
            sys.exit(1)

    from rasterstats import zonal_stats
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    logger.info("=== Slope Unit Logistic Regression ===")

    slope_units = gpd.read_file(str(config.SLOPE_UNITS_GPKG))
    logger.info("  %d slope units loaded", len(slope_units))

    valid_factors = [(p, n) for p, n in FACTOR_PATHS if p.exists()]
    feature_names = [n for _, n in valid_factors]

    logger.info("  Computing zonal statistics for %d factors ...", len(valid_factors))
    for path, name in valid_factors:
        stats = zonal_stats(slope_units, str(path), stats=["mean"],
                            nodata=config.NODATA, all_touched=False)
        slope_units[name] = [s["mean"] for s in stats]
        logger.info("    %s", name)

    nli = gpd.read_file(str(config.USGS_LS_GPKG)).to_crs(slope_units.crs)
    nli_hq = nli[nli["Confidence"] >= 3].copy()
    nli_hq = nli_hq.set_geometry(nli_hq.geometry.centroid)
    nli_all = nli.copy()
    nli_all = nli_all.set_geometry(nli_all.geometry.centroid)

    joined_pos = gpd.sjoin(slope_units[["geometry"]], nli_hq[["geometry"]],
                           how="left", predicate="contains")
    positive_ids = set(joined_pos[joined_pos["index_right"].notna()].index)

    joined_all = gpd.sjoin(slope_units[["geometry"]], nli_all[["geometry"]],
                           how="left", predicate="contains")
    any_ls_ids = set(joined_all[joined_all["index_right"].notna()].index)
    negative_ids = set(slope_units.index) - any_ls_ids

    slope_units["label"] = np.nan
    slope_units.loc[list(positive_ids), "label"] = 1.0
    slope_units.loc[list(negative_ids), "label"] = 0.0

    labeled = slope_units[slope_units["label"].notna()]
    n_pos = int((labeled["label"] == 1).sum())
    n_neg = int((labeled["label"] == 0).sum())
    logger.info("  %d positive units, %d negative units", n_pos, n_neg)

    rng = np.random.default_rng(config.RANDOM_SEED)
    neg_idx = labeled[labeled["label"] == 0].index.tolist()
    sampled_neg = rng.choice(neg_idx, size=min(n_pos * 5, len(neg_idx)), replace=False)
    train_idx = list(positive_ids) + list(sampled_neg)
    train = labeled.loc[train_idx]

    X_raw = train[feature_names].values.astype(np.float32)
    y = train["label"].values

    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X_raw)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imp)

    clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=config.RANDOM_SEED)
    clf.fit(X_scaled, y)
    logger.info("  Model trained on %d units (%d pos / %d neg)", len(train), n_pos, len(sampled_neg))

    import pandas as pd
    coef = clf.coef_[0]
    norm_coef = np.abs(coef) / np.abs(coef).sum()
    coef_df = pd.DataFrame({
        "factor": feature_names,
        "lr_coefficient": coef.round(4).tolist(),
        "lr_normalized_weight": norm_coef.round(4).tolist(),
    }).sort_values("lr_normalized_weight", ascending=False)
    coef_df.to_csv(config.SU_LR_COEFFICIENTS_CSV, index=False)
    logger.info("  Slope unit coefficients:")
    for _, row in coef_df.iterrows():
        logger.info("    %-18s  raw %+.3f  norm %.1f%%",
                    row["factor"], row["lr_coefficient"], row["lr_normalized_weight"] * 100)

    all_X = slope_units[feature_names].values.astype(np.float32)
    has_data = np.any(np.isfinite(all_X), axis=1)
    prob = np.full(len(slope_units), np.nan)
    prob[has_data] = clf.predict_proba(scaler.transform(imputer.transform(all_X[has_data])))[:, 1]
    slope_units["probability"] = prob

    valid_prob = prob[np.isfinite(prob)]
    p30, p50, p70, p85 = np.percentile(valid_prob, [30, 50, 70, 85])
    breaks = [0.0, float(p30), float(p50), float(p70), float(p85), 1.0]
    logger.info("  Percentile breaks: %s", [round(b, 4) for b in breaks])

    def _classify(p):
        if not np.isfinite(p):
            return 0
        for cls, (lo, hi) in enumerate(zip(breaks[:-1], breaks[1:]), start=1):
            if lo <= p <= hi:
                return cls
        return 5

    slope_units["susc_class"] = slope_units["probability"].apply(_classify)
    slope_units["susc_label"] = slope_units["susc_class"].map(config.SUSCEPTIBILITY_LABELS).fillna("")

    out = slope_units[["geometry", "probability", "susc_class", "susc_label"]].copy()
    out = out[out["susc_class"] > 0].copy()

    if config.GEOLOGY_SHP.exists():
        logger.info("  Clipping to land polygon ...")
        from shapely.ops import unary_union
        geo = gpd.read_file(config.GEOLOGY_SHP)
        county = gpd.read_file(config.COUNTY_UTM_SHP)
        county_geom = county.to_crs(geo.crs).union_all()
        geo_land = geo[geo.intersects(county_geom) & (geo["ORIG_LABEL"] != "water")]
        land_poly = unary_union(geo_land.geometry)
        land_gdf = gpd.GeoDataFrame(geometry=[land_poly], crs=geo.crs).to_crs(out.crs)
        out = gpd.clip(out, land_gdf)
        out = out[out.geometry.is_valid & ~out.geometry.is_empty].copy()
        logger.info("  %d units after land clip", len(out))

    out = out.to_crs("EPSG:4326")
    out["geometry"] = out.geometry.simplify(0.0003, preserve_topology=True)
    out = out[out.geometry.is_valid & ~out.geometry.is_empty].reset_index(drop=True)
    out["probability"] = out["probability"].round(4)

    out.to_file(str(config.SLOPE_UNITS_GEOJSON), driver="GeoJSON")
    logger.info("Slope unit GeoJSON → %s  (%d units)", config.SLOPE_UNITS_GEOJSON, len(out))
    logger.info("=== Stage 6 complete ===")


if __name__ == "__main__":
    main()
