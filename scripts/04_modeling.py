"""
04_modeling.py
──────────────
Stage 4 of the landslide susceptibility pipeline.

Random Forest Classifier
    Samples feature values at historical landslide locations (positive class)
    and stratified non-landslide locations (negative class, 5:1 ratio).
    Trains a sklearn RandomForestClassifier, evaluates with ROC-AUC, 5-fold CV,
    confusion matrix, and feature importance. Applies the model to the full
    raster in chunks to avoid memory errors.

The probability surface is classified into 5 classes using Jenks Natural Breaks
and saved as a GeoTIFF.

Validation against the January 9, 2018 Montecito debris flow polygon is
performed if the shapefile is available.

Outputs
-------
    data/outputs/susceptibility_rf_probability.tif
    data/outputs/susceptibility_rf_classified.tif
    data/outputs/training_samples.csv
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
from rasterio.features import rasterize
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import cross_val_score, train_test_split
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config, utils

logger = utils.get_logger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)


# ── Sampling ──────────────────────────────────────────────────────────────────

def generate_training_samples(
    factor_paths: list[Path],
    feature_names: list[str],
    negative_ratio: int = config.RF_NEGATIVE_RATIO,
    random_state: int = config.RANDOM_SEED,
) -> pd.DataFrame:
    """Extract feature values at landslide (positive) and random (negative) locations.

    Positive samples are drawn from the historical landslide inventory.
    Negative samples are placed randomly across the study area, constrained to:
    - Valid (non-NaN) cells in all factor rasters
    - At least 200 m from any known landslide

    Args:
        factor_paths: List of normalised factor raster paths (same order as features).
        feature_names: Column names for the output DataFrame.
        negative_ratio: Ratio of negative to positive samples.
        random_state: Random seed for reproducibility.

    Returns:
        DataFrame with columns ``[*feature_names, 'label', 'x', 'y']``.
    """
    ls_shp = config.PROCESSED_DIR / "landslide_inventory_utm.shp"
    if not ls_shp.exists():
        raise FileNotFoundError(
            f"Landslide inventory not found at {ls_shp}\n"
            "Download from https://www.sciencebase.gov/catalog/item/58572c7be4b01fad86d5ff1f"
        )

    logger.info("Generating training samples …")
    rng = np.random.default_rng(random_state)

    # Stack all factor rasters
    stack = utils.stack_rasters(factor_paths)   # (n_features, H, W)
    n_features, H, W = stack.shape

    # Build valid-cell mask: every feature must be finite
    valid_mask = np.all(np.isfinite(stack), axis=0)  # (H, W)

    # Reference transform and CRS from first raster
    with rasterio.open(factor_paths[0]) as src:
        transform = src.transform
        crs = src.crs
        pixel_size = transform.a

    # ── Positive samples — landslide locations ────────────────────────────────
    ls_gdf = gpd.read_file(ls_shp)
    ls_centroids = ls_gdf.copy()
    ls_centroids["geometry"] = ls_gdf.geometry.centroid

    pos_rows = []
    for _, row in ls_centroids.iterrows():
        cx, cy = row.geometry.x, row.geometry.y
        # Convert world coords to pixel
        col = int((cx - transform.c) / pixel_size)
        r   = int((transform.f - cy) / pixel_size)
        if 0 <= r < H and 0 <= col < W and valid_mask[r, col]:
            feats = stack[:, r, col].tolist()
            pos_rows.append(feats + [1, cx, cy])

    logger.info("  Positive samples: %d (from %d inventory points)",
                len(pos_rows), len(ls_gdf))

    if not pos_rows:
        raise ValueError("No valid positive samples found — check CRS alignment.")

    # ── Negative samples — random non-landslide locations ─────────────────────
    # Rasterise landslide buffer (200 m) for exclusion
    ls_buffer = ls_gdf.copy()
    ls_buffer["geometry"] = ls_gdf.geometry.buffer(200)
    with rasterio.open(factor_paths[0]) as ref:
        exclusion = rasterize(
            [(g, 1) for g in ls_buffer.geometry],
            out_shape=(H, W),
            transform=transform,
            fill=0,
            dtype="uint8",
        )
    candidate_mask = valid_mask & (exclusion == 0)
    candidate_indices = np.argwhere(candidate_mask)

    n_neg = len(pos_rows) * negative_ratio
    if len(candidate_indices) < n_neg:
        logger.warning(
            "Only %d candidate negative locations available (need %d). Using all.",
            len(candidate_indices), n_neg,
        )
        n_neg = len(candidate_indices)

    chosen = rng.choice(len(candidate_indices), size=n_neg, replace=False)
    neg_rows = []
    for idx in chosen:
        r, col = candidate_indices[idx]
        cx = transform.c + col * pixel_size + pixel_size / 2
        cy = transform.f - r  * pixel_size - pixel_size / 2
        feats = stack[:, r, col].tolist()
        neg_rows.append(feats + [0, cx, cy])

    logger.info("  Negative samples: %d (ratio 1:%d)", n_neg, negative_ratio)

    cols = feature_names + ["label", "x", "y"]
    df = pd.DataFrame(pos_rows + neg_rows, columns=cols)
    df.to_csv(config.TRAINING_SAMPLES_CSV, index=False)
    logger.info("Training samples saved → %s (%d rows)", config.TRAINING_SAMPLES_CSV, len(df))
    return df


# ── Random Forest ─────────────────────────────────────────────────────────────

def train_random_forest(
    df: pd.DataFrame,
    feature_names: list[str],
) -> tuple[RandomForestClassifier, dict]:
    """Train and evaluate the Random Forest classifier.

    Args:
        df: Training samples DataFrame with feature columns and a ``label`` column.
        feature_names: List of feature column names.

    Returns:
        Tuple of (trained classifier, metrics dict).
    """
    logger.info("Training Random Forest classifier …")
    X = df[feature_names].values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.RF_TEST_SIZE,
        stratify=y,
        random_state=config.RANDOM_SEED,
    )
    logger.info("  Train: %d samples  Test: %d samples", len(y_train), len(y_test))

    clf = RandomForestClassifier(**config.RF_PARAMS)
    clf.fit(X_train, y_train)

    # ── Evaluation ────────────────────────────────────────────────────────────
    y_pred       = clf.predict(X_test)
    y_prob       = clf.predict_proba(X_test)[:, 1]
    roc_auc      = roc_auc_score(y_test, y_prob)
    fpr, tpr, _  = roc_curve(y_test, y_prob)
    cm           = confusion_matrix(y_test, y_pred)
    report       = classification_report(y_test, y_pred, output_dict=True)

    # Cross-validation
    cv_scores = cross_val_score(
        clf, X, y, cv=config.RF_CV_FOLDS,
        scoring="roc_auc", n_jobs=-1,
    )

    logger.info("  ROC-AUC: %.4f", roc_auc)
    logger.info("  CV AUC:  %.4f ± %.4f", cv_scores.mean(), cv_scores.std())
    logger.info("  Confusion matrix:\n%s", cm)

    importance_dict = dict(zip(feature_names, clf.feature_importances_))
    metrics = {
        "roc_auc": float(roc_auc),
        "cv_auc_mean": float(cv_scores.mean()),
        "cv_auc_std": float(cv_scores.std()),
        "cv_auc_scores": cv_scores.tolist(),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "feature_importances": importance_dict,
        "roc_curve": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
    }

    if roc_auc < 0.85:
        logger.warning("ROC-AUC %.4f is below the 0.85 target. Consider reviewing "
                       "sample quality or feature engineering.", roc_auc)

    return clf, metrics


def apply_rf_to_raster(
    clf: RandomForestClassifier,
    factor_paths: list[Path],
    profile: dict,
    chunk_rows: int = config.CHUNK_ROWS,
) -> np.ndarray:
    """Apply the trained classifier to the full raster in row chunks.

    Processing the full Santa Barbara County 10 m raster in a single pass
    would require ~4 GB of RAM. This function reads and predicts in
    ``chunk_rows`` strips to keep memory manageable.

    Args:
        clf: Trained RandomForestClassifier.
        factor_paths: Ordered list of normalised factor raster paths.
        profile: Rasterio profile for the output raster.
        chunk_rows: Number of rows to process per chunk.

    Returns:
        2-D float32 probability array (P(landslide) ∈ [0, 1]).
    """
    logger.info("Applying RF model to full raster (chunk_rows=%d) …", chunk_rows)

    # Open all sources
    srcs = [rasterio.open(p) for p in factor_paths if p.exists()]
    H = srcs[0].height
    W = srcs[0].width
    prob_canvas = np.full((H, W), np.nan, dtype=np.float32)

    n_chunks = (H + chunk_rows - 1) // chunk_rows
    for chunk_idx in tqdm(range(n_chunks), desc="RF prediction chunks"):
        row_start = chunk_idx * chunk_rows
        row_end   = min(row_start + chunk_rows, H)
        rows      = row_end - row_start

        # Read chunk from each source
        chunks = []
        for src in srcs:
            window = rasterio.windows.Window(0, row_start, W, rows)
            band = src.read(1, window=window, out_dtype=np.float32)
            nodata = src.nodata
            if nodata is not None:
                band = np.where(band == nodata, np.nan, band)
            chunks.append(band)

        stack = np.stack(chunks, axis=0)   # (n_features, rows, W)
        X, _, valid_mask = utils.raster_to_samples(stack)

        if X.shape[0] == 0:
            continue

        proba = clf.predict_proba(X)[:, 1].astype(np.float32)
        chunk_result = utils.samples_to_raster(proba, valid_mask)
        prob_canvas[row_start:row_end, :] = chunk_result

    for src in srcs:
        src.close()

    logger.info("RF prediction complete.")
    return prob_canvas


# ── Montecito Validation ──────────────────────────────────────────────────────

def validate_montecito(
    rf_classified: np.ndarray,
    profile: dict,
) -> dict:
    """Assess model performance within the 2018 Montecito debris flow extent.

    Clips the classified RF susceptibility surface to the debris flow polygon
    and reports the percentage of area in each susceptibility class.

    Args:
        rf_classified: 5-class RF susceptibility array.
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
    import tempfile, rasterio as rio

    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
        tmp_path = tmp.name
    utils.write_raster(rf_classified, profile.copy(), tmp_path)

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
        pct = 100.0 * count / total if total > 0 else 0.0
        class_pct[config.SUSCEPTIBILITY_LABELS[cls]] = round(float(pct), 2)

    high_pct = class_pct.get("High", 0) + class_pct.get("Very High", 0)
    logger.info(
        "  RF — High+Very High within debris flow: %.1f%%  (target: majority)",
        high_pct,
    )

    import os; os.unlink(tmp_path)

    result = {"rf": class_pct}
    df = pd.DataFrame(result)
    df.index.name = "susceptibility_class"
    df.to_csv(config.MONTECITO_VALIDATION_CSV)
    logger.info("Montecito validation saved → %s", config.MONTECITO_VALIDATION_CSV)
    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    """Run the modeling stage."""
    utils.ensure_dirs()

    factor_paths = [
        config.NORM_SLOPE_TIF,
        config.NORM_CURVATURE_TIF,
        config.NORM_TWI_TIF,
        config.NORM_LITHOLOGY_TIF,
        config.NORM_LANDCOVER_TIF,
        config.NORM_FAULT_TIF,
        config.NORM_PRECIP_TIF,
        config.NORM_FIRE_TIF,
    ]
    feature_names = config.FEATURE_COLS

    existing = [p for p in factor_paths if p.exists()]
    if not existing:
        logger.error("No normalised factor rasters found — run 03_factor_layers.py first")
        sys.exit(1)

    with rasterio.open(existing[0]) as src:
        profile = src.profile.copy()

    metrics = {}

    logger.info("=== Random Forest Model ===")
    try:
        samples = generate_training_samples(factor_paths, feature_names)
        clf, rf_metrics = train_random_forest(samples, feature_names)
        metrics.update(rf_metrics)

        rf_prob = apply_rf_to_raster(clf, factor_paths, profile)
        utils.write_raster(rf_prob, profile.copy(), config.SUSCEPTIBILITY_RF_PROB_TIF)
        logger.info("RF probability map → %s", config.SUSCEPTIBILITY_RF_PROB_TIF)

        rf_classified, rf_breaks = utils.reclassify_jenks(rf_prob, n_classes=5)
        logger.info("RF Jenks breaks: %s", [round(b, 4) for b in rf_breaks])
        metrics["rf_jenks_breaks"] = rf_breaks

        utils.write_raster(rf_classified, profile.copy(), config.SUSCEPTIBILITY_RF_TIF)
        logger.info("RF classified map → %s", config.SUSCEPTIBILITY_RF_TIF)

        logger.info("=== Montecito Validation ===")
        val_results = validate_montecito(rf_classified, profile)
        if val_results:
            metrics["montecito_validation"] = val_results

    except FileNotFoundError as exc:
        logger.error("RF model failed: %s", exc)
        sys.exit(1)

    with open(config.MODEL_METRICS_JSON, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    logger.info("Model metrics saved → %s", config.MODEL_METRICS_JSON)

    logger.info("=== Stage 4 complete ===")


if __name__ == "__main__":
    main()
