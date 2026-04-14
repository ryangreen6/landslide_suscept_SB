"""
config.py
─────────
Centralised constants for the Santa Barbara County landslide susceptibility
pipeline.  Every script imports from here instead of hardcoding values.
"""

from pathlib import Path

# ── Coordinate Reference Systems ──────────────────────────────────────────────
CRS_ANALYSIS = "EPSG:26911"   # UTM Zone 11N (metres) — all analysis
CRS_OUTPUT   = "EPSG:4326"    # WGS84 — interactive map export only

# ── Raster Grid Specification ─────────────────────────────────────────────────
RESOLUTION = 10          # target pixel size in metres
BUFFER_M   = 500         # buffer around county boundary for terrain calcs
NODATA     = -9999.0

# ── Random Seed ───────────────────────────────────────────────────────────────
RANDOM_SEED = 42

# ── Project Root and Data Paths ───────────────────────────────────────────────
PROJECT_ROOT  = Path(__file__).resolve().parent.parent
DATA_DIR      = PROJECT_ROOT / "data"
RAW_DIR       = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUTS_DIR   = DATA_DIR / "outputs"
FIGURES_DIR   = OUTPUTS_DIR / "figures"

# ── Raw Input Files (expected after manual download — see README) ─────────────
COUNTY_BOUNDARY_SHP     = RAW_DIR / "sb_county_boundary" / "sb_county_boundary.shp"
DEM_TILES_DIR           = RAW_DIR / "dem_tiles"                  # 1/3 arc-sec 3DEP tiles
DEM_MOSAIC_TIF          = RAW_DIR / "dem_mosaic_raw.tif"         # mosaicked DEM
GEOLOGY_SHP             = RAW_DIR / "ca_geology" / "ca_geology.shp"
NLCD_TIF                = RAW_DIR / "nlcd_2021" / "nlcd_2021_lcmap.tif"   # USGS LCMAP 2021
LANDSLIDE_INVENTORY_SHP = RAW_DIR / "landslide_inventory" / "ls_inventory.shp"
FAULT_LINES_SHP         = RAW_DIR / "quaternary_faults" / "qfaults.shp"
PRISM_PRECIP_TIF        = RAW_DIR / "prism" / "prism_ppt_sb_area.tif"
FIRE_PERIMETERS_SHP     = RAW_DIR / "fire_perimeters" / "California_Fire_Perimeters_all.shp"
MONTECITO_DEBRIS_SHP    = RAW_DIR / "montecito_debris_flow" / "montecito_2018_debris_flow.shp"

# ── Processed Intermediate Files ──────────────────────────────────────────────
DEM_10M_TIF        = PROCESSED_DIR / "dem_10m.tif"
SLOPE_TIF          = PROCESSED_DIR / "slope.tif"
ASPECT_TIF         = PROCESSED_DIR / "aspect.tif"
PROFILE_CURV_TIF   = PROCESSED_DIR / "profile_curvature.tif"
PLAN_CURV_TIF      = PROCESSED_DIR / "plan_curvature.tif"
TWI_TIF            = PROCESSED_DIR / "twi.tif"
FLOW_ACC_TIF       = PROCESSED_DIR / "flow_accumulation.tif"

LITHOLOGY_RISK_TIF  = PROCESSED_DIR / "lithology_risk.tif"
LANDCOVER_RISK_TIF  = PROCESSED_DIR / "landcover_risk.tif"
FAULT_DIST_RISK_TIF = PROCESSED_DIR / "fault_distance_risk.tif"
PRECIP_NORM_TIF     = PROCESSED_DIR / "precipitation_normalized.tif"
FIRE_RISK_TIF       = PROCESSED_DIR / "fire_history_risk.tif"

# Normalized factor rasters (0–1), ready for modeling
NORM_SLOPE_TIF     = PROCESSED_DIR / "norm_slope.tif"
NORM_CURVATURE_TIF = PROCESSED_DIR / "norm_curvature.tif"   # combined profile+plan
NORM_TWI_TIF       = PROCESSED_DIR / "norm_twi.tif"
NORM_LITHOLOGY_TIF = PROCESSED_DIR / "norm_lithology.tif"
NORM_LANDCOVER_TIF = PROCESSED_DIR / "norm_landcover.tif"
NORM_FAULT_TIF     = PROCESSED_DIR / "norm_fault_distance.tif"
NORM_PRECIP_TIF    = PROCESSED_DIR / "norm_precipitation.tif"
NORM_FIRE_TIF      = PROCESSED_DIR / "norm_fire_history.tif"

# County boundary reprojected to analysis CRS
COUNTY_UTM_SHP     = PROCESSED_DIR / "sb_county_utm.shp"
COUNTY_UTM_TIF     = PROCESSED_DIR / "county_mask.tif"

# ── Output Files ──────────────────────────────────────────────────────────────
SUSCEPTIBILITY_RF_PROB_TIF = OUTPUTS_DIR / "susceptibility_rf_probability.tif"
SUSCEPTIBILITY_RF_TIF      = OUTPUTS_DIR / "susceptibility_rf_classified.tif"
TRAINING_SAMPLES_CSV       = OUTPUTS_DIR / "training_samples.csv"
MODEL_METRICS_JSON         = OUTPUTS_DIR / "model_metrics.json"
MONTECITO_VALIDATION_CSV   = OUTPUTS_DIR / "montecito_validation.csv"

# ── Figure Outputs ────────────────────────────────────────────────────────────
FIG_FACTORS       = FIGURES_DIR / "factor_layers_overview.png"
FIG_COMPARISON    = FIGURES_DIR / "susceptibility_comparison.png"
FIG_ROC           = FIGURES_DIR / "roc_curve.png"
FIG_IMPORTANCE    = FIGURES_DIR / "feature_importance.png"
FIG_MONTECITO     = FIGURES_DIR / "montecito_validation.png"
INTERACTIVE_HTML  = OUTPUTS_DIR / "susceptibility_interactive.html"

# ── Susceptibility Class Colors & Labels ──────────────────────────────────────
SUSCEPTIBILITY_COLORS = {
    1: "#1a9641",   # Very Low
    2: "#a6d96a",   # Low
    3: "#ffffbf",   # Moderate
    4: "#fdae61",   # High
    5: "#d7191c",   # Very High
}
SUSCEPTIBILITY_LABELS = {
    1: "Very Low",
    2: "Low",
    3: "Moderate",
    4: "High",
    5: "Very High",
}

# ── Lithology Risk Classification ─────────────────────────────────────────────
# Matched against USGS geology map UNIT_NAME / ROCKTYPE1 substrings (case-insensitive)
# Score 5 = highest failure susceptibility; 1 = lowest
LITHOLOGY_RISK_KEYWORDS = {
    5: ["shale", "mudstone", "claystone", "marl", "diatomite",
        "marine sediment", "pelitic", "argillite"],
    4: ["alluvium", "alluvial", "fan deposit", "younger deposits",
        "quaternary sediment", "flood plain",
        "unconsolidated",          # SGMC GENERALIZE: "Unconsolidated, undifferentiated"
        "sedimentary, clastic",    # SGMC GENERALIZE: clastic sed. (shale/sandstone/mudstone)
        "clastic"],
    3: ["colluvium", "colluvial", "terrace", "older alluvium",
        "landslide deposit", "sand", "gravel", "sedimentary"],
    2: ["volcanic", "basalt", "andesite", "rhyolite", "tuff",
        "schist", "gneiss", "metamorphic", "phyllite", "slate"],
    1: ["granite", "granodiorite", "diorite", "gabbro",
        "crystalline", "plutonic", "quartz", "serpentinite",
        "intrusive"],              # SGMC GENERALIZE: "Igneous, intrusive"
}
LITHOLOGY_DEFAULT_RISK = 3  # for units not matching any keyword

# ── Land Cover Risk Classification (USGS LCMAP v1.3 class codes) ──────────────
# LCMAP primary land cover classes used in place of NLCD:
#   1=Developed, 2=Cropland, 3=Grass/Shrub, 4=Tree Cover,
#   5=Water, 6=Wetland, 7=Ice/Snow, 8=Barren
NLCD_RISK = {
    1: 3,     # Developed — moderate (grading/impervious, variable drainage)
    2: 3,     # Cropland — moderate (low vegetation cover, tillage loosens soil)
    3: 3,     # Grass/Shrub — moderate (partial cover, exposed slopes)
    4: 1,     # Tree Cover — low (roots stabilize soil, interception)
    5: None,  # Water — exclude
    6: None,  # Wetland — exclude
    7: None,  # Ice/Snow — exclude
    8: 5,     # Barren — very high (no vegetation, maximum exposure)
}

# ── Fault Distance Risk Classification ───────────────────────────────────────
# List of (min_metres, max_metres_or_None, risk_score)
FAULT_DISTANCE_BREAKS = [
    (0,    100,  5),
    (100,  500,  4),
    (500,  1000, 3),
    (1000, 2000, 2),
    (2000, None, 1),
]

# ── Fire History Risk Classification ─────────────────────────────────────────
# Reference date: January 9, 2018 (Montecito debris flow)
FIRE_REFERENCE_DATE = "2018-01-09"

# (min_years_ago, max_years_ago_or_None, risk_score)
FIRE_RISK_BREAKS = [
    (0,  1,    5),   # burned within last 1 year — vegetation gone, hydrophobic layer
    (1,  3,    4),   # 1–3 years — partial recovery
    (3,  5,    3),   # 3–5 years — moderate recovery
    (5,  10,   2),   # 5–10 years — substantial recovery
    (10, None, 1),   # >10 years or unburned — baseline risk
]
FIRE_UNBURNED_RISK = 1

# Name fragment used to identify the Thomas Fire in the perimeters dataset
THOMAS_FIRE_NAME = "THOMAS"

# ── Random Forest Hyperparameters ────────────────────────────────────────────
RF_PARAMS = {
    "n_estimators":     200,
    "max_depth":        10,
    "min_samples_leaf": 5,
    "n_jobs":           -1,
    "random_state":     RANDOM_SEED,
}
RF_NEGATIVE_RATIO = 5    # negative:positive sample ratio
RF_TEST_SIZE      = 0.30
RF_CV_FOLDS       = 5

# Raster prediction chunk size (rows) — avoids memory errors on county-wide 10m grids
CHUNK_ROWS = 1000

# ── Feature column names (order must match normalized raster stack) ────────────
FEATURE_COLS = [
    "slope", "curvature", "twi",
    "lithology", "landcover",
    "fault_distance", "rainfall",
    "fire_history",
]
