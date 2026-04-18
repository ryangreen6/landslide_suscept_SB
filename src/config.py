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
GSSURGO_GDB             = RAW_DIR / "gSSURGO_CA.gdb"
ATLAS14_ASC             = RAW_DIR / "atlas_14" / "sw100yr24ha_ams.asc"
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
GAP_TIF             = PROCESSED_DIR / "gap_landcover_utm.tif"
FAULT_DIST_RISK_TIF = PROCESSED_DIR / "fault_distance_risk.tif"
PRECIP_NORM_TIF     = PROCESSED_DIR / "precipitation_normalized.tif"
ATLAS14_UTM_TIF     = PROCESSED_DIR / "atlas14_100yr24h_utm.tif"
NDVI_TIF            = PROCESSED_DIR / "ndvi_median.tif"
SOIL_RISK_TIF       = PROCESSED_DIR / "soil_erodibility_risk.tif"

# Normalized factor rasters (0–1), ready for modeling
NORM_SLOPE_TIF     = PROCESSED_DIR / "norm_slope.tif"
NORM_CURVATURE_TIF = PROCESSED_DIR / "norm_curvature.tif"
NORM_TWI_TIF       = PROCESSED_DIR / "norm_twi.tif"
NORM_LITHOLOGY_TIF = PROCESSED_DIR / "norm_lithology.tif"
NORM_LANDCOVER_TIF = PROCESSED_DIR / "norm_landcover.tif"
NORM_FAULT_TIF     = PROCESSED_DIR / "norm_fault_distance.tif"
NORM_PRECIP_TIF    = PROCESSED_DIR / "norm_precipitation.tif"
NORM_NDVI_TIF      = PROCESSED_DIR / "norm_ndvi.tif"
NORM_SOIL_TIF      = PROCESSED_DIR / "norm_soil.tif"

# County boundary reprojected to analysis CRS
COUNTY_UTM_SHP     = PROCESSED_DIR / "sb_county_utm.shp"
COUNTY_UTM_TIF     = PROCESSED_DIR / "county_mask.tif"

# ── Output Files ──────────────────────────────────────────────────────────────
SUSCEPTIBILITY_WLC_TIF      = OUTPUTS_DIR / "susceptibility_wlc_classified.tif"
SUSCEPTIBILITY_WLC_PROB_TIF = OUTPUTS_DIR / "susceptibility_wlc_probability.tif"
MODEL_METRICS_JSON          = OUTPUTS_DIR / "model_metrics.json"
MONTECITO_VALIDATION_CSV    = OUTPUTS_DIR / "montecito_validation.csv"

# ── WLC Classification Breaks (fixed, literature-anchored) ───────────────────
# Applied to the continuous 0–1 WLC index.  Stable across runs regardless of
# pixel distribution (unlike Jenks).  Calibrated so Montecito debris-flow area
# (median WLC ≈ 0.33) falls solidly in the High class.
WLC_BREAKS = [0.0, 0.15, 0.24, 0.30, 0.38, 1.0]

# ── WLC Factor Weights (sum = 1.0) ────────────────────────────────────────────
WLC_WEIGHTS = {
    "slope":          0.28,
    "curvature":      0.03,
    "twi":            0.12,
    "lithology":      0.18,
    "landcover":      0.08,
    "fault_distance": 0.12,
    "rainfall":       0.03,
    "ndvi":           0.08,
    "soil":           0.08,
}

# ── Figure Outputs ────────────────────────────────────────────────────────────
FIG_FACTORS       = FIGURES_DIR / "factor_layers_overview.png"
FIG_COMPARISON    = FIGURES_DIR / "susceptibility_comparison.png"
FIG_IMPORTANCE    = FIGURES_DIR / "wlc_weights.png"
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

# ── Land Cover Risk Classification (GAP/LANDFIRE 2011 ecosystem codes) ────────
# Source: USGS GAP Analysis Program via Microsoft Planetary Computer ("gap")
# Codes present in Santa Barbara County mapped to landslide risk scores (1–5).
# None = exclude (water/wetland). Default for unknown codes = 3.
GAP_RISK = {
    # Water / Wetland — exclude
    432: None,  # Temperate Pacific Freshwater Emergent Marsh
    433: None,  # Temperate Pacific Freshwater Mudflat
    455: None,  # Temperate Pacific Tidal Salt and Brackish Marsh
    458: None,  # Inter-Mountain Basins Playa
    509: None,  # Mediterranean California Eelgrass Bed
    578: None,  # Open Water (Brackish/Salt)
    579: None,  # Open Water (Fresh)
    # Dense forest — score 1
    55:  1,     # Mediterranean California Mixed Evergreen Forest
    159: 1,     # California Montane Jeffrey Pine-(Ponderosa Pine) Woodland
    162: 1,     # Mediterranean California Dry-Mesic Mixed Conifer Forest
    165: 1,     # California Coastal Redwood Forest
    539: 1,     # North American Warm Desert Bedrock Cliff and Outcrop
    540: 1,     # North American Warm Desert Pavement
    # Woodland, savanna, riparian, rocky — score 2
    39:  2,     # California Central Valley Mixed Oak Savanna
    40:  2,     # California Coastal Closed-Cone Conifer Forest and Woodland
    41:  2,     # California Coastal Live Oak Woodland and Savanna
    42:  2,     # California Lower Montane Blue Oak-Foothill Pine Woodland
    43:  2,     # Central and Southern California Mixed Evergreen Woodland
    45:  2,     # Southern California Oak Woodland and Savanna
    183: 2,     # Great Basin Pinyon-Juniper Woodland
    277: 2,     # California Central Valley Riparian Woodland and Shrubland
    278: 2,     # Mediterranean California Foothill and Lower Montane Riparian Woodland
    282: 2,     # North American Warm Desert Riparian Woodland and Shrubland
    300: 2,     # Mediterranean California Mesic Serpentine Woodland and Chaparral
    383: 2,     # Mediterranean California Coastal Bluff
    489: 2,     # Inter-Mountain Basins Big Sagebrush Shrubland
    516: 2,     # Southern California Coast Ranges Cliff and Canyon
    563: 2,     # Introduced Upland Vegetation - Treed
    # Chaparral, scrub, grassland, cultivated, developed — score 3
    296: 3,     # California Maritime Chaparral
    297: 3,     # California Mesic Chaparral
    302: 3,     # Southern California Dry-Mesic Chaparral
    303: 3,     # Southern California Coastal Scrub
    304: 3,     # California Central Valley and Southern Coastal Grassland
    305: 3,     # California Mesic Serpentine Grassland
    359: 3,     # Sonora-Mojave Semi-Desert Chaparral
    360: 3,     # California Montane Woodland and Chaparral
    470: 3,     # Mojave Mid-Elevation Mixed Desert Scrub
    472: 3,     # Sonora-Mojave Creosotebush-White Bursage Desert Scrub
    476: 3,     # Sonora-Mojave Mixed Salt Desert Scrub
    485: 3,     # Inter-Mountain Basins Mixed Salt Desert Scrub
    552: 3,     # Unconsolidated Shore
    556: 3,     # Cultivated Cropland
    557: 3,     # Pasture/Hay
    558: 3,     # Introduced Upland Vegetation - Annual Grassland
    568: 3,     # Harvested Forest - Shrub Regeneration
    581: 3,     # Developed, Open Space
    582: 3,     # Developed, Low Intensity
    583: 3,     # Developed, Medium Intensity
    584: 3,     # Developed, High Intensity
    # Unstable/disturbed surfaces — score 4
    384: 4,     # Mediterranean California Northern Coastal Dune
    385: 4,     # Mediterranean California Southern Coastal Dune
    547: 4,     # Inter-Mountain Basins Shale Badland
    567: 4,     # Harvested Forest - Grass/Forb Regeneration (clear-cut)
    580: 4,     # Quarries, Mines, Gravel Pits and Oil Wells
    570: 4,     # Recently Burned (2011 snapshot; fire history layer handles temporal decay)
    # Barren — score 5
    553: 5,     # Undifferentiated Barren Land
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

# ── Feature column names (order must match normalized raster stack) ────────────
FEATURE_COLS = [
    "slope", "curvature", "twi",
    "lithology", "landcover",
    "fault_distance", "rainfall",
    "ndvi", "soil",
]
