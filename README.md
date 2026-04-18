# Landslide Susceptibility Mapping — Santa Barbara County

A geospatial pipeline to produce landslide susceptibility maps for Santa Barbara County, California, using a Weighted Linear Combination (WLC) of nine terrain, geology, and environmental factors.

---

## Motivation

On **January 9, 2018**, a catastrophic debris flow struck Montecito, California, killing 23 people and destroying over 100 homes. The event was directly triggered by an intense rainfall storm (0.5 inches in 5 minutes) falling on steep slopes recently denuded by the **Thomas Fire** — which burned 281,893 acres across Ventura and Santa Barbara counties in December 2017.

This project builds a reproducible, data-driven susceptibility model that incorporates fire history as a mechanistically central factor, explicitly validates against the Montecito debris flow, and demonstrates how spatial analysis can support hazard planning in fire-prone regions.

---

## Study Area

Santa Barbara County, California (EPSG:26911 — UTM Zone 11N)
- Latitude: ~34.0°–35.1°N
- Longitude: ~119.4°–120.6°W
- Area: ~7,090 km²

---

## Methodology

```
Raw Data (9 layers)
      ↓
01_data_prep.py        ← Mosaic, reproject, clip, align to 10 m UTM grid
      ↓
02_terrain_analysis.py ← Slope, aspect, curvature, TWI, flow accumulation
      ↓
03_factor_layers.py    ← Lithology risk, land cover risk, fault distance,
                          precipitation, NDVI, soil erodibility
                          + normalise all layers → [0, 1]
      ↓
04_modeling.py         ← Weighted Linear Combination (WLC)
                          Fixed classification breaks (literature-calibrated)
                          Montecito 2018 validation
      ↓
05_visualization.py    ← Static figures + interactive Folium HTML map
```

**WLC Model:** Nine normalized factor rasters are multiplied by literature-derived weights and summed to produce a continuous susceptibility index (0–1). The index is classified into five levels using fixed breaks calibrated against the January 9, 2018 Montecito debris flow event. Weights were assigned based on published landslide susceptibility literature.

| Factor | Weight |
|---|---|
| Slope | 28% |
| Lithology | 18% |
| Topographic Wetness Index (TWI) | 12% |
| Fault Distance | 12% |
| Land Cover | 8% |
| NDVI | 8% |
| Soil Erodibility | 8% |
| Terrain Curvature | 3% |
| Precipitation | 3% |

---

## Input Data Sources

| Dataset | Source |
|---|---|
| 1/3 arc-second 3DEP DEM | USGS National Map |
| California Geology (SGMC) | USGS National Geologic Map Database |
| Geology (supplemental) | Macrostrat API |
| GAP/LANDFIRE Land Cover 2011 | USGS via Microsoft Planetary Computer |
| Historical Landslide Inventory | California Geological Survey (CaLSI) |
| Quaternary Faults | USGS Earthquake Hazards Program |
| NOAA Atlas 14 Precipitation | NOAA (100-yr / 24-hr AMS) |
| CAL FIRE Perimeters | CAL FIRE FRAP |
| Sentinel-2 NDVI | Microsoft Planetary Computer |
| Soil Erodibility (gSSURGO) | USDA NRCS |
| Montecito Debris Flow | CGS / USGS publications |
| Santa Barbara County Boundary | U.S. Census Bureau TIGER/Line |

### Montecito Debris Flow Polygon

The January 9, 2018 debris flow extent polygon can be obtained from:
- USGS Open-File Report 2019–1204 (Kean et al.)
- California Geological Survey Special Report 237
- Supplement to Warrick et al. (2019) *Nature Geoscience*

Place the shapefile at: `data/raw/montecito_debris_flow/montecito_2018_debris_flow.shp`

---

## Fire History Note

The **Thomas Fire** (December 4–January 12, 2017–2018) must be present in the CAL FIRE perimeters dataset as `FIRE_NAME = "THOMAS"`. Relative to the January 9, 2018 reference date, it falls in the "< 1 year" burn category — the highest risk class (score = 5) — correctly reflecting the active hydrophobic soil layer and near-total vegetation loss that made the Montecito slopes so vulnerable.

---

## Susceptibility Classes

| Class | Label | Color |
|---|---|---|
| 1 | Very Low | `#1a9641` |
| 2 | Low | `#a6d96a` |
| 3 | Moderate | `#ffffbf` |
| 4 | High | `#fdae61` |
| 5 | Very High | `#d7191c` |

---

## Data Download

Download all datasets listed in the table above and place them in `data/raw/` following the expected directory structure in `src/config.py`. The pipeline will fail with clear error messages if any required file is missing.

Key paths to populate:
```
data/raw/
├── dem_tiles/              ← 1/3 arc-sec 3DEP tiles from USGS National Map
├── sb_county_boundary/     ← County boundary shapefile
├── ca_geology/             ← USGS SGMC California geology shapefile
├── landslide_inventory/    ← CGS CaLSI landslide inventory shapefile
├── quaternary_faults/      ← USGS Quaternary faults shapefile
├── atlas_14/               ← NOAA Atlas 14 100-yr/24-hr precipitation raster
├── fire_perimeters/        ← CAL FIRE all-years perimeters shapefile
├── gSSURGO_CA.gdb          ← USDA NRCS gSSURGO geodatabase
└── montecito_debris_flow/  ← 2018 Montecito debris flow polygon
```

---

*Copyright Ryan Green, 2026*
