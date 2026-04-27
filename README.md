# Landslide Susceptibility Mapping — Santa Barbara County

A geospatial pipeline that models landslide susceptibility across Santa Barbara County, California using logistic regression trained on recorded landslide locations and eight terrain, climate, and land condition factors.

---

## Motivation

On January 9, 2018, a catastrophic debris flow struck Montecito, California, killing 23 people and destroying over 100 homes. The event was triggered by intense rainfall falling on steep slopes recently burned by the Thomas Fire, which scorched 281,893 acres across Ventura and Santa Barbara counties in December 2017.

This project builds a reproducible, data-driven susceptibility model that incorporates fire history as a core factor, validates against an independent landslide inventory, and demonstrates how spatial analysis can support hazard planning in fire-prone regions.

---

## Study Area

Santa Barbara County, California (EPSG:26911 — UTM Zone 11N)
- Latitude: ~34.0°–35.1°N
- Longitude: ~119.4°–120.6°W
- Area: ~7,090 km²

---

## Methodology

```
Raw Data (8 factor layers)
      ↓
01_data_prep.py        ← Mosaic DEM tiles, reproject, clip, align to 10 m UTM grid
      ↓
02_terrain_analysis.py ← Slope, aspect, curvature, TWI, flow accumulation
      ↓
03_factor_layers.py    ← Lithology risk, land cover risk, precipitation,
                          NDVI, soil erodibility, burn severity
                          + normalize all layers → [0, 1]
      ↓
04_modeling.py         ← Logistic regression (presence / pseudo-absence)
                          Spatial block cross-validation
                          Percentile-based classification
                          External validation against 2023 Santa Ynez inventory
      ↓
05_visualization.py    ← Interactive Folium HTML map
```

### Model

The model uses logistic regression in a presence/pseudo-absence framework. Training presences are 926 polygon centroids from the USGS National Landslide Inventory v3 (Confidence ≥ 3, meaning Likely or High confidence). Pseudo-absences are ~4,600 points drawn randomly from land pixels at least 200 m away from any known landslide location, at a 5:1 ratio to presences.

The modeled probability of landslide occurrence is classified into five levels using percentile breaks on the county-wide land pixel distribution (bottom 30% / 30–50 / 50–70 / 70–85 / top 15%). Topographic Wetness Index (TWI) is excluded due to collinearity with slope. Fault proximity is excluded as a trigger factor rather than an inherent terrain property.

### Model Factors

| Factor | Data Source | LR Influence |
|---|---|---|
| Slope | USGS 3DEP 10-m DEM | 27.5% |
| Precipitation | NOAA Atlas 14 (100-yr/24-hr AMS) | 22.4% |
| Land Cover | USGS GAP/LANDFIRE 2011 | 17.1% |
| Terrain Curvature | USGS 3DEP 10-m DEM | 9.3% |
| Burn Severity | CAL FIRE FRAP, recency-weighted (3-yr decay) | 9.1% |
| Lithology | USGS State Geologic Map Compilation | 7.6% |
| Soil Erodibility | USDA NRCS gSSURGO | 6.7% |
| NDVI | ESA Sentinel-2 L2A via Microsoft Planetary Computer | <1% |

### Burn Severity

Each fire polygon is assigned a recency weight using exponential decay: `exp(-(2024 - fire_year) / 3)`. This gives more recent fires a higher weight, reflecting the elevated debris flow risk that persists for several years after a burn. Pixels are assigned the weight of the most recent fire that burned them; unburned pixels receive a weight of 0.

---

## Performance

- **Cross-validation AUC: 0.719 ± 0.256** (7 spatial blocks, leave-one-block-out). High variance reflects spatial clustering of landslides in the Santa Ynez Mountains.
- **Hold-out hit rate: 49.7%** of withheld NLI centroids (random 20%, n = 185) fell in the High or Very High class.
- **External validation: 92.8%** of 8,323 landslide points from the January 9, 2023 Santa Ynez atmospheric river storm fell in High or Very High — this dataset was not used in training.

---

## Input Data Sources

| Dataset | Source |
|---|---|
| 1/3 arc-second 3DEP DEM | USGS National Map |
| California Geology (SGMC) | USGS National Geologic Map Database |
| Geology (supplemental) | Macrostrat API |
| GAP/LANDFIRE Land Cover 2011 | USGS via Microsoft Planetary Computer |
| USGS National Landslide Inventory v3 | USGS National Landslide Hazards Program |
| USGS 2023 Santa Ynez Mountains Inventory | Thomas et al., 2025, USGS data release |
| Quaternary Faults | USGS Earthquake Hazards Program |
| NOAA Atlas 14 Precipitation | NOAA (100-yr / 24-hr AMS) |
| CAL FIRE Perimeters | CAL FIRE FRAP |
| Sentinel-2 NDVI | ESA via Microsoft Planetary Computer |
| Soil Erodibility (gSSURGO) | USDA NRCS |
| Santa Barbara County Boundary | U.S. Census Bureau TIGER/Line |

---

## Data Download

Download the datasets above and place them in `data/raw/` following the directory structure defined in `src/config.py`. The pipeline will fail with clear error messages if any required file is missing.

```
data/raw/
├── dem_tiles/                    ← 1/3 arc-sec 3DEP tiles from USGS National Map
├── sb_county_boundary/           ← County boundary shapefile
├── ca_geology/                   ← USGS SGMC California geology shapefile
├── landslide_inventory/          ← USGS NLI v3 shapefile
├── quaternary_faults/            ← USGS Quaternary faults shapefile
├── atlas_14/                     ← NOAA Atlas 14 100-yr/24-hr precipitation raster
├── fire_perimeters/              ← CAL FIRE all-years perimeters shapefile
├── gSSURGO_CA.gdb                ← USDA NRCS gSSURGO geodatabase
└── santa_ynez_mountains_2023/    ← USGS 2023 landslide inventory CSV (external validation)
```

---

## Citations

Thomas, M.A., et al., 2025, Landslide, soil, and vegetation measurements following an atmospheric river storm on January 9, 2023, in the Santa Ynez Mountains, California, USA: U.S. Geological Survey data release. https://doi.org/10.5066/P133CHYQ

Kean, J.W., et al., 2019, Inundation, flow dynamics, and damage in the 9 January 2018 Montecito debris-flow event, California, USA: Geosphere, v. 15, no. 4, p. 1140–1163. https://doi.org/10.1130/GES02040.1

---

*Copyright Ryan Green, 2026*
