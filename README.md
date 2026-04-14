# Landslide Susceptibility Mapping — Santa Barbara County

A full geospatial machine learning pipeline to produce landslide susceptibility maps for Santa Barbara County, California, using multi-criteria spatial analysis (AHP) and a Random Forest classifier.

---

## Motivation

On **January 9, 2018**, a catastrophic debris flow struck Montecito, California, killing 23 people and destroying over 100 homes. The event was directly triggered by an intense rainfall storm (0.5 inches in 5 minutes) falling on steep slopes recently denuded by the **Thomas Fire** — which burned 281,893 acres across Ventura and Santa Barbara counties in December 2017.

This project builds a reproducible, data-driven susceptibility model that incorporates fire history as a mechanistically central factor, explicitly validates against the Montecito debris flow, and demonstrates how spatial ML can support hazard planning in fire-prone regions.

---

## Study Area

Santa Barbara County, California (EPSG:26911 — UTM Zone 11N)
- Latitude: ~34.0°–35.1°N
- Longitude: ~119.4°–120.6°W
- Area: ~7,090 km²

---

## Methodology

```
Raw Data (8 layers)
      ↓
01_data_prep.py      ← Mosaic, reproject, clip, align to 10 m UTM grid
      ↓
02_terrain_analysis.py ← Slope, aspect, curvature, TWI, flow accumulation
      ↓
03_factor_layers.py  ← Lithology risk, land cover risk, fault distance,
                        precipitation, fire history (post-fire vulnerability)
                        + normalise all layers → [0, 1]
      ↓
04_modeling.py       ← AHP weighted linear combination (CR check)
                        Random Forest (n=200, 5-fold CV, feature importance)
                        Montecito 2018 validation
      ↓
05_visualization.py  ← Static figures + interactive Folium HTML map
```

**Model A — AHP:** Eigenvector-derived weights from an 8×8 pairwise comparison matrix. Consistency Ratio must be < 0.10 for validity.

**Model B — Random Forest:** 200 trees, max_depth=10, trained on historical landslide points (positive class) and stratified random non-landslide points (5:1 ratio). Evaluated with ROC-AUC, 5-fold cross-validation, confusion matrix, and feature importance.

Both outputs classified into 5 classes (Jenks Natural Breaks).

---

## Input Data Sources

| Dataset | Source | URL |
|---|---|---|
| 1m 3DEP DEM | USGS National Map | https://apps.nationalmap.gov/downloader/ |
| California Geology | USGS National Geologic Map | https://mrdata.usgs.gov/geology/state/ |
| NLCD 2021 | USGS / MRLC | https://www.mrlc.gov/data |
| Landslide Inventory | USGS Landslide Hazards Program | https://www.sciencebase.gov/catalog/item/58572c7be4b01fad86d5ff1f |
| Quaternary Faults | USGS Earthquake Hazards Program | https://www.usgs.gov/programs/earthquake-hazards/faults |
| PRISM Precipitation | PRISM Climate Group | https://prism.oregonstate.edu/normals/ |
| CAL FIRE Perimeters | CAL FIRE FRAP | https://www.fire.ca.gov/what-we-do/fire-resource-assessment-program/fire-perimeters |
| Montecito Debris Flow | CGS / USGS publications | See notes below |
| Santa Barbara County Boundary | CA State Geoportal | https://data.ca.gov/dataset/ca-geographic-boundaries |

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
| 1 | Very Low | ![#1a9641](https://via.placeholder.com/12/1a9641/000000?text=+) `#1a9641` |
| 2 | Low | ![#a6d96a](https://via.placeholder.com/12/a6d96a/000000?text=+) `#a6d96a` |
| 3 | Moderate | ![#ffffbf](https://via.placeholder.com/12/ffffbf/000000?text=+) `#ffffbf` |
| 4 | High | ![#fdae61](https://via.placeholder.com/12/fdae61/000000?text=+) `#fdae61` |
| 5 | Very High | ![#d7191c](https://via.placeholder.com/12/d7191c/000000?text=+) `#d7191c` |

---

## Key Outputs

| File | Description |
|---|---|
| `data/outputs/susceptibility_ahp_classified.tif` | 5-class AHP susceptibility raster |
| `data/outputs/susceptibility_rf_probability.tif` | Continuous RF probability surface (0–1) |
| `data/outputs/susceptibility_rf_classified.tif` | 5-class RF susceptibility raster |
| `data/outputs/model_metrics.json` | AUC, CV scores, CR, feature importances |
| `data/outputs/montecito_validation.csv` | Class distribution within debris flow polygon |
| `data/outputs/susceptibility_interactive.html` | Interactive Folium map |
| `data/outputs/figures/` | All static figures (PNG, 200 dpi) |

---

## Environment Setup

### Requirements

- Python 3.10+
- GDAL system library (`brew install gdal` on macOS; `apt install gdal-bin` on Ubuntu)

### Install

```bash
git clone https://github.com/ryangreen6/landslide_susceptability_SB.git
cd landslide_susceptability_SB
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## Data Download

Download all datasets listed in the table above and place them in `data/raw/` following the expected directory structure in `src/config.py`. The pipeline will fail with clear error messages if any required file is missing.

Key paths to populate:
```
data/raw/
├── dem_tiles/              ← 1m 3DEP tiles from USGS National Map
├── sb_county_boundary/     ← County boundary shapefile
├── ca_geology/             ← USGS California geology shapefile
├── nlcd_2021/              ← NLCD 2021 land cover raster (.img)
├── landslide_inventory/    ← USGS landslide inventory shapefile
├── quaternary_faults/      ← USGS Quaternary faults shapefile
├── prism/                  ← PRISM 30-year annual precipitation (.bil)
├── fire_perimeters/        ← CAL FIRE all-years perimeters shapefile
└── montecito_debris_flow/  ← 2018 Montecito debris flow polygon
```

---

## Running the Pipeline

```bash
# Full pipeline
python run_all.py

# Resume from stage 3 (if stages 1–2 already complete)
python run_all.py --start-from 3

# AHP model only (skip Random Forest — much faster)
python run_all.py --skip-rf

# Individual stage
python scripts/04_modeling.py --skip-rf
```

---

## Known Limitations

1. **DEM resolution:** The 1m 3DEP DEM is resampled to 10m. Fine-scale terrain features (individual gullies, small scarps) are averaged out. A 3m resampling would improve shallow-landslide detection at the cost of ~10× memory.

2. **Fire history proxy:** The fire history layer encodes burn age but not burn severity (DNBR). Incorporating Landsat-derived dNBR from MTBS would improve the model — particularly for distinguishing high-severity Thomas Fire areas from low-severity patches.

3. **Landslide inventory completeness:** The USGS inventory is known to be incomplete for Santa Barbara County. Underreporting in the training data biases the classifier and likely underestimates susceptibility in some areas. CGS supplementary inventories should be incorporated when available.

4. **Temporal static model:** The susceptibility map represents a snapshot in time (January 2018 reference). As vegetation recovers from the Thomas Fire and new fires occur, the fire history layer becomes stale. A dynamic updating framework would be needed for operational use.

5. **Debris flow vs. deep-seated failure:** The model conflates multiple landslide types. Debris flows (the Montecito mechanism) are primarily controlled by slope, burn status, and rainfall intensity. Deep-seated landslides are more influenced by geology and groundwater. Type-specific models would improve accuracy.

6. **AHP subjectivity:** AHP weights encode expert judgment and are inherently subjective. The pairwise comparison matrix in `config.py` was designed to reflect the Montecito triggering mechanism; a different study area or focus would warrant different weights.
