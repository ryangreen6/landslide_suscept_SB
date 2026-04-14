"""
05_visualization.py
───────────────────
Stage 5 of the landslide susceptibility pipeline.

Generates all output figures and the interactive Leaflet/Folium HTML map.

Static figures (saved to data/outputs/figures/)
------------------------------------------------
factor_layers_overview.png    — 8-panel normalised input layer grid
susceptibility_map.png        — RF classified susceptibility map
roc_curve.png                 — ROC curve with AUC annotation
feature_importance.png        — RF feature importance horizontal bar chart
montecito_validation.png      — zoomed Montecito area with debris flow overlay

Interactive map (data/outputs/susceptibility_interactive.html)
--------------------------------------------------------------
Folium map with toggleable layers: RF susceptibility, Thomas Fire perimeter,
all fire perimeters, landslide inventory, Montecito debris flow polygon,
county boundary, legend, and dual basemap options.

Usage
-----
    python scripts/05_visualization.py
    python scripts/05_visualization.py --dpi 150
"""

import argparse
import json
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import numpy as np
import rasterio
from rasterio.warp import transform_bounds
from rasterio.crs import CRS

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config, utils

logger = utils.get_logger(__name__)

# ── Colormap for 5-class susceptibility ───────────────────────────────────────
SUSC_CMAP = mcolors.ListedColormap(
    [config.SUSCEPTIBILITY_COLORS[i] for i in range(1, 6)]
)
SUSC_NORM = mcolors.BoundaryNorm([0.5, 1.5, 2.5, 3.5, 4.5, 5.5], SUSC_CMAP.N)
SUSC_LEGEND_PATCHES = [
    mpatches.Patch(color=config.SUSCEPTIBILITY_COLORS[i],
                   label=config.SUSCEPTIBILITY_LABELS[i])
    for i in range(1, 6)
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _imshow_raster(ax, arr, cmap="viridis", vmin=None, vmax=None,
                   extent=None, nodata=config.NODATA):
    """Plot a raster array on a matplotlib Axes, masking nodata.

    Args:
        ax: Matplotlib Axes object.
        arr: 2-D float array.
        cmap: Colormap name or instance.
        vmin: Minimum display value.
        vmax: Maximum display value.
        extent: [left, right, bottom, top] in data units.
        nodata: Value to treat as transparent.
    """
    masked = np.ma.masked_where((arr == nodata) | ~np.isfinite(arr), arr)
    finite = masked.compressed()
    if vmin is None and len(finite):
        vmin = float(np.percentile(finite, 2))
    if vmax is None and len(finite):
        vmax = float(np.percentile(finite, 98))
    ax.imshow(
        masked, cmap=cmap, vmin=vmin, vmax=vmax,
        extent=extent, origin="upper", interpolation="nearest",
    )


def _raster_extent(path: Path):
    """Return [xmin, xmax, ymin, ymax] extent for matplotlib imshow.

    Args:
        path: Path to raster file.

    Returns:
        List [xmin, xmax, ymin, ymax].
    """
    with rasterio.open(path) as src:
        b = src.bounds
    return [b.left, b.right, b.bottom, b.top]


def _raster_to_wgs84_extent(path: Path):
    """Return extent in WGS84 degrees for Folium imageOverlay bounds.

    Args:
        path: Path to raster file.

    Returns:
        [[south, west], [north, east]] suitable for Folium.
    """
    with rasterio.open(path) as src:
        bounds = transform_bounds(src.crs, CRS.from_epsg(4326), *src.bounds)
    w, s, e, n = bounds
    return [[s, w], [n, e]]


# ── Figure 1: Factor Layers Overview ─────────────────────────────────────────

def fig_factor_layers(dpi: int = 200) -> None:
    """Generate an 8-panel overview of all normalised input factor layers.

    Args:
        dpi: Output image resolution.
    """
    factor_info = [
        (config.NORM_SLOPE_TIF,     "Slope",             "YlOrRd"),
        (config.NORM_CURVATURE_TIF, "Curvature",         "RdBu_r"),
        (config.NORM_TWI_TIF,       "TWI",               "Blues"),
        (config.NORM_LITHOLOGY_TIF, "Lithology Risk",    "OrRd"),
        (config.NORM_LANDCOVER_TIF, "Land Cover Risk",   "YlGn"),
        (config.NORM_FAULT_TIF,     "Fault Distance",    "Purples"),
        (config.NORM_PRECIP_TIF,    "Precipitation",     "PuBu"),
        (config.NORM_FIRE_TIF,      "Fire History Risk", "hot_r"),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(20, 10), facecolor="white")
    fig.suptitle(
        "Normalised Factor Layers — Santa Barbara County Landslide Susceptibility",
        fontsize=14, fontweight="bold", y=1.01,
    )

    for ax, (path, title, cmap) in zip(axes.flatten(), factor_info):
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])
        if not path.exists():
            ax.text(0.5, 0.5, "Layer not available",
                    ha="center", va="center", transform=ax.transAxes, color="gray")
            continue
        arr, _ = utils.read_raster(path)
        extent = _raster_extent(path)
        _imshow_raster(ax, arr, cmap=cmap, vmin=0, vmax=1, extent=extent)
        plt.colorbar(
            plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(0, 1)),
            ax=ax, fraction=0.04, pad=0.02, label="Normalised value",
        )

    plt.tight_layout()
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(config.FIG_FACTORS, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figure saved → %s", config.FIG_FACTORS)


# ── Figure 2: RF Susceptibility Map ──────────────────────────────────────────

def fig_susceptibility_map(dpi: int = 200) -> None:
    """Plot the RF classified susceptibility map.

    Args:
        dpi: Output image resolution.
    """
    fig, ax = plt.subplots(figsize=(10, 8), facecolor="white")
    ax.set_title(
        "Landslide Susceptibility — Santa Barbara County\nRandom Forest Classifier",
        fontsize=13, fontweight="bold",
    )
    ax.set_xlabel("Easting (m, UTM 11N)")
    ax.set_ylabel("Northing (m, UTM 11N)")

    path = config.SUSCEPTIBILITY_RF_TIF
    if not path.exists():
        ax.text(0.5, 0.5, "Output not available",
                ha="center", va="center", transform=ax.transAxes, color="gray")
    else:
        arr, _ = utils.read_raster(path)
        extent = _raster_extent(path)
        ax.imshow(
            np.ma.masked_invalid(arr), cmap=SUSC_CMAP, norm=SUSC_NORM,
            extent=extent, origin="upper", interpolation="nearest",
        )

        county_shp = config.COUNTY_UTM_SHP
        if county_shp.exists():
            county = gpd.read_file(county_shp)
            county.boundary.plot(ax=ax, color="black", linewidth=0.8)

        fire_shp = config.PROCESSED_DIR / "fire_perimeters_utm.shp"
        if fire_shp.exists():
            fires = gpd.read_file(fire_shp)
            thomas = fires[fires.apply(
                lambda r: config.THOMAS_FIRE_NAME in str(
                    r.get("FIRE_NAME", r.get("fire_name", ""))
                ).upper(), axis=1
            )]
            if not thomas.empty:
                thomas.boundary.plot(ax=ax, color="darkorange", linewidth=1.5,
                                     label="Thomas Fire (2017)")

    ax.legend(handles=SUSC_LEGEND_PATCHES,
              loc="upper right", fontsize=8, title="Susceptibility",
              framealpha=0.8)

    plt.tight_layout()
    fig.savefig(config.FIG_COMPARISON, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figure saved → %s", config.FIG_COMPARISON)


# ── Figure 3: ROC Curve ───────────────────────────────────────────────────────

def fig_roc_curve(dpi: int = 200) -> None:
    """Plot the ROC curve for the Random Forest model.

    Reads fpr/tpr arrays from the saved model_metrics.json.

    Args:
        dpi: Output image resolution.
    """
    if not config.MODEL_METRICS_JSON.exists():
        logger.warning("model_metrics.json not found — skipping ROC figure")
        return

    with open(config.MODEL_METRICS_JSON) as f:
        metrics = json.load(f)

    roc_data = metrics.get("roc_curve", {})
    fpr = roc_data.get("fpr")
    tpr = roc_data.get("tpr")
    auc = metrics.get("roc_auc")

    if fpr is None or tpr is None:
        logger.warning("ROC curve data not in metrics — skipping ROC figure")
        return

    fig, ax = plt.subplots(figsize=(7, 6), facecolor="white")
    ax.plot(fpr, tpr, color="#d7191c", lw=2,
            label=f"Random Forest  (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random classifier")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve — Landslide Susceptibility Random Forest", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    cv_mean = metrics.get("cv_auc_mean")
    cv_std  = metrics.get("cv_auc_std")
    if cv_mean is not None:
        ax.text(0.6, 0.15,
                f"5-fold CV AUC: {cv_mean:.4f} ± {cv_std:.4f}",
                transform=ax.transAxes, fontsize=10,
                bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))
    plt.tight_layout()
    fig.savefig(config.FIG_ROC, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figure saved → %s", config.FIG_ROC)


# ── Figure 4: Feature Importance ─────────────────────────────────────────────

def fig_feature_importance(dpi: int = 200) -> None:
    """Horizontal bar chart of Random Forest feature importances.

    Args:
        dpi: Output image resolution.
    """
    if not config.MODEL_METRICS_JSON.exists():
        logger.warning("model_metrics.json not found — skipping feature importance figure")
        return

    with open(config.MODEL_METRICS_JSON) as f:
        metrics = json.load(f)

    importances = metrics.get("feature_importances")
    if not importances:
        logger.warning("Feature importances not in metrics")
        return

    names = list(importances.keys())
    vals  = [importances[n] for n in names]
    order = np.argsort(vals)
    names = [names[i] for i in order]
    vals  = [vals[i]  for i in order]

    fig, ax = plt.subplots(figsize=(8, 5), facecolor="white")
    colors = ["#d7191c" if v == max(vals) else "#fdae61" if v >= np.percentile(vals, 75)
              else "#a6d96a" for v in vals]
    bars = ax.barh(names, vals, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Mean Decrease in Impurity (Gini Importance)", fontsize=11)
    ax.set_title("Random Forest Feature Importances\nLandslide Susceptibility Model", fontsize=12)
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
    ax.set_xlim(0, max(vals) * 1.2)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    fig.savefig(config.FIG_IMPORTANCE, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figure saved → %s", config.FIG_IMPORTANCE)


# ── Figure 5: Montecito Validation ───────────────────────────────────────────

def fig_montecito_validation(dpi: int = 200) -> None:
    """Zoomed map of the Montecito area with susceptibility and debris flow overlay.

    Args:
        dpi: Output image resolution.
    """
    debris_shp = config.PROCESSED_DIR / "montecito_debris_utm.shp"
    if not debris_shp.exists():
        logger.warning("Montecito debris flow shapefile not found — skipping figure")
        return

    src_path = config.SUSCEPTIBILITY_RF_TIF
    if not src_path.exists():
        logger.warning("No susceptibility output found — skipping Montecito figure")
        return

    debris = gpd.read_file(debris_shp)
    bounds = debris.total_bounds   # minx, miny, maxx, maxy
    pad = 5000   # 5 km padding
    xlim = [bounds[0] - pad, bounds[2] + pad]
    ylim = [bounds[1] - pad, bounds[3] + pad]

    arr, _ = utils.read_raster(src_path)
    extent = _raster_extent(src_path)

    fig, ax = plt.subplots(figsize=(10, 8), facecolor="white")
    ax.imshow(
        np.ma.masked_invalid(arr), cmap=SUSC_CMAP, norm=SUSC_NORM,
        extent=extent, origin="upper", interpolation="nearest",
    )
    debris.boundary.plot(ax=ax, color="red", linewidth=2.5,
                         label="2018 Montecito Debris Flow")
    debris.plot(ax=ax, color="red", alpha=0.25)

    fire_shp = config.PROCESSED_DIR / "fire_perimeters_utm.shp"
    if fire_shp.exists():
        fires = gpd.read_file(fire_shp)
        thomas = fires[fires.apply(
            lambda r: config.THOMAS_FIRE_NAME in str(
                r.get("FIRE_NAME", r.get("fire_name", ""))
            ).upper(), axis=1
        )]
        if not thomas.empty:
            thomas.boundary.plot(ax=ax, color="darkorange", linewidth=2,
                                 label="Thomas Fire perimeter (2017)")

    ax.set_xlim(xlim); ax.set_ylim(ylim)
    model_label = "RF"
    ax.set_title(
        f"Montecito Validation — {model_label} Susceptibility\n"
        "January 9, 2018 Debris Flow Extent",
        fontsize=12, fontweight="bold",
    )
    ax.set_xlabel("Easting (m, UTM 11N)")
    ax.set_ylabel("Northing (m, UTM 11N)")

    legend_handles = SUSC_LEGEND_PATCHES + [
        mpatches.Patch(color="red",        alpha=0.7,  label="2018 Debris Flow"),
        mpatches.Patch(color="darkorange", alpha=0.7,  label="Thomas Fire (2017)"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8, framealpha=0.9)

    # Montecito validation stats
    val_csv = config.MONTECITO_VALIDATION_CSV
    if val_csv.exists():
        import pandas as pd
        df = pd.read_csv(val_csv, index_col=0)
        key = "rf" if "rf" in df.columns else df.columns[0]
        stats_text = "\n".join(
            f"{row}: {df[key][row]:.1f}%" for row in df.index
        )
        ax.text(0.02, 0.02, f"Class distribution within\ndebris flow extent:\n{stats_text}",
                transform=ax.transAxes, fontsize=8,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    plt.tight_layout()
    fig.savefig(config.FIG_MONTECITO, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figure saved → %s", config.FIG_MONTECITO)


# ── Interactive Folium Map ────────────────────────────────────────────────────

def build_interactive_map() -> None:
    """Build a self-contained Folium interactive HTML map.

    Layers (all toggleable):
        - Esri Satellite / OpenStreetMap basemaps
        - RF susceptibility (5-class, 70% opacity)
        - Thomas Fire perimeter (orange)
        - All historical fire perimeters (yellow, thinner)
        - Historical landslide inventory points
        - Montecito 2018 debris flow polygon
        - Santa Barbara County boundary
    """
    try:
        import folium
        import base64
        import io as _io
        from PIL import Image as PILImage
    except ImportError as exc:
        logger.error("Interactive map requires folium and Pillow: %s", exc)
        return

    logger.info("Building interactive Folium map …")

    # SB County mainland bounds — excludes Channel Islands (~33.9–34.1°N)
    MAINLAND_BOUNDS = [[34.3, -120.7], [35.15, -119.3]]
    centre = [34.75, -120.0]
    m = folium.Map(location=centre, zoom_start=10, tiles=None, min_zoom=9)
    m.fit_bounds(MAINLAND_BOUNDS)
    m.get_root().html.add_child(folium.Element(
        f"<script>document.addEventListener('DOMContentLoaded',function(){{"
        f"map_{m._id}.setMaxBounds([[34.1,-121.1],[35.35,-119.0]]);}});</script>"
    ))
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/"
              "World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery", name="Satellite",
    ).add_to(m)
    folium.TileLayer(tiles="OpenStreetMap", name="OpenStreetMap").add_to(m)

    def _raster_to_overlay(tif_path: Path, layer_name: str,
                           cmap, norm, opacity: float = 0.7, show: bool = True):
        """Convert a classified raster to a base64 PNG Folium ImageOverlay."""
        if not tif_path.exists():
            logger.warning("  Skipping %s (not found)", layer_name)
            return
        arr, _ = utils.read_raster(tif_path)
        # Reproject to WGS84 for Folium
        import rasterio
        from rasterio.warp import reproject as _reproject, Resampling, transform_bounds
        from rasterio.transform import from_bounds as _fb
        from rasterio.crs import CRS as _CRS

        with rasterio.open(tif_path) as src:
            mb = transform_bounds(src.crs, _CRS.from_epsg(4326), *src.bounds)
        out_w = 1024
        lat_scale = np.cos(np.radians((mb[1] + mb[3]) / 2))
        out_h = max(1, int(out_w * (mb[3] - mb[1]) / ((mb[2] - mb[0]) * lat_scale)))
        dst = np.full((out_h, out_w), np.nan, dtype=np.float32)
        with rasterio.open(tif_path) as src:
            _reproject(
                source=rasterio.band(src, 1),
                destination=dst,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=_fb(*mb, out_w, out_h),
                dst_crs=_CRS.from_epsg(4326),
                resampling=Resampling.nearest,
            )

        rgba = cmap(norm(np.where(np.isfinite(dst), dst, np.nan)))
        rgba[~np.isfinite(dst), 3] = 0.0
        rgba[np.isfinite(dst),  3] = opacity
        rgba_u8 = (rgba * 255).astype(np.uint8)
        img = PILImage.fromarray(rgba_u8, mode="RGBA")
        buf = _io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        url = f"data:image/png;base64,{b64}"
        bounds = [[mb[1], mb[0]], [mb[3], mb[2]]]
        folium.raster_layers.ImageOverlay(
            image=url, bounds=bounds, name=layer_name,
            opacity=1.0, show=show,
        ).add_to(m)
        logger.info("  Added raster overlay: %s", layer_name)

    _raster_to_overlay(config.SUSCEPTIBILITY_RF_TIF,
                       "Landslide Risk", SUSC_CMAP, SUSC_NORM, opacity=0.55)

    # ── Risk lookup for address search ────────────────────────────────────────
    risk_bounds = None
    risk_b64 = None
    if config.SUSCEPTIBILITY_RF_TIF.exists():
        import rasterio as _rio
        from rasterio.warp import reproject as _rp, Resampling as _RS, transform_bounds as _tb
        from rasterio.transform import from_bounds as _fb2
        from rasterio.crs import CRS as _CRS2
        _lw, _lh = 512, 512
        _lookup = np.zeros((_lh, _lw), dtype=np.uint8)
        with _rio.open(config.SUSCEPTIBILITY_RF_TIF) as _src:
            _rb = _tb(_src.crs, _CRS2.from_epsg(4326), *_src.bounds)
            _rp(
                source=_rio.band(_src, 1),
                destination=_lookup,
                src_transform=_src.transform,
                src_crs=_src.crs,
                dst_transform=_fb2(*_rb, _lw, _lh),
                dst_crs=_CRS2.from_epsg(4326),
                resampling=_RS.nearest,
                src_nodata=config.NODATA,
                dst_nodata=0,
            )
        _lookup = np.clip(_lookup, 0, 5)
        risk_bounds = _rb
        risk_b64 = base64.b64encode(_lookup.tobytes()).decode()
        logger.info("  Risk lookup grid generated (%dx%d)", _lw, _lh)

    # ── Factor lookups for address search justification ───────────────────────
    _factor_paths = [
        config.NORM_SLOPE_TIF, config.NORM_CURVATURE_TIF, config.NORM_TWI_TIF,
        config.NORM_LITHOLOGY_TIF, config.NORM_LANDCOVER_TIF,
        config.NORM_FAULT_TIF, config.NORM_PRECIP_TIF, config.NORM_FIRE_TIF,
    ]
    _factor_names = ["slope", "curvature", "twi", "lithology", "landcover",
                     "fault_dist", "rainfall", "fire_history"]
    factor_b64s = {}
    if risk_bounds:
        _fw, _fh = 128, 128
        for _fname, _fpath in zip(_factor_names, _factor_paths):
            if not _fpath.exists():
                continue
            _fdst = np.zeros((_fh, _fw), dtype=np.float32)
            with _rio.open(_fpath) as _fs:
                _rp(
                    source=_rio.band(_fs, 1),
                    destination=_fdst,
                    src_transform=_fs.transform,
                    src_crs=_fs.crs,
                    dst_transform=_fb2(*risk_bounds, _fw, _fh),
                    dst_crs=_CRS2.from_epsg(4326),
                    resampling=_RS.bilinear,
                    src_nodata=config.NODATA,
                    dst_nodata=-1.0,
                )
            _fu8 = np.clip(_fdst * 255, 0, 255).astype(np.uint8)
            _fu8[_fdst < 0] = 0
            factor_b64s[_fname] = base64.b64encode(_fu8.tobytes()).decode()
        logger.info("  Factor lookup grids generated (%d factors)", len(factor_b64s))

    # ── Geology lookup for address search ─────────────────────────────────────
    geo_b64 = None
    geo_cat_js_map = "{}"
    if risk_bounds:
        from rasterio.features import rasterize as _rasterize
        from rasterio.transform import from_bounds as _fb3
        geo_shp = config.PROCESSED_DIR / "geology_utm.shp"
        if geo_shp.exists():
            _geo = gpd.read_file(geo_shp).to_crs("EPSG:4326")
            _geo_cats = sorted(_geo["GENERALIZE"].dropna().unique())
            _geo_int = {cat: i + 1 for i, cat in enumerate(_geo_cats)}
            _gw, _gh = 256, 256
            _geo_shapes = [
                (row.geometry, _geo_int.get(row["GENERALIZE"], 0))
                for _, row in _geo.iterrows()
                if row.geometry is not None and not row.geometry.is_empty
            ]
            _geo_arr = _rasterize(
                _geo_shapes, out_shape=(_gh, _gw),
                transform=_fb3(*risk_bounds, _gw, _gh),
                fill=0, dtype=np.uint8,
            )
            geo_b64 = base64.b64encode(_geo_arr.tobytes()).decode()
            geo_cat_js_map = "{" + ",".join(
                f"{v}:'{k}'" for k, v in _geo_int.items()
            ) + "}"
            logger.info("  Geology lookup grid generated")

    # ── Vector layers ─────────────────────────────────────────────────────────
    # County boundary
    county_shp = config.COUNTY_UTM_SHP
    if county_shp.exists():
        grp = folium.FeatureGroup(name="SB County Boundary", show=True)
        folium.GeoJson(
            gpd.read_file(county_shp).to_crs("EPSG:4326").__geo_interface__,
            style_function=lambda _: {"color": "black", "weight": 2, "fillOpacity": 0},
        ).add_to(grp)
        grp.add_to(m)

    # Fire perimeters — last 10 years (2016–present), single merged layer
    fire_shp = config.PROCESSED_DIR / "fire_perimeters_utm.shp"
    if fire_shp.exists():
        fires = gpd.read_file(fire_shp).to_crs("EPSG:4326")
        year_col = next((c for c in ["YEAR_", "FIRE_YEAR", "year_", "YEAR"] if c in fires.columns), None)
        if year_col:
            recent = fires[pd.to_numeric(fires[year_col], errors="coerce").fillna(0) >= 2016]
        else:
            recent = fires
        if not recent.empty:
            grp = folium.FeatureGroup(name="Fire Perimeters (2016–Present)", show=False)
            tooltip_fields = [f for f in [year_col, "FIRE_NAME"] if f and f in recent.columns]
            folium.GeoJson(
                recent.__geo_interface__,
                style_function=lambda _: {"color": "darkorange", "weight": 1.2,
                                          "fillColor": "orange", "fillOpacity": 0.35},
                tooltip=folium.GeoJsonTooltip(
                    fields=tooltip_fields,
                    aliases=[f.replace("_", " ").title() + ":" for f in tooltip_fields],
                    sticky=True,
                ) if tooltip_fields else None,
            ).add_to(grp)
            grp.add_to(m)
            logger.info("  Added fire perimeters layer (%d fires since 2016)", len(recent))

    # Fault lines
    fault_shp = config.PROCESSED_DIR / "faults_utm.shp"
    if fault_shp.exists():
        grp = folium.FeatureGroup(name="Fault Lines", show=False)
        folium.GeoJson(
            gpd.read_file(fault_shp).to_crs("EPSG:4326").__geo_interface__,
            style_function=lambda _: {"color": "red", "weight": 1.5, "fillOpacity": 0},
        ).add_to(grp)
        grp.add_to(m)
        logger.info("  Added fault lines layer")

    # Geology
    GEO_COLORS = {
        "Sedimentary, clastic":                    "#c2a05a",
        "Unconsolidated, undifferentiated":         "#e8d5a3",
        "Igneous, volcanic":                        "#c1440e",
        "Metamorphic, serpentinite":               "#4f7942",
        "Water":                                   "#4a90d9",
        "Metamorphic, volcanic":                   "#8b7355",
        "Igneous, intrusive":                      "#7b2d8b",
        "Metamorphic, undifferentiated":            "#6b8e23",
        "Igneous and Metamorphic, undifferentiated":"#a0522d",
    }
    geo_shp = config.PROCESSED_DIR / "geology_utm.shp"
    if geo_shp.exists():
        geo_gdf = gpd.read_file(geo_shp).to_crs("EPSG:4326")
        grp = folium.FeatureGroup(name="Geology", show=False)
        for _, row in geo_gdf.iterrows():
            cat = row.get("GENERALIZE", "")
            color = GEO_COLORS.get(cat, "#aaaaaa")
            folium.GeoJson(
                row.geometry.__geo_interface__,
                style_function=lambda _, c=color: {
                    "color": c, "weight": 0.5,
                    "fillColor": c, "fillOpacity": 0.6,
                },
                tooltip=cat,
            ).add_to(grp)
        grp.add_to(m)
        logger.info("  Added geology layer (%d features)", len(geo_gdf))

    # Montecito 2018 debris flow
    debris_shp = config.PROCESSED_DIR / "montecito_debris_utm.shp"
    if debris_shp.exists():
        debris = gpd.read_file(debris_shp).to_crs("EPSG:4326")
        grp = folium.FeatureGroup(name="Montecito 2018 Debris Flow", show=False)
        for _, row in debris.iterrows():
            folium.GeoJson(
                row.geometry.__geo_interface__,
                style_function=lambda _: {"color": "red", "weight": 3,
                                          "fillColor": "red", "fillOpacity": 0.55},
                tooltip="January 9, 2018 Montecito Debris Flow",
                popup=folium.Popup(
                    "<b>Montecito Debris Flow</b><br>"
                    "Date: January 9, 2018<br>"
                    "Triggered by: Intense rainfall on Thomas Fire burn scar",
                    max_width=250,
                ),
            ).add_to(grp)
        grp.add_to(m)

    # Landslide inventory
    ls_shp = config.PROCESSED_DIR / "landslide_inventory_utm.shp"
    if ls_shp.exists():
        ls = gpd.read_file(ls_shp).to_crs("EPSG:4326")
        grp = folium.FeatureGroup(name="Landslide Inventory", show=False)
        for _, row in ls.iterrows():
            lon, lat = row.geometry.centroid.x, row.geometry.centroid.y
            date = str(row.get("date", row.get("Date", row.get("YEAR", "Unknown"))))
            ls_type = str(row.get("type", row.get("Type", row.get("LSTYPE", "Unknown"))))
            folium.CircleMarker(
                location=[lat, lon], radius=4,
                color="darkred", fill=True, fill_color="red", fill_opacity=0.55,
                popup=folium.Popup(
                    f"<b>Landslide</b><br>Date: {date}<br>Type: {ls_type}",
                    max_width=200,
                ),
                tooltip="Historical landslide",
            ).add_to(grp)
        grp.add_to(m)
        logger.info("  Added landslide inventory (%d points)", len(ls))

    # ── Dynamic Legend ────────────────────────────────────────────────────────
    risk_rows_flat = "".join(
        f'<span style="background:{config.SUSCEPTIBILITY_COLORS[i]};display:inline-block;'
        f'width:14px;height:14px;margin-right:4px;border:1px solid #999;vertical-align:middle;"></span>'
        f'{config.SUSCEPTIBILITY_LABELS[i]}<br>'
        for i in range(1, 6)
    )
    geo_rows_flat = "".join(
        f'<span style="background:{color};display:inline-block;width:14px;height:14px;'
        f'margin-right:4px;border:1px solid #999;vertical-align:middle;"></span>{cat}<br>'
        for cat, color in GEO_COLORS.items()
    )
    _leg_sections = {
        "Landslide Risk": f'<b style="font-size:14px">Landslide Risk</b><br>{risk_rows_flat}',
        "Fire Perimeters (2016\u2013Present)": '<hr style="margin:4px 0"><span style="color:darkorange;font-weight:bold">\u2501\u2501</span> Fire Perimeters (2016\u2013Present)<br>',
        "Fault Lines": '<span style="color:red;font-weight:bold">\u2501\u2501</span> Fault Lines<br>',
        "Montecito 2018 Debris Flow": '<span style="background:red;display:inline-block;width:14px;height:14px;margin-right:4px;border:1px solid #999;vertical-align:middle;opacity:0.7;"></span>Montecito 2018 Debris Flow<br>',
        "Geology": f'<hr style="margin:4px 0"><b>Geology</b><br>{geo_rows_flat}',
        "Landslide Inventory": '<span style="display:inline-block;width:10px;height:10px;background:red;border-radius:50%;margin-right:4px;border:1px solid #900;vertical-align:middle;"></span>Landslide Inventory<br>',
    }
    legend_sections_js = "var _legendSections=" + json.dumps(_leg_sections) + ";"
    dynamic_legend_html = (
        '<div id="map-legend" style="position:fixed;bottom:30px;right:10px;z-index:9999;'
        'background:rgba(255,255,255,0.92);border-radius:8px;padding:12px 16px;'
        'box-shadow:0 2px 8px rgba(0,0,0,0.3);font-family:\'Segoe UI\',sans-serif;'
        'font-size:13px;max-height:80vh;overflow-y:auto;"></div>\n'
        f'<script>\n{legend_sections_js}\n'
        f'var _activeLayers={{"Landslide Risk":true}};\n'
        'function _rebuildLegend(){var el=document.getElementById(\'map-legend\');if(!el)return;'
        'var order=["Landslide Risk","Fire Perimeters (2016\u2013Present)","Fault Lines",'
        '"Montecito 2018 Debris Flow","Geology","Landslide Inventory"];'
        'var html=\'\';order.forEach(function(k){if(_activeLayers[k]&&_legendSections[k])html+=_legendSections[k];});'
        'el.innerHTML=html||\'<i style="color:#888">No active layers</i>\';}\n'
        f'document.addEventListener(\'DOMContentLoaded\',function(){{'
        f'var mapObj=map_{m._id};'
        'mapObj.on(\'overlayadd\',function(e){_activeLayers[e.name]=true;_rebuildLegend();});'
        'mapObj.on(\'overlayremove\',function(e){_activeLayers[e.name]=false;_rebuildLegend();});'
        '_rebuildLegend();}});\n</script>'
    )
    m.get_root().html.add_child(folium.Element(dynamic_legend_html))

    folium.LayerControl(collapsed=False).add_to(m)

    # ── Address search ────────────────────────────────────────────────────────
    map_id = m._id
    risk_js = ""
    if risk_bounds and risk_b64:
        w, s, e, n = risk_bounds
        fd_js = ";".join(
            f"_fd['{k}']=new Uint8Array(atob('{v}').split('').map(function(c){{return c.charCodeAt(0);}}));"
            for k, v in factor_b64s.items()
        )
        risk_js = (
            f"var _rb={{w:{w},s:{s},e:{e},n:{n},pw:512,ph:512}};"
            f"var _rv=new Uint8Array(atob('{risk_b64}').split('').map(function(c){{return c.charCodeAt(0);}}) );"
            f"var _rl={{1:'Very Low',2:'Low',3:'Moderate',4:'High',5:'Very High'}};"
            f"var _rc={{1:'#1a9641',2:'#a6d96a',3:'#b8960c',4:'#fdae61',5:'#d7191c'}};"
            f"var _fn={{slope:'slope gradient',curvature:'terrain curvature',twi:'topographic wetness',"
            f"lithology:'geologic instability',landcover:'vegetation cover',"
            f"fault_dist:'fault proximity',rainfall:'annual rainfall',fire_history:'fire history'}};"
            f"var _fd={{}}; {fd_js}"
            f"function _gr(lat,lon){{"
            f"var c=Math.floor((lon-_rb.w)/(_rb.e-_rb.w)*_rb.pw);"
            f"var r=Math.floor((_rb.n-lat)/(_rb.n-_rb.s)*_rb.ph);"
            f"if(c<0||c>=_rb.pw||r<0||r>=_rb.ph)return 0;"
            f"return _rv[r*_rb.pw+c];}}"
            f"function _gf(lat,lon){{"
            f"var c=Math.floor((lon-_rb.w)/(_rb.e-_rb.w)*128);"
            f"var r=Math.floor((_rb.n-lat)/(_rb.n-_rb.s)*128);"
            f"if(c<0||c>=128||r<0||r>=128)return null;"
            f"var v={{}};for(var k in _fd){{v[k]=_fd[k][r*128+c]/255;}}return v;}}"
            f"function _justify(v,risk){{"
            f"if(!v||!_rl[risk])return '';"
            f"var pairs=Object.keys(v).map(function(k){{return[k,v[k]];}});"
            f"pairs.sort(function(a,b){{return b[1]-a[1];}});"
            f"var top=pairs.filter(function(p){{return p[1]>0.1;}}).slice(0,2);"
            f"if(!top.length)return '';"
            f"function adj(x){{return x>0.67?'high':x>0.33?'moderate':'low';}}"
            f"var rw=_rl[risk].toLowerCase()+' risk';"
            f"if(top.length===1){{"
            f"var s=adj(top[0][1])+' '+_fn[top[0][0]];"
            f"return s.charAt(0).toUpperCase()+s.slice(1)+' contributes to '+rw+' at this location.';}}"
            f"var s1=adj(top[0][1])+' '+_fn[top[0][0]];"
            f"var s2=adj(top[1][1])+' '+_fn[top[1][0]];"
            f"return(s1.charAt(0).toUpperCase()+s1.slice(1))+' and '+s2+' contribute to '+rw+' at this location.';}}"
        )
    if geo_b64:
        risk_js += (
            f"var _gb=new Uint8Array(atob('{geo_b64}').split('').map(function(c){{return c.charCodeAt(0);}}) );"
            f"var _gm={geo_cat_js_map};"
            f"function _getGeo(lat,lon){{"
            f"var c=Math.floor((lon-_rb.w)/(_rb.e-_rb.w)*256);"
            f"var r=Math.floor((_rb.n-lat)/(_rb.n-_rb.s)*256);"
            f"if(c<0||c>=256||r<0||r>=256)return null;"
            f"var v=_gb[r*256+c];return v?_gm[v]:null;}}"
        )
    search_html = f"""
<div id="geocoder" style="position:fixed;top:12px;left:50%;transform:translateX(-50%);
    z-index:9999;background:white;padding:8px 12px;border-radius:6px;
    box-shadow:0 2px 8px rgba(0,0,0,0.3);display:flex;gap:8px;align-items:center;
    font-family:'Segoe UI',sans-serif;">
  <input id="addr-input" type="text" placeholder="Search address..."
         style="width:290px;padding:6px 8px;border:1px solid #ccc;border-radius:4px;font-size:13px;"
         onkeydown="if(event.key==='Enter')_sa()">
  <button onclick="_sa()"
          style="padding:6px 14px;background:#333;color:white;border:none;border-radius:4px;cursor:pointer;font-size:13px;">
    Search
  </button>
</div>
<script>
{risk_js}
var _sm=null;
function _sa(){{
  var q=document.getElementById('addr-input').value.trim();
  if(!q)return;
  fetch('https://nominatim.openstreetmap.org/search?q='+encodeURIComponent(q)+'&format=json&limit=1',
    {{headers:{{'Accept-Language':'en-US,en'}}}})
    .then(function(r){{return r.json();}})
    .then(function(data){{
      if(!data.length){{alert('Address not found.');return;}}
      var lat=parseFloat(data[0].lat),lon=parseFloat(data[0].lon);
      var name=data[0].display_name;
      var risk=typeof _gr==='function'?_gr(lat,lon):0;
      var fvals=typeof _gf==='function'?_gf(lat,lon):null;
      var just=typeof _justify==='function'?_justify(fvals,risk):'';
      var geoName=typeof _getGeo==='function'?_getGeo(lat,lon):null;
      var rHtml=risk>0?'<b style="color:'+_rc[risk]+'">'+_rl[risk]+'</b>':'Outside study area';
      var justHtml=just?'<br><span style="font-size:12px;color:#444;font-style:italic">'+just+'</span>':'';
      var geoHtml=geoName?'<br>Geology: '+geoName:'';
      if(_sm)map_{map_id}.removeLayer(_sm);
      _sm=L.marker([lat,lon]).addTo(map_{map_id});
      _sm.bindPopup('<div style="font-family:Segoe UI,sans-serif;font-size:13px;max-width:300px"><b>'+name+'</b><br><br>Landslide Risk: '+rHtml+justHtml+geoHtml+'</div>').openPopup();
      map_{map_id}.setView([lat,lon],14);
    }})
    .catch(function(){{alert('Geocoding failed. Check your connection.');}});
}}
</script>
"""
    m.get_root().html.add_child(folium.Element(search_html))

    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    m.save(str(config.INTERACTIVE_HTML))
    logger.info("Interactive map saved → %s", config.INTERACTIVE_HTML)


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="Stage 5: Visualization for SB landslide susceptibility pipeline"
    )
    parser.add_argument("--dpi", type=int, default=200,
                        help="Figure output resolution in DPI (default: 200)")
    parser.add_argument("--no-interactive", action="store_true",
                        help="Skip interactive Folium map (faster for testing)")
    return parser.parse_args()


def main() -> None:
    """Run all visualizations."""
    args = parse_args()
    utils.ensure_dirs()

    logger.info("Generating factor layers overview …")
    fig_factor_layers(args.dpi)

    logger.info("Generating susceptibility map …")
    fig_susceptibility_map(args.dpi)

    logger.info("Generating ROC curve …")
    fig_roc_curve(args.dpi)

    logger.info("Generating feature importance chart …")
    fig_feature_importance(args.dpi)

    logger.info("Generating Montecito validation figure …")
    fig_montecito_validation(args.dpi)

    if not args.no_interactive:
        build_interactive_map()

    logger.info("=== Stage 5 complete. Figures in %s ===", config.FIGURES_DIR)


if __name__ == "__main__":
    main()
