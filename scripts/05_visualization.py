"""
05_visualization.py
───────────────────
Stage 5 of the landslide susceptibility pipeline.

Generates the interactive Folium HTML map only.

Output: data/outputs/susceptibility_interactive.html

Usage
-----
    python scripts/05_visualization.py
"""

import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config, utils

logger = utils.get_logger(__name__)

SUSC_CMAP = mcolors.ListedColormap(
    [config.SUSCEPTIBILITY_COLORS[i] for i in range(1, 6)]
)
SUSC_NORM = mcolors.BoundaryNorm([0.5, 1.5, 2.5, 3.5, 4.5, 5.5], SUSC_CMAP.N)


def build_interactive_map() -> None:
    try:
        import folium
        import base64
        import io as _io
        from PIL import Image as PILImage
    except ImportError as exc:
        logger.error("Interactive map requires folium and Pillow: %s", exc)
        return

    logger.info("Building interactive Folium map …")

    MAINLAND_BOUNDS = [[34.3, -120.7], [35.15, -119.3]]
    centre = [34.75, -120.0]
    m = folium.Map(location=centre, zoom_start=10, tiles=None, min_zoom=9)
    m.fit_bounds(MAINLAND_BOUNDS)
    m.get_root().html.add_child(folium.Element(
        f"<script>document.addEventListener('DOMContentLoaded',function(){{"
        f"map_{m._id}.setMaxBounds([[33.3,-121.2],[35.35,-118.8]]);}});</script>"
    ))
    folium.TileLayer(tiles="OpenStreetMap", name="OpenStreetMap").add_to(m)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/"
              "World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery", name="Satellite",
    ).add_to(m)

    _county_wgs84 = None
    if config.COUNTY_UTM_SHP.exists():
        _county_wgs84 = gpd.read_file(config.COUNTY_UTM_SHP).to_crs("EPSG:4326")

    def _raster_to_overlay(tif_path, layer_name, cmap, norm, opacity=0.7, show=True, clip_to_county=False):
        if not tif_path.exists():
            logger.warning("  Skipping %s (not found)", layer_name)
            return
        from rasterio.warp import reproject as _reproject, Resampling, transform_bounds
        from rasterio.transform import from_bounds as _fb
        from rasterio.crs import CRS as _CRS
        _MERC = _CRS.from_epsg(3857)
        _WGS84 = _CRS.from_epsg(4326)
        with rasterio.open(tif_path) as src:
            mb_m = transform_bounds(src.crs, _MERC, *src.bounds)
        out_w = 2048
        merc_w = mb_m[2] - mb_m[0]
        merc_h = mb_m[3] - mb_m[1]
        out_h = max(1, int(out_w * merc_h / merc_w))
        dst = np.full((out_h, out_w), np.nan, dtype=np.float32)
        out_transform = _fb(*mb_m, out_w, out_h)
        with rasterio.open(tif_path) as src:
            _reproject(
                source=rasterio.band(src, 1),
                destination=dst,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=out_transform,
                dst_crs=_MERC,
                resampling=Resampling.nearest,
            )
        mb = transform_bounds(_MERC, _WGS84, *mb_m)

        nodata_mask = ~np.isfinite(dst) | (dst == config.NODATA)
        if clip_to_county and _county_wgs84 is not None:
            from rasterio.features import rasterize as _rasterize
            from shapely.geometry import mapping as _mapping
            _county_merc = _county_wgs84.to_crs("EPSG:3857")
            county_raster = _rasterize(
                [(_mapping(g), 1) for g in _county_merc.geometry],
                out_shape=(out_h, out_w),
                transform=out_transform,
                fill=0, dtype=np.uint8,
            )
            nodata_mask = nodata_mask | (county_raster == 0)
        rgba = cmap(norm(np.where(~nodata_mask, dst, np.nan)))
        rgba[nodata_mask,  3] = 0.0
        rgba[~nodata_mask, 3] = opacity

        from scipy.ndimage import distance_transform_edt
        dist = distance_transform_edt(~nodata_mask)
        rgba[..., 3] = rgba[..., 3] * np.clip(dist / 4.0, 0.0, 1.0)

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

    _raster_to_overlay(config.SUSCEPTIBILITY_WLC_TIF,
                       "Landslide Risk", SUSC_CMAP, SUSC_NORM, opacity=0.55)

    _raster_to_overlay(config.NORM_SOIL_TIF, "Soil Erodibility",
                       plt.get_cmap("copper"), mcolors.Normalize(0, 1),
                       opacity=0.6, show=False)
    _raster_to_overlay(config.NORM_PRECIP_TIF, "Precipitation Intensity",
                       plt.get_cmap("Blues"), mcolors.Normalize(0, 1),
                       opacity=0.6, show=False, clip_to_county=True)

    risk_bounds = None
    risk_b64 = None
    if config.SUSCEPTIBILITY_WLC_TIF.exists():
        import rasterio as _rio
        from rasterio.warp import reproject as _rp, Resampling as _RS, transform_bounds as _tb
        from rasterio.transform import from_bounds as _fb2
        from rasterio.crs import CRS as _CRS2
        _lw, _lh = 512, 512
        _lookup = np.zeros((_lh, _lw), dtype=np.uint8)
        with _rio.open(config.SUSCEPTIBILITY_WLC_TIF) as _src:
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

    _factor_paths = [
        config.NORM_SLOPE_TIF, config.NORM_CURVATURE_TIF, config.NORM_TWI_TIF,
        config.NORM_LITHOLOGY_TIF, config.NORM_LANDCOVER_TIF,
        config.NORM_FAULT_TIF, config.NORM_PRECIP_TIF,
        config.NORM_NDVI_TIF, config.NORM_SOIL_TIF,
    ]
    _factor_names = ["slope", "curvature", "twi", "lithology", "landcover",
                     "fault_dist", "rainfall", "ndvi", "soil"]
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

    MACRO_CLASS_COLORS = {
        "Sedimentary": "#c2a05a",
        "Igneous":     "#c1440e",
        "Metamorphic": "#4f7942",
        "Water":       "#4a90d9",
    }
    GEO_COLORS = {
        "Sedimentary, clastic":                     "#c2a05a",
        "Unconsolidated, undifferentiated":          "#e8d5a3",
        "Igneous, volcanic":                         "#c1440e",
        "Metamorphic, serpentinite":                "#4f7942",
        "Water":                                    "#4a90d9",
        "Metamorphic, volcanic":                    "#8b7355",
        "Igneous, intrusive":                       "#7b2d8b",
        "Metamorphic, undifferentiated":             "#6b8e23",
        "Igneous and Metamorphic, undifferentiated": "#a0522d",
    }

    county_shp = config.COUNTY_UTM_SHP
    if county_shp.exists():
        grp = folium.FeatureGroup(name="SB County Boundary", show=True)
        folium.GeoJson(
            gpd.read_file(county_shp).to_crs("EPSG:4326").__geo_interface__,
            style_function=lambda _: {"color": "black", "weight": 2, "fillOpacity": 0},
        ).add_to(grp)
        grp.add_to(m)

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

    fault_shp = config.PROCESSED_DIR / "faults_utm.shp"
    if fault_shp.exists():
        faults_gdf = gpd.read_file(fault_shp).to_crs("EPSG:4326")
        grp = folium.FeatureGroup(name="Fault Lines", show=False)
        for _, row in faults_gdf.iterrows():
            def _clean(val):
                s = str(val).strip() if pd.notna(val) else ""
                return "" if s.lower() in ("nan", "none", "", "unspecified") else s
            name = _clean(row.get("fault_name", ""))
            section = _clean(row.get("section_na", ""))
            slip = _clean(row.get("slip_sense", ""))
            ltype = _clean(row.get("linetype", ""))
            age = _clean(row.get("age", ""))
            rate = _clean(row.get("slip_rate", ""))
            title = (f"{name} — {section}" if section and section.lower() != name.lower() else name) or "Unnamed fault"
            lines = [f"<b>{title}</b>"]
            if slip:  lines.append(f"Slip sense: {slip}")
            if ltype: lines.append(f"Constraint: {ltype}")
            if age:   lines.append(f"Age: {age}")
            if rate:  lines.append(f"Slip rate: {rate}")
            tip = "<br>".join(lines)
            folium.GeoJson(
                row.geometry.__geo_interface__,
                style_function=lambda _: {"color": "red", "weight": 3, "fillOpacity": 0},
                tooltip=folium.Tooltip(tip, style="font-size:13px;"),
            ).add_to(grp)
        grp.add_to(m)
        logger.info("  Added fault lines layer (%d segments)", len(faults_gdf))

    geo_shp = config.PROCESSED_DIR / "geology_utm.shp"
    geo_cache = config.PROCESSED_DIR / "geology_macrostrat_cache.json"
    if geo_shp.exists():
        import concurrent.futures as _cf
        import requests as _req
        geo_gdf = gpd.read_file(geo_shp).to_crs("EPSG:4326")
        if geo_cache.exists():
            with open(geo_cache) as _f:
                _mac = json.load(_f)
            logger.info("  Macrostrat geology cache loaded (%d entries)", len(_mac))
        else:
            logger.info("  Fetching Macrostrat geology for %d polygons ...", len(geo_gdf))
            def _qmac(args):
                idx, lat, lon = args
                try:
                    r = _req.get(
                        "https://macrostrat.org/api/geologic_units/map",
                        params={"lat": lat, "lng": lon, "format": "geojson"},
                        timeout=10,
                    )
                    feats = r.json().get("success", {}).get("data", {}).get("features", [])
                    if feats:
                        p = feats[0]["properties"]
                        return str(idx), {"name": p.get("name", ""), "color": p.get("color", ""), "lith": p.get("lith", "")}
                except Exception:
                    pass
                return str(idx), None
            centroids = geo_gdf.geometry.centroid
            _args = [(i, c.y, c.x) for i, c in enumerate(centroids)]
            _mac = {}
            with _cf.ThreadPoolExecutor(max_workers=10) as ex:
                for k, v in ex.map(_qmac, _args):
                    _mac[k] = v
            with open(geo_cache, "w") as _f:
                json.dump(_mac, _f)
            logger.info("  Macrostrat geology cache built")
        grp = folium.FeatureGroup(name="Geology", show=False)
        for idx, (_, row) in enumerate(geo_gdf.iterrows()):
            info = _mac.get(str(idx))
            sgmc_cat = row.get("GENERALIZE", "")
            mac_name = (info or {}).get("name", "")
            mac_mismatch = (
                mac_name.lower() == "landslide deposit"
                and "landslide" not in sgmc_cat.lower()
            )
            if info and info.get("color") and not mac_mismatch:
                color = info["color"]
                tip = mac_name or sgmc_cat
                if info.get("lith"):
                    tip += f" ({info['lith']})"
            else:
                color = GEO_COLORS.get(sgmc_cat, "#aaaaaa")
                tip = sgmc_cat
            folium.GeoJson(
                row.geometry.__geo_interface__,
                style_function=lambda _, c=color: {"color": c, "weight": 0.5, "fillColor": c, "fillOpacity": 0.6},
                tooltip=folium.Tooltip(tip, style="max-width:960px;white-space:normal;word-wrap:break-word;font-size:13px;"),
            ).add_to(grp)
        grp.add_to(m)
        logger.info("  Added geology layer (%d features)", len(geo_gdf))

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

    ls_shp = config.PROCESSED_DIR / "landslide_inventory_utm.shp"
    if ls_shp.exists():
        ls = gpd.read_file(ls_shp).to_crs("EPSG:4326")
        grp = folium.FeatureGroup(name="Historical Landslides", show=False)
        for _, row in ls.iterrows():
            lon, lat = row.geometry.centroid.x, row.geometry.centroid.y
            date = str(row.get("date", row.get("Date", row.get("YEAR", "Unknown"))))
            ls_type = str(row.get("type", row.get("Type", row.get("LSTYPE", "Unknown"))))
            folium.CircleMarker(
                location=[lat, lon], radius=4,
                color="#cc3300", fill=True, fill_color="#ff5500", fill_opacity=0.9,
                popup=folium.Popup(
                    f"<b>Landslide</b><br>Date: {date}<br>Type: {ls_type}",
                    max_width=200,
                ),
                tooltip="Historical landslide",
            ).add_to(grp)
        grp.add_to(m)
        logger.info("  Added landslide inventory (%d points)", len(ls))

    risk_rows_flat = "".join(
        f'<span style="background:{config.SUSCEPTIBILITY_COLORS[i]};display:inline-block;'
        f'width:14px;height:14px;margin-right:4px;border:1px solid #999;vertical-align:middle;"></span>'
        f'{config.SUSCEPTIBILITY_LABELS[i]}<br>'
        for i in range(1, 6)
    )
    geo_rows_flat = "".join(
        f'<span style="background:{color};display:inline-block;width:14px;height:14px;'
        f'margin-right:4px;border:1px solid #999;vertical-align:middle;"></span>{cls}<br>'
        for cls, color in MACRO_CLASS_COLORS.items()
    )
    _leg_sections = {
        "Landslide Risk": f'<b style="font-size:14px">Landslide Risk</b><br>{risk_rows_flat}',
        "Fire Perimeters (2016\u2013Present)": '<hr style="margin:4px 0"><span style="color:darkorange;font-weight:bold">\u2501\u2501</span> Fire Perimeters (2016\u2013Present)<br>',
        "Fault Lines": '<span style="color:red;font-weight:bold">\u2501\u2501</span> Fault Lines<br><span style="font-size:11px;color:#888;font-style:italic;display:block;margin-top:2px;">Hover over a fault line for details</span>',
        "Montecito 2018 Debris Flow": '<span style="background:red;display:inline-block;width:14px;height:14px;margin-right:4px;border:1px solid #999;vertical-align:middle;opacity:0.7;"></span>Montecito 2018 Debris Flow<br>',
        "Geology": '<hr style="margin:4px 0"><b>Geology</b><br><span style="font-size:11px;color:#888;font-style:italic;display:block;max-width:140px;word-wrap:break-word;">Hover over an area for geological details</span><br>',
        "Historical Landslides": '<span style="display:inline-block;width:10px;height:10px;background:#ff5500;border-radius:50%;margin-right:4px;border:1px solid #cc3300;vertical-align:middle;"></span>Historical Landslides<br>',
        "Soil Erodibility": '<hr style="margin:4px 0"><b>Soil Erodibility</b><br><span style="font-size:11px;color:#888">Low \u2192 High (copper scale)</span><br>',
        "Precipitation Intensity": '<b>Precipitation Intensity</b><br><span style="font-size:11px;color:#888">Low \u2192 High (100-yr/24-hr, blues)</span><br>',
    }
    legend_sections_js = "var _legendSections=" + json.dumps(_leg_sections) + ";"
    dynamic_legend_html = (
        '<div id="map-legend" style="position:fixed;bottom:30px;right:10px;z-index:9999;'
        'background:rgba(45,45,45,0.97);border-radius:8px;padding:12px 16px;'
        'box-shadow:0 2px 8px rgba(0,0,0,0.3);font-family:\'Segoe UI\',sans-serif;'
        'font-size:15px;max-height:80vh;overflow-y:auto;"></div>\n'
        f'<script>\n{legend_sections_js}\n'
        f'var _activeLayers={{"Landslide Risk":true}};\n'
        'function _rebuildLegend(){var el=document.getElementById(\'map-legend\');if(!el)return;'
        'var order=["Landslide Risk","Soil Erodibility","Precipitation Intensity","Fire Perimeters (2016\u2013Present)","Fault Lines",'
        '"Montecito 2018 Debris Flow","Geology","Historical Landslides"];'
        'var html=\'\';order.forEach(function(k){if(_activeLayers[k]&&_legendSections[k])html+=_legendSections[k];});'
        'el.innerHTML=html||\'<i style="color:#888">No active layers</i>\';}\n'
        '_rebuildLegend();\n'
        f'window.addEventListener(\'load\',function(){{'
        f'var mapObj=map_{m._id};'
        'mapObj.on(\'overlayadd\',function(e){_activeLayers[e.name]=true;_rebuildLegend();if(e.name===\'Geology\')e.layer.bringToFront();});'
        'mapObj.on(\'overlayremove\',function(e){_activeLayers[e.name]=false;_rebuildLegend();});'
        '});\n</script>'
    )
    m.get_root().html.add_child(folium.Element(dynamic_legend_html))

    folium.LayerControl(collapsed=False).add_to(m)

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
            f"fault_dist:'fault proximity',rainfall:'annual rainfall',"
            f"ndvi:'vegetation density (NDVI)',soil:'soil erodibility'}};"
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
    search_html = f"""
<div id="geocoder" style="position:fixed;top:12px;left:50%;transform:translateX(-50%);
    z-index:9999;background:rgba(45,45,45,0.97);padding:8px 12px;border-radius:6px;
    box-shadow:0 2px 8px rgba(0,0,0,0.3);display:flex;gap:8px;align-items:center;
    font-family:'Segoe UI',sans-serif;">
  <input id="addr-input" type="text" placeholder="Search address..." onfocus="var _i=this;setTimeout(function(){{_i.select();}},0)"
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
window.addEventListener('load',function(){{
  map_{map_id}.on('click',function(){{if(_sm){{map_{map_id}.removeLayer(_sm);_sm=null;}}}});
}});
function _showPin(lat,lon,name,rHtml,justHtml,geoHtml){{
  if(_sm)map_{map_id}.removeLayer(_sm);
  _sm=L.marker([lat,lon]).addTo(map_{map_id});
  _sm.bindPopup('<div style="font-family:Segoe UI,sans-serif;font-size:13px;max-width:300px"><b>'+name+'</b><br><br>Landslide Risk: '+rHtml+justHtml+geoHtml+'</div>').openPopup();
  map_{map_id}.setView([lat,lon],14);
}}
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
      var rHtml=risk>0?'<b style="color:'+_rc[risk]+'">'+_rl[risk]+'</b>':'Outside study area';
      var justHtml=just?'<br><span style="font-size:12px;color:#444;font-style:italic">'+just+'</span>':'';
      fetch('https://macrostrat.org/api/geologic_units/map?lat='+lat+'&lng='+lon+'&response=long')
        .then(function(gr){{return gr.json();}})
        .then(function(gd){{
          var units=(gd.success&&gd.success.data)?gd.success.data:[];
          var geoHtml=units.length?'<br>Geology: '+units[0].name:'';
          _showPin(lat,lon,name,rHtml,justHtml,geoHtml);
        }})
        .catch(function(){{_showPin(lat,lon,name,rHtml,justHtml,'');}});
    }})
    .catch(function(){{alert('Geocoding failed. Check your connection.');}});
}}
</script>
"""
    m.get_root().html.add_child(folium.Element(search_html))

    datasources_html = """
<div id="ds-btn" onclick="dsToggle()"
     style="position:fixed;bottom:30px;left:10px;z-index:9999;background:rgba(45,45,45,0.97);
            border-radius:8px;padding:7px 12px;box-shadow:0 2px 8px rgba(0,0,0,0.3);
            cursor:pointer;font-family:'Segoe UI',sans-serif;font-size:15px;font-weight:600;
            user-select:none;">
  &#9432; Info
</div>
<div id="ds-panel"
     style="display:none;position:fixed;bottom:30px;left:10px;z-index:9999;
            background:rgba(45,45,45,0.97);border-radius:8px;padding:12px 16px;
            box-shadow:0 2px 8px rgba(0,0,0,0.3);font-family:'Segoe UI',sans-serif;
            font-size:13px;max-height:85vh;overflow-y:auto;max-width:340px;min-width:260px;">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">
    <b style="font-size:14px">Data Sources</b>
    <span onclick="dsToggle()" style="cursor:pointer;color:#666;font-size:18px;line-height:1;">&times;</span>
  </div>
  <p style="margin:0 0 10px;color:#444;font-size:12px;line-height:1.5;">All source datasets are listed below. Factor weights were assigned based on published landslide susceptibility literature and validated against the January 9, 2018 Montecito debris flow event. For referenced landslide literature, see the GitHub repository for this project.</p>
  <b style="font-size:12px;text-transform:uppercase;letter-spacing:0.5px;color:#333;">WLC Data Sources</b>
  <ul style="margin:4px 0 12px;padding-left:16px;line-height:1.85;color:#222;">
    <li><b>Slope</b> &mdash; Derived from USGS 3DEP 1/3 arc-second DEM &mdash; <span style="color:#555;">Weight: 28%</span></li>
    <li><b>Lithology / Geology</b> &mdash; USGS SGMC, supplemented by Macrostrat API &mdash; <span style="color:#555;">Weight: 18%</span></li>
    <li><b>Topographic Wetness Index (TWI)</b> &mdash; Derived from USGS 3DEP 1/3 arc-second DEM &mdash; <span style="color:#555;">Weight: 12%</span></li>
    <li><b>Fault Distance</b> &mdash; USGS Quaternary Fault and Fold Database &mdash; <span style="color:#555;">Weight: 12%</span></li>
    <li><b>Land Cover</b> &mdash; USGS GAP/LANDFIRE 2011 via Microsoft Planetary Computer &mdash; <span style="color:#555;">Weight: 8%</span></li>
    <li><b>NDVI</b> &mdash; Sentinel-2 L2A median composite via Microsoft Planetary Computer &mdash; <span style="color:#555;">Weight: 8%</span></li>
    <li><b>Soil Erodibility</b> &mdash; USDA NRCS gSSURGO (K-factor &amp; hydrologic group) &mdash; <span style="color:#555;">Weight: 8%</span></li>
    <li><b>Terrain Curvature</b> &mdash; Derived from USGS 3DEP 1/3 arc-second DEM &mdash; <span style="color:#555;">Weight: 3%</span></li>
    <li><b>Precipitation</b> &mdash; NOAA Atlas 14 Vol. 1 (100-yr / 24-hr AMS) &mdash; <span style="color:#555;">Weight: 3%</span></li>
  </ul>
  <b style="font-size:12px;text-transform:uppercase;letter-spacing:0.5px;color:#333;">Additional Data Sources</b>
  <ul style="margin:4px 0 0;padding-left:16px;line-height:1.85;color:#222;">
    <li><b>County Boundary</b> &mdash; U.S. Census Bureau TIGER/Line Shapefiles</li>
    <li><b>Fire Perimeters</b> &mdash; CAL FIRE Fire and Resource Assessment Program (FRAP)</li>
    <li><b>Historical Landslides</b> &mdash; California Geological Survey (CGS) California Landslide Inventory (CaLSI)</li>
    <li><b>Montecito Debris Flow</b> &mdash; USGS / CGS 2018 Thomas Fire debris flow mapping</li>
  </ul>
  <p style="margin:12px 0 0;font-size:11px;color:#666;text-align:center;">&copy; Ryan Green, 2026</p>
</div>
<script>
function dsToggle() {
  var p = document.getElementById('ds-panel');
  var b = document.getElementById('ds-btn');
  if (p.style.display === 'none') { p.style.display = 'block'; b.style.display = 'none'; }
  else { p.style.display = 'none'; b.style.display = 'block'; }
}
</script>
"""
    m.get_root().html.add_child(folium.Element(datasources_html))

    welcome_html = """
<div id="info-banner"
     style="position:fixed;top:62px;left:50%;transform:translateX(-50%);z-index:9998;
            background:rgba(45,45,45,0.97);border-radius:8px;padding:14px 18px;
            max-width:480px;width:90%;box-shadow:0 2px 10px rgba(0,0,0,0.5);
            font-family:'Segoe UI',sans-serif;">
  <b style="display:block;font-size:15px;color:#ddd;margin-bottom:8px;text-align:center;">
    Landslide Risk Assessment Tool for Santa Barbara County
  </b>
  <p style="margin:0;font-size:13px;color:#bbb;line-height:1.6;">
    This tool models landslide susceptibility across Santa Barbara County using a Weighted Linear Combination (WLC) of nine geospatial factors (Info tab for details).
  </p>
</div>
<script>
window.addEventListener('load', function() {
  var geocoder = document.getElementById('geocoder');
  var banner = document.getElementById('info-banner');
  if (geocoder && banner) {
    banner.style.width = geocoder.offsetWidth + 'px';
    banner.style.maxWidth = 'none';
  }
  setTimeout(function() {
    document.addEventListener('click', function() {
      if (banner) banner.style.display = 'none';
    }, {once: true});
  }, 300);
});
</script>
"""
    m.get_root().html.add_child(folium.Element(welcome_html))

    darkmode_html = """
<style>
#map-legend, #ds-panel, #ds-btn, #dl-btn, #geocoder, #info-banner,
.leaflet-control-layers, .leaflet-bar a {
  background:rgba(45,45,45,0.97) !important;
  box-shadow:0 2px 8px rgba(0,0,0,0.6) !important;
}
#map-legend, #ds-panel, #ds-btn, #info-banner,
.leaflet-control-layers, .leaflet-control-layers label,
.leaflet-control-layers span { color:#ddd !important; }
#map-legend * { color:#ddd !important; }
#map-legend span[style*="darkorange"] { color:darkorange !important; }
#map-legend span[style*="color:red"] { color:#ff6b6b !important; }
#ds-panel b, #ds-panel li, #ds-panel p, #ds-panel span,
#info-banner b, #info-banner p { color:#ddd !important; }
.leaflet-bar a { color:#ddd !important; border-bottom-color:#555 !important; }
.leaflet-bar a:hover { background:rgba(70,70,70,0.97) !important; }
.leaflet-control-layers { border:1px solid #555 !important; }
.leaflet-control-layers button {
  background:#3a3a3a !important; border-color:#555 !important; color:#ddd !important;
}
#geocoder input { background:#333 !important; border-color:#555 !important; color:#ddd !important; }
#geocoder button { background:#555 !important; color:#ddd !important; }
hr { border-color:#555 !important; }
.leaflet-tooltip {
  background:rgba(45,45,45,0.97) !important;
  color:#ddd !important;
  border-color:#555 !important;
  box-shadow:0 2px 6px rgba(0,0,0,0.5) !important;
}
.leaflet-tooltip-left::before  { border-left-color:#555 !important; }
.leaflet-tooltip-right::before { border-right-color:#555 !important; }
.leaflet-tooltip-top::before   { border-top-color:#555 !important; }
.leaflet-tooltip-bottom::before{ border-bottom-color:#555 !important; }
.leaflet-popup-content-wrapper, .leaflet-popup-tip {
  background:rgba(45,45,45,0.97) !important;
  color:#ddd !important;
  box-shadow:0 2px 8px rgba(0,0,0,0.6) !important;
}
.leaflet-popup-content b { color:#fff !important; }
.leaflet-popup-close-button { color:#aaa !important; }
</style>
"""
    m.get_root().html.add_child(folium.Element(darkmode_html))

    default_layers_html = """
<script>
window.addEventListener('load', function() {
  var _defaults = {
    'Landslide Risk': true,
    'Soil Erodibility': false,
    'Precipitation Intensity': false,
    'SB County Boundary': true,
    'Fire Perimeters (2016\u2013Present)': false,
    'Fault Lines': false,
    'Geology': false,
    'Montecito 2018 Debris Flow': false,
    'Historical Landslides': false,
  };
  var ctrl = document.querySelector('.leaflet-control-layers');
  if (!ctrl) return;

  var dlBtn = document.createElement('div');
  dlBtn.id = 'dl-btn';
  dlBtn.style.cssText = 'position:fixed;top:10px;right:10px;z-index:9999;'
    + 'background:rgba(45,45,45,0.97);border-radius:8px;padding:7px 12px;'
    + 'box-shadow:0 2px 8px rgba(0,0,0,0.3);cursor:pointer;'
    + 'font-family:Segoe UI,sans-serif;font-size:15px;font-weight:600;'
    + 'color:#ddd;user-select:none;display:block;';
  dlBtn.innerHTML = '&#9776; Data Layers';
  dlBtn.onclick = function() { ctrl.style.display = ''; dlBtn.style.display = 'none'; };
  document.body.appendChild(dlBtn);
  ctrl.style.display = 'none';

  var list = ctrl.querySelector('.leaflet-control-layers-list');
  if (list) {
    var hdr = document.createElement('div');
    hdr.style.cssText = 'display:flex;justify-content:space-between;align-items:center;'
      + 'margin-bottom:8px;padding-bottom:6px;border-bottom:1px solid #555;';
    var htitle = document.createElement('b');
    htitle.textContent = 'Data Layers';
    htitle.style.cssText = 'font-size:14px;color:#ddd;';
    var hclose = document.createElement('span');
    hclose.innerHTML = '&times;';
    hclose.style.cssText = 'cursor:pointer;color:#aaa;font-size:18px;line-height:1;margin-left:12px;';
    hclose.onclick = function() { ctrl.style.display = 'none'; dlBtn.style.display = 'block'; };
    hdr.appendChild(htitle);
    hdr.appendChild(hclose);
    list.insertBefore(hdr, list.firstChild);
  }

  var overlaysDiv = ctrl.querySelector('.leaflet-control-layers-overlays');
  if (overlaysDiv) {
    var allLabels = Array.from(overlaysDiv.querySelectorAll('label'));
    var topLabels = allLabels.filter(function(l) {
      var s = l.querySelector('span'); return s && !!_defaults[s.textContent.trim()];
    });
    var restLabels = allLabels.filter(function(l) {
      var s = l.querySelector('span'); return !s || !_defaults[s.textContent.trim()];
    });
    topLabels.concat(restLabels).forEach(function(l) { overlaysDiv.appendChild(l); });
  }

  var sep = document.createElement('div');
  sep.style.cssText = 'border-top:1px solid #555;margin:10px 6px 6px;';
  ctrl.appendChild(sep);
  var btn = document.createElement('button');
  btn.textContent = 'Clear Layers';
  btn.style.cssText = 'display:block;width:calc(100% - 12px);margin:0 6px 6px;padding:5px 0;'
    + 'background:#3a3a3a;border:1px solid #555;border-radius:4px;cursor:pointer;'
    + 'font-family:Segoe UI,sans-serif;font-size:12px;color:#ddd;';
  btn.onmouseenter = function(){this.style.background='#4a4a4a';};
  btn.onmouseleave = function(){this.style.background='#3a3a3a';};
  btn.onclick = function() {
    var overlays = ctrl.querySelectorAll('.leaflet-control-layers-overlays label');
    overlays.forEach(function(label) {
      var span = label.querySelector('span');
      if (!span) return;
      var name = span.textContent.trim();
      var cb = label.querySelector('input[type=checkbox]');
      if (!cb) return;
      var want = !!_defaults[name];
      if (cb.checked !== want) cb.click();
    });
    var bases = ctrl.querySelectorAll('.leaflet-control-layers-base label');
    bases.forEach(function(label) {
      var span = label.querySelector('span');
      if (span && span.textContent.trim() === 'Satellite') {
        var rb = label.querySelector('input[type=radio]');
        if (rb && !rb.checked) rb.click();
      }
    });
  };
  ctrl.appendChild(btn);
});
</script>
"""
    m.get_root().html.add_child(folium.Element(default_layers_html))

    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    m.save(str(config.INTERACTIVE_HTML))
    logger.info("Interactive map saved → %s", config.INTERACTIVE_HTML)


def main() -> None:
    utils.ensure_dirs()
    build_interactive_map()
    logger.info("=== Stage 5 complete ===")


if __name__ == "__main__":
    main()
