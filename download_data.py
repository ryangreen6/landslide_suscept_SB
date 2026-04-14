"""
download_data.py
────────────────
Downloads all raw input datasets for the Santa Barbara County landslide
susceptibility pipeline.  Run this before 01_data_prep.py.

What is automated
-----------------
  ✓ Santa Barbara County boundary  (Census TIGER)
  ✓ USGS 3DEP 1/3 arc-second DEM  (TNM Access API)
  ✓ California geology shapefile   (USGS mrdata)
  ✓ CAL FIRE fire perimeters       (CNRA Open Data)
  ✓ USGS Landslide inventory       (ScienceBase API)
  ✓ USGS Quaternary faults         (ScienceBase API)
  ✓ Montecito 2018 debris flow     (ScienceBase API — Kean et al. 2019)
  ✓ NLCD 2021 land cover           (MRLC WCS endpoint)
  ✓ PRISM 30-yr annual precip      (PRISM FTP mirror)

Usage
-----
    python download_data.py
    python download_data.py --skip-dem   # skip DEM (large files)
    python download_data.py --only dem   # download DEM tiles only
"""

import argparse
import io
import json
import logging
import os
import re
import shutil
import sys
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import requests
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src import config

logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("download_data")

# Santa Barbara County bounding box with margin
BBOX = (-120.65, 33.85, -119.25, 35.25)   # west, south, east, north
W, S, E, N = BBOX
SB_COUNTY_FIPS = "083"   # Santa Barbara FIPS within California (FIPS 06)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _download_file(url: str, dest: Path, label: str = "",
                   chunk_size: int = 1 << 20,
                   headers: Optional[dict] = None) -> bool:
    """Download a file with a tqdm progress bar.

    Args:
        url: URL to download.
        dest: Destination file path.
        label: Human-readable label for the progress bar.
        chunk_size: Streaming chunk size in bytes.
        headers: Optional HTTP headers dict.

    Returns:
        True on success; False on failure.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        logger.info("  Already exists: %s — skipping", dest.name)
        return True

    try:
        h = headers or {}
        h.setdefault("User-Agent", "Mozilla/5.0 (compatible; landslide-pipeline/1.0)")
        resp = requests.get(url, headers=h, stream=True, timeout=120)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))

        with open(dest, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True,
            desc=f"  {label or dest.name[:40]}", leave=False,
        ) as bar:
            for chunk in resp.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                bar.update(len(chunk))

        logger.info("  ✓ Downloaded %s (%.1f MB)", dest.name, dest.stat().st_size / 1e6)
        return True

    except Exception as exc:
        logger.error("  ✗ Failed to download %s: %s", url[:80], exc)
        if dest.exists():
            dest.unlink()
        return False


def _extract_zip(zip_path: Path, dest_dir: Path, label: str = "") -> bool:
    """Extract a ZIP archive to a directory.

    Args:
        zip_path: Path to ZIP file.
        dest_dir: Destination directory.
        label: Human-readable label for logging.

    Returns:
        True on success; False on failure.
    """
    try:
        dest_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(dest_dir)
        logger.info("  ✓ Extracted %s → %s", label or zip_path.name, dest_dir)
        return True
    except Exception as exc:
        logger.error("  ✗ Failed to extract %s: %s", zip_path.name, exc)
        return False


def _sciencebase_files(item_id: str) -> list[dict]:
    """Return the files list from a ScienceBase catalog item.

    Args:
        item_id: ScienceBase item ID.

    Returns:
        List of file dicts from the ScienceBase API (may be empty on failure).
    """
    url = f"https://www.sciencebase.gov/catalog/item/{item_id}?format=json"
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return data.get("files", [])
    except Exception as exc:
        logger.error("  ScienceBase API failed for item %s: %s", item_id, exc)
        return []


def _sciencebase_download_first(
    item_id: str, dest_dir: Path,
    extensions: tuple[str, ...] = (".zip", ".shp"),
    label: str = "",
) -> bool:
    """Download the first matching file from a ScienceBase item.

    Args:
        item_id: ScienceBase item ID.
        dest_dir: Destination directory.
        extensions: Preferred file extensions (in priority order).
        label: Human-readable label for logging.

    Returns:
        True if a file was downloaded; False otherwise.
    """
    files = _sciencebase_files(item_id)
    if not files:
        return False

    # Prefer .zip, then .shp, then anything
    for ext in extensions:
        for f in files:
            name = f.get("name", "")
            if name.lower().endswith(ext):
                url = f.get("downloadUri") or f.get("url", "")
                if url:
                    dest = dest_dir / name
                    return _download_file(url, dest, label or name)

    # Fallback: download first file
    f = files[0]
    name = f.get("name", "file")
    url  = f.get("downloadUri") or f.get("url", "")
    if url:
        return _download_file(url, dest_dir / name, label or name)
    return False


# ── Dataset: County Boundary ──────────────────────────────────────────────────

def download_county_boundary() -> bool:
    """Download California county boundaries from Census TIGER and filter to SB.

    Returns:
        True on success.
    """
    logger.info("─ Santa Barbara County boundary (Census TIGER 2023) …")
    out_dir = config.RAW_DIR / "sb_county_boundary"
    shp = out_dir / "sb_county_boundary.shp"
    if shp.exists():
        logger.info("  Already exists — skipping")
        return True

    url = "https://www2.census.gov/geo/tiger/TIGER2023/COUNTY/tl_2023_06_county.zip"
    zip_path = config.RAW_DIR / "tl_2023_06_county.zip"

    ok = _download_file(url, zip_path, "CA counties (TIGER)")
    if not ok:
        return False

    tmp_dir = config.RAW_DIR / "_tiger_tmp"
    _extract_zip(zip_path, tmp_dir)

    # Filter to Santa Barbara County and save
    try:
        import geopandas as gpd
        ca_counties = gpd.read_file(tmp_dir / "tl_2023_06_county.shp")
        sb = ca_counties[ca_counties["COUNTYFP"] == SB_COUNTY_FIPS].copy()
        if sb.empty:
            logger.error("  Santa Barbara County not found in TIGER file")
            return False
        out_dir.mkdir(parents=True, exist_ok=True)
        sb.to_file(shp)
        logger.info("  ✓ SB County boundary saved → %s", shp)
    except Exception as exc:
        logger.error("  Failed to filter county boundary: %s", exc)
        return False
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        zip_path.unlink(missing_ok=True)

    return True


# ── Dataset: 3DEP DEM ─────────────────────────────────────────────────────────

def download_dem_tiles() -> bool:
    """Query TNM API for 1/3 arc-second DEM tiles and download them.

    Tiles are saved to data/raw/dem_tiles/.

    Returns:
        True if at least one tile was downloaded.
    """
    logger.info("─ USGS 3DEP 1/3 arc-second DEM tiles (TNM API) …")
    tiles_dir = config.DEM_TILES_DIR
    tiles_dir.mkdir(parents=True, exist_ok=True)

    api_url = (
        "https://tnmaccess.nationalmap.gov/api/v1/products"
        f"?bbox={W},{S},{E},{N}"
        "&datasets=Digital%20Elevation%20Model%20%281%2F3%20arc-second%29"
        "&prodFormats=GeoTiff"
        "&outputFormat=JSON"
        "&max=100"
    )

    logger.info("  Querying TNM API for DEM tiles …")
    try:
        resp = requests.get(api_url, timeout=60)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.error("  TNM API query failed: %s", exc)
        # Try alternate dataset name
        try:
            alt_url = (
                "https://tnmaccess.nationalmap.gov/api/v1/products"
                f"?bbox={W},{S},{E},{N}"
                "&datasets=Digital%20Elevation%20Model%20%281%20arc-second%29"
                "&prodFormats=GeoTiff"
                "&outputFormat=JSON"
                "&max=100"
            )
            resp = requests.get(alt_url, timeout=60)
            data = resp.json()
            logger.info("  Using 1 arc-second DEM as fallback")
        except Exception as exc2:
            logger.error("  Fallback TNM query also failed: %s", exc2)
            return False

    items = data.get("items", [])
    logger.info("  Found %d DEM tile(s)", len(items))

    if not items:
        logger.warning("  No DEM tiles found via API. Check TNM manually at:\n"
                       "  https://apps.nationalmap.gov/downloader/")
        return False

    success_count = 0
    for item in items:
        title = item.get("title", "unknown")
        download_url = item.get("downloadURL", "")
        if not download_url:
            continue
        fname = Path(download_url).name
        dest = tiles_dir / fname
        logger.info("  Tile: %s", title)
        if _download_file(download_url, dest, f"DEM: {fname[:50]}"):
            success_count += 1

    logger.info("  ✓ Downloaded %d/%d DEM tile(s)", success_count, len(items))
    return success_count > 0


# ── Dataset: California Geology ───────────────────────────────────────────────

def download_geology() -> bool:
    """Download USGS California state geology shapefile.

    Returns:
        True on success.
    """
    logger.info("─ USGS California geology shapefile …")
    out_dir = config.RAW_DIR / "ca_geology"
    shp = out_dir / "ca_geology.shp"
    if shp.exists():
        logger.info("  Already exists — skipping")
        return True

    url = "https://mrdata.usgs.gov/geology/state/shp/CA.zip"
    zip_path = config.RAW_DIR / "CA_geology.zip"

    ok = _download_file(url, zip_path, "CA geology (USGS mrdata)")
    if not ok:
        # Try alternate URL
        url2 = "https://www.sciencebase.gov/catalog/file/get/5888bf4fe4b05ccb964bab9d?f=__disk__f%2F68%2F88%2Ff6888e67..."
        logger.warning("  Primary URL failed. Try manual download at:\n"
                       "  https://mrdata.usgs.gov/geology/state/")
        return False

    ok = _extract_zip(zip_path, out_dir, "CA geology")
    zip_path.unlink(missing_ok=True)

    # Verify .shp file present
    shp_files = list(out_dir.glob("*.shp"))
    if shp_files:
        # Rename to expected filename if different
        if shp_files[0] != shp:
            shp_files[0].rename(shp)
            for suf in (".dbf", ".prj", ".shx", ".cpg"):
                src = shp_files[0].with_suffix(suf)
                if src.exists():
                    src.rename(shp.with_suffix(suf))
        logger.info("  ✓ Geology shapefile ready → %s", shp)
        return True

    logger.error("  No .shp file found after extraction")
    return False


# ── Dataset: CAL FIRE Perimeters ──────────────────────────────────────────────

def download_fire_perimeters() -> bool:
    """Download California historical fire perimeters shapefile from CNRA.

    Returns:
        True on success.
    """
    logger.info("─ CAL FIRE historical fire perimeters …")
    out_dir = config.RAW_DIR / "fire_perimeters"
    shp_candidates = list(out_dir.glob("*.shp")) if out_dir.exists() else []
    if shp_candidates:
        logger.info("  Already exists (%s) — skipping", shp_candidates[0].name)
        return True

    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = config.RAW_DIR / "ca_fire_perimeters.zip"

    # Primary URL: CNRA Open Data shapefile download
    url = (
        "https://gis.data.cnra.ca.gov/api/download/v1/items/"
        "c3c10388e3b24cec8a954ba10458039d/shapefile?layers=0"
    )
    ok = _download_file(url, zip_path, "CAL FIRE perimeters")

    if not ok:
        # Fallback: ArcGIS REST API GeoJSON export (slower but more reliable)
        logger.info("  Trying ArcGIS REST fallback …")
        rest_url = (
            "https://services1.arcgis.com/jUJYIo9tSA7EHvfZ/arcgis/rest/services/"
            "California_Fire_Perimeters_All/FeatureServer/0/query"
            "?where=1%3D1&outFields=FIRE_NAME%2CYEAR_%2CAGENCY%2CCOUNTY"
            "&geometryType=esriGeometryPolygon"
            f"&geometry={W},{S},{E},{N}&geometryType=esriGeometryEnvelope"
            "&inSR=4326&spatialRel=esriSpatialRelIntersects"
            "&outSR=4326&f=geojson"
        )
        geojson_path = out_dir / "California_Fire_Perimeters_All.geojson"
        ok = _download_file(rest_url, geojson_path, "CAL FIRE (GeoJSON)")
        if ok:
            logger.info("  ✓ Downloaded as GeoJSON — converting to shapefile …")
            try:
                import geopandas as gpd
                gdf = gpd.read_file(geojson_path)
                shp_path = out_dir / "California_Fire_Perimeters_all.shp"
                gdf.to_file(shp_path)
                geojson_path.unlink(missing_ok=True)
                logger.info("  ✓ Fire perimeters shapefile → %s", shp_path)
                return True
            except Exception as exc:
                logger.error("  GeoJSON conversion failed: %s", exc)
                return False
        return False

    _extract_zip(zip_path, out_dir, "CAL FIRE perimeters")
    zip_path.unlink(missing_ok=True)

    shp_candidates = list(out_dir.glob("*.shp"))
    if shp_candidates:
        # Rename to expected name
        target = out_dir / "California_Fire_Perimeters_all.shp"
        if shp_candidates[0] != target:
            for src_f in out_dir.iterdir():
                if src_f.stem == shp_candidates[0].stem:
                    src_f.rename(out_dir / (target.stem + src_f.suffix))
        logger.info("  ✓ Fire perimeters ready")
        return True

    logger.error("  No shapefile found in fire perimeters directory")
    return False


# ── Dataset: USGS Landslide Inventory ────────────────────────────────────────

def download_landslide_inventory() -> bool:
    """Download USGS Global Landslide Inventory from ScienceBase.

    Tries v2.0 (item 5c7065b4e4b0fe48cb43fbd7) first, then the original.

    Returns:
        True on success.
    """
    logger.info("─ USGS Landslide Inventory (ScienceBase) …")
    out_dir = config.RAW_DIR / "landslide_inventory"
    shp_candidates = list(out_dir.glob("*.shp")) if out_dir.exists() else []
    if shp_candidates:
        logger.info("  Already exists — skipping")
        return True

    out_dir.mkdir(parents=True, exist_ok=True)

    # Try v2.0 first, then v1
    item_ids = [
        "5c7065b4e4b0fe48cb43fbd7",   # Global Landslide Inventory v2.0
        "58572c7be4b01fad86d5ff1f",    # Original USGS LS inventory
    ]

    for item_id in item_ids:
        logger.info("  Trying ScienceBase item %s …", item_id)
        files = _sciencebase_files(item_id)
        logger.info("  Available files: %s", [f.get("name") for f in files])

        zip_files = [f for f in files if f.get("name", "").lower().endswith(".zip")]
        if not zip_files:
            continue

        f = zip_files[0]
        url  = f.get("downloadUri") or f.get("url", "")
        name = f.get("name", "ls_inventory.zip")

        if not url:
            # Try constructing the URL
            url = (f"https://www.sciencebase.gov/catalog/file/get/"
                   f"{item_id}?f=__disk__{name}")

        zip_path = out_dir / name
        if _download_file(url, zip_path, f"Landslide inventory ({name})"):
            _extract_zip(zip_path, out_dir, "landslide inventory")
            zip_path.unlink(missing_ok=True)

            shp_files = list(out_dir.glob("*.shp"))
            if shp_files:
                # Standardise filename
                target = out_dir / "ls_inventory.shp"
                if shp_files[0] != target:
                    stem = shp_files[0].stem
                    for suf in (".shp", ".dbf", ".prj", ".shx", ".cpg"):
                        src = out_dir / (stem + suf)
                        if src.exists():
                            src.rename(out_dir / ("ls_inventory" + suf))
                logger.info("  ✓ Landslide inventory ready")
                return True

    logger.warning(
        "  Could not auto-download landslide inventory.\n"
        "  Manual download:\n"
        "  https://www.sciencebase.gov/catalog/item/5c7065b4e4b0fe48cb43fbd7\n"
        "  Place shapefile in: data/raw/landslide_inventory/ls_inventory.shp"
    )
    return False


# ── Dataset: Quaternary Faults ────────────────────────────────────────────────

def download_quaternary_faults() -> bool:
    """Download USGS Quaternary Fault and Fold Database from ScienceBase.

    Returns:
        True on success.
    """
    logger.info("─ USGS Quaternary Faults (ScienceBase) …")
    out_dir = config.RAW_DIR / "quaternary_faults"
    shp_candidates = list(out_dir.glob("*.shp")) if out_dir.exists() else []
    if shp_candidates:
        logger.info("  Already exists — skipping")
        return True

    out_dir.mkdir(parents=True, exist_ok=True)

    item_id = "589097b1e4b072a7ac0cae23"
    files = _sciencebase_files(item_id)
    logger.info("  ScienceBase files: %s", [f.get("name") for f in files])

    zip_files = [f for f in files if f.get("name", "").lower().endswith(".zip")]
    if zip_files:
        f = zip_files[0]
        url  = f.get("downloadUri") or f.get("url", "")
        name = f.get("name", "qfaults.zip")
        zip_path = out_dir / name
        ok = _download_file(url, zip_path, f"Quaternary faults ({name})")
        if ok:
            _extract_zip(zip_path, out_dir, "Quaternary faults")
            zip_path.unlink(missing_ok=True)
            # Standardise
            shp_files = list(out_dir.glob("**/*.shp"))
            if shp_files:
                target = out_dir / "qfaults.shp"
                stem = shp_files[0].stem
                src_dir = shp_files[0].parent
                for suf in (".shp", ".dbf", ".prj", ".shx", ".cpg"):
                    src = src_dir / (stem + suf)
                    if src.exists():
                        src.rename(out_dir / ("qfaults" + suf))
                logger.info("  ✓ Quaternary faults ready")
                return True

    logger.warning(
        "  Could not auto-download Quaternary faults.\n"
        "  Manual download:\n"
        "  https://www.sciencebase.gov/catalog/item/589097b1e4b072a7ac0cae23\n"
        "  Place shapefile in: data/raw/quaternary_faults/qfaults.shp"
    )
    return False


# ── Dataset: Montecito 2018 Debris Flow ───────────────────────────────────────

def download_montecito_debris() -> bool:
    """Download the January 9, 2018 Montecito debris flow polygon from ScienceBase.

    ScienceBase item: 5c115fe6e4b034bf6a836bc2 (Kean et al. 2019)

    Returns:
        True on success.
    """
    logger.info("─ Montecito 2018 debris flow polygon (Kean et al. 2019) …")
    out_dir = config.RAW_DIR / "montecito_debris_flow"
    shp_candidates = list(out_dir.glob("*.shp")) if out_dir.exists() else []
    if shp_candidates:
        logger.info("  Already exists — skipping")
        return True

    out_dir.mkdir(parents=True, exist_ok=True)

    item_id = "5c115fe6e4b034bf6a836bc2"
    files = _sciencebase_files(item_id)
    logger.info("  Available files: %s", [f.get("name") for f in files])

    # Download all zip files; one should contain the debris flow polygon
    any_ok = False
    for f in files:
        name = f.get("name", "")
        url  = f.get("downloadUri") or f.get("url", "")
        if not url or not name:
            continue
        dest = out_dir / name
        if _download_file(url, dest, f"Montecito: {name}"):
            any_ok = True
            if name.lower().endswith(".zip"):
                _extract_zip(dest, out_dir, f"Montecito {name}")
                dest.unlink(missing_ok=True)

    # Look for shapefile
    shp_files = list(out_dir.glob("**/*.shp"))
    if shp_files:
        # Use the first shapefile that sounds like it's the inundation boundary
        boundary_shps = [s for s in shp_files if any(
            k in s.name.lower() for k in ["inundation", "boundary", "debris", "flow"]
        )]
        best = boundary_shps[0] if boundary_shps else shp_files[0]
        target = out_dir / "montecito_2018_debris_flow.shp"
        stem = best.stem
        src_dir = best.parent
        for suf in (".shp", ".dbf", ".prj", ".shx", ".cpg"):
            src = src_dir / (stem + suf)
            if src.exists() and src != out_dir / ("montecito_2018_debris_flow" + suf):
                src.rename(out_dir / ("montecito_2018_debris_flow" + suf))
        logger.info("  ✓ Montecito debris flow shapefile ready")
        return True

    if not any_ok:
        logger.warning(
            "  Could not auto-download Montecito debris flow.\n"
            "  Manual download (Kean et al. 2019 USGS OFR 2019-1185):\n"
            "  https://www.sciencebase.gov/catalog/item/5c115fe6e4b034bf6a836bc2\n"
            "  Place inundation boundary shapefile in:\n"
            "  data/raw/montecito_debris_flow/montecito_2018_debris_flow.shp"
        )
    return False


# ── Dataset: NLCD 2021 ────────────────────────────────────────────────────────

def download_nlcd() -> bool:
    """Download NLCD 2021 land cover for the SB County bbox via MRLC WCS.

    Returns:
        True on success.
    """
    logger.info("─ NLCD 2021 land cover (MRLC WCS) …")
    out_dir = config.RAW_DIR / "nlcd_2021"
    tif_path = out_dir / "nlcd_2021.img"
    if tif_path.exists() or list(out_dir.glob("*.img")) or list(out_dir.glob("*.tif")):
        logger.info("  Already exists — skipping")
        return True
    out_dir.mkdir(parents=True, exist_ok=True)

    # Try MRLC WCS 2.0.1 endpoint
    # Coverage names to try in order
    coverage_names = [
        "NLCD_2021_Land_Cover_L48",
        "NLCD_2021_Land_Cover_L48_20230630",
        "nlcd_2021",
    ]

    # First get capabilities to find the right name
    wcs_base = "https://dmsdata.cr.usgs.gov/geoserver/ows"
    caps_url = f"{wcs_base}?service=WCS&version=2.0.1&request=GetCapabilities"

    coverage_id = None
    try:
        resp = requests.get(caps_url, timeout=60)
        # Search for NLCD 2021 in the XML
        text = resp.text
        for name in coverage_names:
            if name in text:
                coverage_id = name
                break
        if not coverage_id:
            # Extract any NLCD 2021 coverage id
            import re
            match = re.search(r'(NLCD_2021[^<\s"]*)', text)
            if match:
                coverage_id = match.group(1)
    except Exception as exc:
        logger.warning("  WCS GetCapabilities failed: %s", exc)

    if not coverage_id:
        coverage_id = coverage_names[0]
        logger.info("  Using default coverage name: %s", coverage_id)

    # Build GetCoverage URL
    # Convert bbox to pixel sizes: ~100m resolution for download (will be resampled)
    resx = 0.001  # degrees (~100m at this latitude)
    resy = 0.001
    wcs_url = (
        f"{wcs_base}?service=WCS&version=2.0.1&request=GetCoverage"
        f"&coverageId={coverage_id}"
        f"&format=image/tiff"
        f"&subset=Lat({S},{N})"
        f"&subset=Long({W},{E})"
    )

    tif_path2 = out_dir / "nlcd_2021.tif"
    ok = _download_file(wcs_url, tif_path2, "NLCD 2021 (WCS)")

    if not ok:
        # Fallback: try 1.0.0 WCS
        wcs_url_v1 = (
            f"{wcs_base}?service=WCS&VERSION=1.0.0&request=GetCoverage"
            f"&coverage={coverage_id}"
            f"&format=GeoTIFF"
            f"&bbox={W},{S},{E},{N}"
            f"&crs=EPSG:4326"
            f"&resx={resx}&resy={resy}"
        )
        ok = _download_file(wcs_url_v1, tif_path2, "NLCD 2021 (WCS v1)")

    if not ok:
        logger.warning(
            "  Could not auto-download NLCD 2021.\n"
            "  Manual download:\n"
            "  1. Go to https://www.mrlc.gov/viewer/\n"
            "  2. Draw the SB County bounding box\n"
            "  3. Download NLCD 2021 Land Cover\n"
            "  4. Place file in: data/raw/nlcd_2021/"
        )
        return False

    logger.info("  ✓ NLCD 2021 downloaded → %s", tif_path2)
    return True


# ── Dataset: PRISM Precipitation ─────────────────────────────────────────────

def download_prism() -> bool:
    """Download PRISM 30-year normal annual precipitation (4 km grid).

    Returns:
        True on success.
    """
    logger.info("─ PRISM 30-year annual precipitation …")
    out_dir = config.RAW_DIR / "prism"
    existing = list(out_dir.glob("*.bil")) + list(out_dir.glob("*.tif")) + list(out_dir.glob("*.zip"))
    if existing:
        logger.info("  Already exists — skipping")
        return True
    out_dir.mkdir(parents=True, exist_ok=True)

    # PRISM FTP-like mirror — 1991-2020 normals
    urls = [
        "https://data.prism.oregonstate.edu/normals/an91/4km/PRISM_ppt_30yr_normal_4kmM4_annual_asc.zip",
        "https://data.prism.oregonstate.edu/normals/an81/4km/PRISM_ppt_30yr_normal_4kmM3_annual_bil.zip",
        "https://data.prism.oregonstate.edu/normals/an81/800m/PRISM_ppt_30yr_normal_800mM3_annual_bil.zip",
    ]

    for url in urls:
        fname = Path(url).name
        zip_path = out_dir / fname
        ok = _download_file(url, zip_path, f"PRISM precipitation ({fname})")
        if ok:
            _extract_zip(zip_path, out_dir, "PRISM precip")
            zip_path.unlink(missing_ok=True)
            bil_files = list(out_dir.glob("*.bil")) + list(out_dir.glob("*.tif"))
            if bil_files:
                # Update config path to match downloaded file
                logger.info("  ✓ PRISM precipitation downloaded: %s", bil_files[0].name)
                return True

    logger.warning(
        "  Could not auto-download PRISM.\n"
        "  Manual download:\n"
        "  1. Go to https://prism.oregonstate.edu/normals/\n"
        "  2. Download '30-year normals → Annual → 4km'\n"
        "  3. Place BIL/TIF in: data/raw/prism/"
    )
    return False


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="Download all raw datasets for the SB landslide pipeline"
    )
    parser.add_argument("--skip-dem", action="store_true",
                        help="Skip DEM tile downloads (large files)")
    parser.add_argument("--only", type=str, default=None,
                        metavar="DATASET",
                        help="Download only one dataset: county|dem|geology|fire|"
                             "landslide|faults|montecito|nlcd|prism")
    return parser.parse_args()


def main() -> None:
    """Run all data downloads."""
    args = parse_args()

    # Ensure raw directories exist
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    config.DEM_TILES_DIR.mkdir(parents=True, exist_ok=True)

    downloaders = {
        "county":    download_county_boundary,
        "dem":       download_dem_tiles,
        "geology":   download_geology,
        "fire":      download_fire_perimeters,
        "landslide": download_landslide_inventory,
        "faults":    download_quaternary_faults,
        "montecito": download_montecito_debris,
        "nlcd":      download_nlcd,
        "prism":     download_prism,
    }

    if args.only:
        key = args.only.lower()
        if key not in downloaders:
            logger.error("Unknown dataset: %s. Choose from: %s",
                         key, list(downloaders.keys()))
            sys.exit(1)
        downloaders[key]()
        return

    results = {}
    for key, fn in downloaders.items():
        if key == "dem" and args.skip_dem:
            logger.info("─ DEM tiles: SKIPPED (--skip-dem)")
            results[key] = None
            continue
        results[key] = fn()

    # Summary
    logger.info("")
    logger.info("═" * 55)
    logger.info("Download summary:")
    for key, ok in results.items():
        status = "✓" if ok else ("—" if ok is None else "✗ MISSING")
        logger.info("  %-14s %s", key, status)
    logger.info("═" * 55)

    missing = [k for k, v in results.items() if v is False]
    if missing:
        logger.warning(
            "\nSome datasets could not be auto-downloaded: %s\n"
            "Review the warnings above for manual download instructions.\n"
            "The pipeline will continue with available data, using defaults\n"
            "for missing layers where possible.",
            missing,
        )
    else:
        logger.info("\nAll datasets ready. Run: python run_all.py")


if __name__ == "__main__":
    main()
