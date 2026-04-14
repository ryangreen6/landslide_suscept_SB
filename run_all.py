"""
run_all.py
──────────
Master pipeline runner for the Santa Barbara County landslide susceptibility
analysis.  Executes all five stages in order with error handling: if any stage
fails, the error is logged and the pipeline halts.

Usage
-----
    python run_all.py                    # full pipeline
    python run_all.py --start-from 3     # resume from stage 3
    python run_all.py --only 4           # run stage 4 only
    python run_all.py --dpi 150          # pass --dpi flag to stage 5
"""

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  run_all — %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("run_all")

SCRIPTS_DIR = Path(__file__).parent / "scripts"

PIPELINE = [
    (1, "01_data_prep.py",       "Data preparation (mosaic, reproject, clip)"),
    (2, "02_terrain_analysis.py","Terrain analysis (slope, aspect, TWI, curvature)"),
    (3, "03_factor_layers.py",   "Factor layers (lithology, land cover, faults, fire)"),
    (4, "04_modeling.py",        "Susceptibility modeling (Random Forest)"),
    (5, "05_visualization.py",   "Visualisation (figures + interactive map)"),
]


def run_stage(
    stage_num: int,
    script_name: str,
    description: str,
    extra_args: list[str],
) -> bool:
    """Run a single pipeline stage as a subprocess.

    Args:
        stage_num: Stage number (for logging).
        script_name: Filename of the script in ``scripts/``.
        description: Human-readable stage description.
        extra_args: Additional CLI arguments to forward to the script.

    Returns:
        True if the stage completed successfully; False otherwise.
    """
    script_path = SCRIPTS_DIR / script_name
    if not script_path.exists():
        logger.error("Script not found: %s", script_path)
        return False

    cmd = [sys.executable, str(script_path)] + extra_args
    logger.info("")
    logger.info("━" * 60)
    logger.info("Stage %d/%d — %s", stage_num, len(PIPELINE), description)
    logger.info("Command: %s", " ".join(cmd))
    logger.info("━" * 60)

    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(Path(__file__).parent))
    elapsed = time.time() - t0

    if result.returncode != 0:
        logger.error(
            "Stage %d FAILED (exit code %d) after %.1f s",
            stage_num, result.returncode, elapsed,
        )
        return False

    logger.info("Stage %d completed in %.1f s ✓", stage_num, elapsed)
    return True


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="Run the full Santa Barbara landslide susceptibility pipeline"
    )
    parser.add_argument(
        "--start-from", type=int, default=1, metavar="N",
        help="Start from stage N (default: 1)"
    )
    parser.add_argument(
        "--only", type=int, default=None, metavar="N",
        help="Run only stage N"
    )
    parser.add_argument(
        "--skip-dem", action="store_true",
        help="Pass --skip-dem to stage 1 (DEM already prepared)"
    )
    parser.add_argument(
        "--dpi", type=int, default=200,
        help="Pass --dpi to stage 5 (figure resolution)"
    )
    parser.add_argument(
        "--no-interactive", action="store_true",
        help="Pass --no-interactive to stage 5 (skip Folium map)"
    )
    return parser.parse_args()


def main() -> None:
    """Execute the pipeline."""
    args = parse_args()

    # Build stage-specific extra args
    stage_extra: dict[int, list[str]] = {
        1: ["--skip-dem"] if args.skip_dem else [],
        5: ["--dpi", str(args.dpi)] + (["--no-interactive"] if args.no_interactive else []),
    }

    stages_to_run = [s for s in PIPELINE if s[0] >= args.start_from]
    if args.only is not None:
        stages_to_run = [s for s in PIPELINE if s[0] == args.only]
        if not stages_to_run:
            logger.error("--only %d: no stage with that number", args.only)
            sys.exit(1)

    total_start = time.time()
    logger.info("Starting pipeline: %d stage(s) to run", len(stages_to_run))

    for stage_num, script_name, description in stages_to_run:
        extra = stage_extra.get(stage_num, [])
        success = run_stage(stage_num, script_name, description, extra)
        if not success:
            logger.error("Pipeline halted at stage %d.", stage_num)
            sys.exit(1)

    total_elapsed = time.time() - total_start
    logger.info("")
    logger.info("═" * 60)
    logger.info("Pipeline complete. Total time: %.1f s (%.1f min)",
                total_elapsed, total_elapsed / 60)
    logger.info("Outputs in: %s", Path(__file__).parent / "data" / "outputs")
    logger.info("═" * 60)


if __name__ == "__main__":
    main()
