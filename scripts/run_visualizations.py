"""Supplemental cross-stage QC visualisations (Altair).

Outputs under ``<run-dir>/visualizations/figures/``:

  - ``metadata_by_sex``               metadata histograms stratified by sex
  - ``smoking_by_sex``                smoking-status bar chart by sex
  - ``rib_length_by_level``           mean rib length per (rib, side)
  - ``shape_by_rib_<feature>``        per-feature mirrored ridgelines (PNG/SVG)
  - ``interactive_shape_by_rib.html`` shape-parameter browser
  - ``methodology``                   (if --methodology-figure-json) nine-panel
                                      one-patient methodology overview

Usage::

    python scripts/run_visualizations.py --run-dir <run-dir>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from settings import apply_publication_style             # noqa: E402
from utils.logging import get_logger, log_step           # noqa: E402
from utils.paths import stage_dir                        # noqa: E402
from visualizations.vis_altair import (                  # noqa: E402
    plot_metadata_by_sex,
    plot_rib_length_by_level,
    plot_shape_by_rib,
    plot_smoking_by_sex,
)

apply_publication_style()

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Supplemental visualisations runner")
    p.add_argument(
        "--run-dir", type=Path, required=True,
        help="Top-level pipeline run dir.  Reads "
             "<run-dir>/ingestion/analytic_clean.parquet; writes figures to "
             "<run-dir>/visualizations/figures/.",
    )
    p.add_argument(
        "--methodology-figure-json", type=str, default=None,
        help="JSON-encoded methodology_figure preset block. When present, "
             "renders the nine-panel one-patient methodology overview in "
             "addition to the QC visualisations.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir
    out_dir = stage_dir(run_dir, "visualizations") / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    analytic_path = stage_dir(run_dir, "ingestion") / "analytic_clean.parquet"
    if not analytic_path.exists():
        raise FileNotFoundError(
            f"{analytic_path} missing; run the ingestion stage first."
        )
    logger.info(f"Reading {analytic_path}")
    df = pd.read_parquet(analytic_path)

    with log_step(logger, "metadata + smoking + rib-length plots"):
        plot_metadata_by_sex(df, out_dir)
        plot_smoking_by_sex(df, out_dir)
        plot_rib_length_by_level(df, out_dir)

    with log_step(logger, "shape-by-rib ridgelines"):
        plot_shape_by_rib(df, out_dir)

    if args.methodology_figure_json:
        from utils.config import load_config                       # noqa: E402
        from visualizations.methodology_figure import render as mf_render  # noqa: E402
        mf_cfg = json.loads(args.methodology_figure_json)
        with log_step(logger, f"methodology figure (patient {mf_cfg['display_patient']})"):
            mf_render(
                run_dir=run_dir,
                data_config=load_config(),
                preset_cfg=mf_cfg,
                out_stem=out_dir / "methodology",
            )

    logger.info(f"Done. Figures under: {out_dir}")


if __name__ == "__main__":
    main()
