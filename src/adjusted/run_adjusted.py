"""Descriptor-analysis runner: unadjusted, adjusted, and targeted (DAG) models.

Usage::

    cd /path/to/codebase
    .venv/bin/python src/adjusted/run_adjusted.py --run-dir <run-dir>

Reads ``<run-dir>/ingestion/analytic_clean.parquet`` and writes all outputs
under ``<run-dir>/adjusted/``.

Outputs
-------
  <run-dir>/adjusted/descriptor_unadjusted.csv
  <run-dir>/adjusted/descriptor_adjusted.csv
  <run-dir>/adjusted/descriptor_targeted.csv
  <run-dir>/adjusted/figures/adj_heatmap_beta.png
  <run-dir>/adjusted/figures/adj_heatmap_partial_r2.png
  <run-dir>/adjusted/figures/adj_forest_plots.png
  <run-dir>/adjusted/figures/unadj_heatmap_beta.png
  <run-dir>/adjusted/figures/targeted_heatmap_beta.png
  <run-dir>/adjusted/figures/targeted_heatmap_partial_r2.png

Models
------
  Unadjusted: each descriptor (within-rib z) on each predictor marginally.
  Adjusted:   each descriptor on the full CORE predictor set in one OLS
              (sex, age, height, weight, body fat, ever-smoker, pack-years),
              with FWL partial R². Patient-clustered standard errors.
  Targeted:   each descriptor on each exposure with that exposure's DAG-based
              back-door adjustment set (settings.DAG_ADJUSTMENT_SETS); the focal
              coefficient and its FWL partial R², tagged with its estimand.
              Patient-clustered SE.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from adjusted.adjusted import run_descriptor_models  # noqa: E402
from adjusted.plots_adjusted_altair import (  # noqa: E402
    plot_adj_heatmap,
    plot_forest_plots,
    plot_partial_r2_heatmap,
)
from data_ingestion.loaders import ANALYSIS_SHAPE_COLS  # noqa: E402
from settings import FDR_DISPLAY_ALPHA, apply_publication_style  # noqa: E402
from utils.config import read_parquet        # noqa: E402
from utils.logging import get_logger         # noqa: E402
from utils.paths import stage_dir            # noqa: E402

apply_publication_style()

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Adjusted + unadjusted descriptor-analysis runner")
    p.add_argument(
        "--run-dir", type=Path, required=True, metavar="DIR",
        help="Per-run directory (created by run_pipeline.py). "
             "Reads <run-dir>/ingestion/analytic_clean.parquet; "
             "writes outputs under <run-dir>/adjusted/.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir
    OUT  = stage_dir(run_dir, "adjusted")
    FIGS = OUT / "figures"
    OUT.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)

    pipeline_t0 = time.monotonic()

    logger.info("Loading analytic_clean.parquet …")
    df = read_parquet(stage_dir(run_dir, "ingestion") / "analytic_clean.parquet")
    logger.info(f"  {len(df):,} rib rows | {df['patient_id'].nunique():,} patients")

    logger.info("Descriptor models (within-rib z; patient-clustered) …")
    t0 = time.monotonic()
    unadj, adj, targeted = run_descriptor_models(df, ANALYSIS_SHAPE_COLS)
    n_sig_un  = int((unadj["p_value_fdr"] < FDR_DISPLAY_ALPHA).sum())
    n_sig_adj = int((adj["p_value_fdr"] < FDR_DISPLAY_ALPHA).sum())
    n_sig_tg  = int((targeted["p_value_fdr"] < FDR_DISPLAY_ALPHA).sum())
    unadj.to_csv(OUT / "descriptor_unadjusted.csv", index=False)
    adj.to_csv(OUT / "descriptor_adjusted.csv", index=False)
    targeted.to_csv(OUT / "descriptor_targeted.csv", index=False)
    logger.info(
        f"  unadjusted {len(unadj)} pairs (FDR q<{FDR_DISPLAY_ALPHA}: {n_sig_un}) · "
        f"adjusted {len(adj)} pairs (FDR q<{FDR_DISPLAY_ALPHA}: {n_sig_adj}) · "
        f"targeted {len(targeted)} pairs (FDR q<{FDR_DISPLAY_ALPHA}: {n_sig_tg})  "
        f"({time.monotonic()-t0:.1f}s)"
    )

    logger.info("Figures …")
    plot_adj_heatmap(adj, FIGS / "adj_heatmap_beta")
    plot_partial_r2_heatmap(adj, FIGS / "adj_heatmap_partial_r2")
    plot_forest_plots(adj, ANALYSIS_SHAPE_COLS, FIGS / "adj_forest_plots")
    plot_adj_heatmap(unadj, FIGS / "unadj_heatmap_beta",
                     title="Unadjusted OLS – standardised β")
    plot_adj_heatmap(targeted, FIGS / "targeted_heatmap_beta",
                     title="DAG-based per-exposure OLS – standardised β")
    plot_partial_r2_heatmap(targeted, FIGS / "targeted_heatmap_partial_r2",
                            title="DAG-based per-exposure OLS – partial R²")

    total = time.monotonic() - pipeline_t0
    m, s = divmod(total, 60)
    logger.info(f"Adjusted analysis complete – total wall time {m:.0f}m{s:.0f}s")
    logger.info(f"  All results → {OUT}/")


if __name__ == "__main__":
    main()
