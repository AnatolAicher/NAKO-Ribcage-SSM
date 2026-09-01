"""Regression of PCA scores on patient metadata (three-model design).

Three HC3-robust layers per PC: ``unadjusted`` (one marginal OLS per predictor –
total association), ``adjusted`` (one multivariable OLS – partial coefficients +
Frisch-Waugh-Lovell partial R²), and ``targeted`` (one OLS per exposure with its
DAG-based back-door adjustment set). Sex is additionally summarised by Welch's
t-test for the conventional Cohen's d. BH-FDR is applied within each layer.
"""
from __future__ import annotations

import logging

import pandas as pd

from bivariate.bivariate import run_welch_ttest
from settings import ADJ_PREDICTORS, DAG_ADJUSTMENT_SETS, DAG_TOTAL_EFFECT, UNADJ_PREDICTORS
from utils.regression import adjusted_models, marginal_models, targeted_models

logger = logging.getLogger(__name__)


def run_pc_regression(
    scores_df: pd.DataFrame,
    pc_cols: list[str],
) -> dict[str, pd.DataFrame]:
    """Unadjusted + adjusted HC3 regressions of each PC on metadata, plus a sex t-test.

    Returns a dict with keys ``unadjusted``, ``adjusted``, ``targeted`` (all with a
    ``pc`` column), and ``ttest`` (sex Cohen's d per PC).
    """
    logger.info(f"  unadjusted (HC3): {len(pc_cols)} PCs × {len(UNADJ_PREDICTORS)} predictors")
    unadj = marginal_models(scores_df, pc_cols, UNADJ_PREDICTORS, se_mode="hc3")

    logger.info(f"  adjusted (HC3): {len(pc_cols)} PCs × {len(ADJ_PREDICTORS)} predictors")
    adj = adjusted_models(scores_df, pc_cols, ADJ_PREDICTORS, se_mode="hc3")

    logger.info(f"  targeted (HC3, DAG): {len(pc_cols)} PCs × {len(DAG_ADJUSTMENT_SETS)} exposures")
    targeted = targeted_models(scores_df, pc_cols, DAG_ADJUSTMENT_SETS,
                               total_effect=DAG_TOTAL_EFFECT, se_mode="hc3")

    logger.info(f"  sex Welch t-test: {len(pc_cols)} PCs")
    ttest = run_welch_ttest(scores_df, pc_cols)

    for df in (unadj, adj, targeted):
        if "outcome" in df.columns:
            df.rename(columns={"outcome": "pc"}, inplace=True)
    if "shape_param" in ttest.columns:
        ttest.rename(columns={"shape_param": "pc"}, inplace=True)

    return {"unadjusted": unadj, "adjusted": adj, "targeted": targeted, "ttest": ttest}
