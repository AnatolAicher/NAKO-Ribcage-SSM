"""Adjusted + unadjusted rib-level descriptor analysis (within-rib z-scored).

Each descriptor is z-scored within its rib identity (level × side) across
patients, then pooled across ribs. The adjusted model regresses every descriptor
on the full CORE predictor set in one OLS (partial coefficients + Frisch-Waugh-
Lovell partial R²); the unadjusted model fits each predictor marginally. Standard
errors are patient-clustered (HC1) across the rib rows per patient. Smoking is
ever/never; pack-years adds the within-smoker dose. BH-FDR within each layer.
"""
from __future__ import annotations

import logging

import pandas as pd

from settings import (
    ADJ_PREDICTORS,
    DAG_ADJUSTMENT_SETS,
    DAG_TOTAL_EFFECT,
    MIN_N_OLS,
    UNADJ_PREDICTORS,
)
from utils.regression import _fwl_partial_r2 as _fwl_pair  # re-exported for tests/test_fwl.py
from utils.regression import (
    adjusted_models,
    marginal_models,
    targeted_models,
    zscore_within,
)

logger = logging.getLogger(__name__)

PREDICTOR_LABELS: dict[str, str] = {
    "is_female":    "Sex (Female)",
    "age":          "Age",
    "height_cm":    "Height",
    "weight_kg":    "Weight",
    "bmi":          "BMI",
    "body_fat_pct": "Body fat (%)",
    "ever_smoker":  "Ever-smoker",
    "pack_years":   "Pack-years",
}


def _with_rib_label(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["rib_label"] = out["vert_level"].astype(str) + "_" + out["side"].astype(str)
    return out


def run_descriptor_models(
    df: pd.DataFrame,
    shape_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Unadjusted, adjusted, and targeted descriptor models on within-rib-z data.

    Returns ``(unadjusted, adjusted, targeted)``, each with a ``shape_param`` column
    and patient-clustered standard errors; the adjusted frame carries FWL partial
    R², the targeted frame one OLS per exposure with its DAG back-door set.
    """
    d = _with_rib_label(df)
    avail = [c for c in shape_cols if c in d.columns]
    d = zscore_within(d, avail, group_col="rib_label")

    logger.info(f"  unadjusted (cluster): {len(avail)} descriptors × {len(UNADJ_PREDICTORS)} predictors")
    unadj = marginal_models(d, avail, UNADJ_PREDICTORS, se_mode="cluster",
                            cluster_col="patient_id", min_n=MIN_N_OLS)
    logger.info(f"  adjusted (cluster): {len(avail)} descriptors × {len(ADJ_PREDICTORS)} predictors")
    adj = adjusted_models(d, avail, ADJ_PREDICTORS, se_mode="cluster",
                          cluster_col="patient_id", min_n=MIN_N_OLS)
    logger.info(f"  targeted (cluster, DAG): {len(avail)} descriptors × {len(DAG_ADJUSTMENT_SETS)} exposures")
    targeted = targeted_models(d, avail, DAG_ADJUSTMENT_SETS, total_effect=DAG_TOTAL_EFFECT,
                               se_mode="cluster", cluster_col="patient_id", min_n=MIN_N_OLS)

    for r in (unadj, adj, targeted):
        if "outcome" in r.columns:
            r.rename(columns={"outcome": "shape_param"}, inplace=True)
    return unadj, adj, targeted
