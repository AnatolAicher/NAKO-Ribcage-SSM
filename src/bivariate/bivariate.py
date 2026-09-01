"""Welch's t-test helper for sex × shape, shared by the PC-score analysis.

Helper library, not a pipeline stage: :mod:`ssm.pc_regression` uses
``run_welch_ttest`` for the conventional Cohen's d on the PC scores. The
rib-level unadjusted descriptor layer lives in :mod:`adjusted.adjusted`.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from statsmodels.stats.multitest import multipletests

logger = logging.getLogger(__name__)


def _bh_correct(pvals: pd.Series) -> np.ndarray:
    """BH-FDR correction; NaNs are preserved unchanged."""
    valid = pvals.notna()
    result = np.full(len(pvals), np.nan)
    if valid.sum() > 0:
        _, corrected, _, _ = multipletests(pvals[valid], method="fdr_bh")
        result[valid.values] = corrected
    return result


def run_welch_ttest(
    df_pt: pd.DataFrame,
    shape_cols: list[str],
    sex_col: str = "sex",
    ref: str = "Male",
    alt: str = "Female",
) -> pd.DataFrame:
    """Welch's t-test for sex × each outcome.

    Cohen's d = ``(mean_alt − mean_ref) / pooled SD``.  Positive d means
    ``alt > ref`` (Female > Male by default).

    Returns
    -------
    DataFrame with columns: ``shape_param``, ``n_male``, ``n_female``,
    ``mean_male``, ``mean_female``, ``cohen_d``, ``t_stat``, ``df``,
    ``p_value``, ``p_value_fdr``.
    """
    rows = []
    for outcome in shape_cols:
        if outcome not in df_pt.columns or sex_col not in df_pt.columns:
            continue
        sub = df_pt[[outcome, sex_col]].dropna()
        a = sub.loc[sub[sex_col] == ref, outcome].values
        b = sub.loc[sub[sex_col] == alt, outcome].values
        if len(a) < 2 or len(b) < 2:
            continue

        t, p = scipy_stats.ttest_ind(a, b, equal_var=False)

        s1, s2, n1, n2 = (
            float(np.std(a, ddof=1)),
            float(np.std(b, ddof=1)),
            len(a), len(b),
        )
        # Welch-Satterthwaite df
        df_ws = (s1**2/n1 + s2**2/n2)**2 / (
            (s1**2/n1)**2/(n1-1) + (s2**2/n2)**2/(n2-1)
        )
        # Cohen's d – root-mean-square SD denominator (Welch's t-test
        # already assumes unequal variances).
        denom = float(np.sqrt((s1**2 + s2**2) / 2))
        d = (float(np.mean(b)) - float(np.mean(a))) / denom if denom > 0 else np.nan

        rows.append({
            "shape_param": outcome,
            f"n_{ref.lower()}": n1,
            f"n_{alt.lower()}": n2,
            f"mean_{ref.lower()}": float(np.mean(a)),
            f"mean_{alt.lower()}": float(np.mean(b)),
            "cohen_d": d,
            "t_stat": float(t),
            "df": float(df_ws),
            "p_value": float(p),
        })

    if not rows:
        return pd.DataFrame(columns=[
            "shape_param",
            f"n_{ref.lower()}", f"n_{alt.lower()}",
            f"mean_{ref.lower()}", f"mean_{alt.lower()}",
            "cohen_d", "t_stat", "df", "p_value", "p_value_fdr",
        ])
    res = pd.DataFrame(rows)
    res["p_value_fdr"] = _bh_correct(res["p_value"])
    return res
