"""Shared OLS engine for the marginal (unadjusted) and adjusted models.

Both the PC-score and descriptor analyses use these. ``marginal_models`` fits one
OLS per (outcome, predictor); ``adjusted_models`` fits one multivariable OLS per
outcome and adds Frisch-Waugh-Lovell partial R² per predictor. Standardised betas
and BH-FDR (one family across the whole returned grid) are included. Standard
errors are HC3 (heteroscedasticity-robust) for patient-level outcomes or
patient-clustered for rib-level outcomes; ``sex``/``smoking_status`` are encoded
to ``is_female``/``ever_smoker`` via :mod:`utils.design`.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

from utils.design import add_design_columns

logger = logging.getLogger(__name__)


def _bh(pvals: pd.Series) -> np.ndarray:
    """BH-FDR across all non-NaN p-values; NaNs preserved."""
    valid = pvals.notna()
    out = np.full(len(pvals), np.nan)
    if valid.sum() > 0:
        _, q, _, _ = multipletests(pvals[valid], method="fdr_bh")
        out[valid.to_numpy()] = q
    return out


def _fit(y: np.ndarray, X: np.ndarray, se_mode: str, groups: pd.Series | None):
    """OLS fit with HC3, patient-clustered, or classical SEs."""
    model = sm.OLS(y, X, missing="drop")
    if se_mode == "hc3":
        return model.fit(cov_type="HC3")
    if se_mode == "cluster":
        return model.fit(cov_type="cluster", cov_kwds={"groups": groups})
    return model.fit()


def marginal_models(
    df: pd.DataFrame,
    outcomes: list[str],
    predictors: list[str],
    se_mode: str = "hc3",
    cluster_col: str | None = None,
    ever_only: tuple[str, ...] = ("pack_years",),
    min_n: int = 30,
) -> pd.DataFrame:
    """One OLS per (outcome, predictor); marginal/total association.

    ``ever_only`` predictors (pack-years) are fit on rows where the predictor is
    > 0 (ever-smokers), the only sample in which they vary.
    """
    d = add_design_columns(df)
    rows: list[dict] = []
    for outcome in outcomes:
        if outcome not in d.columns:
            continue
        for pred in predictors:
            if pred not in d.columns:
                continue
            cols = [outcome, pred] + ([cluster_col] if cluster_col else [])
            sub = d[cols].dropna()
            if pred in ever_only:
                sub = sub[sub[pred] > 0]
            if sub[outcome].nunique() < 2 or len(sub) < min_n:
                continue

            X = sm.add_constant(sub[[pred]].to_numpy(dtype=float), has_constant="add")
            res = _fit(sub[outcome].to_numpy(dtype=float), X, se_mode,
                       groups=sub[cluster_col] if cluster_col else None)

            beta = float(res.params[1])
            y_sd = float(sub[outcome].std(ddof=1))
            x_sd = float(sub[pred].std(ddof=1))
            ci = res.conf_int()
            rows.append({
                "outcome":     outcome,
                "predictor":   pred,
                "n":           int(len(sub)),
                "beta":        beta,
                "beta_std":    beta * x_sd / y_sd if y_sd > 0 else np.nan,
                "se":          float(res.bse[1]),
                "ci_low":      float(ci[1, 0]),
                "ci_high":     float(ci[1, 1]),
                "r_squared":   float(res.rsquared),
                "p_value":     float(res.pvalues[1]),
            })

    if not rows:
        return pd.DataFrame(columns=["outcome", "predictor", "n", "beta", "beta_std",
                                     "se", "ci_low", "ci_high", "r_squared",
                                     "p_value", "p_value_fdr"])
    res_df = pd.DataFrame(rows)
    res_df["p_value_fdr"] = _bh(res_df["p_value"])
    return res_df


def _fwl_partial_r2(y: np.ndarray, x: np.ndarray, z: np.ndarray) -> tuple[float, float]:
    """Squared partial correlation of x on y given z (FWL); signed partial r."""
    Z = sm.add_constant(z, has_constant="add")
    e_y = sm.OLS(y, Z).fit().resid
    e_x = sm.OLS(x, Z).fit().resid
    if e_x.std() < 1e-12 or e_y.std() < 1e-12:
        return np.nan, np.nan
    r = float(np.corrcoef(e_y, e_x)[0, 1])
    return r ** 2, r


def adjusted_models(
    df: pd.DataFrame,
    outcomes: list[str],
    predictors: list[str],
    se_mode: str = "hc3",
    cluster_col: str | None = None,
    min_n: int = 100,
    partial_r2: bool = True,
) -> pd.DataFrame:
    """One multivariable OLS per outcome; partial coefficients + FWL partial R²."""
    d = add_design_columns(df)
    rows: list[dict] = []
    for outcome in outcomes:
        if outcome not in d.columns:
            continue
        cols = [outcome, *predictors] + ([cluster_col] if cluster_col else [])
        sub = d[[c for c in cols if c in d.columns]].dropna()
        if len(sub) < min_n:
            continue

        Xdf = sub[predictors].astype(float)
        X = sm.add_constant(Xdf, has_constant="add")
        res = _fit(sub[outcome].to_numpy(dtype=float), X, se_mode,
                   groups=sub[cluster_col] if cluster_col else None)
        y_sd = float(sub[outcome].std(ddof=1))
        full_r2 = float(res.rsquared)
        ci = res.conf_int()

        for pred in predictors:
            if pred not in res.params.index:
                continue
            beta = float(res.params[pred])
            x_sd = float(sub[pred].std(ddof=1))
            pr2 = pr = np.nan
            if partial_r2:
                others = [p for p in predictors if p != pred]
                z = sub[others].to_numpy(dtype=float) if others else np.empty((len(sub), 0))
                pr2, pr = _fwl_partial_r2(sub[outcome].to_numpy(dtype=float),
                                          sub[pred].to_numpy(dtype=float), z)
            rows.append({
                "outcome":     outcome,
                "predictor":   pred,
                "n":           int(len(sub)),
                "beta":        beta,
                "beta_std":    beta * x_sd / y_sd if y_sd > 0 else np.nan,
                "se":          float(res.bse[pred]),
                "ci_low":      float(ci.loc[pred, 0]),
                "ci_high":     float(ci.loc[pred, 1]),
                "t_stat":      float(res.tvalues[pred]),
                "r_squared":   full_r2,
                "partial_r2":  pr2,
                "partial_r":   pr,
                "p_value":     float(res.pvalues[pred]),
            })

    if not rows:
        return pd.DataFrame(columns=["outcome", "predictor", "n", "beta", "beta_std",
                                     "se", "ci_low", "ci_high", "t_stat", "r_squared",
                                     "partial_r2", "partial_r", "p_value", "p_value_fdr"])
    res_df = pd.DataFrame(rows)
    res_df["p_value_fdr"] = _bh(res_df["p_value"])
    return res_df


def targeted_models(
    df: pd.DataFrame,
    outcomes: list[str],
    adjustment_sets: dict[str, list[str]],
    se_mode: str = "hc3",
    cluster_col: str | None = None,
    total_effect: frozenset[str] = frozenset(),
    ever_only: tuple[str, ...] = ("pack_years",),
    min_n: int = 30,
    partial_r2: bool = True,
) -> pd.DataFrame:
    """One OLS per (outcome, exposure): ``outcome ~ exposure + its DAG adjustment set``.

    Returns the focal exposure coefficient per fit plus its Frisch-Waugh-Lovell
    partial R² (variance uniquely attributable to the exposure given its
    adjustment set), tagged with the estimand (``total_effect`` for
    ``total_effect`` members, else ``adjusted_association``) and the covariates
    conditioned on. ``ever_only`` exposures (pack-years) are fit on ever-smokers.
    BH-FDR is one family across the returned coefficients.
    """
    d = add_design_columns(df)
    rows: list[dict] = []
    for outcome in outcomes:
        if outcome not in d.columns:
            continue
        for exp, adj in adjustment_sets.items():
            cols = [outcome, exp, *adj] + ([cluster_col] if cluster_col else [])
            sub = d[[c for c in cols if c in d.columns]].dropna()
            if exp in ever_only:
                sub = sub[sub[exp] > 0]
            if exp not in sub.columns or sub[outcome].nunique() < 2 or len(sub) < min_n:
                continue

            present = [a for a in adj if a in sub.columns]
            X = sm.add_constant(sub[[exp, *present]].to_numpy(dtype=float), has_constant="add")
            res = _fit(sub[outcome].to_numpy(dtype=float), X, se_mode,
                       groups=sub[cluster_col] if cluster_col else None)

            beta = float(res.params[1])
            y_sd = float(sub[outcome].std(ddof=1))
            x_sd = float(sub[exp].std(ddof=1))
            ci = res.conf_int()
            pr2 = pr = np.nan
            if partial_r2:
                z = sub[present].to_numpy(dtype=float) if present else np.empty((len(sub), 0))
                pr2, pr = _fwl_partial_r2(sub[outcome].to_numpy(dtype=float),
                                          sub[exp].to_numpy(dtype=float), z)
            rows.append({
                "outcome":      outcome,
                "predictor":    exp,
                "n":            int(len(sub)),
                "beta":         beta,
                "beta_std":     beta * x_sd / y_sd if y_sd > 0 else np.nan,
                "se":           float(res.bse[1]),
                "ci_low":       float(ci[1, 0]),
                "ci_high":      float(ci[1, 1]),
                "partial_r2":   pr2,
                "partial_r":    pr,
                "p_value":      float(res.pvalues[1]),
                "estimand":     "total_effect" if exp in total_effect else "adjusted_association",
                "adjusted_for": ", ".join(present) or "(none)",
            })

    if not rows:
        return pd.DataFrame(columns=["outcome", "predictor", "n", "beta", "beta_std",
                                     "se", "ci_low", "ci_high", "partial_r2", "partial_r",
                                     "p_value", "p_value_fdr", "estimand", "adjusted_for"])
    res_df = pd.DataFrame(rows)
    res_df["p_value_fdr"] = _bh(res_df["p_value"])
    return res_df


def zscore_within(df: pd.DataFrame, value_cols: list[str], group_col: str) -> pd.DataFrame:
    """Return a copy with ``value_cols`` z-scored within each ``group_col`` level."""
    out = df.copy()
    g = out.groupby(group_col)
    for c in value_cols:
        if c in out.columns:
            mean = g[c].transform("mean")
            std = g[c].transform("std")
            out[c] = np.where(std > 0, (out[c] - mean) / std, np.nan)
    return out
