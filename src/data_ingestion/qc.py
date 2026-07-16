"""Quality control, descriptive statistics, and cohort audits.

Public API
----------
apply_exclusions(merged, rib_audit, seg_components_audit, cfg)
                                            -> (included_df, exclusion_table_df)
audit_rib_counts(df)                        -> per-patient rib count summary
audit_missingness(df, shape_cols, meta)     -> missingness report
compute_table1(df, continuous, categorical) -> Table 1 DataFrame
normality_summary(df, cols)                 -> per-column skew/kurt/Shapiro

Figures for these tables are emitted by the Altair module :mod:`data_ingestion.qc_altair`.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

from utils.rib_labels import display_from_seg, display_from_vert

logger = logging.getLogger(__name__)

# 12 vertebral levels (T8–T19) × 2 sides = 24 rib-side rows per patient.
EXPECTED_RIB_SIDES = 24


# ── Exclusions ───────────────────────────────────────────────────────────────

def apply_exclusions(
    merged: pd.DataFrame,
    rib_audit: pd.DataFrame,
    seg_components_audit: pd.DataFrame,
    cfg: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a per-patient exclusion table and return ``(included_df, exclusion_df)``.

    Exclusion criteria (read from ``cfg["exclusion_criteria"]``):

      1. Rib count outside ``[min_rib_sides, max_rib_sides]``.
      2. Any required metadata column is missing.
      3. Any rib label (40–51) does not yield exactly two connected components.
      4. Any rib has ``seg_at_border`` set.
      5. Any rib has ``split_start`` or ``split_end`` set.

    ``exclusion_df`` has one row per patient with reason columns (one per
    criterion).  Empty reason string means the criterion did not apply.
    """
    excl_cfg = cfg["exclusion_criteria"]
    min_sides = excl_cfg["rib_count"]["min_rib_sides"]
    max_sides = excl_cfg["rib_count"]["max_rib_sides"]
    meta_cols = excl_cfg["metadata_completeness"]["require_complete_columns"]
    comp_cfg  = excl_cfg.get("segmentation_components", {})
    check_components = comp_cfg.get("require_two_components_per_rib", False)
    seg_cfg   = excl_cfg.get("segmentation_quality", {})
    check_border = seg_cfg.get("exclude_if_any_seg_at_border", False)
    check_split  = seg_cfg.get("exclude_if_any_split", False)

    meta_per_patient = (
        merged.drop_duplicates("patient_id")
        .set_index("patient_id")[
            [c for c in meta_cols if c in merged.columns]
        ]
    )
    rib_map = rib_audit.set_index("patient_id")["n_rib_sides"].to_dict()

    def _affected_ribs(df: pd.DataFrame, flag_col: str) -> dict[int, str]:
        """``{patient_id: 'Rib 1 Right; Rib 3 Left'}`` for rows where ``flag_col`` is truthy."""
        affected = df[df[flag_col].fillna(False).astype(bool)].copy()
        if affected.empty:
            return {}
        affected["label"] = affected.apply(
            lambda r: display_from_vert(r["vert_level"], r["side"], side_long=True),
            axis=1,
        )
        return affected.groupby("patient_id")["label"].apply(lambda x: "; ".join(x)).to_dict()

    components_map: dict[int, str] = {}
    if check_components and not seg_components_audit.empty:
        anomalous = seg_components_audit[seg_components_audit["n_components"] != 2]
        if not anomalous.empty:
            def _format_row(r: pd.Series) -> str:
                if r["rib_label"] < 0:
                    return "NIfTI missing"
                return f"{display_from_seg(r['rib_label'], side_long=False)} ({r['n_components']} comps)"
            anomalous = anomalous.copy()
            anomalous["label"] = anomalous.apply(_format_row, axis=1)
            components_map = (
                anomalous.groupby("patient_id")["label"]
                .apply(lambda x: "; ".join(x))
                .to_dict()
            )

    border_map: dict = _affected_ribs(merged, "seg_at_border") if check_border else {}
    split_map:  dict = {}
    if check_split:
        split_rows = merged[
            merged["split_start"].fillna(False).astype(bool)
            | merged["split_end"].fillna(False).astype(bool)
        ].copy()
        if not split_rows.empty:
            split_rows["label"] = split_rows.apply(
                lambda r: (
                    f"{display_from_vert(r['vert_level'], r['side'], side_long=True)} "
                    f"({'+'.join(f for f in ('split_start', 'split_end') if r.get(f))})"
                ),
                axis=1,
            )
            split_map = split_rows.groupby("patient_id")["label"].apply(lambda x: "; ".join(x)).to_dict()

    all_pids = sorted(merged["patient_id"].unique())
    records = []
    for pid in tqdm(all_pids, desc="Building exclusion table", unit="patient"):
        n_sides = rib_map.get(pid, 0)

        if n_sides < min_sides or n_sides > max_sides:
            expected = (
                f"{min_sides}"
                if min_sides == max_sides
                else f"{min_sides}–{max_sides}"
            )
            r_rib = f"{n_sides} sides (expected {expected})"
        else:
            r_rib = ""

        missing = []
        if pid in meta_per_patient.index:
            row = meta_per_patient.loc[pid]
            for col in meta_cols:
                if col in row.index and pd.isna(row[col]):
                    missing.append(col)
        r_meta = "; ".join(missing)

        r_components = components_map.get(pid, "")
        r_border = border_map.get(pid, "")
        r_split  = split_map.get(pid, "")

        records.append({
            "patient_id": pid,
            "n_rib_sides": n_sides,
            "excluded": bool(r_rib or r_meta or r_components or r_border or r_split),
            "reason_rib_count": r_rib,
            "reason_missing_metadata": r_meta,
            "reason_seg_components": r_components,
            "reason_seg_at_border": r_border,
            "reason_split": r_split,
        })

    excl_df = pd.DataFrame(records)

    n_rib    = (excl_df["reason_rib_count"] != "").sum()
    n_meta   = (excl_df["reason_missing_metadata"] != "").sum()
    n_comp   = (excl_df["reason_seg_components"] != "").sum()
    n_border = (excl_df["reason_seg_at_border"] != "").sum()
    n_split  = (excl_df["reason_split"] != "").sum()
    n_excl   = excl_df["excluded"].sum()
    n_total  = len(excl_df)

    logger.info(
        f"Exclusions — rib count: {n_rib}, missing metadata: {n_meta}, "
        f"seg components: {n_comp}, seg_at_border: {n_border}, split: {n_split} | "
        f"total: {n_excl}/{n_total} ({100 * n_excl / n_total:.1f}%)"
    )

    included_ids = excl_df.loc[~excl_df["excluded"], "patient_id"].values
    included_df = merged[merged["patient_id"].isin(included_ids)].copy()
    logger.info(
        f"Included cohort: {included_df['patient_id'].nunique():,} patients | "
        f"{len(included_df):,} rib-side rows"
    )
    return included_df, excl_df


# ── Rib-count audit ──────────────────────────────────────────────────────────

def audit_rib_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Per-patient audit of rib-side count.

    Returns a DataFrame indexed by ``patient_id`` with columns
    ``n_rib_sides``, ``vert_levels_present``, ``n_vert_levels``, ``anomaly``,
    ``anomaly_type``.
    """
    def _agg(g):
        return pd.Series({
            "n_rib_sides": len(g),
            "vert_levels_present": sorted(g["vert_level"].unique().tolist()),
            "n_vert_levels": g["vert_level"].nunique(),
        })

    audit = df.groupby("patient_id").apply(_agg, include_groups=False).reset_index()
    audit["anomaly"] = audit["n_rib_sides"] != EXPECTED_RIB_SIDES

    def _classify(row):
        if not row["anomaly"]:
            return "normal"
        if row["n_rib_sides"] < EXPECTED_RIB_SIDES:
            return "fewer_than_24_ribs"
        return "more_than_24_ribs"

    audit["anomaly_type"] = audit.apply(_classify, axis=1)

    n_anom = audit["anomaly"].sum()
    n_total = len(audit)
    logger.info(
        f"Rib count audit: {n_anom}/{n_total} patients have anomalous counts "
        f"({100 * n_anom / n_total:.1f}%)"
    )
    dist = audit["n_rib_sides"].value_counts().sort_index()
    logger.info(f"Distribution of rib-side counts:\n{dist.to_string()}")

    return audit


# ── Missingness audit ────────────────────────────────────────────────────────

def audit_missingness(
    df: pd.DataFrame, shape_cols: list[str], meta_cols: list[str]
) -> pd.DataFrame:
    """Report missingness per column, sorted by ``%`` missing (desc).

    Only columns with at least one missing value are returned.
    """
    target_cols = [c for c in shape_cols + meta_cols if c in df.columns]
    n = len(df)
    miss = df[target_cols].isnull().sum()
    report = pd.DataFrame({
        "n_missing": miss,
        "pct_missing": (miss / n * 100).round(2),
        "n_present": n - miss,
    })
    report = report[report["n_missing"] > 0].sort_values("pct_missing", ascending=False)
    logger.info(f"Missingness (cols with any missing):\n{report.to_string()}")
    return report


# ── Table 1 ──────────────────────────────────────────────────────────────────

def _fmt_mean_sd(series: pd.Series) -> str:
    return f"{series.mean():.2f} ± {series.std():.2f}"


def _fmt_median_iqr(series: pd.Series) -> str:
    q25, q50, q75 = series.quantile([0.25, 0.50, 0.75])
    return f"{q50:.2f} [{q25:.2f}–{q75:.2f}]"


def _is_skewed(series: pd.Series, threshold: float = 1.0) -> bool:
    return abs(series.dropna().skew()) > threshold


def compute_table1(
    df: pd.DataFrame,
    continuous_cols: list[str],
    categorical_cols: list[str],
    strat_col: str = "sex",
) -> pd.DataFrame:
    """Standard Table 1: descriptive statistics overall and stratified by ``strat_col``.

    Continuous cols: mean ± SD; if ``|skewness| > 1`` use median [IQR] instead.
    Categorical cols: ``n (%)``.
    Two-group p-value: Welch t-test (continuous) or chi-squared (categorical).
    """
    groups = (
        df[strat_col].cat.categories.tolist()
        if hasattr(df[strat_col], "cat")
        else df[strat_col].unique()
    )
    group_dfs = {g: df[df[strat_col] == g] for g in groups}
    rows = []

    def _row(variable, overall_str, group_strs, p_val, note=""):
        r = {"Variable": variable, "Overall": overall_str}
        r.update(group_strs)
        r["p-value"] = f"{p_val:.3f}" if p_val is not None else "—"
        r["Note"] = note
        return r

    n_overall = len(df)
    n_groups = {g: len(gdf) for g, gdf in group_dfs.items()}
    rows.append(_row(
        "N",
        str(n_overall),
        {str(g): str(n) for g, n in n_groups.items()},
        None,
    ))

    for col in continuous_cols:
        if col not in df.columns:
            continue
        s = df[col].dropna()
        skewed = _is_skewed(s)
        note = "median [IQR] (skewed)" if skewed else "mean ± SD"
        fmt = _fmt_median_iqr if skewed else _fmt_mean_sd

        overall_str = fmt(s)
        group_strs = {str(g): fmt(gdf[col].dropna()) for g, gdf in group_dfs.items()}

        p_val = None
        if len(groups) == 2:
            g0, g1 = groups
            a = group_dfs[g0][col].dropna()
            b = group_dfs[g1][col].dropna()
            if len(a) > 1 and len(b) > 1:
                _, p_val = stats.ttest_ind(a, b, equal_var=False)

        rows.append(_row(col, overall_str, group_strs, p_val, note))

    for col in categorical_cols:
        if col not in df.columns:
            continue
        vc = df[col].value_counts(dropna=False)
        overall_str = "<br>".join(f"{k}: {v} ({100*v/n_overall:.1f}%)" for k, v in vc.items())
        group_strs = {}
        is_strat_row = col == strat_col
        for g, gdf in group_dfs.items():
            vc_g = gdf[col].value_counts(dropna=False)
            ng = len(gdf)
            if is_strat_row:
                # Stratifier's own column: just the count in this group.
                group_strs[str(g)] = str(int(vc_g.get(g, 0)))
            else:
                group_strs[str(g)] = "<br>".join(
                    f"{k}: {v} ({100*v/ng:.1f}%)" for k, v in vc_g.items()
                )

        p_val = None
        try:
            contingency = pd.crosstab(df[col].dropna(), df[strat_col][df[col].notna()])
            if contingency.shape[0] > 1 and contingency.shape[1] > 1:
                _, p_val, _, _ = stats.chi2_contingency(contingency)
        except ValueError as exc:
            logger.debug("Chi-squared test failed for %s: %s", col, exc)

        rows.append(_row(col, overall_str, group_strs, p_val, "n (%)"))

    return pd.DataFrame(rows)


def normality_summary(df: pd.DataFrame, cols: list[str], sample_n: int = 5_000) -> pd.DataFrame:
    """Shapiro-Wilk test on a random sample per column.

    With large N, Shapiro-Wilk almost always rejects; treat as a guide alongside
    visual inspection (skewness, Q-Q plots).
    """
    sample = df.sample(min(len(df), sample_n), random_state=42)
    results = []
    for col in tqdm(cols, desc="Normality tests", unit="col"):
        if col not in df.columns:
            continue
        vals = sample[col].dropna().values
        if len(vals) < 20:
            continue
        skew = float(pd.Series(vals).skew())
        kurt = float(pd.Series(vals).kurtosis())
        try:
            w, p = stats.shapiro(vals[:5000])
        except ValueError as exc:
            logger.debug("Shapiro-Wilk test failed for %s: %s", col, exc)
            w, p = np.nan, np.nan
        results.append({
            "column": col,
            "n": len(vals),
            "skewness": round(skew, 3),
            "kurtosis": round(kurt, 3),
            "shapiro_W": round(w, 4) if not np.isnan(w) else np.nan,
            "shapiro_p": round(p, 4) if not np.isnan(p) else np.nan,
            "likely_normal": bool(abs(skew) < 1.0 and abs(kurt) < 2.0),
        })
    return pd.DataFrame(results).set_index("column")
