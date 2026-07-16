"""Per-(PC, rib, feature) regressions linking SSM PC scores to radiomics features.

For each (PC *j*, rib *r* = (vert_level, side), feature *f*) triplet, fit a
univariate OLS across subjects::

    feature[r, f] ~ β · z(PC_j) + α

where ``z(PC_j)`` is the PC score divided by its column SD (so β has units of
"feature unit per 1 SD of PC"). The standardised effect ``β_std`` is obtained
by additionally z-scoring the feature column — equivalent to Pearson *r*.

BH-FDR is applied per-PC family (24 × 14 tests per PC), mirroring the
per-family convention in :mod:`ssm.pc_regression`.

The module is pure-function: inputs are DataFrames + column lists, output is a
single tidy DataFrame.  I/O lives in :mod:`ssm.run_radiomics_correlation`.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

from data_ingestion.loaders import ANALYSIS_SHAPE_COLS
from utils.paths import stage_dir
from utils.rib_labels import vert_to_anatomical

logger = logging.getLogger(__name__)


def load_inputs(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Load PC scores and per-rib radiomics from a pipeline run dir.

    Parameters
    ----------
    run_dir
        Top-level per-run directory.  Must contain
        ``ssm_pca/pc_scores_surface.csv`` and
        ``ingestion/analytic_clean.parquet``.

    Returns
    -------
    pc_df          : one row per patient — ``patient_id, PC_1, …, PC_K``.
    radiomics_df   : one row per (patient, vert_level, side) — radiomics columns.
    pc_cols        : sorted list of PC column names.
    """
    run_dir = Path(run_dir)
    pc_path       = stage_dir(run_dir, "ssm_pca")    / "pc_scores_surface.csv"
    parquet_path  = stage_dir(run_dir, "ingestion") / "analytic_clean.parquet"
    if not pc_path.exists():
        raise FileNotFoundError(f"PC scores not found: {pc_path}")
    if not parquet_path.exists():
        raise FileNotFoundError(f"Radiomics parquet not found: {parquet_path}")

    pc_df = pd.read_csv(pc_path)
    pc_cols = sorted(
        (c for c in pc_df.columns if c.startswith("PC_")),
        key=lambda c: int(c.removeprefix("PC_")),
    )
    pc_df = pc_df[["patient_id", *pc_cols]].copy()

    radiomics_df = pd.read_parquet(parquet_path)
    keep_cols = ["patient_id", "vert_level", "side", *ANALYSIS_SHAPE_COLS]
    missing = [c for c in keep_cols if c not in radiomics_df.columns]
    if missing:
        raise KeyError(
            f"Radiomics parquet missing expected columns: {missing}. "
            f"Got: {list(radiomics_df.columns)}"
        )
    radiomics_df = radiomics_df[keep_cols].copy()

    logger.info(
        f"Loaded {len(pc_df):,} patients with {len(pc_cols)} PCs "
        f"and {len(radiomics_df):,} rib-rows ({radiomics_df['patient_id'].nunique():,} patients)"
    )
    return pc_df, radiomics_df, pc_cols


def join_pc_with_ribs(
    pc_df: pd.DataFrame,
    radiomics_df: pd.DataFrame,
) -> pd.DataFrame:
    """Inner-join PC scores with per-rib radiomics on ``patient_id``.

    Each subject's PC vector is broadcast across their (up to 24) rib rows.
    """
    merged = radiomics_df.merge(pc_df, on="patient_id", how="inner")
    logger.info(
        f"Joined: {len(merged):,} rib-rows  "
        f"({merged['patient_id'].nunique():,} patients × up to 24 ribs)"
    )
    return merged


def _rib_groups(joined: pd.DataFrame) -> list[tuple[int, str]]:
    """Return the (vert_level, side) pairs present in the join, ordered."""
    pairs = (
        joined[["vert_level", "side"]]
        .drop_duplicates()
        .sort_values(["vert_level", "side"], kind="stable")
    )
    return list(map(tuple, pairs.itertuples(index=False, name=None)))


def compute_effects(
    joined: pd.DataFrame,
    pc_cols: list[str],
    feature_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Run all per-(PC, rib, feature) OLS regressions and return a tidy table.

    The PC score is z-scored across the full cohort (mean 0, sd 1) so β reads
    as "change in feature per +1 SD of PC".  ``β_std`` further z-scores the
    feature column (equivalent to Pearson *r*).

    FDR is applied per-PC family (all ribs × features for that PC) via BH.

    Parameters
    ----------
    joined
        Output of :func:`join_pc_with_ribs`.
    pc_cols
        PC column names to regress on (e.g. ``["PC_1", ..., "PC_K"]``).
    feature_cols
        Radiomics columns to regress.  Defaults to
        :data:`data_ingestion.loaders.ANALYSIS_SHAPE_COLS`.

    Returns
    -------
    DataFrame with columns ``pc, vert_level, anatomical_rib, side, feature,
    beta_native, beta_std, ci_low_native, ci_high_native, se_native,
    p_value, q_value, n``.  One row per (PC, rib, feature) regression.
    """
    if feature_cols is None:
        feature_cols = list(ANALYSIS_SHAPE_COLS)

    # Z-score PCs once across the cohort — patient-level, not rib-level.
    # Each patient appears up to 24 times in `joined`, but the PC value is
    # identical across their rows, so computing SD from drop_duplicates is
    # the patient-level (correct) statistic.
    pc_stats = (
        joined.drop_duplicates("patient_id")[pc_cols]
        .agg(["mean", "std"])
    )
    pc_z = joined[pc_cols].copy()
    for c in pc_cols:
        mu  = float(pc_stats.loc["mean", c])
        sd  = float(pc_stats.loc["std",  c])
        if sd > 0:
            pc_z[c] = (joined[c] - mu) / sd
        else:
            pc_z[c] = 0.0
            logger.warning(f"PC column {c!r} has zero SD; regressions on it will be NaN.")

    rib_pairs = _rib_groups(joined)
    rows: list[dict] = []
    log_every = max(1, len(pc_cols) * len(rib_pairs) // 20)
    progress = 0
    total = len(pc_cols) * len(rib_pairs) * len(feature_cols)

    for pc in pc_cols:
        pc_block_rows: list[dict] = []
        for vert_level, side in rib_pairs:
            mask_rib = (joined["vert_level"] == vert_level) & (joined["side"] == side)
            x_full = pc_z.loc[mask_rib, pc].to_numpy()
            for feat in feature_cols:
                y_full = joined.loc[mask_rib, feat].to_numpy()
                ok = np.isfinite(x_full) & np.isfinite(y_full)
                n = int(ok.sum())
                row = {
                    "pc":               pc,
                    "vert_level":       int(vert_level),
                    "anatomical_rib":   int(vert_to_anatomical(vert_level)),
                    "side":             str(side),
                    "feature":          feat,
                    "beta_native":      float("nan"),
                    "beta_std":         float("nan"),
                    "ci_low_native":    float("nan"),
                    "ci_high_native":   float("nan"),
                    "se_native":        float("nan"),
                    "p_value":          float("nan"),
                    "n":                n,
                }
                if n >= 3:
                    x = x_full[ok]
                    y = y_full[ok]
                    y_sd = float(np.std(y, ddof=1))
                    try:
                        model = sm.OLS(y, sm.add_constant(x)).fit()
                        beta   = float(model.params[1])
                        se     = float(model.bse[1])
                        p_val  = float(model.pvalues[1])
                        ci     = model.conf_int()
                        ci_lo  = float(ci[1, 0])
                        ci_hi  = float(ci[1, 1])
                        row.update(
                            beta_native    = beta,
                            beta_std       = beta / y_sd if y_sd > 0 else float("nan"),
                            ci_low_native  = ci_lo,
                            ci_high_native = ci_hi,
                            se_native      = se,
                            p_value        = p_val,
                        )
                    except (np.linalg.LinAlgError, ValueError) as exc:
                        logger.debug(
                            "OLS failed for pc=%s vert=%d side=%s feat=%s: %s",
                            pc, vert_level, side, feat, exc,
                        )
                pc_block_rows.append(row)
            progress += len(feature_cols)
            if progress % log_every == 0 or progress == total:
                logger.info(f"  regressions: {progress:,}/{total:,}")

        # Per-PC BH-FDR across the (rib × feature) family for this PC.
        p_arr = np.asarray([r["p_value"] for r in pc_block_rows], dtype=float)
        q_arr = np.full_like(p_arr, np.nan)
        valid = np.isfinite(p_arr)
        if valid.any():
            _, q_valid, _, _ = multipletests(p_arr[valid], method="fdr_bh")
            q_arr[valid] = q_valid
        for r, q in zip(pc_block_rows, q_arr):
            r["q_value"] = float(q)
        rows.extend(pc_block_rows)

    effects = pd.DataFrame(rows, columns=[
        "pc", "vert_level", "anatomical_rib", "side", "feature",
        "beta_native", "beta_std", "ci_low_native", "ci_high_native",
        "se_native", "p_value", "q_value", "n",
    ])
    logger.info(
        f"Computed {len(effects):,} effects "
        f"({len(pc_cols)} PCs × {len(rib_pairs)} ribs × {len(feature_cols)} features)"
    )
    return effects


def top_effects(effects: pd.DataFrame, n: int = 200) -> pd.DataFrame:
    """Return the top-``n`` rows ranked by |beta_std|."""
    return (
        effects.assign(_abs=lambda d: d["beta_std"].abs())
        .sort_values("_abs", ascending=False, kind="stable")
        .drop(columns="_abs")
        .head(n)
        .reset_index(drop=True)
    )
