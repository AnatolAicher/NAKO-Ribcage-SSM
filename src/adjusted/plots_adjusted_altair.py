"""Altair figures for the adjusted (confounder-corrected) analysis."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

import altair as alt
import numpy as np

from adjusted.adjusted import PREDICTOR_LABELS
from bivariate.plots_bivariate_altair import _build_forest_facet, _forest_long
from settings import FDR_DISPLAY_ALPHA
from utils.altair_theme import (
    STAR_LEGEND,
    fdr_masked_heatmap,
    make_title,
    predictor_scale,
    width_for,
)
from utils.figure_export_altair import save_chart
from utils.shape_labels import shape_label as _slabel


def _plabel(pred: str) -> str:
    return PREDICTOR_LABELS.get(pred, pred)


def _strip_format_suffix(p: Path) -> Path:
    return p.with_suffix("") if p.suffix in {".png", ".svg", ".pdf", ".html"} else p


# ── Coefficient heatmap (β_std, diverging) ───────────────────────────────────

def plot_adj_heatmap(
    results: pd.DataFrame,
    out_stem: Path,
    *,
    value_col: str = "beta_std",
    title: str | None = None,
    model_label: str | None = None,
) -> None:
    """Heatmap of standardised β with FDR-non-significant cells masked."""
    out_stem = _strip_format_suffix(Path(out_stem))
    if title is None:
        prefix = f"Model {model_label} – " if model_label else ""
        title_text = f"{prefix}Adjusted OLS – standardised β"
    else:
        title_text = title
    subtitle = (f"Cluster-robust SE · FDR-masked at q ≥ {FDR_DISPLAY_ALPHA} "
                f"(empty cells) · {STAR_LEGEND}")

    pred_order  = [p for p in PREDICTOR_LABELS if p in results["predictor"].unique()]
    pred_labels = [_plabel(p) for p in pred_order]

    z = results.pivot(index="shape_param", columns="predictor", values=value_col)
    q = results.pivot(index="shape_param", columns="predictor", values="p_value_fdr")
    z = z.reindex(columns=pred_order); q = q.reindex(columns=pred_order)
    z.columns = q.columns = pred_labels
    new_idx = [_slabel(c) for c in z.index]
    z.index = q.index = new_idx

    chart = fdr_masked_heatmap(
        z, q,
        value_label="Std. β",
        fdr_threshold=FDR_DISPLAY_ALPHA,
        width_class="full",
        title=make_title(title_text, subtitle=subtitle),
        tickangle_x=-25,
        col_order=pred_labels,
        row_order=new_idx,
        row_title="Shape",
        col_title="Predictor",
    )
    save_chart(chart, out_stem, title=title_text, width_class="full")


# ── Partial R² heatmap (sequential 0..max) ───────────────────────────────────

def plot_partial_r2_heatmap(
    results: pd.DataFrame,
    out_stem: Path,
    *,
    title: str | None = None,
    model_label: str | None = None,
) -> None:
    """Heatmap of partial R² values; FDR-non-significant cells masked."""
    out_stem = _strip_format_suffix(Path(out_stem))

    pred_order  = [p for p in PREDICTOR_LABELS if p in results["predictor"].unique()]
    pred_labels = [_plabel(p) for p in pred_order]

    z = results.pivot(index="shape_param", columns="predictor", values="partial_r2")
    q = results.pivot(index="shape_param", columns="predictor", values="p_value_fdr")
    z = z.reindex(columns=pred_order); q = q.reindex(columns=pred_order)
    z.columns = q.columns = pred_labels
    new_idx = [_slabel(c) for c in z.index]
    z.index = q.index = new_idx

    if title is None:
        prefix = f"Model {model_label} – " if model_label else ""
        title_text = f"{prefix}Adjusted OLS – partial R²"
    else:
        title_text = title
    subtitle = (f"FDR-masked at q ≥ {FDR_DISPLAY_ALPHA} (empty cells) · "
                f"{STAR_LEGEND}")
    chart = fdr_masked_heatmap(
        z, q,
        value_label="Partial R²",
        text_format=".3f",
        diverging=False,
        sequential_var="magnitude",
        fdr_threshold=FDR_DISPLAY_ALPHA,
        width_class="full",
        title=make_title(title_text, subtitle=subtitle),
        tickangle_x=-25,
        col_order=pred_labels,
        row_order=new_idx,
        row_title="Shape",
        col_title="Predictor",
        show_stars=True,
    )
    save_chart(chart, out_stem, title=title_text, width_class="full")


# ── Forest plots per shape parameter (point + 95% CI) ────────────────────────

def plot_forest_plots(
    results: pd.DataFrame,
    shape_cols: list[str],
    out_stem: Path,
    *,
    model_label: str | None = None,
) -> None:
    """One panel per shape parameter: predictors as horizontal CI bars + dot for β."""
    out_stem = _strip_format_suffix(Path(out_stem))
    available = [c for c in shape_cols if c in results["shape_param"].values]
    if not available:
        return

    pred_order  = [p for p in PREDICTOR_LABELS if p in results["predictor"].unique()]
    pred_labels = [_plabel(p) for p in pred_order]
    sub = results[results["predictor"].isin(pred_order)].copy()
    long = _forest_long(sub, pred_label=_plabel)
    long["shape_label"] = long["shape_param"].map(_slabel)
    shape_order = [_slabel(s) for s in available]

    prefix = f"Model {model_label} – " if model_label else ""
    title_text = f"{prefix}Adjusted OLS – standardised β with 95 % CI"
    subtitle = "Cluster-robust SE · BH-FDR"
    chart = _build_forest_facet(
        long, pred_order=pred_labels, label_col="shape_label",
        label_order=shape_order,
        columns=2, width_px=width_for("half"),
        title=make_title(title_text, subtitle=subtitle),
    )
    save_chart(chart, out_stem, title="Adjusted OLS – forest plots",
               width_class="full")

