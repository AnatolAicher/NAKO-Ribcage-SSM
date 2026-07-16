"""Forest-plot helpers (rule + point + 95% CI) shared by the adjusted analysis.

The forest builders are consumed by :mod:`adjusted.plots_adjusted_altair`.
"""
from __future__ import annotations

import altair as alt
import pandas as pd

from settings import FDR_DISPLAY_ALPHA
from utils.shape_labels import shape_label as _label


def _forest_long(
    lm_results: pd.DataFrame,
    *,
    pred_label: callable | None = None,
) -> pd.DataFrame:
    """Build long-form (shape, predictor, beta_std, lo, hi, q, sig_cat) for forests."""
    df = lm_results.copy()
    if "shape_label" not in df.columns:
        df["shape_label"] = df["shape_param"].map(_label)
    pred_col = "predictor"
    if pred_label is not None:
        df["pred_label"] = df[pred_col].map(pred_label)
    else:
        df["pred_label"] = df[pred_col].astype(str)

    if "ci_low" in df.columns and "ci_high" in df.columns:
        se_raw = (df["ci_high"].astype(float) - df["ci_low"].astype(float)) / (2 * 1.96)
        raw_beta = df["beta"].astype(float).abs().replace(0, 1.0)
        beta = df["beta_std"].astype(float)
        se_std = se_raw * (beta.abs() / raw_beta)
    else:
        se = df["se"].astype(float)
        raw_beta = df["beta"].astype(float).abs().replace(0, 1.0)
        beta = df["beta_std"].astype(float)
        se_std = se * (beta.abs() / raw_beta)

    df["beta_std"] = beta
    df["lo"] = beta - 1.96 * se_std
    df["hi"] = beta + 1.96 * se_std
    q = df["p_value_fdr"].astype(float)
    sig = q < FDR_DISPLAY_ALPHA
    df["sig_cat"] = "ns"
    df.loc[sig & (beta >= 0), "sig_cat"] = "positive"
    df.loc[sig & (beta < 0),  "sig_cat"] = "negative"
    df["q"] = q
    return df.dropna(subset=["beta_std"]).reset_index(drop=True)


_SIG_CAT_SCALE = alt.Scale(
    domain=["positive", "negative", "ns"],
    range=["#0072B2", "#A8195C", "#9A9A9A"],
)


def _assign_grid(
    df: pd.DataFrame,
    label_col: str,
    *,
    columns: int = 2,
    order: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Annotate ``df`` with ``_row``/``_col`` Becker-grid indices.

    Each unique value in ``label_col`` is placed in a fixed ``(row, col)`` cell
    (row-major), so the chart can ``.facet(row=..., column=...)`` and any cell
    whose label has no data renders as truly empty (no axis, no rule, no rule).
    Returns ``(df_with_indices, ordered_labels)``.
    """
    labels = list(order) if order is not None else sorted(df[label_col].unique().tolist())
    pos = {lbl: (i // columns, i % columns) for i, lbl in enumerate(labels)}
    out = df.copy()
    out["_row"] = out[label_col].map(lambda v: pos.get(v, (0, 0))[0])
    out["_col"] = out[label_col].map(lambda v: pos.get(v, (0, 0))[1])
    return out, labels


def _build_forest_facet(
    long: pd.DataFrame,
    *,
    pred_order: list[str],
    label_col: str = "shape_label",
    label_order: list[str] | None = None,
    columns: int = 2,
    width_px: int,
    title: alt.TitleParams | str,
) -> alt.Chart:
    """Becker-wrapped forest plot: rule + point + zero, faceted on (row, col)."""
    import settings as S
    long, labels = _assign_grid(long, label_col, columns=columns, order=label_order)
    sig_sel = alt.selection_point(fields=["sig_cat"], bind="legend", name="sig_sel")
    sig_opacity = alt.condition(sig_sel, alt.value(1.0), alt.value(0.15))
    base = alt.Chart(long).encode(
        y=alt.Y("pred_label:N", sort=pred_order, title=None),
        color=alt.Color("sig_cat:N", scale=_SIG_CAT_SCALE,
                        legend=alt.Legend(title="Direction · sig.")),
        tooltip=[alt.Tooltip(f"{label_col}:N", title="Shape"),
                 alt.Tooltip("pred_label:N", title="Predictor"),
                 alt.Tooltip("beta_std:Q", title="β (std)", format=".3f"),
                 alt.Tooltip("lo:Q", title="CI low", format=".3f"),
                 alt.Tooltip("hi:Q", title="CI high", format=".3f"),
                 alt.Tooltip("q:Q", title="q (FDR)", format=".2g")],
    )
    rule  = base.mark_rule(strokeWidth=1.5).encode(
        x="lo:Q", x2="hi:Q", opacity=sig_opacity)
    point = base.mark_point(filled=True, size=70, stroke="#222",
                            strokeWidth=0.5).encode(
        x=alt.X("beta_std:Q", title="Std. β"), opacity=sig_opacity)
    # Bold solid β=0 reference line (forest-plot convention).
    zero  = base.mark_rule(color="#222", strokeWidth=2.0).encode(x=alt.datum(0))

    panel = ((rule + point + zero)
             .properties(width=width_px)
             .add_params(sig_sel))
    header = alt.Header(labels=True, labelOrient="top",
                         labelAnchor="start",
                         labelFontSize=S.FONT_SIZE_TITLE_PT * 4 / 3,
                         labelFontWeight="bold")
    facet_field = (alt.Facet(label_col, type="nominal", sort=labels,
                              title=None, header=header)
                   if label_order is not None
                   else alt.Facet(label_col, type="nominal", title=None,
                                  header=header))
    chart = (panel.facet(facet=facet_field, columns=columns)
                  .resolve_scale(x="shared")
                  .properties(title=title))
    return chart

