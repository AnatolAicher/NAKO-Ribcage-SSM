"""Altair 2D figures for the surface SSM (scree, loadings, PC regression,
pair-plots, β-vector field).

The 3D mosaics (``plot_pc_deformations``, ``plot_mean_shape_gallery``) have no
Altair equivalent and live in :mod:`ssm.plots_ssm`.
"""
from __future__ import annotations

import logging
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd

from settings import ADJ_PREDICTORS, DAG_ADJUSTMENT_SETS, FDR_DISPLAY_ALPHA, N_PCS_DISPLAY
from utils.altair_theme import (
    STAR_LEGEND,
    continuous_scale,
    fdr_masked_heatmap,
    kde_long,
    make_title,
    pc_axis,
    rib_side_color,
    sex_scale,
    smoking_scale,
    subscript,
    width_for,
)
from utils.altair_binning import bin_xy
from utils.colors import PALETTE, color
from utils.design import add_design_columns
from utils.figure_export_altair import save_chart, save_chart_split
from utils.rib_labels import seg_to_anatomical

logger = logging.getLogger(__name__)


def _strip_format_suffix(p: Path) -> Path:
    return p.with_suffix("") if p.suffix in {".png", ".svg", ".pdf", ".html"} else p


def _pc_int(label: str) -> int:
    s = str(label)
    return int(s.removeprefix("PC_")) if s.startswith("PC_") else 999


def pc_arrow_data(
    beta_df: pd.DataFrame, predictor: str, ev, pc_cols: list[str],
    *, scale_sd: float = 2.0,
) -> pd.DataFrame:
    """β-vector arrow per off-diagonal PC pair, in native PC-score units.

    For a Δ of ``scale_sd`` predictor SDs the shift on PC_k is
    ``β_std · sd(PC_k) · scale_sd`` (sd(PC_k) = √ev_k), so the arrow lands in the
    same units as the pair-plot bubbles.
    """
    if beta_df.empty or "predictor" not in beta_df.columns:
        return pd.DataFrame(columns=["pc_x", "pc_y", "bx", "by", "zero"])
    sub = beta_df[beta_df["predictor"] == predictor]
    bmap = dict(zip(sub["pc"], sub["beta_std"]))
    ev = np.asarray(ev, dtype=float)
    rows: list[dict] = []
    for i, px in enumerate(pc_cols):
        for py in pc_cols[i + 1:]:
            bx = float(bmap.get(px, 0.0)) * float(np.sqrt(ev[_pc_int(px) - 1])) * scale_sd
            by = float(bmap.get(py, 0.0)) * float(np.sqrt(ev[_pc_int(py) - 1])) * scale_sd
            rows.append({"pc_x": px.replace("PC_", "PC"), "pc_y": py.replace("PC_", "PC"),
                         "bx": bx, "by": by, "zero": 0.0})
    return pd.DataFrame(rows)


# ── Regression heatmap (PC × predictor, FDR-masked) ──────────────────────────

def plot_regression_heatmap_surface(
    lm_results, out_stem: str | Path, *,
    title: str | None = None, subtitle: str | None = None,
) -> None:
    """Standardised β heatmap for PC-score regression."""
    out_stem = _strip_format_suffix(Path(out_stem))
    if isinstance(lm_results, pd.DataFrame) and lm_results.empty:
        return

    z = lm_results.pivot(index="predictor", columns="pc", values="beta_std")
    q = lm_results.pivot(index="predictor", columns="pc", values="p_value_fdr")
    sorted_cols = sorted(z.columns, key=_pc_int)
    z = z[sorted_cols]; q = q[sorted_cols]

    title_text = title or "Surface SSM — Adjusted PC regression (standardised β)"
    subtitle = subtitle or (f"HC3-robust SE · FDR-masked at q ≥ {FDR_DISPLAY_ALPHA} (empty cells) · "
                            f"{STAR_LEGEND}")
    common = dict(
        value_label="Std. β",
        fdr_threshold=FDR_DISPLAY_ALPHA,
        width_class="full",
        title=make_title(title_text, subtitle=subtitle),
        y_axis_title="Predictor",
        row_title="Predictor",
        col_title="PC",
    )
    html_chart = fdr_masked_heatmap(
        z, q, col_order=sorted_cols,
        x_axis=pc_axis(sorted_cols, every=5, title="PC"),
        annotate=False, show_stars=False,
        **common,
    )
    top_cols = sorted_cols[:N_PCS_DISPLAY]
    static_chart = fdr_masked_heatmap(
        z[top_cols], q[top_cols], col_order=top_cols,
        x_axis_title="PC",
        **common,
    )
    save_chart_split(
        html_chart, static_chart, out_stem,
        title=title_text, width_class="full",
    )
    logger.info("Saved regression heatmap → %s", out_stem)


# ── PC × rib loading magnitude heatmap (sequential) ──────────────────────────

def plot_pc_loadings_per_rib(
    pca,
    rib_offsets: np.ndarray,
    rib_ids: list[str] | None,
    out_stem: Path,
    *,
    n_pcs: int | None = None,
) -> None:
    """Per-rib mean displacement magnitude for each PC mode (row-normalised)."""
    out_stem = _strip_format_suffix(Path(out_stem))
    n_pts = pca.components_.shape[1] // 3
    rib_offsets = list(rib_offsets) + [n_pts]
    n_ribs = len(rib_offsets) - 1
    n_pcs = pca.n_components_ if n_pcs is None else min(n_pcs, pca.n_components_)

    if rib_ids is None or len(rib_ids) < n_ribs:
        rib_ids = [f"Rib {i+1}" for i in range(n_ribs)]
    rib_ids = list(rib_ids)[:n_ribs]
    pc_labels = [f"PC{k+1}" for k in range(n_pcs)]

    z = np.zeros((n_pcs, n_ribs), dtype=float)
    for k in range(n_pcs):
        comp = pca.components_[k].reshape(n_pts, 3)
        mags = np.linalg.norm(comp, axis=1)
        for ri in range(n_ribs):
            s, e = rib_offsets[ri], rib_offsets[ri + 1]
            z[k, ri] = float(mags[s:e].mean()) if e > s else 0.0
    row_norm = z / np.maximum(z.sum(axis=1, keepdims=True), 1e-12)

    z_pivot = pd.DataFrame(row_norm, index=pc_labels, columns=rib_ids)

    title_text = "PC loading concentration per rib"
    subtitle = "Row-normalised mean ‖component‖ per rib identity"
    common = dict(
        q_pivot=None,
        value_label="rel. ‖Δ‖",
        text_format=".3f",
        annotate=False,
        diverging=False,
        sequential_var="magnitude",
        vmax=float(row_norm.max()),
        width_class="full",
        title=make_title(title_text, subtitle=subtitle),
        tickangle_x=-45,
        col_order=rib_ids,
        row_title="PC",
        col_title="Rib",
    )
    html_chart = fdr_masked_heatmap(
        z_pivot, row_order=pc_labels,
        y_axis=pc_axis(pc_labels, every=5, title="PC"),
        **common,
    )
    top_pcs = pc_labels[:N_PCS_DISPLAY]
    static_chart = fdr_masked_heatmap(
        z_pivot.loc[top_pcs], row_order=top_pcs,
        **common,
    )
    save_chart_split(
        html_chart, static_chart, out_stem,
        title=title_text, width_class="full",
    )


# ── PC loadings per rib — mirrored-bar companion ─────────────────────────────

def plot_pc_loadings_per_rib_histo(
    pca,
    rib_offsets: np.ndarray,
    rib_ids: list[str] | None,
    out_stem: Path,
    *,
    n_pcs_display: int = N_PCS_DISPLAY,
) -> None:
    """Population-pyramid view of the top ``n_pcs_display`` PC loadings.

    y-axis = anatomical rib (1..12), x-axis = signed relative ‖component‖
    (L stays positive, R negated to the left). Single chart, facetted by PC.
    """
    out_stem = _strip_format_suffix(Path(out_stem))
    n_pts = pca.components_.shape[1] // 3
    offsets = list(rib_offsets) + [n_pts]
    n_ribs  = len(offsets) - 1
    n_pcs   = min(n_pcs_display, pca.n_components_)
    if rib_ids is None or len(rib_ids) < n_ribs:
        rib_ids = [f"Rib {i+1} L" for i in range(n_ribs)]
    rib_ids = list(rib_ids)[:n_ribs]

    def _parse(rid: str) -> tuple[int, str]:
        parts = str(rid).split()
        if len(parts) >= 3 and parts[0] == "Rib":
            return int(parts[1]), parts[2].upper()
        return 0, "L"

    parsed = [_parse(r) for r in rib_ids]

    rows: list[dict] = []
    for k in range(n_pcs):
        comp = pca.components_[k].reshape(n_pts, 3)
        mags = np.linalg.norm(comp, axis=1)
        per_rib_mag = []
        row_total = 0.0
        for ri in range(n_ribs):
            s, e = offsets[ri], offsets[ri + 1]
            m = float(mags[s:e].mean()) if e > s else 0.0
            per_rib_mag.append(m)
            row_total += m
        norm = row_total if row_total > 0 else 1.0
        for (rib, side), mag in zip(parsed, per_rib_mag):
            rows.append({
                "pc": f"PC{k+1}",
                "rib": rib,
                "side": side,
                "value": mag / norm,
                "signed_value": (mag / norm) * (-1.0 if side == "R" else 1.0),
            })
    long = pd.DataFrame(rows)
    if long.empty:
        return

    pc_order = [f"PC{k+1}" for k in range(n_pcs)]
    abs_max = float(long["signed_value"].abs().max()) * 1.05

    rib_levels = sorted(long["rib"].unique())
    long["color_key"] = long["rib"].astype(str) + "-" + long["side"]
    rib_color, color_sel = rib_side_color(rib_levels)

    title_text = "PC loadings per rib · mirrored bars"
    subtitle = f"Top {n_pcs} PCs · y = anatomical rib · x = signed relative ‖component‖"

    bars = (
        alt.Chart(long)
        .mark_bar()
        .encode(
            x=alt.X("signed_value:Q", title="rel. ‖Δ‖",
                    scale=alt.Scale(domain=[-abs_max, abs_max]),
                    stack=None,
                    axis=alt.Axis(labelExpr="abs(datum.value)")),
            x2=alt.X2(datum=0),
            y=alt.Y("rib:O", title="Rib (anatomical, 1 = T8)",
                    sort=rib_levels),
            color=rib_color,
            opacity=alt.condition(color_sel, alt.value(0.85), alt.value(0.1)),
            tooltip=[alt.Tooltip("pc:N", title="PC"),
                     alt.Tooltip("rib:O", title="Rib"),
                     alt.Tooltip("side:N"),
                     alt.Tooltip("value:Q", title="rel. ‖Δ‖", format=".3f")],
        )
        .add_params(color_sel)
        .properties(width=width_for("third"), height=180)
    )
    chart = (bars
             .facet(column=alt.Column("pc:N", sort=pc_order, title=None))
             .resolve_scale(x="shared")
             .properties(title=make_title(title_text, subtitle=subtitle)))
    save_chart(chart, out_stem, title=title_text, width_class="full")


# ── GPA convergence (line + markers, log y) ──────────────────────────────────

def plot_gpa_convergence(
    rms_history: np.ndarray | list[float],
    out_stem: Path,
    *,
    title: str | None = None,
) -> None:
    """Line plot of relative mean-shape change vs. GPA iteration (log y)."""
    out_stem = _strip_format_suffix(Path(out_stem))
    history = np.asarray(rms_history, dtype=float)
    if history.size == 0:
        return
    df = pd.DataFrame({"iter": np.arange(1, len(history) + 1), "rms": history})

    # Render the y-axis label with unicode subscripts (vega/SVG has no LaTeX).
    mu_new = subscript("μ", "new")
    mu_old = subscript("μ", "old")
    fro    = subscript("", "F")
    y_label = f"‖{mu_new} − {mu_old}‖{fro} / ‖{mu_old}‖{fro}"

    title_text = title or "GPA convergence"
    subtitle = "Relative mean-shape change per iteration (log y)"

    base = alt.Chart(df).encode(
        x=alt.X("iter:Q", title="GPA iteration",
                axis=alt.Axis(tickMinStep=1)),
        y=alt.Y("rms:Q", title=y_label, scale=alt.Scale(type="log")),
        tooltip=[alt.Tooltip("iter:Q"),
                 alt.Tooltip("rms:Q", format=".2e")],
    )
    chart = (
        (base.mark_line(color=color("sex", "Male"), strokeWidth=1.8)
         + base.mark_point(color=color("sex", "Male"), filled=True, size=50))
        .properties(width=width_for("half"),
                    title=make_title(title_text, subtitle=subtitle))
    )
    save_chart(chart, out_stem, title=title_text, width_class="half")


# ── Scree plot (per-PC bar + cumulative line on independent y-axes) ──────────

def plot_scree_surface(pca, out_stem: str | Path) -> None:
    """Per-PC variance bar + cumulative variance line + 80 % / 95 % reference."""
    out_stem = _strip_format_suffix(Path(out_stem))
    evr  = np.asarray(pca.explained_variance_ratio_, dtype=float)
    cumr = np.cumsum(evr)
    k    = len(evr)
    df = pd.DataFrame({
        "pc": [f"{i+1}" for i in range(k)],
        "pc_idx": np.arange(1, k+1),
        "evr": evr * 100, "cumr": cumr * 100,
    })

    bar_color  = color("sex", "Male")
    line_color = color("sex", "Female")

    # Shared colour scale → single legend with "Per-PC variance" and
    # "Cumulative variance" entries. Threshold rules + labels stay neutral
    # grey (no legend entry) since they're reference markers.
    series_scale = alt.Scale(
        domain=["Per-PC variance", "Cumulative variance"],
        range=[bar_color, line_color],
    )
    series_legend = alt.Legend(title=None, orient="right")

    # All k PCs are shown but only every 5th gets a tick label (PC 1 included).
    pc_label_expr = (
        "(toNumber(datum.value) === 1 || toNumber(datum.value) % 5 === 0) "
        "? datum.value : ''"
    )
    x_axis = alt.Axis(title="Principal component (PC)", labelAngle=0,
                      labelExpr=pc_label_expr)
    x_enc = alt.X("pc:O", sort=df["pc"].tolist(), axis=x_axis)

    bar = (alt.Chart(df).mark_bar(opacity=0.85)
           .transform_calculate(series="'Per-PC variance'")
           .encode(x=x_enc,
                   y=alt.Y("evr:Q",
                           title="Per-PC variance (%)",
                           scale=alt.Scale(domain=[0, 100]),
                           axis=alt.Axis(orient="left")),
                   color=alt.Color("series:N", scale=series_scale,
                                   legend=series_legend),
                   tooltip=[alt.Tooltip("pc:O", title="PC"),
                            alt.Tooltip("evr:Q", title="EVR (%)", format=".2f"),
                            alt.Tooltip("cumr:Q", title="cumulative (%)", format=".2f")]))

    # Right-side y-axis: only the line layer renders the axis; later layers
    # encode y= with axis=None so the cumulative-variance title isn't redrawn
    # on top of itself by every layer.
    cumulative_line = (alt.Chart(df).mark_line(strokeWidth=2)
                       .transform_calculate(series="'Cumulative variance'")
                       .encode(x=x_enc,
                               y=alt.Y("cumr:Q",
                                       title="Cumulative variance (%)",
                                       scale=alt.Scale(domain=[0, 100]),
                                       axis=alt.Axis(orient="right")),
                               color=alt.Color("series:N", scale=series_scale,
                                               legend=series_legend)))
    cumulative_pt   = (alt.Chart(df).mark_point(color=line_color, filled=True, size=40)
                       .encode(x=x_enc,
                               y=alt.Y("cumr:Q", axis=None)))

    thresh_df = pd.DataFrame({"y": [80.0, 95.0], "label": ["80 %", "95 %"]})
    thresh_rule = (alt.Chart(thresh_df)
                   .mark_rule(color="#888", strokeDash=[2, 2], strokeWidth=0.6)
                   .encode(y=alt.Y("y:Q",
                                   scale=alt.Scale(domain=[0, 100]),
                                   axis=None)))
    thresh_text = (alt.Chart(thresh_df)
                   .mark_text(align="right", color="#666", dx=-4, dy=-4, fontSize=9)
                   .encode(y=alt.Y("y:Q",
                                   scale=alt.Scale(domain=[0, 100]),
                                   axis=None),
                           x=alt.value(width_for("full") - 6),
                           text="label:N"))

    # Two top-level layers ⇒ resolve_scale(y="independent") yields exactly two
    # y-scales (left: bar, right: line+points+thresholds), not five.
    right_side = alt.layer(cumulative_line, cumulative_pt, thresh_rule, thresh_text)
    title_text = f"Surface SSM — PCA scree (K={k} components)"
    chart = (alt.layer(bar, right_side)
             .resolve_scale(y="independent")
             .properties(width=width_for("full"),
                         title=make_title(title_text,
                                          subtitle="Per-PC variance (bars) and cumulative (line)")))
    save_chart(chart, out_stem, title=title_text, width_class="full")
    logger.info("Saved scree plot → %s", out_stem)


# ── PC scores pair-plot (faceted SPLOM) ──────────────────────────────────────

def _score_color_enc(color_by: str, mode: str) -> alt.Color | None:
    """Colour encoding for PC-score plots; ``None`` when nothing maps (fixed colour)."""
    if color_by == "sex":
        return alt.Color("sex:N", scale=sex_scale(), legend=alt.Legend(title="Sex"))
    if color_by in ("smoking_status", "smoking"):
        return alt.Color(f"{color_by}:N", scale=smoking_scale(),
                         legend=alt.Legend(title="Smoking"))
    if mode == "continuous":
        return alt.Color(f"{color_by}:Q", scale=continuous_scale(color_by),
                         legend=alt.Legend(title=color_by))
    if mode == "categorical":
        return alt.Color(f"{color_by}:N", legend=alt.Legend(title=color_by))
    return None


def plot_pc_scores_pairs(
    scores_df: pd.DataFrame,
    out_stem: Path,
    *,
    n_pcs: int = 4,
    color_by: str = "sex",
    bins: int = 25,
    min_count: int = 5,
    title: str | None = None,
    arrow: pd.DataFrame | None = None,
) -> None:
    """SPLOM of leading PC scores: bubble counts off-diagonal, KDE on-diagonal.

    Off-diagonal cells aggregate per-patient PC scores to 2D bin counts in
    pandas; bubble size encodes count, colour encodes ``color_by``. Bins
    with ``n < min_count`` are dropped (k-anonymity). The diagonal carries
    each PC's marginal KDE (peak-normalised, per group when categorical).
    For categorical ``color_by`` the legend is bound to a point selection
    so HTML viewers can toggle groups on/off — both bubbles and KDE dim.
    """
    out_stem = _strip_format_suffix(Path(out_stem))
    pc_cols = sorted(
        [c for c in scores_df.columns if c.startswith("PC_")], key=_pc_int,
    )[:n_pcs]
    if len(pc_cols) < 2:
        return

    in_df = color_by in scores_df.columns
    if not in_df:
        mode = "none"
    elif color_by in PALETTE and not isinstance(PALETTE[color_by], dict):
        mode = "continuous"
    else:
        mode = "categorical"

    # Off-diagonal bubble data — LOWER triangle. For each pair (px=smaller PC,
    # py=larger PC) we place px on x and py on y so the lower triangle
    # (row > column) gets the bubbles; upper triangle stays empty.
    pieces: list[pd.DataFrame] = []
    for i, px in enumerate(pc_cols):
        for py in pc_cols[i + 1:]:
            kwargs: dict[str, object] = {"bins": bins, "min_count": min_count}
            if mode == "categorical":
                kwargs["group"] = [color_by]
            elif mode == "continuous":
                kwargs["agg_col"] = color_by
            binned = bin_xy(scores_df, px, py, **kwargs)
            if binned.empty:
                continue
            binned = binned.rename(columns={px: "x", py: "y_value"})
            binned["pc_x"] = px.replace("PC_", "PC")
            binned["pc_y"] = py.replace("PC_", "PC")
            binned["_kind"] = "bubble"
            pieces.append(binned)
    if not pieces:
        return
    bubble_data = pd.concat(pieces, ignore_index=True)

    # Diagonal KDE data (one curve per PC, optionally per group).
    kde_input = scores_df if mode == "categorical" else scores_df.assign(_all="all")
    kde_groupby = [color_by] if mode == "categorical" else ["_all"]
    kde_pieces: list[pd.DataFrame] = []
    for pc_k in pc_cols:
        d = kde_long(kde_input, kde_groupby, pc_k)
        if d.empty:
            continue
        d = d.rename(columns={pc_k: "x", "density": "y_value"})
        if mode != "categorical":
            d = d.drop(columns=["_all"])
        d["pc_x"] = pc_k.replace("PC_", "PC")
        d["pc_y"] = pc_k.replace("PC_", "PC")
        d["_kind"] = "kde"
        kde_pieces.append(d)
    kde_data = (
        pd.concat(kde_pieces, ignore_index=True) if kde_pieces else pd.DataFrame()
    )
    arrow_rows = (
        arrow.assign(_kind="arrow", x=0.0, y_value=0.0)
        if arrow is not None and not arrow.empty else None
    )
    parts = [bubble_data, kde_data]
    if arrow_rows is not None:
        parts.append(arrow_rows)
    data = pd.concat(parts, ignore_index=True)

    color_enc = _score_color_enc(color_by, mode)
    if color_enc is None:
        color_enc = alt.value(color("sex", "Male"))

    # KDE has no continuous mean to colour by; use a neutral fill instead.
    kde_color_enc = color_enc if mode != "continuous" else alt.value("#888888")

    tooltip: list[alt.Tooltip] = []
    if mode == "categorical":
        tooltip.append(alt.Tooltip(f"{color_by}:N", title=color_by))
    elif mode == "continuous":
        tooltip.append(alt.Tooltip(f"{color_by}:Q", title=color_by, format=".2f"))
    tooltip.extend([
        alt.Tooltip("pc_x:N", title="X axis"),
        alt.Tooltip("pc_y:N", title="Y axis"),
        alt.Tooltip("x:Q", title="x", format=".2f"),
        alt.Tooltip("y_value:Q", title="y", format=".2f"),
        alt.Tooltip("n:Q", title="count"),
    ])

    sel = None
    if mode == "categorical":
        sel = alt.selection_point(fields=[color_by], bind="legend")
        bubble_opacity: object = alt.condition(sel, alt.value(0.6), alt.value(0.05))
        kde_opacity: object = alt.condition(sel, alt.value(0.45), alt.value(0.05))
    else:
        bubble_opacity = alt.value(0.6)
        kde_opacity = alt.value(0.45)

    bubble_chart = (
        alt.Chart(data)
        .mark_circle(stroke="rgba(0,0,0,0.25)", strokeWidth=0.3)
        .transform_filter("datum._kind == 'bubble'")
        .encode(
            x=alt.X("x:Q", scale=alt.Scale(zero=False), axis=alt.Axis(title=None)),
            y=alt.Y("y_value:Q", scale=alt.Scale(zero=False),
                    axis=alt.Axis(title=None)),
            size=alt.Size("n:Q", legend=alt.Legend(title="count"),
                          scale=alt.Scale(range=[10, 400], zero=False)),
            color=color_enc,
            opacity=bubble_opacity,
            tooltip=tooltip,
        )
    )

    kde_tooltip = [
        alt.Tooltip("pc_x:N", title="PC"),
        alt.Tooltip("x:Q", title="value", format=".2f"),
        alt.Tooltip("y_value:Q", title="density", format=".2f"),
    ]
    if mode == "categorical":
        kde_tooltip.insert(0, alt.Tooltip(f"{color_by}:N", title=color_by))

    # The kde layer's y-scale must NOT pin to a fixed domain. Vega-Lite resolves
    # the layered y-scale from the bubble layer (zero=False, data-driven) and
    # rejects a sibling scale with an explicit domain; the spec then degrades
    # to a single column of cells in the facet grid instead of the SPLOM.
    kde_chart = (
        alt.Chart(data)
        .mark_area(line={"strokeWidth": 0.6})
        .transform_filter("datum._kind == 'kde'")
        .encode(
            x=alt.X("x:Q", scale=alt.Scale(zero=False), axis=alt.Axis(title=None)),
            y=alt.Y("y_value:Q",
                    axis=alt.Axis(title=None, labels=False, ticks=False)),
            color=kde_color_enc,
            opacity=kde_opacity,
            tooltip=kde_tooltip,
        )
    )

    static_layers = [bubble_chart, kde_chart]
    if arrow_rows is not None:
        arrow_rule = (
            alt.Chart(data).mark_rule(color="#222", strokeWidth=1.3)
            .transform_filter("datum._kind == 'arrow'")
            .encode(x=alt.X("x:Q", scale=alt.Scale(zero=False)), y="y_value:Q",
                    x2="bx:Q", y2="by:Q")
        )
        arrow_head = (
            alt.Chart(data).mark_point(color="#222", size=26, filled=True, shape="triangle-up")
            .transform_filter("datum._kind == 'arrow'")
            .encode(x=alt.X("bx:Q", scale=alt.Scale(zero=False)), y="by:Q")
        )
        static_layers += [arrow_rule, arrow_head]
    layered = alt.layer(*static_layers).properties(width=130, height=130)
    if sel is not None:
        layered = layered.add_params(sel)

    # Map the color_by key to a display label for the title.
    color_label = {"sex": "Sex", "smoking_status": "Smoking",
                   "smoking": "Smoking"}.get(color_by, color_by.title())
    title_text  = title or f"PC score pair-plot — {color_label}"
    subtitle    = f"(PC1..PC{len(pc_cols)})"

    chart = (
        layered.facet(
            row=alt.Row("pc_y:N", title=None,
                        header=alt.Header(labelOrient="left")),
            column=alt.Column("pc_x:N", title=None,
                              header=alt.Header(labelOrient="top")),
        )
        .resolve_scale(x="independent", y="independent")
        # Strip the per-cell view stroke so empty (upper-triangle) cells render
        # as truly empty instead of an outlined empty rectangle.
        .configure_view(stroke=None)
        .properties(
            title=make_title(title_text, subtitle=subtitle)
        )
    )

    save_chart(chart, out_stem, title=title_text, width_class="full")


# ── Adjusted (partial-residual) PC pair-plot ─────────────────────────────────

def _residualize_pcs(
    scores_df: pd.DataFrame, pc_cols: list[str], controls: list[str],
) -> pd.DataFrame:
    """Return a copy with each PC column residualised on ``controls`` (+ intercept)."""
    d = add_design_columns(scores_df)
    ctrl = [c for c in controls if c in d.columns]
    sub = d.dropna(subset=[*pc_cols, *ctrl]).copy()
    if not ctrl or len(sub) < len(ctrl) + 2:
        return sub
    X = np.column_stack([np.ones(len(sub)), *(sub[c].to_numpy(dtype=float) for c in ctrl)])
    for pc in pc_cols:
        y = sub[pc].to_numpy(dtype=float)
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        sub[pc] = y - X @ coef
    return sub


# Design columns dropped from the control set when residualising for a given
# colour variable (so the variable's own effect is not partialled out).
_RESID_DROP: dict[str, list[str]] = {
    "sex":            ["is_female"],
    "smoking_status": ["ever_smoker", "pack_years"],
    "ever_smoker":    ["ever_smoker", "pack_years"],
}


def plot_pc_scores_pairs_adjusted(
    scores_df: pd.DataFrame,
    out_stem: Path,
    *,
    color_by: str = "sex",
    n_pcs: int = N_PCS_DISPLAY,
    bins: int = 25,
    min_count: int = 5,
    arrow: pd.DataFrame | None = None,
) -> None:
    """Partial-residual PC pair-plot: PCs residualised on CORE minus ``color_by``."""
    pc_cols = sorted(
        [c for c in scores_df.columns if c.startswith("PC_")], key=_pc_int,
    )[:n_pcs]
    drop = _RESID_DROP.get(color_by, [color_by])
    controls = [p for p in ADJ_PREDICTORS if p not in drop]
    resid = _residualize_pcs(scores_df, pc_cols, controls)
    label = {"sex": "Sex", "smoking_status": "Smoking"}.get(color_by, color_by.title())
    plot_pc_scores_pairs(
        resid, out_stem, n_pcs=n_pcs, color_by=color_by, bins=bins,
        min_count=min_count, title=f"PC pair-plot, adjusted residuals — {label}",
        arrow=arrow,
    )


# ── DAG (back-door) PC pair-plot ─────────────────────────────────────────────

# Map the pair-plot colour key to the DAG exposure whose back-door set the PCs
# are residualised on.
_DAG_EXPOSURE: dict[str, str] = {
    "sex":            "is_female",
    "smoking_status": "ever_smoker",
}


def plot_pc_scores_pairs_targeted(
    scores_df: pd.DataFrame,
    out_stem: Path,
    *,
    color_by: str = "sex",
    n_pcs: int = N_PCS_DISPLAY,
    bins: int = 25,
    min_count: int = 5,
    arrow: pd.DataFrame | None = None,
) -> None:
    """DAG back-door partial-residual pair-plot: PCs residualised on the exposure's adjustment set."""
    pc_cols = sorted(
        [c for c in scores_df.columns if c.startswith("PC_")], key=_pc_int,
    )[:n_pcs]
    exposure = _DAG_EXPOSURE.get(color_by, color_by)
    controls = DAG_ADJUSTMENT_SETS.get(exposure, [])
    resid = _residualize_pcs(scores_df, pc_cols, controls)
    label = {"sex": "Sex", "smoking_status": "Smoking"}.get(color_by, color_by.title())
    plot_pc_scores_pairs(
        resid, out_stem, n_pcs=n_pcs, color_by=color_by, bins=bins,
        min_count=min_count, title=f"PC pair-plot, DAG-adjusted residuals — {label}",
        arrow=arrow,
    )


# ── β-vector field (direction of each predictor's effect in PC space) ────────

def plot_pc_beta_vectors(
    unadj_df: pd.DataFrame,
    adj_df: pd.DataFrame,
    out_stem: Path,
    *,
    tgt_df: pd.DataFrame | None = None,
    predictors: list[str] | None = None,
    n_pcs: int = N_PCS_DISPLAY,
) -> None:
    """Per-PC-pair arrows of each predictor's standardised β (unadjusted, adjusted, DAG-targeted)."""
    out_stem = _strip_format_suffix(Path(out_stem))
    if (adj_df.empty or "predictor" not in adj_df.columns
            or unadj_df.empty or "predictor" not in unadj_df.columns):
        return

    def _pivot(df: pd.DataFrame) -> pd.DataFrame:
        return df.pivot(index="predictor", columns="pc", values="beta_std")

    piv = {"unadjusted": _pivot(unadj_df), "adjusted": _pivot(adj_df)}
    if tgt_df is not None and not tgt_df.empty and "predictor" in tgt_df.columns:
        piv["targeted"] = _pivot(tgt_df)
    pcs = sorted({c for pv in piv.values() for c in pv.columns}, key=_pc_int)[:n_pcs]
    preds = predictors or list(adj_df["predictor"].drop_duplicates())

    rows: list[dict] = []
    for model, pv in piv.items():
        for i, px in enumerate(pcs):
            for py in pcs[i + 1:]:
                for pred in preds:
                    if pred not in pv.index:
                        continue
                    bx = pv.loc[pred, px] if px in pv.columns else np.nan
                    by = pv.loc[pred, py] if py in pv.columns else np.nan
                    rows.append({
                        "pc_x": px.replace("PC_", "PC"),
                        "pc_y": py.replace("PC_", "PC"),
                        "predictor": pred, "model": model, "zero": 0.0,
                        "bx": 0.0 if pd.isna(bx) else float(bx),
                        "by": 0.0 if pd.isna(by) else float(by),
                    })
    data = pd.DataFrame(rows)
    if data.empty:
        return

    sel_pred = alt.selection_point(fields=["predictor"], bind="legend")
    sel_model = alt.selection_point(fields=["model"], bind="legend")
    color_enc = alt.Color("predictor:N", scale=alt.Scale(scheme="tableau10"),
                          legend=alt.Legend(title="Predictor"))
    dash_enc = alt.StrokeDash(
        "model:N",
        scale=alt.Scale(domain=["unadjusted", "adjusted", "targeted"],
                        range=[[1, 0], [4, 3], [2, 2]]),
        legend=alt.Legend(title="Model"),
    )
    opacity_rule = alt.condition(sel_pred & sel_model, alt.value(0.85), alt.value(0.06))
    opacity_head = alt.condition(sel_pred & sel_model, alt.value(0.9), alt.value(0.06))
    base = alt.Chart(data).properties(width=130, height=130)
    rule = base.mark_rule().encode(
        x=alt.X("zero:Q", axis=alt.Axis(title=None)),
        y=alt.Y("zero:Q", axis=alt.Axis(title=None)),
        x2="bx:Q", y2="by:Q",
        color=color_enc, strokeDash=dash_enc, opacity=opacity_rule,
        tooltip=[alt.Tooltip("predictor:N"), alt.Tooltip("model:N"),
                 alt.Tooltip("bx:Q", title="beta PCx", format=".2f"),
                 alt.Tooltip("by:Q", title="beta PCy", format=".2f")],
    )
    head = (
        base.transform_filter("datum.model == 'targeted'")
        .mark_point(size=14, filled=True)
        .encode(x="bx:Q", y="by:Q", color=color_enc, opacity=opacity_head)
    )

    chart = (
        alt.layer(rule, head)
        .add_params(sel_pred, sel_model)
        .facet(row=alt.Row("pc_y:N", title=None, header=alt.Header(labelOrient="left")),
               column=alt.Column("pc_x:N", title=None, header=alt.Header(labelOrient="top")))
        .resolve_scale(x="independent", y="independent")
        .configure_view(stroke=None)
        .properties(title=make_title(
            "PC β-vector field",
            subtitle="standardised β per predictor",
        ))
    )
    save_chart(chart, out_stem, title="PC beta-vector field", width_class="full")
    logger.info("Saved beta-vector field → %s", out_stem)


# ── Residual distribution (per-rib violin of per-vertex residuals) ───────────

def plot_residual_distribution(
    per_patient: dict,
    rib_ids: list[str],
    out_stem: Path,
    *,
    direction: str = "forward",
    title: str | None = None,
) -> None:
    """Per-level mirrored ridgeline of pooled per-vertex residuals.

    One row per rib level (anatomical 1-12). Within each row, the left rib's
    density curve sits on the right side of x=0 (medical "patient's left =
    viewer's right" convention) and the right rib's density on the left side
    (x negated). Densities are normalised to peak=1 by :func:`kde_long`.
    """
    from ssm.eval_residuals import _display_from_internal

    if not per_patient or not rib_ids:
        return
    out_stem = out_stem.with_suffix("") if out_stem.suffix else out_stem

    rng = np.random.default_rng(42)
    per_rib_samples: dict[str, np.ndarray] = {}
    for rib_id in rib_ids:
        pieces = [pdat[rib_id][direction]
                  for pid, pdat in per_patient.items() if rib_id in pdat]
        if not pieces:
            continue
        arr = np.concatenate(pieces)
        if arr.size > 2_000:
            sel = rng.choice(arr.size, size=2_000, replace=False)
            arr = arr[sel]
        per_rib_samples[rib_id] = arr
    if not per_rib_samples:
        return

    def _parse(rib_id: str) -> tuple[int, str]:
        try:
            head, side = rib_id.rsplit("_", 1)
            return int(head.removeprefix("rib")), side
        except (ValueError, AttributeError):
            return 0, "L"

    rows = []
    for rid in sorted(per_rib_samples.keys()):
        seg, side = _parse(rid)
        lv = seg_to_anatomical(seg)
        for v in per_rib_samples[rid]:
            rows.append({
                "level": lv, "side": side,
                "level_label": _display_from_internal(rid).rsplit(" ", 1)[0],
                "value": float(v),
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return

    dens = kde_long(df, ["level", "side", "level_label"], "value")
    if dens.empty:
        return
    # Mirror the x-axis: L stays positive, R goes negative.
    dens["x"] = dens["value"] * np.where(dens["side"] == "R", -1.0, 1.0)
    dens["color_key"] = dens["level"].astype(str) + "-" + dens["side"]

    level_order = sorted(dens["level"].unique())
    rib_color, color_sel = rib_side_color(level_order)

    title_text = title or "Per-rib registration residuals"
    subtitle = f"{direction} pass · |residual| in mm · densities peak-normalised"

    # Symmetric x-domain so the L and R halves are visually balanced.
    abs_max = float(dens["x"].abs().max()) * 1.05

    curves = (
        alt.Chart(dens)
        .mark_area(interpolate="monotone")
        .encode(
            x=alt.X("x:Q", title="Residual (mm)",
                    scale=alt.Scale(domain=[-abs_max, abs_max]),
                    axis=alt.Axis(labelExpr="abs(datum.value)")),
            y=alt.Y("density:Q", title=None,
                    axis=alt.Axis(labels=False, ticks=False, grid=False),
                    scale=alt.Scale(domain=[0, 1.05])),
            color=rib_color,
            detail=alt.Detail("side:N"),
            opacity=alt.condition(color_sel, alt.value(0.85), alt.value(0.1)),
            tooltip=[alt.Tooltip("level_label:N", title="Rib"),
                     alt.Tooltip("side:N", title="Side"),
                     alt.Tooltip("value:Q", title="Residual (mm)", format=".3f"),
                     alt.Tooltip("density:Q", title="Rel. density", format=".2f")],
        )
        .add_params(color_sel)
        .properties(width=width_for("full"), height=42)
    )
    chart = (curves
             .facet(row=alt.Row("level:O", title=None, sort=level_order,
                                header=alt.Header(labelOrient="left",
                                                  labels=True,
                                                  labelExpr="'Rib ' + datum.value")))
             .resolve_scale(x="shared", y="independent")
             .properties(title=make_title(title_text, subtitle=subtitle)))
    save_chart(chart, out_stem, title=title_text, width_class="full")
