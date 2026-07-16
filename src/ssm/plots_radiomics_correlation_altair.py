"""Altair PC × radiomics heatmaps.

``emit_all`` writes the per-PC and per-feature FDR-masked heatmaps via
:func:`utils.altair_theme.fdr_masked_heatmap`.
"""
from __future__ import annotations

import html
import logging
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd

from settings import FDR_DISPLAY_ALPHA, N_PCS_DISPLAY
from utils.altair_theme import (
    fdr_masked_heatmap,
    make_title,
    pc_axis,
    rib_side_color,
    width_for,
)
from utils.figure_export_altair import save_chart, save_chart_split
from utils.rib_labels import display_from_vert, vert_to_anatomical
from utils.shape_labels import shape_label as _slabel

logger = logging.getLogger(__name__)


# Native unit per feature — colorbar / hover unit on per-feature heatmaps.
# Dimensionless features (ratios, sphericity) are left blank.
_FEATURE_UNIT: dict[str, str] = {
    "original_shape_Elongation":              "",
    "original_shape_Flatness":                "",
    "original_shape_LeastAxisLength":         "mm",
    "original_shape_MajorAxisLength":         "mm",
    "original_shape_Maximum2DDiameterColumn": "mm",
    "original_shape_Maximum2DDiameterRow":    "mm",
    "original_shape_Maximum2DDiameterSlice":  "mm",
    "original_shape_Maximum3DDiameter":       "mm",
    "original_shape_MeshVolume":              "mm³",
    "original_shape_MinorAxisLength":         "mm",
    "original_shape_Sphericity":              "",
    "original_shape_SurfaceArea":             "mm²",
    "original_shape_SurfaceVolumeRatio":      "1/mm",
    "rib_length":                             "mm",
}


def _funit(col: str) -> str:
    return _FEATURE_UNIT.get(col, "")


def _rib_axis(effects: pd.DataFrame) -> tuple[list[tuple[int, str]], list[str]]:
    """Ordered ``(vert_level, side)`` pairs and their display labels.

    All 12 left ribs (T8…T19) first, then all 12 right ribs, so same-side
    ribs group together and left/right asymmetries read by eye.
    """
    pairs = (
        effects[["vert_level", "side"]]
        .drop_duplicates()
        .sort_values(["side", "vert_level"], kind="stable")
    )
    rib_pairs = list(map(tuple, pairs.itertuples(index=False, name=None)))
    labels = [display_from_vert(int(v), str(s)) for v, s in rib_pairs]
    return rib_pairs, labels


def _sorted_pcs(effects: pd.DataFrame) -> list[str]:
    return sorted(
        effects["pc"].unique().tolist(),
        key=lambda c: int(str(c).removeprefix("PC_")),
    )


def _pivot_for_pc(
    effects: pd.DataFrame,
    pc: str,
    feature_cols: list[str],
    rib_pairs: list[tuple[int, str]],
    value_col: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Pivot effects for one PC into (n_ribs × n_features) arrays of value + q."""
    sub = effects[effects["pc"] == pc].copy()
    key = list(zip(sub["vert_level"].astype(int), sub["side"].astype(str)))
    sub = sub.assign(_rib_key=key)
    val_pivot = sub.pivot(index="_rib_key", columns="feature", values=value_col)
    q_pivot   = sub.pivot(index="_rib_key", columns="feature", values="q_value")
    val_pivot = val_pivot.reindex(index=rib_pairs, columns=feature_cols)
    q_pivot   = q_pivot.reindex(  index=rib_pairs, columns=feature_cols)
    return val_pivot.values.astype(float), q_pivot.values.astype(float)


def _pivot_for_feature(
    effects: pd.DataFrame,
    feature: str,
    pc_cols: list[str],
    rib_pairs: list[tuple[int, str]],
    value_col: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Pivot effects for one feature into (n_ribs × n_pcs) arrays of value + q."""
    sub = effects[effects["feature"] == feature].copy()
    key = list(zip(sub["vert_level"].astype(int), sub["side"].astype(str)))
    sub = sub.assign(_rib_key=key)
    val_pivot = sub.pivot(index="_rib_key", columns="pc", values=value_col)
    q_pivot   = sub.pivot(index="_rib_key", columns="pc", values="q_value")
    val_pivot = val_pivot.reindex(index=rib_pairs, columns=pc_cols)
    q_pivot   = q_pivot.reindex(  index=rib_pairs, columns=pc_cols)
    return val_pivot.values.astype(float), q_pivot.values.astype(float)


# ── Per-PC heatmap (rows = ribs, cols = features; β_std) ─────────────────────

def build_per_pc_heatmap(
    effects: pd.DataFrame,
    pc: str,
    feature_cols: list[str],
    *,
    global_vmax: float | None = None,
    mask_nonsig: bool = True,
) -> alt.LayerChart:
    rib_pairs, rib_labels = _rib_axis(effects)
    z, q = _pivot_for_pc(effects, pc, feature_cols, rib_pairs, value_col="beta_std")
    feat_labels = [_slabel(f) for f in feature_cols]

    z_pivot = pd.DataFrame(z, index=rib_labels, columns=feat_labels)
    q_pivot = pd.DataFrame(q, index=rib_labels, columns=feat_labels) if mask_nonsig else None

    title_text = f"PC × radiomics — {pc}"
    sig_clause = (f" · FDR-masked at q ≥ {FDR_DISPLAY_ALPHA} (empty cells)"
                  if mask_nonsig else "")
    subtitle = (f"Standardised β across {len(rib_labels)} ribs × "
                f"{len(feature_cols)} features{sig_clause}")
    return fdr_masked_heatmap(
        z_pivot, q_pivot,
        value_label="Std. β",
        annotate=False,
        show_stars=False,
        fdr_threshold=FDR_DISPLAY_ALPHA,
        vmax=global_vmax,
        width_class="full",
        title=make_title(title_text, subtitle=subtitle),
        x_axis_title="Radiomics feature",
        y_axis_title="Rib",
        tickangle_x=-25,
        row_order=rib_labels,
        col_order=feat_labels,
        row_title="Rib",
        col_title="Feature",
    )


# ── Per-feature heatmap (rows = ribs, cols = PCs; β_native) ──────────────────

def build_per_feature_heatmap(
    effects: pd.DataFrame,
    feature: str,
    pc_cols: list[str],
    *,
    mask_nonsig: bool = True,
) -> tuple[alt.LayerChart, alt.LayerChart]:
    """Build (full-113-PC HTML chart, top-N_PCS_DISPLAY static chart) pair."""
    rib_pairs, rib_labels = _rib_axis(effects)
    z, q = _pivot_for_feature(effects, feature, pc_cols, rib_pairs,
                              value_col="beta_native")
    pc_axis_labels = [f"PC{c.removeprefix('PC_')}" for c in pc_cols]

    z_pivot = pd.DataFrame(z, index=rib_labels, columns=pc_axis_labels)
    q_pivot = pd.DataFrame(q, index=rib_labels, columns=pc_axis_labels) if mask_nonsig else None

    unit = _funit(feature)
    bar_title = f"β ({unit} per 1 SD PC)" if unit else "β (per 1 SD PC)"
    sig_clause = (f" · FDR-masked at q ≥ {FDR_DISPLAY_ALPHA} (empty cells)"
                  if mask_nonsig else "")
    title_text = f"PC × radiomics — {_slabel(feature)}"
    subtitle = f"β in {unit or 'native units'} per 1 SD PC{sig_clause}"

    common = dict(
        value_label=bar_title,
        text_format=".4g",
        annotate=False,
        show_stars=False,
        fdr_threshold=FDR_DISPLAY_ALPHA,
        width_class="full",
        title=make_title(title_text, subtitle=subtitle),
        y_axis_title="Rib",
        row_order=rib_labels,
        row_title="Rib",
        col_title="PC",
    )
    html_chart = fdr_masked_heatmap(
        z_pivot, q_pivot,
        col_order=pc_axis_labels,
        x_axis=pc_axis(pc_axis_labels, every=5, title="PC"),
        **common,
    )
    top_pcs = pc_axis_labels[:N_PCS_DISPLAY]
    static_q = q_pivot[top_pcs] if q_pivot is not None else None
    static_chart = fdr_masked_heatmap(
        z_pivot[top_pcs], static_q,
        col_order=top_pcs,
        x_axis_title="PC",
        tickangle_x=0,
        **common,
    )
    return html_chart, static_chart


# ── Mirrored bar alternates (population-pyramid layout) ──────────────────────

def _mirrored_bar_long(
    effects: pd.DataFrame,
    *,
    pc: str | None = None,
    feature: str | None = None,
    value_col: str = "beta_std",
) -> pd.DataFrame:
    """Long-form (rib, side, item, value, signed_value) for the histo charts.

    Filters to one PC or one feature (exactly one of ``pc`` / ``feature``).
    ``signed_value`` mirrors L bars to positive x and R bars to negative x;
    Altair labels the x-axis with the absolute value via labelExpr so readers
    still see the magnitude on both sides.
    """
    if (pc is None) == (feature is None):
        raise ValueError("Pass exactly one of pc= or feature=.")
    sub = effects[effects["pc"] == pc] if pc else effects[effects["feature"] == feature]
    if sub.empty:
        return pd.DataFrame()
    sub = sub.copy()
    sub["rib"]  = sub["vert_level"].astype(int).apply(vert_to_anatomical)
    sub["side"] = sub["side"].astype(str).str[0].str.upper()
    sub = sub[sub["side"].isin(["L", "R"])]
    sub["signed_value"] = np.where(sub["side"] == "R", -sub[value_col], sub[value_col])
    if pc is None:
        sub["item_raw"]   = sub["pc"]
        sub["item_label"] = sub["pc"].astype(str).str.replace("PC_", "PC", regex=False)
    else:
        sub["item_raw"]   = sub["feature"]
        sub["item_label"] = sub["feature"].astype(str).map(_slabel)
    return sub


def build_per_pc_histo(
    effects: pd.DataFrame,
    pc: str,
    feature_cols: list[str],
    *,
    value_col: str = "beta_std",
    title: str | None = None,
) -> alt.Chart | None:
    """Mirrored bar version of ``build_per_pc_heatmap``.

    Faceted by feature in a 2×7 grid (one panel per feature). Per panel:
    y = anatomical rib (1-12), x = signed β (L grows right of x=0, R left).
    Per-(level, side) colour via the rib_left / rib_right palettes; the two-
    swatch side legend is hconcat'd beside the faceted grid.
    """
    long = _mirrored_bar_long(effects, pc=pc, value_col=value_col)
    if long.empty:
        return None
    long = long[long["feature"].isin(feature_cols)].copy()
    if long.empty:
        return None
    feat_order = [_slabel(f) for f in feature_cols if f in long["feature"].unique()]
    long["color_key"] = long["rib"].astype(str) + "-" + long["side"]
    rib_levels = sorted(long["rib"].unique())
    rib_color, color_sel = rib_side_color(rib_levels)

    title_text = title or f"PC × radiomics — {pc} · mirrored bars"
    subtitle = f"y = anatomical rib · x = signed β · {len(feat_order)} features"
    abs_max = float(long["signed_value"].abs().max() or 0.05) * 1.05

    bars = (
        alt.Chart(long)
        .mark_bar()
        .encode(
            x=alt.X("signed_value:Q", title="β (Std.)",
                    scale=alt.Scale(domain=[-abs_max, abs_max]),
                    stack=None,
                    axis=alt.Axis(labelExpr="abs(datum.value)")),
            x2=alt.X2(datum=0),
            y=alt.Y("rib:O", title="Rib (anatomical, 1 = T8)",
                    sort=rib_levels),
            color=rib_color,
            opacity=alt.condition(color_sel, alt.value(0.85), alt.value(0.1)),
            tooltip=[alt.Tooltip("rib:O", title="Rib"),
                     alt.Tooltip("side:N", title="Side"),
                     alt.Tooltip("item_label:N", title="Feature"),
                     alt.Tooltip(f"{value_col}:Q", title="β (signed)", format=".3f")],
        )
        .add_params(color_sel)
        .properties(width=width_for("half"), height=180)
    )
    chart = (bars
             .facet(facet=alt.Facet("item_label:N", title=None,
                                     sort=feat_order),
                    columns=2)
             .resolve_scale(x="shared", y="shared")
             .properties(title=make_title(title_text, subtitle=subtitle)))
    return chart


def build_per_feature_histo(
    effects: pd.DataFrame,
    feature: str,
    pc_cols: list[str],
    *,
    value_col: str = "beta_native",
    title: str | None = None,
    top_n: int = N_PCS_DISPLAY,
) -> alt.Chart | None:
    """Mirrored bar version of ``build_per_feature_heatmap``.

    Single-column grid (one row per PC) over the top ``top_n`` PCs by ``|β|``.
    Per panel: y = anatomical rib, x = signed β (L right, R left), per-
    (level, side) colour via the rib palette; two-swatch side legend on
    the right.
    """
    long = _mirrored_bar_long(effects, feature=feature, value_col=value_col)
    if long.empty:
        return None
    long = long[long["pc"].isin(pc_cols)]
    # Rank PCs by mean |β| across (rib, side) — keeps the top_n.
    rank = (long.assign(_abs=long["signed_value"].abs())
                 .groupby("pc")["_abs"].mean()
                 .sort_values(ascending=False))
    keep = rank.head(top_n).index.tolist()
    # Sort kept PCs by PC number for the panel order.
    keep_sorted = sorted(keep, key=lambda c: int(str(c).removeprefix("PC_")))
    long = long[long["pc"].isin(keep_sorted)].copy()
    if long.empty:
        return None
    pc_labels = [c.replace("PC_", "PC") for c in keep_sorted]
    long["color_key"] = long["rib"].astype(str) + "-" + long["side"]
    rib_levels = sorted(long["rib"].unique())
    rib_color, color_sel = rib_side_color(rib_levels)

    title_text = title or f"PC × radiomics — {_slabel(feature)} · mirrored bars"
    unit = _funit(feature)
    subtitle = (f"y = anatomical rib · x = signed β"
                f"{f' ({unit} per 1 SD PC)' if unit else ''}"
                f" · top {len(pc_labels)} PCs by mean |β|")
    abs_max = float(long["signed_value"].abs().max() or 0.05) * 1.05

    bars = (
        alt.Chart(long)
        .mark_bar()
        .encode(
            x=alt.X("signed_value:Q",
                    title=f"β ({unit} per 1 SD PC)" if unit else "β (per 1 SD PC)",
                    scale=alt.Scale(domain=[-abs_max, abs_max]),
                    stack=None,
                    axis=alt.Axis(labelExpr="abs(datum.value)")),
            x2=alt.X2(datum=0),
            y=alt.Y("rib:O", title="Rib (anatomical, 1 = T8)",
                    sort=rib_levels),
            color=rib_color,
            opacity=alt.condition(color_sel, alt.value(0.85), alt.value(0.1)),
            tooltip=[alt.Tooltip("rib:O", title="Rib"),
                     alt.Tooltip("side:N", title="Side"),
                     alt.Tooltip("item_label:N", title="PC"),
                     alt.Tooltip(f"{value_col}:Q", title="β (signed)", format=".4g")],
        )
        .add_params(color_sel)
        .properties(width=width_for("half"), height=180)
    )
    chart = (bars
             .facet(facet=alt.Facet("item_label:N", title=None,
                                     sort=pc_labels),
                    columns=1)
             .resolve_scale(x="shared", y="shared")
             .properties(title=make_title(title_text, subtitle=subtitle)))
    return chart


# ── Orchestrator: write every per-PC + per-feature figure ────────────────────

def emit_all(
    effects: pd.DataFrame,
    out_dir: Path,
    feature_cols: list[str],
    *,
    mask_nonsig: bool = True,
    formats: tuple[str, ...] = ("html", "svg", "png"),
) -> None:
    """Write per-PC and per-feature heatmap + histo families under ``out_dir``."""
    pc_cols = _sorted_pcs(effects)
    per_pc_dir      = out_dir / "per_pc"
    per_feat_dir    = out_dir / "per_feature"
    per_pc_histo    = out_dir / "per_pc_histo"
    per_feat_histo  = out_dir / "per_feature_histo"
    for d in (per_pc_dir, per_feat_dir, per_pc_histo, per_feat_histo):
        d.mkdir(parents=True, exist_ok=True)

    if mask_nonsig:
        sig = effects.loc[effects["q_value"] < FDR_DISPLAY_ALPHA, "beta_std"]
    else:
        sig = effects["beta_std"]
    sig_finite = sig[np.isfinite(sig)]
    global_vmax = float(sig_finite.abs().max()) if not sig_finite.empty else None

    n_digits = max(2, len(str(len(pc_cols))))
    static_formats = tuple(f for f in formats if f != "html")
    html_formats   = tuple(f for f in formats if f == "html")

    for pc in pc_cols:
        idx = int(pc.removeprefix("PC_"))
        chart = build_per_pc_heatmap(
            effects, pc, feature_cols,
            global_vmax=global_vmax, mask_nonsig=mask_nonsig,
        )
        save_chart(
            chart, per_pc_dir / f"pc{idx:0{n_digits}d}_std",
            formats=formats,
            title=f"PC{idx} — standardised β heatmap (rib × feature)",
            width_class="full",
        )
        # Mirrored-bar companion next to the heatmap.
        histo = build_per_pc_histo(effects, pc, feature_cols)
        if histo is not None:
            save_chart(
                histo, per_pc_histo / f"pc{idx:0{n_digits}d}_std",
                formats=formats,
                title=f"PC{idx} — mirrored-bar β (rib × feature)",
                width_class="full",
            )

    for feat in feature_cols:
        html_chart, static_chart = build_per_feature_heatmap(
            effects, feat, pc_cols, mask_nonsig=mask_nonsig,
        )
        out_stem = per_feat_dir / f"{_slabel(feat)}_native"
        title = f"{_slabel(feat)} — native-unit β heatmap (rib × PC)"
        if html_formats and static_formats:
            save_chart_split(
                html_chart, static_chart, out_stem,
                html_formats=html_formats, static_formats=static_formats,
                title=title, width_class="full",
            )
        elif html_formats:
            save_chart(html_chart, out_stem, formats=html_formats,
                       title=title, width_class="full")
        else:
            save_chart(static_chart, out_stem, formats=static_formats,
                       title=title, width_class="full")
        # Mirrored-bar companion (top-N_PCS_DISPLAY PCs).
        histo = build_per_feature_histo(effects, feat, pc_cols)
        if histo is not None:
            save_chart(
                histo, per_feat_histo / f"{_slabel(feat)}_native",
                formats=formats,
                title=f"{_slabel(feat)} — mirrored-bar β (rib × top PCs)",
                width_class="full",
            )

    if "html" in formats:
        per_pc_html = [(f"PC {int(pc.removeprefix('PC_'))}",
                       f"pc{int(pc.removeprefix('PC_')):0{n_digits}d}_std.html")
                       for pc in pc_cols]
        _write_index_html(
            per_pc_dir / "index.html",
            title="PC × radiomics — per-PC browser",
            subtitle="Pick a principal component to view its rib × feature heatmap.",
            options=per_pc_html,
            select_label="PC:",
        )
        _write_index_html(
            per_pc_histo / "index.html",
            title="PC × radiomics — per-PC mirrored bars",
            subtitle="Pick a PC to view its rib × feature mirrored-bar chart.",
            options=per_pc_html,
            select_label="PC:",
        )
        per_feat_html = [(_slabel(f), f"{_slabel(f)}_native.html") for f in feature_cols]
        _write_index_html(
            per_feat_dir / "index.html",
            title="PC × radiomics — per-feature browser",
            subtitle="Pick a radiomics feature to view its rib × PC heatmap.",
            options=per_feat_html,
            select_label="Feature:",
        )
        _write_index_html(
            per_feat_histo / "index.html",
            title="PC × radiomics — per-feature mirrored bars",
            subtitle="Pick a feature to view its rib × top-PC mirrored-bar chart.",
            options=per_feat_html,
            select_label="Feature:",
        )

    logger.info(
        f"Wrote {len(pc_cols)} per-PC + {len(feature_cols)} per-feature "
        f"heatmaps (and histo companions) under {out_dir}"
    )


# ── Static iframe-switcher HTML index ────────────────────────────────────────

_INDEX_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <style>
    body {{ font-family: Inter, Helvetica, Arial, sans-serif; margin: 16px; }}
    h1   {{ font-size: 16px; margin: 0 0 4px 0; }}
    p    {{ font-size: 12px; color: #555; margin: 0 0 12px 0; }}
    label {{ font-size: 13px; margin-right: 8px; }}
    select {{ font-size: 13px; padding: 2px 6px; }}
    iframe {{ display: block; width: 100%; height: 86vh; border: 1px solid #ddd;
              margin-top: 10px; }}
  </style>
</head>
<body>
  <h1>{title}</h1>
  <p>{subtitle}</p>
  <label for="picker">{select_label}</label>
  <select id="picker" onchange="document.getElementById('frame').src = this.value">
{options_html}
  </select>
  <iframe id="frame" src="{first_src}" loading="lazy"></iframe>
</body>
</html>
"""


def _write_index_html(
    path: Path,
    *,
    title: str,
    subtitle: str,
    options: list[tuple[str, str]],
    select_label: str,
) -> None:
    """Write a static HTML page with a <select> that swaps an <iframe> src."""
    if not options:
        return
    opt_lines = "\n".join(
        f'    <option value="{html.escape(src)}">{html.escape(label)}</option>'
        for label, src in options
    )
    path.write_text(_INDEX_TEMPLATE.format(
        title=html.escape(title),
        subtitle=html.escape(subtitle),
        select_label=html.escape(select_label),
        options_html=opt_lines,
        first_src=html.escape(options[0][1]),
    ), encoding="utf-8")
    logger.info("Wrote PC-browser index → %s", path)
