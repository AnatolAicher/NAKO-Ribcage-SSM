"""Altair figures for the supplemental cross-stage QC visualisations."""
from __future__ import annotations

import logging
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd

from data_ingestion.loaders import ANALYSIS_SHAPE_COLS
from utils.altair_theme import (
    kde_long,
    make_title,
    rib_side_color,
    sex_scale,
    smoking_scale,
    width_for,
)
from utils.colors import SMOKE_ORDER, cmap, rib_colors
from utils.figure_export_altair import save_chart
from utils.rib_labels import vert_to_anatomical
from utils.shape_labels import shape_label as _slabel

logger = logging.getLogger(__name__)


# ── Patient-level helpers ────────────────────────────────────────────────────

_META_COLS_DEFAULT = ["age", "pack_years", "height_cm", "weight_kg", "bmi", "body_fat_pct"]


def _patients(df: pd.DataFrame) -> pd.DataFrame:
    """One row per ``patient_id`` (drops rib-level duplicates)."""
    return df.drop_duplicates("patient_id")


# ── metadata_by_sex (overlay density histograms per metadata col) ────────────

def plot_metadata_by_sex(df: pd.DataFrame, out_dir: Path) -> None:
    """6-panel histograms of metadata cols, colour-split by sex.

    Uses discrete bin counts (np.histogram) rather than KDE so the displayed
    range stays bounded by the data. pack_years drops the zero bin
    (never-smokers) so the long tail is visible.
    """
    meta_cols = [c for c in _META_COLS_DEFAULT if c in df.columns]
    if not meta_cols or "sex" not in df.columns:
        return
    out_dir = Path(out_dir)
    pat = _patients(df)
    n_pat = len(pat)

    bin_rows: list[dict] = []
    for col in meta_cols:
        sub = pat[[col, "sex"]].dropna()
        if col == "pack_years":
            sub = sub[sub[col] > 0]
        if sub.empty:
            continue
        # Shared bin edges per variable so the per-sex bars align horizontally.
        edges = np.histogram_bin_edges(sub[col].to_numpy(dtype=float), bins=30)
        bin_width = float(np.mean(np.diff(edges)))
        for sex_val, grp in sub.groupby("sex", observed=True):
            n_sex = int(len(grp))
            if n_sex == 0:
                continue
            counts, _ = np.histogram(grp[col].to_numpy(dtype=float), bins=edges)
            # Convert to probability density: density = count / (n × bin_width)
            # so the histogram integrates to 1 per (variable, sex) group and
            # M and F are directly comparable despite different sample sizes.
            for c, lo, hi in zip(counts, edges[:-1], edges[1:]):
                bin_rows.append({
                    "variable": col, "sex": sex_val,
                    "bin_lo": float(lo), "bin_hi": float(hi),
                    "count": int(c),
                    "density": float(c) / (n_sex * bin_width) if bin_width > 0 else 0.0,
                })
    if not bin_rows:
        return
    hist_df = pd.DataFrame(bin_rows)

    sex_sel = alt.selection_point(fields=["sex"], bind="legend")
    title_text = "Metadata by sex"
    subtitle = (f"Patient-level density histograms · n={n_pat:,} · "
                "pack_years zero bin (never-smokers) dropped")

    panel = (
        alt.Chart(hist_df)
        .mark_bar(opacity=0.55)
        .encode(
            x=alt.X("bin_lo:Q", title=None, scale=alt.Scale(zero=False)),
            x2=alt.X2("bin_hi:Q"),
            y=alt.Y("density:Q", title="density", stack=None),
            y2=alt.Y2(datum=0),
            color=alt.Color("sex:N", scale=sex_scale(),
                            legend=alt.Legend(title="Sex")),
            opacity=alt.condition(sex_sel, alt.value(0.55), alt.value(0.10)),
            tooltip=[alt.Tooltip("variable:N"),
                     alt.Tooltip("sex:N"),
                     alt.Tooltip("bin_lo:Q", title="from", format=".2f"),
                     alt.Tooltip("bin_hi:Q", title="to",   format=".2f"),
                     alt.Tooltip("count:Q"),
                     alt.Tooltip("density:Q", format=".3g")],
        )
        .add_params(sex_sel)
        .properties(width=width_for("third"), height=90)
    )
    chart = (panel.facet(facet=alt.Facet("variable:N", title=None,
                                          sort=meta_cols), columns=3)
                  .resolve_scale(x="independent", y="independent")
                  .properties(title=make_title(title_text, subtitle=subtitle)))
    save_chart(chart, out_dir / "metadata_by_sex",
               title=title_text, width_class="full")


# ── smoking_by_sex (count bars, faceted by sex) ──────────────────────────────

def plot_smoking_by_sex(df: pd.DataFrame, out_dir: Path) -> None:
    """Bar chart of smoking status counts, faceted by sex."""
    if "smoking_status" not in df.columns or "sex" not in df.columns:
        return
    out_dir = Path(out_dir)
    pat = _patients(df)
    counts = (pat.groupby(["sex", "smoking_status"], observed=True)
                 .size().rename("n").reset_index())
    if counts.empty:
        return

    title_text = "Smoking status by sex"
    subtitle = f"Included cohort (n = {len(pat):,})"
    chart = (
        alt.Chart(counts)
        .mark_bar()
        .encode(
            x=alt.X("smoking_status:N", title=None, sort=list(SMOKE_ORDER)),
            y=alt.Y("n:Q", title="N patients"),
            color=alt.Color("smoking_status:N", scale=smoking_scale(),
                            sort=list(SMOKE_ORDER),
                            legend=alt.Legend(title="Smoking")),
            tooltip=[alt.Tooltip("sex:N"),
                     alt.Tooltip("smoking_status:N", title="Smoking"),
                     alt.Tooltip("n:Q", title="N", format=",")],
        )
        .properties(width=width_for("third"), height=140)
        .facet(column=alt.Column("sex:N", title=None))
        .properties(title=make_title(title_text, subtitle=subtitle))
    )
    save_chart(chart, out_dir / "smoking_by_sex",
               title=title_text, width_class="full")


# ── rib_length_by_level (population-pyramid layout) ──────────────────────────

def plot_rib_length_by_level(df: pd.DataFrame, out_dir: Path) -> None:
    """Mean rib length per anatomical rib level, mirrored by side.

    Vertical y-axis = rib level (anatomical 1-12, 1 at top). x-axis = mean
    length (mm); left ribs grow rightward (positive x), right ribs grow
    leftward (negative x), matching the population-pyramid pattern.
    """
    if "rib_length" not in df.columns:
        return
    out_dir = Path(out_dir)
    work = df.dropna(subset=["rib_length", "vert_level", "side"]).copy()
    if work.empty:
        return
    work["rib"] = work["vert_level"].apply(vert_to_anatomical)
    work["side_norm"] = work["side"].astype(str).str[0].str.upper().map(
        {"L": "Left", "R": "Right"}
    )
    work = work.dropna(subset=["side_norm"])
    means = (work.groupby(["rib", "side_norm"], observed=True)
                 ["rib_length"].mean().reset_index())
    if means.empty:
        return
    means["signed_length"] = np.where(means["side_norm"] == "Right",
                                       -means["rib_length"], means["rib_length"])

    rib_levels = sorted(means["rib"].unique())
    means["side"] = means["side_norm"].str[0].str.upper()
    means["color_key"] = (means["rib"].astype(str) + "-" + means["side"])
    rib_color, color_sel = rib_side_color(rib_levels)

    abs_max = float(means["rib_length"].abs().max()) * 1.05

    title_text = "Mean rib length by level and side"
    subtitle = "Patient-level mean (mm)"

    chart = (
        alt.Chart(means)
        .mark_bar()
        .encode(
            x=alt.X("signed_length:Q", title="Mean rib length (mm)",
                    scale=alt.Scale(domain=[-abs_max, abs_max]),
                    axis=alt.Axis(labelExpr="abs(datum.value)")),
            y=alt.Y("rib:O", title="Rib (anatomical, 1 = T8)",
                    sort=rib_levels),
            color=rib_color,
            opacity=alt.condition(color_sel, alt.value(1.0), alt.value(0.1)),
            tooltip=[alt.Tooltip("rib:O", title="Rib"),
                     alt.Tooltip("side_norm:N", title="Side"),
                     alt.Tooltip("rib_length:Q", title="Mean length (mm)",
                                 format=".2f")],
        )
        .add_params(color_sel)
        .properties(width=width_for("half"), height=260,
                    title=make_title(title_text, subtitle=subtitle))
    )
    save_chart(chart, out_dir / "rib_length_by_level",
               title=title_text, width_class="half")


# ── shape_by_rib (per-feature mirrored ridgeline) ────────────────────────────

def plot_shape_by_rib(df: pd.DataFrame, out_dir: Path) -> None:
    """One ridgeline per shape feature: rows = rib level, mirrored by side.

    Static (PNG / SVG): one figure per feature, written as
    ``shape_by_rib_<feature>.{svg,png}``. Interactive (HTML): a single combined
    file ``interactive_shape_by_rib.html`` carries a ``<select>`` so all
    features can be browsed without inlining 14 specs.
    """
    cols = [c for c in ANALYSIS_SHAPE_COLS if c in df.columns]
    if not cols or "vert_level" not in df.columns or "side" not in df.columns:
        return
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    work = df[["vert_level", "side"] + cols].dropna(subset=["vert_level", "side"])
    if work.empty:
        return
    work = work.copy()
    work["rib"] = work["vert_level"].apply(vert_to_anatomical)
    work["side_norm"] = work["side"].astype(str).str[0].str.upper()
    work = work[work["side_norm"].isin(["L", "R"])]
    if work.empty:
        return
    rib_levels = sorted(work["rib"].unique())

    rib_color, color_sel = rib_side_color(rib_levels)

    static_paths: list[tuple[str, str]] = []
    for col in cols:
        label = _slabel(col)
        sub = work[["rib", "side_norm", col]].rename(columns={col: "value"})
        sub = sub.dropna(subset=["value"])
        if sub.empty:
            continue
        dens = kde_long(sub, ["rib", "side_norm"], "value")
        if dens.empty:
            continue
        dens["x"] = dens["value"] * np.where(dens["side_norm"] == "R", -1.0, 1.0)
        dens["side"] = dens["side_norm"]
        dens["color_key"] = (dens["rib"].astype(str) + "-" + dens["side"])

        title_text = f"{label} by rib"
        subtitle = "Density peak-normalised"

        curves = (
            alt.Chart(dens)
            .mark_area(interpolate="monotone")
            .encode(
                x=alt.X("x:Q", title=label),
                y=alt.Y("density:Q", title=None,
                        axis=alt.Axis(labels=False, ticks=False, grid=False),
                        scale=alt.Scale(domain=[0, 1.05])),
                color=rib_color,
                detail=alt.Detail("side:N"),
                opacity=alt.condition(color_sel, alt.value(0.85), alt.value(0.1)),
                tooltip=[alt.Tooltip("rib:O", title="Rib"),
                         alt.Tooltip("side:N", title="Side"),
                         alt.Tooltip("value:Q", title=label, format=".3f"),
                         alt.Tooltip("density:Q", title="Rel. density",
                                     format=".2f")],
            )
            .add_params(color_sel)
            .properties(width=width_for("full"), height=42)
        )
        chart = (curves
                 .facet(row=alt.Row("rib:O", title=None, sort=rib_levels,
                                    header=alt.Header(labelOrient="left",
                                                      labelExpr="'Rib ' + datum.value")))
                 .resolve_scale(x="shared", y="independent")
                 .properties(title=make_title(title_text, subtitle=subtitle)))
        stem = out_dir / f"shape_by_rib_{label}"
        save_chart(chart, stem, formats=("svg", "png"),
                   title=title_text, width_class="full")
        static_paths.append((label, f"shape_by_rib_{label}.html"))
        # Also write per-feature HTML so the picker iframe can switch.
        save_chart(chart, stem, formats=("html",),
                   title=title_text, width_class="full",
                   record_manifest=False)

    if static_paths:
        _write_shape_browser(out_dir / "interactive_shape_by_rib.html",
                             options=static_paths)
        logger.info("Wrote shape-by-rib browser → %s",
                    out_dir / "interactive_shape_by_rib.html")


_BROWSER_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Shape by rib — browser</title>
  <style>
    body {{ font-family: Inter, Helvetica, Arial, sans-serif; margin: 16px; }}
    h1 {{ font-size: 16px; margin: 0 0 4px 0; }}
    p {{ font-size: 12px; color: #555; margin: 0 0 12px 0; }}
    select {{ font-size: 13px; padding: 2px 6px; margin-bottom: 10px; }}
    iframe {{ display: block; width: 100%; height: 84vh; border: 1px solid #ddd; }}
  </style>
</head>
<body>
  <h1>Shape by rib</h1>
  <p>Pick a shape feature to view its mirrored ridgeline (L right, R left of x=0).</p>
  <label for="picker">Feature: </label>
  <select id="picker" onchange="document.getElementById('frame').src = this.value">
{options_html}
  </select>
  <iframe id="frame" src="{first_src}" loading="lazy"></iframe>
</body>
</html>
"""


def _write_shape_browser(path: Path, *, options: list[tuple[str, str]]) -> None:
    import html as _html
    opt_lines = "\n".join(
        f'    <option value="{_html.escape(src)}">{_html.escape(label)}</option>'
        for label, src in options
    )
    path.write_text(_BROWSER_TEMPLATE.format(
        options_html=opt_lines,
        first_src=_html.escape(options[0][1]),
    ), encoding="utf-8")
