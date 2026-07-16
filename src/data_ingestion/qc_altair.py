"""Altair QC figures for the audits and tables computed in :mod:`data_ingestion.qc`."""
from __future__ import annotations

import logging
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd

from utils.altair_theme import correlation_scale, make_title, width_for
from utils.colors import _rgba_to_hex, cmap, color
from utils.figure_export_altair import save_chart
from utils.rib_labels import vert_to_anatomical
from utils.shape_labels import shape_label as _label

logger = logging.getLogger(__name__)


# ── Correlation matrix ───────────────────────────────────────────────────────

def plot_correlation_matrix(
    df: pd.DataFrame,
    cols: list[str],
    out_stem: Path,
    *,
    title: str = "Correlation matrix",
    subtitle: str | None = None,
    method: str = "pearson",
    flag_threshold: float = 0.7,
) -> pd.DataFrame:
    """Half-masked correlation matrix (lower triangle, diagonal hidden)."""
    cols = [c for c in cols if c in df.columns]
    out_stem = Path(out_stem)
    if out_stem.suffix in {".png", ".svg", ".pdf", ".html"}:
        out_stem = out_stem.with_suffix("")

    display_cols = [_label(c) for c in cols]
    corr = (df[cols].rename(columns=dict(zip(cols, display_cols)))
                    .corr(method=method))
    n = len(cols)
    z = corr.values.astype(float).copy()
    mask_upper = np.triu(np.ones_like(z, dtype=bool))
    z[mask_upper] = np.nan

    long = (pd.DataFrame(z, index=display_cols, columns=display_cols)
            .rename_axis(index="row", columns="col")
            .stack()
            .rename("r")
            .reset_index())
    long = long.dropna(subset=["r"]).reset_index(drop=True)

    annotate = n <= 20
    # Spearman is rank-based → ρ (rho); Pearson is r. The legend carries only
    # the symbol, the method (used in subtitle) avoids the duplicate "spearman"
    # appearing both in title and legend.
    method_symbol = "ρ" if method == "spearman" else "r"
    legend = alt.Legend(title=method_symbol)

    base = alt.Chart(long).encode(
        x=alt.X("col:N", sort=display_cols,
                axis=alt.Axis(labelAngle=-30, title=None)),
        y=alt.Y("row:N", sort=display_cols, axis=alt.Axis(title=None)),
    )
    heat = base.mark_rect(stroke="white", strokeWidth=1).encode(
        color=alt.Color("r:Q", scale=correlation_scale(vmax=1.0),
                        legend=legend),
        tooltip=[alt.Tooltip("row:N", title="Variable"),
                 alt.Tooltip("col:N", title="Variable"),
                 alt.Tooltip("r:Q", title=method_symbol, format=".3f")],
    )
    layers = [heat]
    if annotate:
        # Use the same |value| > 0.55 contrast test as fdr_masked_heatmap.
        dark_expr = "abs(datum.r) > 0.55"
        layers.append(base.mark_text(baseline="middle").encode(
            text=alt.Text("r:Q", format=".2f"),
            color=alt.condition(dark_expr, alt.value("white"), alt.value("#222")),
        ))

    title_text = title
    if subtitle is None:
        subtitle = f"{method} ({method_symbol}) · lower triangle"
    chart = (alt.layer(*layers)
             .properties(width=width_for("full"),
                         title=make_title(title_text, subtitle=subtitle)))

    save_chart(chart, out_stem, title=title_text, width_class="full")

    upper_only = corr.where(~mask_upper)
    high = upper_only.stack()
    high = high[high.abs() >= flag_threshold].sort_values(ascending=False)
    if not high.empty:
        logger.warning(
            f"High collinearity pairs (|{method} r| ≥ {flag_threshold}):\n"
            + "\n".join(f"  {a} × {b}: {v:.3f}" for (a, b), v in high.items())
        )
    else:
        logger.info(f"No collinear pairs above |r| = {flag_threshold}")
    return corr


# ── Missingness h-bar ────────────────────────────────────────────────────────

def plot_missingness(
    miss_report: pd.DataFrame,
    out_stem: Path,
    *,
    title: str = "Missingness per variable",
) -> None:
    """Horizontal bar of percent-missing per variable."""
    if miss_report is None or miss_report.empty:
        logger.info("Missingness report empty — no figure produced.")
        return
    rep = miss_report.sort_values("pct_missing", ascending=False).reset_index()
    rep = rep.rename(columns={rep.columns[0]: "variable"})

    title_text = title or "Missingness per variable"
    subtitle = "% of patients with missing value"
    # Extreme-negative end of the (RdYlGn) diverging "correlation" ramp.
    bar_hex = _rgba_to_hex(cmap("correlation")(0.0))
    chart = (
        alt.Chart(rep)
        .mark_bar(color=bar_hex)
        .encode(
            x=alt.X("pct_missing:Q", title="% missing"),
            y=alt.Y("variable:N", sort="-x", title=None),
            tooltip=[alt.Tooltip("variable:N"),
                     alt.Tooltip("n_missing:Q", title="Missing", format=","),
                     alt.Tooltip("n_present:Q", title="Present", format=","),
                     alt.Tooltip("pct_missing:Q", title="% missing", format=".2f")],
        )
        .properties(width=width_for("half"),
                    title=make_title(title_text, subtitle=subtitle))
    )
    save_chart(chart, Path(out_stem), title=title_text, width_class="half")


# ── Distribution diagnostics (histogram + Q-Q grid) ──────────────────────────

def plot_distributions(
    df: pd.DataFrame,
    cols: list[str],
    out_dir: Path,
    *,
    sample_n: int = 10_000,
    out_stem: str = "distribution_hist_qq",
) -> None:
    """Per-column histogram + Q-Q plot, faceted vertically.

    When ``df`` has a ``vert_level`` column (i.e. is rib-level shape data),
    histograms and Q-Q dots are stratified by anatomical rib (1-12) using
    the side-independent ``PALETTE['rib_level']`` ramp and an interactive
    legend that toggles both layers together. Otherwise, falls back to a
    single colour.
    """
    from scipy import stats

    out_dir = Path(out_dir)
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return
    # Cap at 5000 for the Altair path: 14 cols × N × 2 facet grids in the spec
    # OOMs vl-convert/V8 above ~300k records; hist + Q-Q look the same at 5k.
    sample_n = min(int(sample_n), 5_000)
    work_cols = list(cols)
    has_rib = "vert_level" in df.columns
    if has_rib:
        work_cols = work_cols + ["vert_level"]
    sample = df[work_cols].dropna(subset=cols, how="all").sample(
        min(len(df), sample_n), random_state=42)

    hist_rows: list[dict] = []
    qq_rows: list[dict] = []
    overall_hist_rows: list[dict] = []
    overall_qq_rows:   list[dict] = []
    if has_rib:
        sample = sample.assign(_rib=sample["vert_level"].apply(vert_to_anatomical))
        # Drop out-of-range vert_levels (e.g. 20 → anatomical 13) before plotting.
        sample = sample[sample["_rib"].between(1, 12)]
        rib_groups = sorted(int(r) for r in sample["_rib"].dropna().unique())
    else:
        rib_groups = [0]
        sample = sample.assign(_rib=0)

    for col in cols:
        label = _label(col)
        # Per-rib panels.
        for rib in rib_groups:
            vals = sample.loc[sample["_rib"] == rib, col].dropna().values
            if vals.size == 0:
                continue
            counts, edges = np.histogram(vals, bins=30)
            bin_width = float(edges[1] - edges[0]) if edges.size > 1 else 1.0
            n_rib = int(vals.size)
            for c, lo, hi in zip(counts, edges[:-1], edges[1:]):
                density = (float(c) / (n_rib * bin_width)
                           if bin_width > 0 else 0.0)
                hist_rows.append({"variable": label, "rib": int(rib),
                                  "bin_lo": float(lo), "bin_hi": float(hi),
                                  "bin_mid": float((lo + hi) / 2),
                                  "count": int(c),
                                  "density": density})
            if vals.size < 3:
                continue
            (osm, osr), _ = stats.probplot(vals, dist="norm")
            # z-standardise the ordered values so every panel shares a common
            # scale and the normal reference is the identity diagonal y = x.
            sd = osr.std(ddof=1)
            osr_z = (osr - osr.mean()) / sd if sd > 0 else np.zeros_like(osr)
            for t, s in zip(osm, osr_z):
                qq_rows.append({"variable": label, "rib": int(rib),
                                "theoretical": float(t), "sample_z": float(s)})
        # Pooled "all ribs" overlay.
        all_vals = sample[col].dropna().values
        if all_vals.size:
            counts, edges = np.histogram(all_vals, bins=30)
            bin_width = float(edges[1] - edges[0]) if edges.size > 1 else 1.0
            n_pooled = int(all_vals.size)
            for c, lo, hi in zip(counts, edges[:-1], edges[1:]):
                density = (float(c) / (n_pooled * bin_width)
                           if bin_width > 0 else 0.0)
                overall_hist_rows.append({"variable": label,
                                          "bin_mid": float((lo + hi) / 2),
                                          "count": int(c),
                                          "density": density})
            if all_vals.size >= 3:
                (osm, osr), _ = stats.probplot(all_vals, dist="norm")
                sd = osr.std(ddof=1)
                osr_z = (osr - osr.mean()) / sd if sd > 0 else np.zeros_like(osr)
                for t, s in zip(osm, osr_z):
                    overall_qq_rows.append({"variable": label,
                                            "theoretical": float(t),
                                            "sample_z": float(s)})

    if not hist_rows:
        return

    hist_df = pd.DataFrame(hist_rows)
    qq_df   = pd.DataFrame(qq_rows)
    overall_hist_df = pd.DataFrame(overall_hist_rows)
    overall_qq_df   = pd.DataFrame(overall_qq_rows)
    var_order = [_label(c) for c in cols]

    if has_rib:
        rib_palette = cmap("rib_level")
        n_ribs = max(len(rib_groups), 1)
        rib_hex = [
            "#{:02X}{:02X}{:02X}".format(
                *(int(255 * c) for c in rib_palette(i / max(n_ribs - 1, 1))[:3])
            ) for i in range(n_ribs)
        ]
        color_scale = alt.Scale(domain=rib_groups, range=rib_hex)
        color_field = "rib:O"
        legend_kw = alt.Legend(title="Rib", orient="right", columns=1)
        rib_sel = alt.selection_point(fields=["rib"], bind="legend")
    else:
        color_scale = None
        color_field = None
        legend_kw = None
        rib_sel = None

    panel_height = 120

    def _maybe_color(enc_kw: dict) -> dict:
        if color_field is None:
            return enc_kw
        enc_kw["color"] = alt.Color(color_field, scale=color_scale, legend=legend_kw)
        return enc_kw

    # Combine per-rib + all-ribs data so the layered facet keeps a single
    # top-level data binding (Altair facets reject layered charts with
    # different inner datas).
    hist_bars_df = hist_df.assign(_kind="rib")
    hist_overall_df = overall_hist_df.assign(_kind="all", rib=-1, bin_lo=np.nan,
                                              bin_hi=np.nan)
    hist_combined = pd.concat([hist_bars_df, hist_overall_df], ignore_index=True)

    qq_points_df = qq_df.assign(_kind="rib")
    qq_overall_df = overall_qq_df.assign(_kind="all", rib=-1)
    qq_combined = pd.concat([qq_points_df, qq_overall_df], ignore_index=True)

    # Per-rib bars and pooled "all ribs" overlay use separate y-scales: the
    # per-rib density tops out lower than the pooled density (less smoothing
    # by aggregation), so independent y-scales keep both layers readable.
    hist_bars_enc = _maybe_color(dict(
        x=alt.X("bin_lo:Q", title=None),
        x2=alt.X2("bin_hi:Q"),
        y=alt.Y("density:Q", title="density (per rib)", stack=None,
                axis=alt.Axis(orient="left")),
        y2=alt.Y2(datum=0),
        tooltip=[alt.Tooltip("variable:N"),
                 alt.Tooltip("rib:O", title="Rib"),
                 alt.Tooltip("bin_lo:Q", title="from", format=".3f"),
                 alt.Tooltip("bin_hi:Q", title="to",   format=".3f"),
                 alt.Tooltip("count:Q"),
                 alt.Tooltip("density:Q", format=".3g")],
    ))
    if rib_sel is not None:
        hist_bars_enc["opacity"] = alt.condition(rib_sel, alt.value(0.45),
                                                  alt.value(0.05))
    hist_bars = (alt.Chart(hist_combined)
                   .transform_filter("datum._kind == 'rib'")
                   .mark_bar()
                   .encode(**hist_bars_enc))
    hist_overall = (alt.Chart(hist_combined)
                      .transform_filter("datum._kind == 'all'")
                      .mark_line(color="#C0392B", strokeWidth=1.5, opacity=0.9,
                                 interpolate="step-after")
                      .encode(x=alt.X("bin_mid:Q", title=None),
                              y=alt.Y("density:Q", title="density (pooled)",
                                      axis=alt.Axis(orient="right"))))
    hist = (alt.layer(hist_bars, hist_overall)
            .resolve_scale(y="independent")
            .properties(width=width_for("third"), height=panel_height))

    qq_enc = _maybe_color(dict(
        x=alt.X("theoretical:Q", title=None),
        y=alt.Y("sample_z:Q", title="Sample (z)"),
        tooltip=[alt.Tooltip("variable:N"),
                 alt.Tooltip("rib:O", title="Rib"),
                 alt.Tooltip("theoretical:Q", format=".3f"),
                 alt.Tooltip("sample_z:Q", title="sample (z)", format=".3f")],
    ))
    if rib_sel is not None:
        qq_enc["opacity"] = alt.condition(rib_sel, alt.value(0.6), alt.value(0.05))
    # Faint y = x identity line: a normal sample lands on it (both axes in
    # standard-normal units).
    qq_identity = (alt.Chart(qq_combined)
                     .mark_line(color="#888888", strokeWidth=0.75, opacity=0.5,
                                strokeDash=[3, 3])
                     .encode(x=alt.X("theoretical:Q", title=None),
                             y=alt.Y("theoretical:Q", title="Sample (z)")))
    qq_points = (alt.Chart(qq_combined)
                   .transform_filter("datum._kind == 'rib'")
                   .mark_circle(size=8, opacity=0.4)
                   .encode(**qq_enc))
    qq_overall = (alt.Chart(qq_combined)
                    .transform_filter("datum._kind == 'all'")
                    .mark_line(color="#C0392B", strokeWidth=1.2, opacity=0.9)
                    .encode(x=alt.X("theoretical:Q", title=None),
                            y=alt.Y("sample_z:Q")))
    qq = alt.layer(qq_identity, qq_points, qq_overall).properties(
        width=width_for("third"), height=panel_height)

    hist_facet = (hist.facet(row=alt.Row("variable:N", sort=var_order, title=None,
                                          header=alt.Header(labelAngle=0,
                                                            labelAnchor="start")))
                      .resolve_scale(x="independent", y="independent"))
    qq_facet = (qq.facet(row=alt.Row("variable:N", sort=var_order, title=None,
                                      header=alt.Header(labels=False)))
                  .resolve_scale(x="independent", y="independent"))
    concat = alt.hconcat(hist_facet, qq_facet, spacing=10)
    if rib_sel is not None:
        concat = concat.add_params(rib_sel)
    title_text = "Distribution diagnostics"
    subtitle = f"{len(cols)} parameters · histogram (left) · Q-Q-plot (right)"
    chart = concat.properties(title=make_title(title_text, subtitle=subtitle))
    save_chart(chart, out_dir / out_stem, title=title_text, width_class="full")


# ── Inclusion flow (bars — Vega-Lite has no Sankey) ──────────────────────────

def plot_inclusion_flow(
    excl_df: pd.DataFrame,
    out_stem: Path,
) -> None:
    """Horizontal bar chart of cohort inclusion: total, per-criterion exclusion counts (non-exclusive), and included."""
    reason_cols = [
        ("reason_rib_count",         "Rib count"),
        ("reason_missing_metadata",  "Missing metadata"),
        ("reason_seg_components",    "Anomalous components"),
        ("reason_seg_at_border",     "Seg at border"),
        ("reason_split",             "Rib split"),
    ]
    n_total    = len(excl_df)
    n_included = int((~excl_df["excluded"]).sum())
    n_excluded = n_total - n_included

    excluded = excl_df[excl_df["excluded"]]

    def _flagged(col: str) -> pd.Series:
        if col not in excluded.columns:
            return pd.Series(False, index=excluded.index)
        return excluded[col].fillna("").astype(str).str.strip() != ""

    # Count every reason a patient trips; a patient with multiple flags is counted
    # under each, so the exclusion bars are non-exclusive and do not sum to n_excluded.
    n_by_reason = {col: int(_flagged(col).sum()) for col, _ in reason_cols}

    named_mask = pd.Series(False, index=excluded.index)
    for col, _ in reason_cols:
        named_mask |= _flagged(col)
    n_other = int((~named_mask).sum()) if not excluded.empty else 0

    rows = [{"step": "All candidates", "count": n_total, "kind": "total"}]
    for col, lbl in reason_cols:
        rows.append({"step": f"– {lbl}", "count": n_by_reason[col], "kind": "exclusion"})
    if n_other > 0:
        rows.append({"step": "– Other / unflagged", "count": n_other, "kind": "exclusion"})
    rows.append({"step": "Included", "count": n_included, "kind": "included"})

    df = pd.DataFrame(rows)
    df["pct"] = df["count"] * 100.0 / max(n_total, 1)
    df["label"] = df.apply(lambda r: f"{int(r['count']):,} ({r['pct']:.1f}%)", axis=1)
    df["step_order"] = np.arange(len(df))

    # Excluded rows take the extreme-negative end of the RdYlGn ramp (deep red);
    # the included row takes the extreme-positive end (deep green); the "All
    # candidates" total stays a neutral grey so the eye reads it as context.
    kind_scale = alt.Scale(
        domain=["total", "exclusion", "included"],
        range=[
            "#9A9A9A",
            _rgba_to_hex(cmap("correlation")(0.0)),
            _rgba_to_hex(cmap("correlation")(1.0)),
        ],
    )
    base = alt.Chart(df).encode(
        y=alt.Y("step:N", sort=df["step"].tolist(), title=None),
    )
    bars = base.mark_bar(opacity=0.9).encode(
        x=alt.X("count:Q", title="Patients"),
        color=alt.Color("kind:N", scale=kind_scale, legend=None),
        tooltip=[alt.Tooltip("step:N"), alt.Tooltip("count:Q", format=","),
                 alt.Tooltip("pct:Q", title="% of total", format=".2f")],
    )
    text = base.mark_text(align="left", baseline="middle", dx=4, color="#222",
                          fontSize=10).encode(
        x=alt.X("count:Q"),
        text="label:N",
    )
    title_text = "Cohort inclusion"
    subtitle = (f"Total n = {n_total:,} · excluded n = {n_excluded:,} · "
                f"included n = {n_included:,} ({100*n_included/max(n_total,1):.1f}%) · "
                f"reasons overlap, not additive")
    chart = (bars + text).properties(
        width=width_for("full"),
        title=make_title(title_text, subtitle=subtitle),
    )
    save_chart(chart, Path(out_stem), title=title_text, width_class="full")


# ── Normality summary (scatter w/ shape encoding) ────────────────────────────

def plot_normality_summary(
    norm_df: pd.DataFrame,
    out_stem: Path,
    *,
    title: str | None = None,
) -> None:
    """Scatter of (skew, kurt) with one colour per variable."""
    if norm_df is None or norm_df.empty:
        return
    df = norm_df.reset_index()
    if "column" not in df.columns:
        df = df.rename(columns={df.columns[0]: "column"})
    df["variable"] = df["column"].map(_label)

    var_order = sorted(df["variable"].unique().tolist())
    title_text = title or "Normality diagnostics"
    subtitle = "Skewness × kurtosis"
    var_sel = alt.selection_point(fields=["variable"], bind="legend",
                                   name="variable_sel")
    chart = (
        alt.Chart(df)
        .mark_circle(size=160, stroke="#222", strokeWidth=0.5)
        .encode(
            x=alt.X("skewness:Q", title="Skewness"),
            y=alt.Y("kurtosis:Q", title="Excess kurtosis"),
            color=alt.Color("variable:N", sort=var_order,
                            scale=alt.Scale(scheme="tableau20"),
                            legend=alt.Legend(title="Variable")),
            opacity=alt.condition(var_sel, alt.value(0.85), alt.value(0.15)),
            tooltip=[alt.Tooltip("variable:N"),
                     alt.Tooltip("n:Q", title="n"),
                     alt.Tooltip("skewness:Q", format=".3f"),
                     alt.Tooltip("kurtosis:Q", format=".3f"),
                     alt.Tooltip("shapiro_W:Q", title="Shapiro W", format=".3f"),
                     alt.Tooltip("shapiro_p:Q", title="Shapiro p", format=".2g")],
        )
        .add_params(var_sel)
        .properties(width=width_for("half"),
                    title=make_title(title_text, subtitle=subtitle))
        .interactive()
    )
    save_chart(chart, Path(out_stem), title=title_text, width_class="half")
