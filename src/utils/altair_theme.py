"""Altair color adapter, width helper, and shared chart factories.

Bridges :mod:`utils.colors` (matplotlib-based ``PALETTE``) to Altair
``Scale`` objects, and exposes ``width_for(class_)`` so charts honour the
project's print widths (170/85/56 mm). Deliberately thin – no font /
layout / template registration.

Public API::

    from utils.altair_theme import (
        sex_scale, smoking_scale,
        correlation_scale, continuous_scale,
        width_for, fdr_masked_heatmap,
        make_title, pc_axis, subscript,
    )
"""
from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd

import settings as S
from utils.colors import (
    PALETTE,
    SEX_ORDER,
    SMOKE_ORDER,
    _rgba_to_hex,
    color,
    cmap,
    predictor_color,
    rib_colors,
)
from utils.plotly_theme import sig_stars

alt.data_transformers.disable_max_rows()


# ── Unit conversion ──────────────────────────────────────────────────────────

def _mm_to_px(mm: float) -> int:
    return int(round(mm * S.SCREEN_DPI / 25.4))


_WIDTH_MM: dict[str, float] = {
    "full":  S.FIG_WIDTH_FULL_MM,
    "half":  S.FIG_WIDTH_HALF_MM,
    "third": S.FIG_WIDTH_THIRD_MM,
}


def width_for(class_: str | None) -> int | None:
    """Return chart width in pixels for ``class_`` (``"full"`` / ``"half"`` / ``"third"``)."""
    if class_ is None:
        return None
    try:
        return _mm_to_px(_WIDTH_MM[class_])
    except KeyError:
        raise KeyError(
            f"altair_theme.width_for(): unknown class {class_!r}; "
            f"expected one of {sorted(_WIDTH_MM)}."
        ) from None


# ── Categorical scales ───────────────────────────────────────────────────────

def sex_scale() -> alt.Scale:
    """``alt.Scale`` mapping ``Male`` / ``Female`` to project hexes."""
    return alt.Scale(
        domain=list(SEX_ORDER),
        range=[color("sex", v) for v in SEX_ORDER],
    )


def smoking_scale() -> alt.Scale:
    """``alt.Scale`` mapping smoking status to project hexes."""
    return alt.Scale(
        domain=list(SMOKE_ORDER),
        range=[color("smoking", v) for v in SMOKE_ORDER],
    )


def predictor_scale(raw_keys: list[str], labels: list[str]) -> alt.Scale:
    """``alt.Scale`` mapping predictor display labels to predictor hexes.

    ``raw_keys`` and ``labels`` are parallel (one per predictor); the raw key
    drives the hex lookup via :func:`utils.colors.predictor_color`.
    """
    return alt.Scale(
        domain=list(labels),
        range=[predictor_color(k) for k in raw_keys],
    )


# ── Continuous / diverging scales ────────────────────────────────────────────

def _cmap_to_hex_list(variable: str, n: int = 256) -> list[str]:
    cm = cmap(variable)
    out: list[str] = []
    for t in np.linspace(0.0, 1.0, n):
        r, g, b, _ = cm(float(t))
        out.append(f"#{int(r*255):02X}{int(g*255):02X}{int(b*255):02X}")
    return out


def correlation_scale(vmax: float | None = None) -> alt.Scale:
    """Diverging scale for signed effects, symmetric about zero.

    Domain is ``[-vmax, +vmax]`` (default ``vmax=1.0``, the correlation
    convention). The full colormap ramp is handed to Vega so zero maps to the
    midpoint colour and mid-tones match the matplotlib / Plotly heatmaps.
    """
    if vmax is None:
        vmax = 1.0
    vmax = float(vmax)
    return alt.Scale(domain=[-vmax, vmax],
                     range=_cmap_to_hex_list("correlation"),
                     type="linear")


def continuous_scale(variable: str,
                     domain: tuple[float, float] | None = None) -> alt.Scale:
    """Sequential scale for a continuous variable defined in PALETTE.

    ``domain`` is optional; when omitted the chart-level scale picks data extent.
    """
    rng = _cmap_to_hex_list(variable)
    if domain is None:
        return alt.Scale(range=rng, type="linear")
    return alt.Scale(domain=list(domain), range=rng, type="linear")


# ── Direct hex accessors ─────────────────────────────────────────────────────
# Plot code that needs a single hex (e.g. a constant mark colour) can call
# ``color()`` directly from utils.colors. Re-exported here for proximity.


# ── Pre-computed KDE for violins ─────────────────────────────────────────────
# Vega-Lite's ``transform_density`` ignores ``resolve_scale(y='independent')``
# inside facets, so wide-range groups get squashed to a global y-domain.
# Pre-computing per-group KDE in pandas and normalising each curve to a
# peak density of 1 sidesteps that and also equalises violin widths across
# facets with vastly different value scales.

def kde_long(
    df: pd.DataFrame,
    groupby_cols: list[str],
    value_col: str,
    *,
    steps: int = 200,
    pad: float = 0.10,
) -> pd.DataFrame:
    """Per-group Gaussian KDE; returns long-form (groupby + value + density)."""
    from scipy.stats import gaussian_kde

    rows: list[dict] = []
    for key, sub in df.groupby(groupby_cols, sort=False):
        vals = sub[value_col].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size < 2 or np.ptp(vals) == 0:
            continue
        try:
            kde = gaussian_kde(vals)
        except (np.linalg.LinAlgError, ValueError):
            continue
        lo, hi = float(vals.min()), float(vals.max())
        span = (hi - lo) or 1.0
        xs = np.linspace(lo - pad * span, hi + pad * span, steps)
        dens = kde(xs)
        dmax = float(dens.max())
        if dmax <= 0:
            continue
        dens = dens / dmax
        key_dict = dict(zip(groupby_cols,
                            key if isinstance(key, tuple) else (key,)))
        for x, d in zip(xs, dens):
            rows.append({**key_dict, value_col: float(x), "density": float(d)})
    return pd.DataFrame(rows)


__all__ = [
    "sex_scale",
    "smoking_scale",
    "predictor_scale",
    "correlation_scale",
    "continuous_scale",
    "width_for",
    "fdr_masked_heatmap",
    "kde_long",
    "make_title",
    "pc_axis",
    "subscript",
    "rib_side_color",
    "STAR_LEGEND",
    "PALETTE",
    "color",
]


# ── Title + subtitle helper ──────────────────────────────────────────────────

STAR_LEGEND: str = "* p<0.05 · ** p<0.01 · *** p<0.001"


def make_title(text: str, *, subtitle: str | list[str] | None = None) -> alt.TitleParams:
    """``alt.TitleParams`` with project font sizes and left-anchored layout."""
    kwargs: dict = dict(
        text=text,
        fontSize=S.FONT_SIZE_TITLE_PT * 4 / 3,
        subtitleFontSize=S.FONT_SIZE_LEGEND_PT * 4 / 3,
        anchor="start",
        offset=6,
    )
    if subtitle is not None:
        kwargs["subtitle"] = subtitle
    return alt.TitleParams(**kwargs)


# ── PC axis helper (label every Nth PC, drop "PC" prefix) ───────────────────

def pc_axis(pc_labels: list[str], *, every: int = 5,
            title: str | None = "PC") -> alt.Axis:
    """Axis that labels every Nth PC and shortens labels (``PC7`` → ``7``).

    ``pc_labels`` is the ordered list of PC tick values exactly as they appear
    in the encoded field. The axis ``values`` keeps every Nth and the first
    label always; ``labelExpr`` strips the ``PC`` prefix.
    """
    if not pc_labels:
        return alt.Axis(title=title)
    n = len(pc_labels)
    keep = [pc_labels[0]] + [pc_labels[i] for i in range(every - 1, n, every)
                              if pc_labels[i] != pc_labels[0]]
    return alt.Axis(
        title=title,
        values=keep,
        labelExpr="replace(replace(datum.value, 'PC_', ''), 'PC', '')",
        labelAngle=0,
    )


# ── Unicode subscript helper ────────────────────────────────────────────────

_SUBSCRIPT_TABLE = str.maketrans({
    "0": "₀", "1": "₁", "2": "₂", "3": "₃", "4": "₄",
    "5": "₅", "6": "₆", "7": "₇", "8": "₈", "9": "₉",
    "+": "₊", "-": "₋", "=": "₌", "(": "₍", ")": "₎",
    "a": "ₐ", "e": "ₑ", "h": "ₕ", "i": "ᵢ", "j": "ⱼ",
    "k": "ₖ", "l": "ₗ", "m": "ₘ", "n": "ₙ", "o": "ₒ",
    "p": "ₚ", "r": "ᵣ", "s": "ₛ", "t": "ₜ", "u": "ᵤ",
    "v": "ᵥ", "x": "ₓ",
})


def subscript(text: str, sub: str) -> str:
    """Append ``sub`` to ``text`` as unicode subscripts (digits + select letters)."""
    return f"{text}{str(sub).translate(_SUBSCRIPT_TABLE)}"


# ── Rib side colours (Portland-neg for L, Portland-pos for R) ────────────────

def rib_side_color(
    rib_levels: list[int],
    *,
    color_field: str = "color_key",
    legend_title: str = "Rib · side",
) -> tuple[alt.Color, alt.Parameter]:
    """Composite ``(level, side)`` → hex colour encoding with a visible 24-entry legend.

    Mirrored L/R plots shade each bar by its ``"{level}-{side}"`` key (T8 L
    lighter than T19 L, etc.) drawn from the rib_left / rib_right ramps. The
    legend lists every ``(level, side)`` swatch; ``labelExpr`` renders the
    cryptic ``"7-L"`` key as ``"Rib 7 L"``. The encoding is bound to a
    ``selection_point`` so clicking a swatch toggles its bars.

    Consumers set a ``color_field`` column (``"{level}-{side}"``) on their data.

    Returns ``(color, color_selection)``:
      - ``color``: ``alt.Color(color_field:N, …)`` with the visible legend.
      - ``color_selection``: ``selection_point(fields=[color_field], bind="legend")``.
        Consumers add ``opacity=alt.condition(color_selection, …)`` to their
        marks and ``.add_params(color_selection)`` before faceting.
    """
    levels = sorted(set(int(lv) for lv in rib_levels))
    keys = [f"{lv}-L" for lv in levels] + [f"{lv}-R" for lv in levels]
    hexes = rib_colors(levels + levels, ["L"] * len(levels) + ["R"] * len(levels))
    composite_scale = alt.Scale(domain=keys, range=hexes)
    color_selection = alt.selection_point(fields=[color_field], bind="legend",
                                           name="rib_side_sel")
    color = alt.Color(
        f"{color_field}:N", scale=composite_scale, sort=keys,
        legend=alt.Legend(
            title=legend_title, columns=2, symbolLimit=0,
            labelExpr="'Rib ' + split(datum.value, '-')[0] + ' ' "
                      "+ split(datum.value, '-')[1]",
        ),
    )
    return color, color_selection


# ── Shared chart factories ───────────────────────────────────────────────────

def fdr_masked_heatmap(
    z_pivot: pd.DataFrame,
    q_pivot: pd.DataFrame | None = None,
    *,
    value_label: str = "value",
    text_format: str = ".2f",
    annotate: bool = True,
    show_stars: bool = True,
    diverging: bool = True,
    fdr_threshold: float = 0.05,
    vmax: float | None = None,
    width_class: str | None = "full",
    title: str | alt.TitleParams | None = None,
    x_axis: alt.Axis | None = None,
    y_axis: alt.Axis | None = None,
    x_axis_title: str | None = None,
    y_axis_title: str | None = None,
    row_title: str = "row",
    col_title: str = "col",
    tickangle_x: float = -30.0,
    sequential_var: str = "magnitude",
    row_order: list[str] | None = None,
    col_order: list[str] | None = None,
    contrast_threshold: float = 0.55,
) -> alt.LayerChart:
    """Heatmap of ``z_pivot`` (rows × cols), optionally FDR-masked by ``q_pivot``.

    Cells with ``q >= fdr_threshold`` (or NaN q) are drawn as empty white cells
    with a faint border rather than dropped, so the full row × col grid stays
    visible even when an entire row/column – or the whole plot – is
    non-significant (a background rect layer over every cell pins the axes).
    ``diverging=True`` (default) uses :func:`correlation_scale` symmetric at
    zero; ``diverging=False`` uses :func:`continuous_scale(sequential_var)``
    from zero to ``vmax`` (auto-detected if ``None``).

    Annotation text colour flips to white on dark cells when
    ``|value|/vmax > contrast_threshold``.

    When ``x_axis`` / ``y_axis`` are provided, they override the auto-built
    axes (used to inject :func:`pc_axis` on multi-PC heatmaps).
    """
    rows = list(z_pivot.index) if row_order is None else list(row_order)
    cols = list(z_pivot.columns) if col_order is None else list(col_order)
    z = z_pivot.reindex(index=rows, columns=cols).values.astype(float)
    if q_pivot is not None:
        q = q_pivot.reindex(index=rows, columns=cols).values.astype(float)
        mask = (q >= fdr_threshold) | np.isnan(q)
    else:
        q = np.full_like(z, np.nan)
        mask = np.isnan(z)
    z_disp = np.where(mask, np.nan, z)

    if vmax is None:
        finite = np.abs(z_disp)[np.isfinite(z_disp)]
        vmax = float(finite.max()) if finite.size else (1.0 if diverging else 0.5)
    vmax = max(float(vmax), 1e-9)

    # Every (row, col) goes into df_all (so the grid is always complete); only
    # FDR-significant, finite cells go into df_sig (the coloured + annotated layer).
    all_recs, sig_recs = [], []
    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            finite = bool(np.isfinite(z[i, j]))
            qv = float(q[i, j]) if np.isfinite(q[i, j]) else None
            all_recs.append({
                "row": r, "col": c,
                "value": float(z[i, j]) if finite else None,
                "q": qv,
            })
            if mask[i, j] or not finite:
                continue
            stars = sig_stars(q[i, j]) if (show_stars and np.isfinite(q[i, j])) else ""
            text = format(z[i, j], text_format)
            sig_recs.append({
                "row": r, "col": c, "value": float(z[i, j]), "q": qv, "stars": stars,
                "text": f"{text}\n{stars}".rstrip() if (annotate and stars) else text,
            })
    if not all_recs:
        return alt.layer(alt.Chart(pd.DataFrame({"row": [], "col": []})).mark_rect())
    df_all = pd.DataFrame.from_records(all_recs)
    df_sig = pd.DataFrame.from_records(sig_recs)

    if x_axis is None:
        x_axis = alt.Axis(labelAngle=tickangle_x, title=x_axis_title)
    if y_axis is None:
        y_axis = alt.Axis(title=y_axis_title)
    x_enc = alt.X("col:N", sort=cols, axis=x_axis)
    y_enc = alt.Y("row:N", sort=rows, axis=y_axis)
    tooltip = [alt.Tooltip("row:N", title=row_title),
               alt.Tooltip("col:N", title=col_title),
               alt.Tooltip("value:Q", title=value_label, format=".3g"),
               alt.Tooltip("q:Q", title="q", format=".2g")]

    # Background grid: a white cell with a faint border for every (row, col) –
    # keeps masked/missing cells visible and pins the axis domains.
    bg = alt.Chart(df_all).mark_rect(
        fill="white", stroke="#d9d9d9", strokeWidth=0.5,
    ).encode(x=x_enc, y=y_enc, tooltip=tooltip)
    layers: list[alt.Chart] = [bg]

    if len(df_sig):
        color_enc = alt.Color(
            "value:Q",
            scale=(correlation_scale(vmax=vmax) if diverging
                   else continuous_scale(sequential_var, domain=(0.0, vmax))),
            legend=alt.Legend(title=value_label),
        )
        heat = alt.Chart(df_sig).mark_rect(stroke="white", strokeWidth=1).encode(
            x=x_enc, y=y_enc, color=color_enc, tooltip=tooltip,
        )
        layers.append(heat)
        if annotate and not df_sig["text"].eq("").all():
            # Diverging scales saturate at ±vmax, sequential at vmax – same test
            # for both: |value|/vmax > threshold ⇒ dark fill ⇒ flip text to white.
            dark_expr = f"abs(datum.value) > {vmax * contrast_threshold}"
            layers.append(alt.Chart(df_sig).mark_text(baseline="middle").encode(
                x=x_enc, y=y_enc, text=alt.Text("text:N"),
                color=alt.condition(dark_expr, alt.value("white"), alt.value("#222")),
            ))

    chart = alt.layer(*layers)
    if width_class is not None:
        chart = chart.properties(width=width_for(width_class))
    if title is not None:
        chart = chart.properties(title=title)
    return chart
