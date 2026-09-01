"""Plotly publication theme.

Installs ``pio.templates["nako"]`` (the project-wide visual default) and
exposes:

  * :func:`apply_layout` – canonical width / height / margins. Figures size
    themselves via ``width_class``; no raw ``width=`` at call sites.
  * :func:`add_panel_label` – bold ``a`` / ``b`` / ``c`` panel label.

All figure-shaping values come from :mod:`settings`.
"""
from __future__ import annotations

from typing import Iterable, Literal, Mapping

import plotly.graph_objects as go
import plotly.io as pio

import settings as S
from utils import colors as C


# ── mm ↔ px and pt ↔ px conversion ───────────────────────────────────────────

def mm_to_px(mm: float) -> int:
    """Convert millimetres at print scale to Plotly layout pixels (at SCREEN_DPI)."""
    return int(round(mm * S.SCREEN_DPI / 25.4))


def pt_to_px(pt: float) -> float:
    """Convert physical points (at print scale) to Plotly ``font.size`` pixels.

    Plotly's ``font.size`` is in CSS pixels at 96 dpi, not points; multiplying
    by ``SCREEN_DPI / 72`` (= 4/3) yields the pixel value that renders at the
    requested physical point size when the figure is printed at its layout
    width. Matplotlib ``fontsize=`` is already in points – do not wrap it.
    """
    return pt * S.SCREEN_DPI / 72


WIDTH_CLASS = Literal["full", "half", "third"]
_WIDTH_MM: dict[str, float] = {
    "full":  S.FIG_WIDTH_FULL_MM,
    "half":  S.FIG_WIDTH_HALF_MM,
    "third": S.FIG_WIDTH_THIRD_MM,
}


# ── Default colorway ─────────────────────────────────────────────────────────
# Project palette + Okabe-Ito extension for traces beyond the 4th.

_COLORWAY: list[str] = [
    C.color("sex", "Male"),         # #0072B2  Okabe-Ito blue
    C.color("sex", "Female"),       # #A8195C  deep magenta
    C.color("smoking", "Current"),  # #5A2A0F  deep umber
    "#009E73",                      # Okabe-Ito green       (extension)
    "#E69F00",                      # Okabe-Ito orange      (extension)
    "#56B4E9",                      # Okabe-Ito sky blue    (extension)
    "#CC79A7",                      # Okabe-Ito reddish purple
    "#000000",                      # neutral black
]


# ── Template ─────────────────────────────────────────────────────────────────

def _build_template() -> go.layout.Template:
    """Build the project-wide Plotly template.

    Encodes typography, axes, legend, and margin defaults.  Colorway and
    diverging colorscale come from :mod:`utils.colors`.
    """
    base_font = dict(
        family=S.FONT_FAMILY,
        size=pt_to_px(S.FONT_SIZE_TICK_PT),
        color="#222222",
    )
    axis_kwargs = dict(
        showline=True,
        linewidth=S.LINE_WIDTH * 0.6,
        linecolor="#222222",
        ticks="outside",
        tickwidth=S.LINE_WIDTH * 0.6,
        tickcolor="#222222",
        ticklen=3,
        showgrid=True,
        gridwidth=S.GRID_WIDTH,
        gridcolor="rgba(0,0,0,0.07)",
        zeroline=False,
        mirror=False,
        automargin=True,
        title=dict(font=dict(family=S.FONT_FAMILY, size=pt_to_px(S.FONT_SIZE_AXIS_PT))),
    )

    return go.layout.Template(
        layout=dict(
            font=base_font,
            plot_bgcolor="white",
            paper_bgcolor="white",
            xaxis=axis_kwargs,
            yaxis=axis_kwargs,
            title=dict(
                font=dict(family=S.FONT_FAMILY, size=pt_to_px(S.FONT_SIZE_TITLE_PT),
                          color="#111111"),
                x=0.5,
                xanchor="center",
                automargin=True,
                pad=dict(l=0, r=0, t=8, b=8),
            ),
            legend=dict(
                font=dict(family=S.FONT_FAMILY, size=pt_to_px(S.FONT_SIZE_LEGEND_PT)),
                bgcolor="rgba(0,0,0,0)",
                bordercolor="rgba(0,0,0,0)",
                borderwidth=0,
            ),
            colorway=list(_COLORWAY),
            colorscale=dict(
                diverging=C.colorscale("correlation"),
                sequential=C.colorscale("magnitude"),
                sequentialminus=C.colorscale("magnitude"),
            ),
            margin=dict(**S.MARGIN_PX),
            hoverlabel=dict(
                font=dict(family=S.FONT_FAMILY, size=pt_to_px(S.FONT_SIZE_LEGEND_PT)),
            ),
            # 3D scenes: same fonts; light grid; equal aspect by default.
            scene=dict(
                xaxis=dict(
                    backgroundcolor="white",
                    gridcolor="rgba(0,0,0,0.08)",
                    showbackground=False,
                    title=dict(font=dict(family=S.FONT_FAMILY,
                                         size=pt_to_px(S.FONT_SIZE_AXIS_PT))),
                ),
                yaxis=dict(
                    backgroundcolor="white",
                    gridcolor="rgba(0,0,0,0.08)",
                    showbackground=False,
                    title=dict(font=dict(family=S.FONT_FAMILY,
                                         size=pt_to_px(S.FONT_SIZE_AXIS_PT))),
                ),
                zaxis=dict(
                    backgroundcolor="white",
                    gridcolor="rgba(0,0,0,0.08)",
                    showbackground=False,
                    title=dict(font=dict(family=S.FONT_FAMILY,
                                         size=pt_to_px(S.FONT_SIZE_AXIS_PT))),
                ),
                aspectmode="data",
            ),
        ),
    )


_TEMPLATE_NAME = "nako"


def install_template() -> None:
    """Install the ``"nako"`` template into ``plotly.io`` and make it default.

    Idempotent.  Called by :func:`settings.apply_publication_style`; safe
    to call directly too.
    """
    pio.templates[_TEMPLATE_NAME] = _build_template()
    pio.templates.default = _TEMPLATE_NAME


# ── apply_layout ─────────────────────────────────────────────────────────────

_TITLE_CHAR_BUDGET: dict[str, int] = {"full": 90, "half": 44, "third": 28}


def _wrap_title(text: str, width_class: WIDTH_CLASS) -> str:
    """Insert ``<br>`` at word boundaries so each line fits the figure width.

    Honours pre-existing ``<br>`` in the input. The per-class budget is a rough
    glyph count at ``FONT_SIZE_TITLE_PT`` – enough to keep titles within the
    plot's print width without measuring text metrics.
    """
    budget = _TITLE_CHAR_BUDGET[width_class]
    out_lines: list[str] = []
    for paragraph in text.split("<br>"):
        line = ""
        for word in paragraph.split(" "):
            if not line:
                line = word
            elif len(line) + 1 + len(word) <= budget:
                line += " " + word
            else:
                out_lines.append(line)
                line = word
        if line:
            out_lines.append(line)
    return "<br>".join(out_lines)


def apply_layout(
    fig: go.Figure,
    width_class: WIDTH_CLASS = "full",
    *,
    n_rows: int = 1,
    height_mm: float | None = None,
    margin: Mapping[str, int] | None = None,
    title: str | None = None,
) -> go.Figure:
    """Mutate ``fig`` with canonical width / height / margins / template.

    Parameters
    ----------
    fig
        The Plotly figure to mutate.
    width_class
        ``"full"`` (170 mm), ``"half"`` (85 mm), or ``"third"`` (56 mm).
        Default ``"full"`` per the project-wide consistency requirement
        – only opt into a narrower width when the figure is genuinely a
        side-by-side small element.
    n_rows
        Number of plot rows.  When ``height_mm`` is not given, the
        figure height becomes ``n_rows * settings.FIG_ROW_HEIGHT_MM``.
    height_mm
        Override the computed height with an explicit value (mm).
    margin
        Override the default margin (Plotly ``layout.margin`` pixels).
    title
        Figure title. Word-wrapped to fit ``width_class``; Plotly's
        ``title.automargin`` (set in the template) auto-grows the top margin
        so the title never overlaps the plot area.

    Returns
    -------
    The same ``fig`` (mutated, returned for chaining).
    """
    if width_class not in _WIDTH_MM:
        raise ValueError(
            f"width_class must be one of {tuple(_WIDTH_MM)}; got {width_class!r}"
        )
    width_mm = _WIDTH_MM[width_class]
    h_mm = float(height_mm) if height_mm is not None else float(n_rows) * S.FIG_ROW_HEIGHT_MM

    if title is not None:
        fig.update_layout(title=dict(text=_wrap_title(title, width_class)))

    requested = dict(margin) if margin is not None else dict(S.MARGIN_PX)
    existing = fig.layout.margin
    final_margin = {
        side: max(int(requested[side]), int(getattr(existing, side) or 0))
        for side in ("l", "r", "t", "b")
    }

    fig.update_layout(
        template=_TEMPLATE_NAME,
        width=mm_to_px(width_mm),
        height=mm_to_px(h_mm),
        margin=final_margin,
    )
    return fig


# ── add_panel_label ──────────────────────────────────────────────────────────

def add_panel_label(
    fig: go.Figure,
    label: str,
    *,
    xref: str = "paper",
    yref: str = "paper",
    x: float = 0.0,
    y: float = 1.0,
    xshift: int = -28,
    yshift: int = 12,
) -> None:
    """Add a bold panel label (``a`` / ``b`` / ``c``) at the top-left.

    Defaults place the label slightly outside the axes area, in line
    with Nature / Cell convention.  ``xshift`` / ``yshift`` are pixel
    offsets that survive plot resizing.
    """
    fig.add_annotation(
        text=f"<b>{label}</b>",
        xref=xref, yref=yref,
        x=x, y=y,
        xshift=xshift, yshift=yshift,
        showarrow=False,
        font=dict(family=S.FONT_FAMILY, size=pt_to_px(S.FONT_SIZE_PANEL_PT), color="#111111"),
        align="left",
    )


# ── Helpers used by plot modules ─────────────────────────────────────────────

_LEGEND_RESERVE_PX = 24


def annotate_n(fig: go.Figure, text: str) -> None:
    """Add a small bottom-left figure annotation stating ``n``, the test, etc.

    Bumps ``layout.margin.b`` enough to host the annotation *and* a horizontal
    legend below it (the legend, if present, anchors to container bottom).
    Plotly's ``annotation`` does not support ``yref="container"`` and does
    not auto-grow margins, so we reserve manually.
    """
    text_px = int(pt_to_px(S.FONT_SIZE_ANNOT_PT)) + 6
    target_b = int(S.MARGIN_PX["b"]) + text_px + _LEGEND_RESERVE_PX
    current_b = fig.layout.margin.b
    if current_b is None or current_b < target_b:
        fig.update_layout(margin=dict(b=target_b))
    fig.add_annotation(
        text=text,
        xref="paper", yref="paper",
        x=0.0, y=0.0,
        xanchor="left", yanchor="top",
        yshift=-(int(S.MARGIN_PX["b"]) + _LEGEND_RESERVE_PX + 2),
        showarrow=False,
        font=dict(family=S.FONT_FAMILY, size=pt_to_px(S.FONT_SIZE_ANNOT_PT), color="#444444"),
        align="left",
    )


def place_legend_bottom(fig: go.Figure, *, title: str | None = None) -> None:
    """Anchor a horizontal legend to the bottom edge of the figure container.

    Container-relative coordinates mean the legend never overflows. Bumps
    ``layout.margin.b`` to reserve clearance between the plot area and the
    legend; :func:`annotate_n` already reserves a matching slot, so call
    order between the two does not matter.
    """
    legend = dict(
        orientation="h",
        xref="container", yref="container",
        x=0.0, y=0.0,
        xanchor="left", yanchor="bottom",
    )
    if title is not None:
        legend["title"] = dict(text=title)
    target_b = int(S.MARGIN_PX["b"]) + _LEGEND_RESERVE_PX
    current_b = fig.layout.margin.b
    if current_b is None or current_b < target_b:
        fig.update_layout(margin=dict(b=target_b))
    fig.update_layout(legend=legend)


def place_colorbar_right(
    *,
    title: str | None = None,
    thickness: int = 10,
    length_fraction: float = 0.6,
    pad: int = 4,
) -> dict:
    """Return a ``colorbar=`` dict that hugs the right edge of the plot area.

    ``xanchor='left'`` plus ``x=1.0`` puts the colorbar *just outside* the plot
    box; Plotly does not auto-expand the right margin for colorbars, so callers
    must ensure ``layout.margin.r`` is wide enough – the ``MARGIN_PX['r']``
    floor (40 px) accommodates the default thickness/title; bump per-figure
    via :func:`reserve_colorbar_margin` only for unusually wide titles.
    """
    cb = dict(
        xref="paper",
        x=1.0,
        xanchor="left",
        thickness=thickness,
        len=length_fraction,
        lenmode="fraction",
        xpad=pad,
    )
    if title is not None:
        cb["title"] = dict(
            text=title,
            side="top",
            font=dict(family=S.FONT_FAMILY, size=pt_to_px(S.FONT_SIZE_AXIS_PT)),
        )
    return cb


def reserve_colorbar_margin(fig: go.Figure, *, extra_px: int = 0) -> None:
    """Ensure ``layout.margin.r`` is at least ``MARGIN_PX['r'] + extra_px``."""
    target = int(S.MARGIN_PX["r"] + extra_px)
    current_r = fig.layout.margin.r
    if current_r is None or current_r < target:
        fig.update_layout(margin=dict(r=target))


def grid_spacing(n_rows: int, row_height_mm: float, gap_mm: float = 4.0) -> float:
    """Proportional ``vertical_spacing`` for ``make_subplots`` that targets an
    inter-panel gap of ``gap_mm`` regardless of ``n_rows``.

    Plotly's ``vertical_spacing`` is a fraction of the inner plot area
    (figure − top − bottom margin); we approximate that inner area as
    ``n_rows × row_height_mm − 25 mm`` (≈ default margin total at 96 dpi).
    Pick ``gap_mm`` ≥ 12 for 2D subplots whose top row has visible tick
    labels (otherwise subplot titles can land on the ticks above).
    """
    if n_rows <= 1:
        return 0.0
    inner_mm = max(float(n_rows) * float(row_height_mm) - 25.0, 1.0)
    return float(gap_mm) / inner_mm


def sig_stars(p: float) -> str:
    """``***`` for *p* < 0.001, ``**`` < 0.01, ``*`` < 0.05, ``ns`` otherwise."""
    if p is None or (isinstance(p, float) and p != p):  # NaN
        return ""
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


def categorical_palette_for(variable: str, values: Iterable[str]) -> dict[str, str]:
    """Return a ``{value: hex}`` dict for use as ``color_discrete_map=`` in plotly.express."""
    return {v: C.color(variable, v) for v in values}


# ── Axis-reference helpers ───────────────────────────────────────────────────
# Plotly names the *first* x-axis ``"x"`` (no ``"1"`` suffix); subsequent
# ones are ``"x2"``, ``"x3"``, ….  Annotations / shapes that need to bind
# to a specific subplot must therefore special-case ``idx == 1``.

def axref(kind: str, idx: int, *, domain: bool = False) -> str:
    """Return the Plotly axis-reference string for ``(kind='x'|'y', idx)``.

    ``idx`` is 1-based.  When ``domain=True``, ``" domain"`` is appended.
    """
    if kind not in ("x", "y"):
        raise ValueError(f"axref kind must be 'x' or 'y'; got {kind!r}")
    if idx < 1:
        raise ValueError(f"axref idx must be ≥ 1; got {idx}")
    name = kind if idx == 1 else f"{kind}{idx}"
    return f"{name} domain" if domain else name
