"""Project-wide colour palette.

Continuous variables use sequential colormaps (truncated at the light end);
categorical variables use discrete hex codes; signed values around zero use a
diverging colormap. Categorical hexes pass a ΔE76 ≥ 18 separation test under
deuteranopia / protanopia simulation (see ``tests/test_palette.py``).

Public API::

    from utils.colors import PALETTE, cmap, color, colorscale, rib_colors

    plt.scatter(x, y, c=df['age'], cmap=cmap('age'))      # continuous
    sex_color = color('sex', 'Male')                       # categorical
    fig.add_trace(go.Heatmap(z=corr, colorscale=colorscale('correlation'),
                              zmin=-1, zmax=1))            # Plotly

Variable assignments
--------------------
Continuous (single-hue or perceptual sequential, truncated):
    age          Greens          bmi          Purples
    weight       Blues           body_fat     Teal
    height       PuRd            pack_years   turbid
    rib_left     Portland negative half (t ∈ [0.30, 0.0]: blues; T8 light → T19 dark)
    rib_right    Portland positive half (t ∈ [0.70, 1.0]: oranges→reds; T8 light → T19 dark)
    rib_level    Sunsetdark (side-independent; used when L/R split isn't carried)
    magnitude    PuBu (matplotlib sequential light→deep blue)
    residual     RdYlGn negative half (yellow → deep red; shared with correlation)

Categorical:
    sex          Female = deep magenta,  Male = Okabe-Ito blue
    smoking      turbid 3-point ramp (t = 0 / 0.5 / 1): never → previous → current

Diverging:
    correlation  RdYlGn (red = negative, green = positive)
    
Run this file directly to print the palette as a reference card.
"""
from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import matplotlib as mpl
import matplotlib.colors as mcolors
import numpy as np


# ── Helpers ──────────────────────────────────────────────────────────────────

def _truncate(cmap_, vmin: float = 0.25, vmax: float = 1.0,
              n: int = 256, name: str | None = None):
    """Crop a colormap so it starts at ``vmin`` (0 = light end, 1 = dark end)."""
    if isinstance(cmap_, str):
        cmap_ = mpl.colormaps[cmap_]
    base = name or getattr(cmap_, "name", "cmap")
    return mcolors.LinearSegmentedColormap.from_list(
        f"{base}_trunc", cmap_(np.linspace(vmin, vmax, n))
    )


def _rgba_to_hex(rgba) -> str:
    r, g, b, _ = rgba
    return f"#{int(r * 255):02X}{int(g * 255):02X}{int(b * 255):02X}"


# ── CARTOcolors ramps (not built into matplotlib) ────────────────────────────
# Source: https://carto.com/carto-colors/

TEAL = mcolors.LinearSegmentedColormap.from_list("Teal", [
    "#d1eeea", "#a8dbd9", "#85c4c9", "#68abb8",
    "#4f90a6", "#3b738f", "#2a5674",
])

# Plotly diverging.Portland — used for per-side rib ramps (negative half →
# rib_left, positive half → rib_right) so left/right share a single tonal
# family without competing with the diverging "correlation" scale.
PORTLAND = mcolors.LinearSegmentedColormap.from_list("Portland", [
    "#0c3383", "#0a88ba", "#f2d338", "#f28f38", "#d91e1e",
])

# CARTO sequential Sunsetdark — side-independent rib_level ramp.
SUNSETDARK = mcolors.LinearSegmentedColormap.from_list("Sunsetdark", [
    "#fcde9c", "#faa476", "#f0746e", "#e34f6f",
    "#dc3977", "#b9257a", "#7c1d6f",
])

# cmocean sequential turbid (matches Plotly's `turbid`) — pack_years dose ramp
# and the 3-point smoking scale (sampled at t = 0 / 0.5 / 1).
TURBID = mcolors.LinearSegmentedColormap.from_list("turbid", [
    "#e9f6ab", "#dfe292", "#d7cf7b", "#cfbc66", "#c8a954", "#bf9747",
    "#b58740", "#a8773c", "#9a6a3b", "#8a5e3a", "#795338", "#674835",
    "#563e30", "#44342a", "#332a23", "#221f1b",
])

# ── Categorical colour mappings ──────────────────────────────────────────────
# Long-form and short-form aliases (``"Male"`` / ``"M"`` / ``"male"``) all map
# to the same hex so plot code doesn't need to know the upstream convention.

_SEX = {
    "Male":   "#0072B2",  # Okabe-Ito blue
    "Female": "#A8195C",  # deep magenta
}
_SEX_ALIAS = {"M": _SEX["Male"], "F": _SEX["Female"],
              "male": _SEX["Male"], "female": _SEX["Female"]}

# turbid sampled at t = 0 / 0.5 / 1 — an ordered ramp whose three stops clear
# the ΔE76 ≥ 18 colour-blind separation test (tests/test_palette.py).
_SMOKE = {
    "Never":     _rgba_to_hex(TURBID(0.0)),  # pale yellow
    "Ex-smoker": _rgba_to_hex(TURBID(0.5)),  # tan
    "Current":   _rgba_to_hex(TURBID(1.0)),  # near-black
}
_SMOKE_ALIAS = {
    "never":     _SMOKE["Never"],
    "ex":        _SMOKE["Ex-smoker"],
    "ex_smoker": _SMOKE["Ex-smoker"],
    "previous":  _SMOKE["Ex-smoker"],
    "current":   _SMOKE["Current"],
    "smoker":    _SMOKE["Current"],
}


def _sex_dict()  -> dict[str, str]: return {**_SEX,   **_SEX_ALIAS}
def _smoke_dict()-> dict[str, str]: return {**_SMOKE, **_SMOKE_ALIAS}


# ── PALETTE ──────────────────────────────────────────────────────────────────

PALETTE: dict[str, object] = {
    # --- continuous (sequential) ---
    "age":          _truncate("Greens", vmin=0.30),
    "bmi":          _truncate("Purples", vmin=0.30),
    "weight":       _truncate("Blues", vmin=0.30),
    "weight_kg":    _truncate("Blues", vmin=0.30),
    "body_fat":     _truncate(TEAL, vmin=0.0, name="Teal"),
    "body_fat_pct": _truncate(TEAL, vmin=0.0, name="Teal"),
    "body_fat_kg":  _truncate(TEAL, vmin=0.0, name="Teal"),
    "height":       _truncate("PuRd", vmin=0.30),
    "height_cm":    _truncate("PuRd", vmin=0.30),
    "pack_years":   _truncate(TURBID, vmin=0.0, name="turbid"),

    # --- per-side rib ramps; position along ramp = rib level (T8 → T19).
    # Both halves of Portland diverging — left = blues, right = oranges/reds.
    # _truncate accepts reversed bounds (vmin > vmax via np.linspace) so the
    # LIGHT end of each truncated cmap (t=0.0) corresponds to T8 and the DARK
    # end (t=1.0) to T19. Stops chosen inside [0, 0.30] and [0.70, 1.0] to
    # stay clear of Portland's yellow center crossover (t≈0.5).
    "rib_left":     _truncate(PORTLAND, vmin=0.30, vmax=0.0, name="Portland-neg"),
    "rib_right":    _truncate(PORTLAND, vmin=0.70, vmax=1.0, name="Portland-pos"),
    # Side-independent rib level ramp for plots that don't carry L/R.
    "rib_level":    _truncate(SUNSETDARK, vmin=0.0, name="Sunsetdark"),

    # --- generic non-negative magnitude (displacement, residual, etc.) ---
    # Distinct from the rib ramps so "rib k" colour is never confused with
    # "displacement value" colour.
    "magnitude":    _truncate("PuBu", vmin=0.0, name="PuBu"),

    # Rib-cage shape change (PC-mode |Δ| and interactive viewer displacement).
    # Distinct from "magnitude" so changes to one scale don't bleed into the
    # other.
    "displacement": _truncate("plasma", vmin=0.0, name="plasma"),

    # Negative (red) half of the diverging correlation scale, repurposed as a
    # sequential ramp for fit-error magnitudes: t=0 → yellow (no error), t=1 →
    # deep red (worst). Reversed bounds map the colormap's neutral midpoint to
    # the light end and its dark-red endpoint to the dark end.
    "residual":     _truncate("RdYlGn", vmin=0.5, vmax=0.0, name="RdYlGn-neg"),

    # --- categorical ---
    "sex":            _sex_dict(),
    "smoking":        _smoke_dict(),
    "smoking_status": _smoke_dict(),

    # --- diverging (signed values around zero) ---
    "correlation":  mpl.colormaps["RdYlGn"],
}


# ── Strict accessors ─────────────────────────────────────────────────────────
# Unknown keys raise; silent fallthrough is the easiest way to reintroduce
# hardcoded colours.

def cmap(variable: str):
    """Return the matplotlib ``Colormap`` for a continuous / diverging variable."""
    try:
        entry = PALETTE[variable]
    except KeyError:
        raise KeyError(
            f"colors.cmap(): variable {variable!r} is not in PALETTE. "
            f"Add it to utils.colors.PALETTE first."
        ) from None
    if isinstance(entry, Mapping):
        raise KeyError(
            f"colors.cmap(): variable {variable!r} is categorical; use color() instead."
        )
    return entry


def color(variable: str, value) -> str:
    """Return the hex colour for a categorical ``(variable, value)`` pair."""
    try:
        entry = PALETTE[variable]
    except KeyError:
        raise KeyError(
            f"colors.color(): variable {variable!r} is not in PALETTE. "
            f"Add it to utils.colors.PALETTE first."
        ) from None
    if not isinstance(entry, Mapping):
        raise KeyError(
            f"colors.color(): variable {variable!r} is continuous; use cmap() instead."
        )
    try:
        return entry[value]
    except KeyError:
        raise KeyError(
            f"colors.color(): no entry for {variable}={value!r}. "
            f"Known values: {sorted(set(entry))}"
        ) from None


def colorscale(variable: str, n: int = 256) -> list[list]:
    """Return a Plotly-format colorscale ``[[t0, hex0], …, [tN, hexN]]``."""
    cm = cmap(variable)
    samples = np.linspace(0.0, 1.0, n)
    out: list[list] = []
    for t in samples:
        r, g, b, _ = cm(float(t))
        out.append([float(t), f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"])
    return out


def categorical_colors(variable: str, values: Iterable[str]) -> list[str]:
    """Ordered hex list for the given categorical values (preserves order)."""
    return [color(variable, v) for v in values]


# ── Predictor / metadata-variable colours ────────────────────────────────────
# Single source of truth so the Altair scatters (modelA-vs-B, normality)
# share the same predictor → hex map.

_PREDICTOR_OVERRIDES: dict[str, str] = {
    "sex":            _SEX["Female"],
    "sex_Female":     _SEX["Female"],
    "sex_Male":       _SEX["Male"],
    "smoking":        _SMOKE["Current"],
    "smoking_status": _SMOKE["Current"],
    "smoke_Current":  _SMOKE["Current"],
    "smoke_Ex":       _SMOKE["Ex-smoker"],
    "smoke_Never":    _SMOKE["Never"],
}


def predictor_color(key: str) -> str:
    """Hex colour for a predictor / metadata variable.

    Categorical predictors hit ``_PREDICTOR_OVERRIDES``. Continuous predictors
    sample their PALETTE colormap at t=0.7 (the dark end of the truncated ramp)
    so each variable gets a distinct, identifiable hex.
    """
    if key in _PREDICTOR_OVERRIDES:
        return _PREDICTOR_OVERRIDES[key]
    if key in PALETTE and not isinstance(PALETTE[key], Mapping):
        return _rgba_to_hex(cmap(key)(0.7))
    return "#7F7F7F"


# ── Per-rib colours (side-conditional) ───────────────────────────────────────

SEX_ORDER:   list[str] = ["Male", "Female"]
SMOKE_ORDER: list[str] = ["Never", "Ex-smoker", "Current"]

# Vert-level (T8..T19) and segmentation-label (40..51) both map onto the same
# 12-position range (anatomical ribs 1..12); ``rib_colors`` normalises whichever
# the caller passes. Display strings come from ``utils.rib_labels.display_from_*``.
_VERT_LO, _VERT_HI = 8, 19
_SEG_LO,  _SEG_HI  = 40, 51


def _normalise_level(lv: int) -> float:
    """Map a rib level (T8..T19 or seg-label 40..51) onto ``[0, 1]``."""
    if lv >= _SEG_LO:
        lo, hi = _SEG_LO, _SEG_HI
    else:
        lo, hi = _VERT_LO, _VERT_HI
    span = max(hi - lo, 1)
    t = (lv - lo) / span
    return float(min(max(t, 0.0), 1.0))


def _normalise_side(side: str) -> str:
    """Return ``"L"`` or ``"R"`` from any ``L/R/Left/Right`` (case-insensitive)."""
    s = str(side).strip().lower()
    if s in ("l", "left"):  return "L"
    if s in ("r", "right"): return "R"
    raise KeyError(f"colors.rib_colors(): unknown side {side!r}; expected L/R/Left/Right.")


def rib_colors(
    levels: Sequence[int],
    sides: Sequence[str],
) -> list[str]:
    """Per-rib hex colours, side-conditional.

    Left ribs sample ``PALETTE['rib_left']`` (teal), right ribs ``rib_right`` (red);
    position along each ramp is the rib level (T8 lightest, T19 darkest).

    Parameters
    ----------
    levels
        Rib level per output entry. Vertebral T8..T19 or seg-label 40..51 —
        the function infers which.
    sides
        Side per entry. Accepts ``L``/``R`` or ``Left``/``Right`` (case-insensitive).
    """
    levels = list(levels)
    sides  = list(sides)
    if len(levels) != len(sides):
        raise ValueError(
            f"colors.rib_colors(): len(levels)={len(levels)} but "
            f"len(sides)={len(sides)} — they must be parallel."
        )
    cm_left  = cmap("rib_left")
    cm_right = cmap("rib_right")
    out: list[str] = []
    for lv, side in zip(levels, sides):
        t  = _normalise_level(int(lv))
        cm = cm_left if _normalise_side(side) == "L" else cm_right
        out.append(_rgba_to_hex(cm(t)))
    return out


# ── Reference card ───────────────────────────────────────────────────────────

def preview() -> mpl.figure.Figure:
    """Render every palette entry as a labelled reference card (matplotlib)."""
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    items = list(PALETTE.items())
    n_rows = len(items)
    fig, ax = plt.subplots(figsize=(9, 0.5 * n_rows + 1))
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.6, n_rows - 0.4)
    ax.invert_yaxis()
    ax.axis("off")

    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    canonical_categorical = {"sex": _SEX, "smoking": _SMOKE,
                             "smoking_status": _SMOKE}

    for i, (name, entry) in enumerate(items):
        ax.text(0, i, name, va="center", fontsize=10, family="monospace")

        if isinstance(entry, Mapping):
            unique = list(canonical_categorical.get(name, entry).items())
            n = len(unique)
            width = 6.0 / n
            for j, (label, hex_) in enumerate(unique):
                x0 = 3 + j * width
                ax.add_patch(mpatches.Rectangle(
                    (x0, i - 0.3), width - 0.05, 0.6,
                    facecolor=hex_, edgecolor="none"))
                r, g, b = mcolors.to_rgb(hex_)
                luma = 0.299 * r + 0.587 * g + 0.114 * b
                txt = "white" if luma < 0.55 else "#222"
                ax.text(x0 + width / 2, i, str(label),
                        va="center", ha="center", fontsize=9, color=txt)
        else:
            ax.imshow(gradient, aspect="auto", cmap=entry,
                      extent=[3, 9, i - 0.3, i + 0.3])

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    preview()
    plt.show()
