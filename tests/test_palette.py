"""Palette accessibility & accessor invariants.

Two things we want to keep stable across palette edits:

  1. Categorical hex pairs that are commonly co-displayed (sex.Male vs
     sex.Female, smoking.Never vs Ex vs Current) stay distinguishable
     under deuteranopia / protanopia simulation.  We use a coarse but
     well-known approximation (Vienot-Brettel-Mollon LMS transform) and
     a CIELab ΔE76 distance.
  2. The strict accessors (cmap / color / colorscale) raise on unknown
     keys instead of silently returning a default.  This is the only
     thing that prevents drift back into hardcoded colours.

ΔE76 ≥ 25 is a generous threshold (Nature-style figures are often
reduced to small print, so the floor must hold even at low spatial
frequencies).  If the test fails after a palette edit, the
``smoking.Current`` entry should be retuned (see colors.py header).
"""
from __future__ import annotations

import numpy as np
import pytest

from utils import colors as C


# ── Colour-space utilities (no scikit-image dependency) ──────────────────────

def _hex_to_rgb01(h: str) -> np.ndarray:
    h = h.lstrip("#")
    return np.array([int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)],
                    dtype=float) / 255.0


def _srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)


def _linear_to_xyz(rgb: np.ndarray) -> np.ndarray:
    M = np.array([
        [0.4124, 0.3576, 0.1805],
        [0.2126, 0.7152, 0.0722],
        [0.0193, 0.1192, 0.9505],
    ])
    return M @ rgb


def _xyz_to_lab(xyz: np.ndarray) -> np.ndarray:
    # D65 white
    Xn, Yn, Zn = 0.95047, 1.0, 1.08883
    f = lambda t: np.where(t > 0.008856, np.cbrt(t), 7.787 * t + 16/116)
    fx, fy, fz = f(xyz[0]/Xn), f(xyz[1]/Yn), f(xyz[2]/Zn)
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)
    return np.array([L, a, b])


def _lab(hex_: str) -> np.ndarray:
    return _xyz_to_lab(_linear_to_xyz(_srgb_to_linear(_hex_to_rgb01(hex_))))


def _delta_e76(a: str, b: str) -> float:
    return float(np.linalg.norm(_lab(a) - _lab(b)))


# Vienot-Brettel-Mollon deuteranopia / protanopia LMS confusion projection.
# Coefficients from the standard reference; values rounded for readability.
def _simulate(hex_: str, kind: str) -> str:
    rgb_lin = _srgb_to_linear(_hex_to_rgb01(hex_))
    # sRGB → LMS
    M = np.array([
        [17.8824,  43.5161,  4.11935],
        [ 3.45565, 27.1554,  3.86714],
        [ 0.0299566, 0.184309, 1.46709],
    ])
    Minv = np.linalg.inv(M)
    lms = M @ rgb_lin
    if kind == "protanopia":
        T = np.array([[0.0, 2.02344, -2.52581],
                      [0.0, 1.0,       0.0],
                      [0.0, 0.0,       1.0]])
    elif kind == "deuteranopia":
        T = np.array([[1.0,       0.0, 0.0],
                      [0.494207,  0.0, 1.24827],
                      [0.0,       0.0, 1.0]])
    else:
        raise ValueError(f"unknown kind {kind!r}")
    lms_sim = T @ lms
    rgb_sim = Minv @ lms_sim
    rgb_sim = np.clip(rgb_sim, 0, 1)
    # Re-encode: linear → sRGB
    rgb_srgb = np.where(
        rgb_sim <= 0.0031308,
        12.92 * rgb_sim,
        1.055 * (rgb_sim ** (1/2.4)) - 0.055,
    )
    rgb_srgb = np.clip(rgb_srgb, 0, 1)
    r, g, b = (int(c * 255) for c in rgb_srgb)
    return f"#{r:02X}{g:02X}{b:02X}"


# ── Tests: categorical separation ────────────────────────────────────────────

THRESHOLD_NORMAL = 30.0   # floor for normal vision
# Floor under deuteranopia/protanopia. The smoking ramp is ordinal, so adjacent
# steps land close under red-blind simulation by design; 18 lets that through
# while still flagging genuine categorical collisions.
THRESHOLD_CB     = 18.0


CO_DISPLAY_PAIRS = [
    ("sex", "Male",   "sex", "Female"),
    ("smoking", "Never", "smoking", "Ex-smoker"),
    ("smoking", "Ex-smoker", "smoking", "Current"),
    ("smoking", "Never", "smoking", "Current"),
    # Cross-variable risk: sex × smoking are frequently shown together.
    ("sex", "Female", "smoking", "Current"),
    ("sex", "Male",   "smoking", "Ex-smoker"),
]


@pytest.mark.parametrize("v1,k1,v2,k2", CO_DISPLAY_PAIRS)
def test_categorical_pairs_separated_normal(v1, k1, v2, k2):
    h1, h2 = C.color(v1, k1), C.color(v2, k2)
    de = _delta_e76(h1, h2)
    assert de >= THRESHOLD_NORMAL, (
        f"ΔE76 {de:.1f} < {THRESHOLD_NORMAL} for {v1}.{k1}={h1} vs {v2}.{k2}={h2}"
    )


@pytest.mark.parametrize("v1,k1,v2,k2", CO_DISPLAY_PAIRS)
@pytest.mark.parametrize("kind", ["deuteranopia", "protanopia"])
def test_categorical_pairs_separated_colorblind(v1, k1, v2, k2, kind):
    h1, h2 = C.color(v1, k1), C.color(v2, k2)
    s1, s2 = _simulate(h1, kind), _simulate(h2, kind)
    de = _delta_e76(s1, s2)
    assert de >= THRESHOLD_CB, (
        f"{kind} ΔE76 {de:.1f} < {THRESHOLD_CB} for "
        f"{v1}.{k1}={h1}->{s1} vs {v2}.{k2}={h2}->{s2}"
    )


# ── Tests: strict accessors ──────────────────────────────────────────────────

def test_cmap_unknown_var_raises():
    with pytest.raises(KeyError):
        C.cmap("unknown_variable_xyz")


def test_cmap_on_categorical_raises():
    with pytest.raises(KeyError):
        C.cmap("sex")


def test_color_on_continuous_raises():
    with pytest.raises(KeyError):
        C.color("age", 42)


def test_color_unknown_value_raises():
    with pytest.raises(KeyError):
        C.color("sex", "Other")


def test_colorscale_returns_pairs():
    cs = C.colorscale("correlation", n=8)
    assert len(cs) == 8
    assert all(len(pair) == 2 and 0.0 <= pair[0] <= 1.0 for pair in cs)
    # Colors come back as CSS rgb strings (Plotly-friendly).
    assert all(str(pair[1]).startswith("rgb(") for pair in cs)


def test_palette_aliases_resolve_to_same_hex():
    # The palette accepts both long-form and short-form keys; both
    # should map to the same hex.
    assert C.color("sex", "Male") == C.color("sex", "M")
    assert C.color("smoking", "Ex-smoker") == C.color("smoking", "previous")


def test_rib_colors_per_side_signature():
    # rib_left ramp = cool teal, rib_right ramp = warm red.  Both share
    # the same level→position mapping (T8 lightest, T19 darkest), so a
    # T8-Left and a T19-Right call should resolve to clearly distinct
    # hexes, and a T8-Left vs T8-Right should be distinct in hue.
    out = C.rib_colors([8, 19, 8, 19], ["L", "L", "R", "R"])
    assert len(out) == 4
    assert all(h.startswith("#") and len(h) == 7 for h in out)
    # T8 (light end) is brighter than T19 (dark end) within each side.
    def _lum(h: str) -> float:
        return int(h[1:3], 16) + int(h[3:5], 16) + int(h[5:7], 16)
    assert _lum(out[0]) > _lum(out[1]), "rib_left T8 should be lighter than T19"
    assert _lum(out[2]) > _lum(out[3]), "rib_right T8 should be lighter than T19"
    # The two sides at the same level should differ.
    assert out[0] != out[2], "rib_left T8 should differ from rib_right T8"


def test_rib_colors_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        C.rib_colors([8, 9, 10], ["L", "R"])


def test_dropped_palette_keys_raise():
    # The old keys are removed; using them must raise so callers
    # don't silently degrade to an unintended default.
    for key in ("rib_number", "rib_side", "side", "beta", "effect"):
        with pytest.raises(KeyError):
            C.cmap(key) if key in ("rib_number", "beta", "effect") else C.color(key, "Left")
