"""Cross-cutting tests for figure-level consistency.

Verifies that figure-shaping is governed by ``settings`` and that the
``apply_layout`` / ``save_fig`` plumbing produces:

  * canonical pixel widths matching ``FIG_WIDTH_FULL_MM`` /
    ``FIG_WIDTH_HALF_MM`` / ``FIG_WIDTH_THIRD_MM``;
  * the ``"nako"`` Plotly template installed and used as default;
  * the publication font family on the layout;
  * an HTML file written by ``save_fig`` (kaleido is not assumed
    available in CI; SVG / PNG are skipped if missing).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import pytest

import settings as S
from utils import figure_meta, figure_export
from utils.plotly_theme import apply_layout, install_template, mm_to_px


@pytest.fixture(autouse=True)
def _install_template():
    install_template()
    yield


def _make_fig() -> go.Figure:
    fig = go.Figure(go.Scatter(x=[1, 2, 3], y=[3, 1, 2], mode="lines+markers"))
    return fig


def test_template_installed_and_default():
    assert "nako" in pio.templates
    assert pio.templates.default == "nako"
    tmpl = pio.templates["nako"]
    # Font family flows from settings.
    assert tmpl.layout.font.family == S.FONT_FAMILY


def test_apply_layout_widths_match_settings():
    for cls, mm in (("full", S.FIG_WIDTH_FULL_MM),
                    ("half", S.FIG_WIDTH_HALF_MM),
                    ("third", S.FIG_WIDTH_THIRD_MM)):
        fig = _make_fig()
        apply_layout(fig, width_class=cls)
        assert fig.layout.width == mm_to_px(mm), (
            f"width_class={cls} expected {mm_to_px(mm)} px, got {fig.layout.width}"
        )


def test_apply_layout_rejects_unknown_width():
    fig = _make_fig()
    with pytest.raises(ValueError):
        apply_layout(fig, width_class="enormous")


def test_apply_layout_height_default_scales_with_rows():
    fig = _make_fig()
    apply_layout(fig, width_class="full", n_rows=3)
    assert fig.layout.height == mm_to_px(3 * S.FIG_ROW_HEIGHT_MM)


def test_apply_layout_height_override():
    fig = _make_fig()
    apply_layout(fig, width_class="full", height_mm=42.0)
    assert fig.layout.height == mm_to_px(42.0)


def test_save_fig_writes_html(tmp_path: Path):
    fig = _make_fig()
    apply_layout(fig, width_class="full")
    out_stem = tmp_path / "fig"
    figure_export.save_fig(fig, out_stem, formats=("html",))
    assert (tmp_path / "fig.html").exists()


def test_save_fig_appends_manifest(tmp_path: Path):
    fig = _make_fig()
    apply_layout(fig, width_class="half")
    figure_export.save_fig(fig, tmp_path / "fig_a", formats=("html",),
                           title="A", width_class="half")
    figure_export.save_fig(fig, tmp_path / "fig_b", formats=("html",),
                           title="B", width_class="half")
    manifest = figure_meta.read_manifest(tmp_path)
    assert "fig_a" in manifest["figures"]
    assert "fig_b" in manifest["figures"]
    assert manifest["palette_version"] == S.PALETTE_VERSION
    assert manifest["schema"] == "nako-figures-manifest/2"
    rec = manifest["figures"]["fig_a"]
    assert "stem" not in rec and "files" not in rec
    assert {"formats", "title", "width_class",
            "git_rev", "palette_version", "written_at_utc"} <= rec.keys()


def test_save_fig_rejects_bad_format(tmp_path: Path):
    fig = _make_fig()
    apply_layout(fig, width_class="full")
    with pytest.raises(ValueError):
        figure_export.save_fig(fig, tmp_path / "bad", formats=("xyz",))
