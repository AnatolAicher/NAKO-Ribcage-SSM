"""Single export entry point for all figures.

``save_fig(fig, out_stem, formats=("html", "svg", "png"))`` writes a Plotly
figure to the requested formats next to ``out_stem`` (no extension on the
path). Raster export uses Kaleido at ``settings.KALEIDO_SCALE``.

For 3D figures requiring a print-quality render, pass ``static_renderer``:
it receives the destination path and writes the static-format export
(typically a matplotlib ``Poly3DCollection`` mosaic for headless safety).

Each call appends a row to ``figures_manifest.json`` (one per figure dir).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Iterable, Sequence

import plotly.graph_objects as go
import plotly.io as pio

import settings as S
from utils import figure_meta as M

logger = logging.getLogger(__name__)


# ── Format helpers ───────────────────────────────────────────────────────────

_RASTER = {"png", "jpg", "jpeg", "webp"}
_VECTOR = {"svg", "pdf", "eps"}
_HTML   = {"html"}
_VALID_FORMATS = _RASTER | _VECTOR | _HTML


def _validate_formats(formats: Sequence[str]) -> tuple[str, ...]:
    out = tuple(f.lower() for f in formats)
    bad = set(out) - _VALID_FORMATS
    if bad:
        raise ValueError(
            f"Unsupported export formats: {sorted(bad)}. "
            f"Valid: {sorted(_VALID_FORMATS)}"
        )
    return out


# ── Kaleido availability ─────────────────────────────────────────────────────

_KALEIDO_OK: bool | None = None


def _kaleido_available() -> bool:
    """Probe whether kaleido is importable; cache the result."""
    global _KALEIDO_OK
    if _KALEIDO_OK is None:
        try:
            import kaleido  # noqa: F401
            _KALEIDO_OK = True
        except ImportError as exc:
            logger.warning(
                "kaleido is not available (%s); SVG/PDF/PNG export will be skipped. "
                "`pip install kaleido` to enable static export.",
                exc,
            )
            _KALEIDO_OK = False
    return _KALEIDO_OK


# ── save_fig ─────────────────────────────────────────────────────────────────

def save_fig(
    fig: go.Figure,
    out_stem: str | Path,
    formats: Sequence[str] = S.EXPORT_FORMATS_DEFAULT,
    *,
    title: str | None = None,
    width_class: str | None = None,
    static_renderer: Callable[[Path], None] | None = None,
    skip_existing: bool = False,
    record_manifest: bool = True,
) -> dict:
    """Write ``fig`` to ``out_stem.{html,svg,pdf,png}`` for the requested formats.

    Parameters
    ----------
    fig
        A Plotly figure (already laid out via
        :func:`utils.plotly_theme.apply_layout`).
    out_stem
        Destination path **without** extension. Parent dir is created.
    formats
        Iterable from ``{"html", "svg", "pdf", "png"}``. HTML is interactive;
        the rest go through Kaleido.
    static_renderer
        Optional ``(path: Path) -> None`` callable; used as the raster/vector
        renderer for 3D figures (typically a matplotlib closure).
    skip_existing
        If True, skip exports whose file already exists with non-zero size.
    record_manifest
        If True (default), append an entry to ``figures_manifest.json``.

    Returns
    -------
    dict with ``stem``, ``formats``, ``files``, ``title``, ``width_class``.
    """
    out_stem = Path(out_stem)
    out_stem.parent.mkdir(parents=True, exist_ok=True)

    formats = _validate_formats(formats)
    written: dict[str, Path] = {}

    for ext in formats:
        out = out_stem.with_suffix(f".{ext}")
        if skip_existing and out.exists() and out.stat().st_size > 0:
            written[ext] = out
            continue

        if ext in _HTML:
            _write_html(fig, out)
        elif ext in _VECTOR or ext in _RASTER:
            ok = False
            if static_renderer is not None and ext in (_VECTOR | _RASTER):
                try:
                    static_renderer(out)
                    ok = out.exists() and out.stat().st_size > 0
                except Exception as exc:  # third-party renderers vary
                    logger.warning(
                        "Static renderer failed for %s (%s) — falling back to Plotly export.",
                        out.name, exc,
                    )
            if not ok:
                _write_static(fig, out, ext)

        written[ext] = out

    record = dict(
        stem=str(out_stem),
        formats=list(written.keys()),
        files={k: str(v) for k, v in written.items()},
        title=title,
        width_class=width_class,
    )
    if record_manifest:
        M.append_manifest(out_stem.parent, out_stem.name, record)
    logger.info("Saved figure: %s  (%s)", out_stem.name, ", ".join(written.keys()))
    return record


# ── HTML / static writers ────────────────────────────────────────────────────

def _write_html(fig: go.Figure, out: Path) -> None:
    """Self-contained interactive HTML, responsive to its container."""
    # Strip fixed pixel layout so the embed fills its iframe; static
    # (SVG/PNG) exports still see the original figure with its print sizing.
    layout_w = fig.layout.width
    layout_h = fig.layout.height
    fig_html = go.Figure(fig)
    fig_html.update_layout(autosize=True, width=None, height=None)
    html = pio.to_html(
        fig_html,
        include_plotlyjs="cdn",
        full_html=True,
        default_width="100%",
        default_height="100%",
        config=dict(
            displaylogo=False,
            modeBarButtonsToRemove=["lasso2d", "select2d"],
            toImageButtonOptions=dict(format="svg"),
            responsive=True,
        ),
    )
    # Portrait layouts (multi-row 3D mosaics) would otherwise be squashed to
    # viewport height — pin body height to viewport width × layout aspect so
    # panels keep their print aspect and the page scrolls vertically.
    if layout_w and layout_h and float(layout_h) > float(layout_w):
        aspect = float(layout_h) / float(layout_w)
        body_css = f"body{{width:100vw;height:calc(100vw * {aspect:.4f});}}"
    else:
        body_css = "body{height:100%;}"
    html = html.replace(
        "<head>",
        f"<head><style>html,body{{margin:0;padding:0;}}html{{height:100%;}}{body_css}</style>",
        1,
    )
    out.write_text(html, encoding="utf-8")


def _write_static(fig: go.Figure, out: Path, ext: str) -> None:
    """Vector / raster export via Kaleido."""
    if not _kaleido_available():
        # Placeholder so the missing path is visible without crashing.
        out.write_bytes(b"")
        return

    # Vector: scale=1 keeps font metrics correct (size unaffected).
    scale = S.KALEIDO_SCALE if ext in _RASTER else 1.0

    pio.write_image(
        fig,
        file=str(out),
        format=ext,
        scale=scale,
        validate=True,
    )
