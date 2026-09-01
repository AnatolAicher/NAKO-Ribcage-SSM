"""Single export entry point for Altair charts.

``save_chart(chart, out_stem, formats=("html", "svg", "png"))`` writes an
Altair chart to the requested formats next to ``out_stem`` (no extension on
the path). Raster export uses vl-convert at ``settings.KALEIDO_SCALE``
(``EXPORT_RASTER_DPI / SCREEN_DPI``).

Each call appends a row to ``figures_manifest.json`` (one per figure dir),
sharing the schema with :mod:`utils.figure_export`. Unlike the Plotly path,
vl-convert failures **raise** rather than writing a zero-byte placeholder.

Static (svg/pdf/png) renders run in a short-lived worker subprocess capped
at ``_TASKS_PER_CHILD`` renders: vl-convert keeps a single V8 isolate per
process, and over a long batch (rerender_figures.py) the V8 old-space heap
accumulates and OOMs near the ~1.4 GB default. The pool recycles the
worker before its isolate fills.
"""
from __future__ import annotations

import atexit
import json
import logging
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from typing import Sequence

import altair as alt

import settings as S
from utils import figure_meta as M

logger = logging.getLogger(__name__)


# ── vl-convert worker pool ───────────────────────────────────────────────────

# One render per worker: vl-convert's V8 isolate accumulates heap and OOMs
# at ~1.4 GB across multi-chart batches. Each render gets a fresh isolate.
_TASKS_PER_CHILD = 1
_pool: ProcessPoolExecutor | None = None


def _get_pool() -> ProcessPoolExecutor:
    global _pool
    if _pool is None:
        _pool = ProcessPoolExecutor(
            max_workers=1, max_tasks_per_child=_TASKS_PER_CHILD,
        )
        atexit.register(_pool.shutdown, wait=True)
    return _pool


def _reset_pool() -> None:
    global _pool
    if _pool is not None:
        _pool.shutdown(wait=False, cancel_futures=True)
        _pool = None


def _render_static_in_worker(spec: dict, ext: str, scale: float) -> bytes | str:
    # SVG is rendered by V8 (Vega → SVG). Raster/PDF then go through resvg
    # in pure Rust via svg_to_*, so V8 never has to allocate the pixel
    # buffer – that's what OOMs `vegalite_to_png` at high DPI for big
    # SPLOMs even when the spec itself is small.
    import vl_convert as vlc
    svg = vlc.vegalite_to_svg(spec)
    if ext == "svg":
        return svg
    if ext == "pdf":
        return vlc.svg_to_pdf(svg)
    if ext in ("jpg", "jpeg"):
        return vlc.svg_to_jpeg(svg, scale=scale)
    return vlc.svg_to_png(svg, scale=scale)


# ── Format helpers ───────────────────────────────────────────────────────────

_RASTER = {"png", "jpg", "jpeg", "webp"}
_VECTOR = {"svg", "pdf"}
_HTML   = {"html"}
_VALID_FORMATS = _RASTER | _VECTOR | _HTML

# vega-embed menu: keep the PNG/SVG export entry, drop the source / compiled-spec
# / Vega-editor links. scaleFactor matches the project raster DPI so a reader's
# in-browser "Save as PNG" matches the static export.
_EMBED_OPTIONS = {
    "actions": {"export": True, "source": False, "compiled": False, "editor": False},
    "scaleFactor": S.KALEIDO_SCALE,
}


def _validate_formats(formats: Sequence[str]) -> tuple[str, ...]:
    out = tuple(f.lower() for f in formats)
    bad = set(out) - _VALID_FORMATS
    if bad:
        raise ValueError(
            f"Unsupported export formats: {sorted(bad)}. "
            f"Valid: {sorted(_VALID_FORMATS)}"
        )
    return out


# ── save_chart ───────────────────────────────────────────────────────────────

def save_chart(
    chart: alt.TopLevelMixin,
    out_stem: str | Path,
    formats: Sequence[str] = S.EXPORT_FORMATS_DEFAULT,
    *,
    title: str | None = None,
    width_class: str | None = None,
    skip_existing: bool = False,
    record_manifest: bool = True,
) -> dict:
    """Write ``chart`` to ``out_stem.{html,svg,pdf,png}`` for the requested formats.

    Parameters
    ----------
    chart
        An Altair chart (any subclass of ``alt.TopLevelMixin``).
    out_stem
        Destination path **without** extension. Parent dir is created.
    formats
        Iterable from ``{"html", "svg", "pdf", "png"}``.
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
    spec: dict | None = None
    written: dict[str, Path] = {}

    for ext in formats:
        out = out_stem.with_suffix(f".{ext}")
        if skip_existing and out.exists() and out.stat().st_size > 0:
            written[ext] = out
            continue

        if ext == "html":
            # inline=True bundles vega/vega-lite/vega-embed into the file so it
            # renders offline (no CDN fetch at view time).
            chart.save(str(out), format="html", inline=True,
                       embed_options=_EMBED_OPTIONS)
        else:
            if spec is None:
                spec = chart.to_dict()
            _write_static(spec, out, ext)

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
    logger.info("Saved chart: %s  (%s)", out_stem.name, ", ".join(written.keys()))
    return record


# ── PNG / HTML split export ──────────────────────────────────────────────────

def save_chart_split(
    html_chart: alt.TopLevelMixin,
    static_chart: alt.TopLevelMixin,
    out_stem: str | Path,
    *,
    html_formats: Sequence[str] = ("html",),
    static_formats: Sequence[str] = ("svg", "png"),
    title: str | None = None,
    width_class: str | None = None,
    skip_existing: bool = False,
    record_manifest: bool = True,
) -> dict:
    """Save two specs to the same ``out_stem`` – one for HTML, one for raster/vector.

    Multi-PC charts use this to keep all 113 PCs in interactive HTML while the
    static PNG/SVG only carries the leading ``N_PCS_DISPLAY`` so axis labels
    stay legible at print scale.
    """
    out_stem = Path(out_stem)
    out_stem.parent.mkdir(parents=True, exist_ok=True)

    rec_html = save_chart(
        html_chart, out_stem, formats=html_formats, title=title,
        width_class=width_class, skip_existing=skip_existing,
        record_manifest=False,
    )
    rec_static = save_chart(
        static_chart, out_stem, formats=static_formats, title=title,
        width_class=width_class, skip_existing=skip_existing,
        record_manifest=False,
    )
    written = {**rec_html["files"], **rec_static["files"]}
    record = dict(
        stem=str(out_stem),
        formats=list(written.keys()),
        files=written,
        title=title,
        width_class=width_class,
    )
    if record_manifest:
        M.append_manifest(out_stem.parent, out_stem.name, record)
    return record


# ── Static writers ───────────────────────────────────────────────────────────

def _dump_failing_spec(spec: dict, out: Path, exc: BaseException) -> Path:
    dump_path = out.with_suffix(".failed.json")
    try:
        dump_path.write_text(
            json.dumps(spec, indent=2, default=str), encoding="utf-8"
        )
    except OSError:
        return out
    logger.error(
        "vl-convert failed for %s.%s (%s) – spec dumped to %s",
        out.stem, out.suffix.lstrip("."), exc.__class__.__name__, dump_path,
    )
    return dump_path


def _write_static(spec: dict, out: Path, ext: str) -> None:
    try:
        result = _get_pool().submit(
            _render_static_in_worker, spec, ext, S.KALEIDO_SCALE,
        ).result()
    except BrokenProcessPool:
        logger.warning(
            "vl-convert worker crashed rendering %s.%s; resetting pool and "
            "retrying once.", out.stem, ext,
        )
        _reset_pool()
        try:
            result = _get_pool().submit(
                _render_static_in_worker, spec, ext, S.KALEIDO_SCALE,
            ).result()
        except BrokenProcessPool:
            # Second crash on the same chart → genuinely too large for V8.
            # Reset so the next (different) chart starts clean.
            _reset_pool()
            _dump_failing_spec(spec, out, BrokenProcessPool("twice"))
            raise
    except ValueError as exc:
        # vl-convert raises ValueError for Vega-Lite compile failures. Dump the
        # spec so the underlying null/type mismatch can be inspected offline.
        _dump_failing_spec(spec, out, exc)
        raise
    if ext == "svg":
        out.write_text(result, encoding="utf-8")
    elif ext == "pdf" or ext in _RASTER:
        out.write_bytes(result)
    else:  # pragma: no cover – guarded by _validate_formats
        raise ValueError(f"Unsupported static format: {ext}")
