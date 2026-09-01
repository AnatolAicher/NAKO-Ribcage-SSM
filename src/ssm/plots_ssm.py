"""3D shape figures for the surface-based SSM (Plotly Mesh3d HTML + matplotlib print).

The 2D PC and regression figures live in the Altair module
:mod:`ssm.plots_ssm_altair`.

Functions
---------
plot_pc_deformations            ±n_sd shape modes (Plotly Mesh3d HTML +
                                matplotlib Poly3DCollection for print).
plot_mean_shape_gallery         12-level × L/R mean-shape thumbnails.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import settings as S
from utils import colors as C
from utils.figure_export import save_fig
from utils.plotly_theme import apply_layout, grid_spacing, place_colorbar_right

logger = logging.getLogger(__name__)


def _strip_format_suffix(p: Path) -> Path:
    return p.with_suffix("") if p.suffix in {".png", ".svg", ".pdf", ".html"} else p


# ── PC deformation modes – dual-render (Plotly Mesh3d + PyVista print) ──────

def _build_pc_deformation_figure(
    mean_shape: np.ndarray,
    pca,
    template_faces: np.ndarray,
    n_pcs: int,
    n_sd: float,
) -> go.Figure:
    """Build one Plotly figure with N_PCs × 2 (±SD) Mesh3d subplots."""
    n_pts = mean_shape.shape[0]
    sds   = np.sqrt(np.asarray(pca.explained_variance_, dtype=float))
    n_pcs_actual = min(n_pcs, pca.n_components_)

    titles = []
    for k in range(n_pcs_actual):
        evr = pca.explained_variance_ratio_[k]
        titles.extend([f"PC{k+1}  −{n_sd:.0f} SD  ({evr:.1%})",
                       f"PC{k+1}  +{n_sd:.0f} SD"])

    specs = [[{"type": "scene"}, {"type": "scene"}] for _ in range(n_pcs_actual)]
    # 2 % vertical gap is tight enough that the per-PC rows aren't visually
    # split by whitespace. 4 % left noticeable bands that pushed the title
    # off the visible area when n_pcs_actual was large.
    fig = make_subplots(
        rows=n_pcs_actual, cols=2,
        specs=specs,
        subplot_titles=titles,
        horizontal_spacing=0.0,
        vertical_spacing=0.02,
    )

    cs = C.colorscale("displacement")
    cmin, cmax = 0.0, None  # set per trace from displacement max

    for k in range(n_pcs_actual):
        component = pca.components_[k].reshape(n_pts, 3)

        for col_idx, sign in enumerate([-1, +1], start=1):
            deformed = mean_shape + sign * n_sd * sds[k] * component
            disp     = np.linalg.norm(deformed - mean_shape, axis=1)
            cmax_val = float(disp.max()) if disp.size else 1.0

            fig.add_trace(
                go.Mesh3d(
                    x=deformed[:, 0], y=deformed[:, 1], z=deformed[:, 2],
                    i=template_faces[:, 0],
                    j=template_faces[:, 1],
                    k=template_faces[:, 2],
                    intensity=disp,
                    colorscale=cs,
                    cmin=0.0, cmax=cmax_val,
                    showscale=(col_idx == 2 and k == 0),  # one shared colourbar
                    colorbar=(
                        place_colorbar_right(title="|Δ| (mm)", length_fraction=0.6)
                        if (col_idx == 2 and k == 0) else None
                    ),
                    flatshading=False,
                    lighting=dict(
                        ambient=0.45, diffuse=0.7, specular=0.15,
                        roughness=0.85, fresnel=0.2,
                    ),
                    lightposition=dict(x=400, y=200, z=300),
                    name=f"PC{k+1} {'+' if sign > 0 else '−'}",
                    hovertemplate="|Δ|=%{intensity:.2f} mm<extra></extra>",
                ),
                row=k + 1, col=col_idx,
            )
        # Equal-aspect 3D scenes; identical for both panels of one PC.
        for col_idx in (1, 2):
            scene_id = f"scene{(k * 2 + col_idx) if (k * 2 + col_idx) > 1 else ''}"
            fig.layout[scene_id].update(
                aspectmode="data",
                xaxis=dict(visible=False), yaxis=dict(visible=False),
                zaxis=dict(visible=False),
            )

    return fig


def _mpl_pc_deformations_renderer(
    *,
    mean_shape: np.ndarray,
    pca,
    template_faces: np.ndarray,
    n_pcs: int,
    n_sd: float,
) -> callable:
    """Return a closure that renders the PC-deformation mosaic via matplotlib.

    The closure takes a single ``path`` argument (so it slots into
    :func:`utils.figure_export.save_fig`'s ``static_renderer`` slot)
    and writes a single composite PNG. Headless-safe: no PyVista / VTK
    rendering, so it runs in any container without OSMesa / EGL / X.
    """
    import matplotlib.pyplot as plt  # local – lazy
    from utils.mesh_mpl import add_mesh, set_cube_bounds, style_axis

    def _render(path: Path) -> None:
        n_pts = mean_shape.shape[0]
        sds   = np.sqrt(np.asarray(pca.explained_variance_, dtype=float))
        n_pcs_actual = min(n_pcs, pca.n_components_)
        faces = np.asarray(template_faces)

        # Taller per row + slightly wider so the per-panel "PC N ±k SD" titles
        # and the 3D scenes aren't squeezed.
        fig = plt.figure(figsize=(9.0, 3.6 * n_pcs_actual))
        ax_label_pairs: list[tuple] = []  # (ax, label) for post-layout fig.text
        for k in range(n_pcs_actual):
            component = pca.components_[k].reshape(n_pts, 3)
            evr = pca.explained_variance_ratio_[k]
            for col_idx, sign in enumerate([-1, +1]):
                deformed = mean_shape + sign * n_sd * sds[k] * component
                disp     = np.linalg.norm(deformed - mean_shape, axis=1)

                ax = fig.add_subplot(
                    n_pcs_actual, 2, k * 2 + col_idx + 1,
                    projection="3d",
                )
                add_mesh(
                    ax, deformed, faces,
                    scalars=disp, cmap=C.cmap("displacement"),
                    clim=(0.0, max(float(disp.max()), 1e-6)),
                )
                # Loosened bounds (1.25 vs 1.05) so the rib tips aren't cropped.
                set_cube_bounds(ax, [deformed], padding=1.25)
                style_axis(ax)
                ax_label_pairs.append((
                    ax,
                    f"PC{k+1}  {'+' if sign > 0 else '−'}{n_sd:.0f} SD  ({evr:.1%})",
                ))
        fig.suptitle("Surface SSM – leading PC deformations",
                     fontsize=S.FONT_SIZE_TITLE_PT + 3, y=0.99)
        # Reserve 3 % at the top for the suptitle so it lives inside the
        # figure frame instead of floating outside.
        fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.97], h_pad=0.4, w_pad=0.4)
        # Per-panel labels at figure coords ABOVE each subplot – ax.set_title
        # on a 3D axes can render inside the cube volume and become illegible.
        for ax, label in ax_label_pairs:
            bbox = ax.get_position()
            fig.text(
                (bbox.x0 + bbox.x1) / 2,
                bbox.y1 + 0.005,
                label,
                transform=fig.transFigure, ha="center", va="bottom",
                fontsize=S.FONT_SIZE_TITLE_PT + 2,
            )
        fig.savefig(path, dpi=S.EXPORT_RASTER_DPI)
        plt.close(fig)

    return _render


def _mpl_mean_shape_gallery_renderer(
    *,
    mean_shape: np.ndarray,
    faces: np.ndarray,
    rib_offsets: list[int],
    rib_ids: list[str],
    rib_colours: list[str],
) -> callable:
    """Return a closure that renders the per-rib mean-shape gallery via matplotlib."""
    import matplotlib.pyplot as plt  # local – lazy
    from utils.mesh_mpl import add_mesh, set_cube_bounds, style_axis

    def _render(path: Path) -> None:
        n_ribs = len(rib_offsets) - 1
        ncols = 4
        nrows = (n_ribs + ncols - 1) // ncols

        ends = np.asarray(rib_offsets[1:], dtype=np.int64)
        faces_i64 = np.asarray(faces, dtype=np.int64)
        v0 = np.searchsorted(ends, faces_i64[:, 0], side="right")
        v1 = np.searchsorted(ends, faces_i64[:, 1], side="right")
        v2 = np.searchsorted(ends, faces_i64[:, 2], side="right")
        face_rib = np.where((v0 == v1) & (v1 == v2), v0, -1)

        fig = plt.figure(figsize=(8.0, 2.0 * nrows))
        for ri in range(n_ribs):
            s, e = rib_offsets[ri], rib_offsets[ri + 1]
            verts = mean_shape[s:e]
            f_sub = faces_i64[face_rib == ri] - s
            ax = fig.add_subplot(nrows, ncols, ri + 1, projection="3d")
            if verts.size and f_sub.size:
                add_mesh(ax, verts, f_sub, color=rib_colours[ri])
                set_cube_bounds(ax, [verts])
            style_axis(ax)
            ax.set_title(rib_ids[ri], fontsize=S.FONT_SIZE_TITLE_PT)
        fig.suptitle(f"GPA mean shape – per-rib gallery ({n_ribs} ribs)",
                     fontsize=S.FONT_SIZE_TITLE_PT + 1)
        fig.tight_layout()
        fig.savefig(path, dpi=S.EXPORT_RASTER_DPI)
        plt.close(fig)

    return _render


def _static_3d_renderer(kind: str, **kw):
    """Dispatcher: PyVista offscreen first, matplotlib fallback under ``auto``.

    ``kind`` is ``"pc_deformations"`` or ``"mean_shape_gallery"``. All other
    args (mesh data, PCA, etc.) are forwarded as keyword args to the backend.
    """
    backend = S.STATIC_3D_BACKEND

    def _render(path: Path) -> None:
        if backend in ("auto", "pyvista"):
            try:
                from utils import mesh_pv
                if kind == "pc_deformations":
                    mesh_pv.render_pc_deformations(path=path, **kw)
                else:
                    mesh_pv.render_mean_shape_gallery(path=path, **kw)
                return
            except Exception as exc:  # why-broad: VTK/EGL/OSMesa raise varies
                if backend == "pyvista":
                    raise
                logger.warning(
                    "PyVista offscreen render failed (%s); falling back to matplotlib.",
                    exc,
                )
        if kind == "pc_deformations":
            _mpl_pc_deformations_renderer(**kw)(path)
        else:
            _mpl_mean_shape_gallery_renderer(**kw)(path)

    return _render


def plot_pc_deformations(
    mean_shape: np.ndarray,
    pca,
    template_faces: np.ndarray,
    out_stem: str | Path,
    *,
    n_pcs: int = 5,
    n_sd: float = 2.0,
    formats: tuple[str, ...] = ("html", "png"),
) -> None:
    """Dual-render leading PC deformation modes.

    Plotly Mesh3d HTML for interactive exploration (one composite figure
    with ``n_pcs × 2`` panels) plus a static raster mosaic for print.
    Writes ``<out_stem>.{html,png}`` directly – no subdirectory.
    """
    out_stem = Path(out_stem)
    out_stem.parent.mkdir(parents=True, exist_ok=True)

    fig = _build_pc_deformation_figure(mean_shape, pca, template_faces, n_pcs, n_sd)
    n_pcs_actual = min(n_pcs, pca.n_components_)
    apply_layout(
        fig, width_class="full",
        # 85 mm/PC: each row matches the half-width of one of the two columns,
        # so panels are roughly square at full figure width.
        height_mm=max(85.0 * n_pcs_actual, 80.0),
        title=f"Surface SSM – leading {n_pcs_actual} PC deformation modes",
    )
    # Detail moves under the title as an annotation so the title itself fits
    # on a single line at the chart's full width.
    fig.add_annotation(
        text=f"±{n_sd:.0f} SD · vertex colour = displacement magnitude (mm)",
        xref="paper", yref="paper",
        x=0.5, y=1.0, xanchor="center", yanchor="bottom",
        showarrow=False, font=dict(size=10, color="#555"),
        yshift=2,
    )

    needs_static = any(f in formats for f in ("svg", "pdf", "png", "jpg", "jpeg"))
    renderer = (
        _static_3d_renderer(
            "pc_deformations",
            mean_shape=mean_shape, pca=pca, template_faces=template_faces,
            n_pcs=n_pcs, n_sd=n_sd,
        )
        if needs_static else None
    )
    save_fig(
        fig, out_stem,
        formats=formats,
        static_renderer=renderer,
        title="PC deformation modes (Plotly HTML + static raster print)",
        width_class="full",
    )


# ── Mean-shape per-rib gallery ───────────────────────────────────────────────

def plot_mean_shape_gallery(
    mean_shape: np.ndarray,
    template_faces: np.ndarray,
    rib_offsets: np.ndarray,
    rib_ids: list[str] | None,
    out_stem: Path,
) -> None:
    """One thumbnail per rib of the GPA mean shape (Plotly Mesh3d HTML).

    The interactive HTML is the primary product; PyVista isn't used
    here because each thumbnail is small enough that Mesh3d quality is
    sufficient at the default lighting.
    """
    out_stem = _strip_format_suffix(Path(out_stem))
    rib_offsets = list(rib_offsets) + [mean_shape.shape[0]]
    n_ribs = len(rib_offsets) - 1
    if rib_ids is None or len(rib_ids) < n_ribs:
        rib_ids = [f"Rib {i+1}" for i in range(n_ribs)]
    rib_ids = list(rib_ids)[:n_ribs]

    ncols = 4
    nrows = (n_ribs + ncols - 1) // ncols
    specs = [[{"type": "scene"}] * ncols for _ in range(nrows)]
    fig = make_subplots(
        rows=nrows, cols=ncols,
        specs=specs,
        subplot_titles=rib_ids,
        horizontal_spacing=0.02,
        vertical_spacing=grid_spacing(nrows, row_height_mm=40.0, gap_mm=4.0),
    )

    # Per-rib colours, side-conditional. Accepts either display form
    # ("Rib 7 L") or seg-label form ("rib40_L"); returns a seg-label int.
    def _parse(rib_id: str) -> tuple[int, str]:
        try:
            if rib_id.startswith("Rib "):
                _, n_str, side = rib_id.split()
                return int(n_str) + 39, side  # anatomical → seg label
            head, side = rib_id.rsplit("_", 1)
            lab = int(head.removeprefix("rib"))
            return lab, side
        except (ValueError, AttributeError):
            return 0, "L"

    parsed = [_parse(rid) for rid in rib_ids]
    rib_colours = C.rib_colors([lv for lv, _ in parsed],
                               [sd for _, sd in parsed])

    # Assign each face to a rib by looking up its first vertex's index in
    # the offsets table.  ``np.searchsorted(offsets[1:], v, side='right')``
    # returns the rib k such that ``offsets[k] <= v < offsets[k+1]``.  We
    # then verify all three vertices land in the same rib – if a face
    # straddles a rib boundary it is logged as `cross-rib` and dropped.
    # This is more robust than the naive ``min ≥ s & max < e`` test,
    # which silently drops a whole rib's faces if any single index in
    # ``template_faces.npy`` is off-by-one.
    faces  = template_faces.astype(np.int64)
    ends   = np.asarray(rib_offsets[1:], dtype=np.int64)
    v0_rib = np.searchsorted(ends, faces[:, 0], side="right")
    v1_rib = np.searchsorted(ends, faces[:, 1], side="right")
    v2_rib = np.searchsorted(ends, faces[:, 2], side="right")
    same_rib = (v0_rib == v1_rib) & (v1_rib == v2_rib)
    n_cross  = int((~same_rib).sum())
    if n_cross:
        logger.warning(
            "plot_mean_shape_gallery: %d/%d faces span a rib boundary and "
            "will be dropped (template_faces.npy / rib_offsets.npy may be "
            "out of sync).", n_cross, len(faces),
        )
    face_rib = np.where(same_rib, v0_rib, -1)

    # Per-rib diagnostic – prints face / vertex counts so empty ribs are
    # visible.  Saved alongside the gallery for later inspection.
    diag_rows = []
    for ri in range(n_ribs):
        s, e = rib_offsets[ri], rib_offsets[ri + 1]
        n_v = int(e - s)
        n_f = int((face_rib == ri).sum())
        diag_rows.append((rib_ids[ri], n_v, n_f))
    n_skipped = sum(1 for _, nv, nf in diag_rows if nv == 0 or nf == 0)
    if n_skipped:
        logger.warning(
            "plot_mean_shape_gallery: %d/%d ribs have zero vertices or zero "
            "faces and will not render (see per-rib counts below).",
            n_skipped, n_ribs,
        )
    logger.info("plot_mean_shape_gallery per-rib counts:")
    for name, n_v, n_f in diag_rows:
        marker = "  " if (n_v and n_f) else "✗ "
        logger.info("  %s%-12s  vertices=%6d  faces=%6d", marker, name, n_v, n_f)

    for ri in range(n_ribs):
        s, e = rib_offsets[ri], rib_offsets[ri + 1]
        f_sub  = faces[face_rib == ri] - s
        verts  = mean_shape[s:e]
        if verts.size == 0 or f_sub.size == 0:
            continue

        col = (ri % ncols) + 1
        row = (ri // ncols) + 1
        fig.add_trace(
            go.Mesh3d(
                x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                i=f_sub[:, 0], j=f_sub[:, 1], k=f_sub[:, 2],
                color=rib_colours[ri],
                opacity=0.95,
                flatshading=False,
                lighting=dict(ambient=0.45, diffuse=0.7, specular=0.15,
                              roughness=0.85, fresnel=0.2),
                lightposition=dict(x=300, y=200, z=200),
                showlegend=False,
                name=rib_ids[ri],
                hovertemplate=f"{rib_ids[ri]}<extra></extra>",
            ),
            row=row, col=col,
        )

    # Hide axes in every scene.
    for k in range(1, n_ribs + 1):
        scene_id = f"scene{k if k > 1 else ''}"
        fig.layout[scene_id].update(
            aspectmode="data",
            xaxis=dict(visible=False), yaxis=dict(visible=False),
            zaxis=dict(visible=False),
        )

    apply_layout(
        fig, width_class="full",
        height_mm=max(40.0 * nrows, 80.0),
        title=f"GPA mean shape – per-rib gallery ({n_ribs} ribs)",
    )
    save_fig(fig, out_stem,
             title="GPA mean shape – per-rib gallery",
             width_class="full",
             formats=("html", "png"),
             static_renderer=_static_3d_renderer(
                 "mean_shape_gallery",
                 mean_shape=mean_shape, faces=faces, rib_offsets=rib_offsets,
                 rib_ids=rib_ids, rib_colours=list(rib_colours),
             ))

