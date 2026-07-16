"""PyVista off-screen mesh rasterizers.

VTK-backed alternative to ``utils.mesh_mpl`` for the two large 3D SSM
figures (`plot_pc_deformations`, `plot_mean_shape_gallery`).  Camera and
bounds conventions mirror ``utils.mesh_mpl`` so the rendered output stays
visually comparable across backends.
"""
from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np

import settings as S
from utils.mesh_mpl import BOUNDS_PADDING, CAMERA_AZIM, CAMERA_ELEV

logger = logging.getLogger(__name__)

_PV_CONFIGURED: bool = False


def _configure_pv() -> None:
    """One-time theme setup — silences VTK warnings, disables notebook backend."""
    global _PV_CONFIGURED
    if _PV_CONFIGURED:
        return
    import pyvista as pv
    import vtk

    pv.OFF_SCREEN = True
    pv.global_theme.notebook = False
    pv.global_theme.background = "white"
    pv.global_theme.transparent_background = False
    vtk.vtkObject.GlobalWarningDisplayOff()
    _PV_CONFIGURED = True


def _window_size(width_mm: float, dpi: int, *, aspect: float = 0.75) -> tuple[int, int]:
    w = int(round(width_mm / 25.4 * dpi))
    return w, int(round(w * aspect))


def _faces_to_vtk(faces: np.ndarray) -> np.ndarray:
    f = np.asarray(faces, dtype=np.int64)
    n_tri = f.shape[0]
    return np.hstack([np.full((n_tri, 1), 3, dtype=np.int64), f]).ravel()


def _set_camera(plotter, verts: np.ndarray, *, padding: float = BOUNDS_PADDING) -> None:
    """Match the ``mesh_mpl`` camera (elev=20°, azim=-60°) with parallel projection
    scaled to fit the full bbox extent."""
    center = verts.mean(axis=0)
    extent = float(np.linalg.norm(verts.max(axis=0) - verts.min(axis=0)))
    r = max(extent, 1.0)
    elev_rad = math.radians(CAMERA_ELEV)
    azim_rad = math.radians(CAMERA_AZIM)
    position = (
        center[0] + r * 1.5 * math.cos(elev_rad) * math.cos(azim_rad),
        center[1] + r * 1.5 * math.cos(elev_rad) * math.sin(azim_rad),
        center[2] + r * 1.5 * math.sin(elev_rad),
    )
    plotter.camera_position = [tuple(position), tuple(center), (0.0, 0.0, 1.0)]
    plotter.camera.parallel_projection = True
    plotter.camera.parallel_scale = r * 0.6 * padding
    plotter.reset_camera_clipping_range()


_LIGHTING = dict(
    smooth_shading=True,
    ambient=0.45,
    diffuse=0.70,
    specular=0.15,
    specular_power=8.0,
)


def render_pc_deformations(
    *,
    mean_shape: np.ndarray,
    pca,
    template_faces: np.ndarray,
    n_pcs: int,
    n_sd: float,
    path: Path,
    dpi: int = S.EXPORT_RASTER_DPI,
) -> None:
    """Write a single composite PNG: ``n_pcs × 2`` (±SD) deformation panels."""
    _configure_pv()
    import pyvista as pv

    n_pts = mean_shape.shape[0]
    sds = np.sqrt(np.asarray(pca.explained_variance_, dtype=float))
    n_pcs_actual = int(min(n_pcs, pca.n_components_))
    faces_vtk = _faces_to_vtk(template_faces)

    width_mm = S.FIG_WIDTH_FULL_MM
    aspect = max(0.5, min(2.0, 0.40 * n_pcs_actual))
    w_px, h_px = _window_size(width_mm, dpi, aspect=aspect)

    plotter = pv.Plotter(
        off_screen=True,
        shape=(n_pcs_actual, 2),
        window_size=(w_px, h_px),
        border=False,
    )

    title_font = max(int(round(S.FONT_SIZE_TITLE_PT * dpi / S.SCREEN_DPI / 4)), 8)

    for k in range(n_pcs_actual):
        component = pca.components_[k].reshape(n_pts, 3)
        evr = float(pca.explained_variance_ratio_[k])

        cmax = 0.0
        per_panel: list[tuple[np.ndarray, np.ndarray]] = []
        for sign in (-1, +1):
            deformed = mean_shape + sign * n_sd * sds[k] * component
            disp = np.linalg.norm(deformed - mean_shape, axis=1)
            cmax = max(cmax, float(disp.max()) if disp.size else 0.0)
            per_panel.append((deformed, disp))

        cmax = max(cmax, 1e-6)
        for col_idx, ((deformed, disp), sign) in enumerate(zip(per_panel, (-1, +1))):
            plotter.subplot(k, col_idx)
            mesh = pv.PolyData(np.ascontiguousarray(deformed), faces_vtk)
            mesh.point_data["disp"] = disp
            is_anchor = (k == 0 and col_idx == 1)
            plotter.add_mesh(
                mesh,
                scalars="disp",
                cmap="viridis",
                clim=(0.0, cmax),
                show_scalar_bar=is_anchor,
                scalar_bar_args=(
                    dict(title="|Δ| (mm)", n_labels=4, vertical=True,
                         position_x=0.90, position_y=0.20,
                         width=0.04, height=0.60)
                    if is_anchor else None
                ),
                **_LIGHTING,
            )
            plotter.add_text(
                f"PC{k+1}  {'+' if sign > 0 else '−'}{n_sd:.0f} SD  ({evr:.1%})",
                position="upper_edge",
                font_size=title_font,
                color="black",
            )
            _set_camera(plotter, deformed)

    plotter.screenshot(str(path), transparent_background=False, return_img=False)
    plotter.close()


def render_mean_shape_gallery(
    *,
    mean_shape: np.ndarray,
    faces: np.ndarray,
    rib_offsets: list[int],
    rib_ids: list[str],
    rib_colours: list[str],
    path: Path,
    dpi: int = S.EXPORT_RASTER_DPI,
) -> None:
    """Write a single composite PNG: 4-column gallery of per-rib mean-shape thumbnails."""
    _configure_pv()
    import pyvista as pv

    n_ribs = len(rib_offsets) - 1
    ncols = 4
    nrows = (n_ribs + ncols - 1) // ncols

    ends = np.asarray(rib_offsets[1:], dtype=np.int64)
    faces_i64 = np.asarray(faces, dtype=np.int64)
    v0 = np.searchsorted(ends, faces_i64[:, 0], side="right")
    v1 = np.searchsorted(ends, faces_i64[:, 1], side="right")
    v2 = np.searchsorted(ends, faces_i64[:, 2], side="right")
    face_rib = np.where((v0 == v1) & (v1 == v2), v0, -1)

    width_mm = S.FIG_WIDTH_FULL_MM
    aspect = max(0.5, 0.30 * nrows)
    w_px, h_px = _window_size(width_mm, dpi, aspect=aspect)

    plotter = pv.Plotter(
        off_screen=True,
        shape=(nrows, ncols),
        window_size=(w_px, h_px),
        border=False,
    )
    title_font = max(int(round(S.FONT_SIZE_TITLE_PT * dpi / S.SCREEN_DPI / 4)), 8)

    for ri in range(n_ribs):
        s, e = rib_offsets[ri], rib_offsets[ri + 1]
        f_sub = faces_i64[face_rib == ri] - s
        verts = mean_shape[s:e]
        row, col = ri // ncols, ri % ncols
        plotter.subplot(row, col)
        if verts.size == 0 or f_sub.size == 0:
            plotter.add_text(rib_ids[ri], position="upper_edge",
                             font_size=title_font, color="black")
            continue
        mesh = pv.PolyData(np.ascontiguousarray(verts), _faces_to_vtk(f_sub))
        plotter.add_mesh(
            mesh,
            color=rib_colours[ri],
            show_scalar_bar=False,
            **_LIGHTING,
        )
        plotter.add_text(rib_ids[ri], position="upper_edge",
                         font_size=title_font, color="black")
        _set_camera(plotter, verts)

    # Any leftover empty subplot positions: leave default (white background).
    plotter.screenshot(str(path), transparent_background=False, return_img=False)
    plotter.close()
