"""Mean ribcage rendered from three orthogonal views.

Coronal, sagittal, and axial panels are stitched into a single PNG. The
off-screen rasteriser is selected via ``settings.STATIC_3D_BACKEND``.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

import settings as S
from utils import figure_meta as M
from utils.colors import rib_colors
from utils.mesh_pv import _configure_pv, _faces_to_vtk

logger = logging.getLogger(__name__)


_VIEWS: list[tuple[str, tuple[float, float, float], tuple[float, float, float]]] = [
    # (label, view_axis_unit_vector, view_up_unit_vector)
    ("Coronal (anterior)",  ( 0.0,  1.0,  0.0), (0.0, 0.0, 1.0)),
    ("Sagittal (left)",     (-1.0,  0.0,  0.0), (0.0, 0.0, 1.0)),
    ("Axial (inferior)",    ( 0.0,  0.0, -1.0), (0.0, 1.0, 0.0)),
]


def _parse_rib_id(rib_id: str) -> tuple[int, str]:
    try:
        if rib_id.startswith("Rib "):
            parts = rib_id.split()
            return int(parts[1]) + 39, parts[2]
        head, side = rib_id.rsplit("_", 1)
        return int(head.removeprefix("rib")), side
    except (ValueError, AttributeError):
        return 0, "L"


def _common_setup(
    mean_shape: np.ndarray,
    template_faces: np.ndarray,
    rib_offsets: np.ndarray,
    rib_ids: list[str] | None,
) -> tuple[list[int], list[str], list[str], np.ndarray, np.ndarray]:
    """Return ``(offsets, rib_ids, rib_colours, faces_int64, face_rib_index)``."""
    offsets = list(rib_offsets) + [mean_shape.shape[0]]
    n_ribs = len(offsets) - 1
    if rib_ids is None or len(rib_ids) < n_ribs:
        rib_ids = [f"Rib {i+1}" for i in range(n_ribs)]
    rib_ids = list(rib_ids)[:n_ribs]
    parsed = [_parse_rib_id(rid) for rid in rib_ids]
    colours = rib_colors([lv for lv, _ in parsed], [sd for _, sd in parsed])

    faces = template_faces.astype(np.int64)
    ends = np.asarray(offsets[1:], dtype=np.int64)
    v0 = np.searchsorted(ends, faces[:, 0], side="right")
    v1 = np.searchsorted(ends, faces[:, 1], side="right")
    v2 = np.searchsorted(ends, faces[:, 2], side="right")
    face_rib = np.where((v0 == v1) & (v1 == v2), v0, -1)
    return offsets, rib_ids, colours, faces, face_rib


def _pv_render(
    mean_shape: np.ndarray,
    template_faces: np.ndarray,
    rib_offsets: np.ndarray,
    rib_ids: list[str],
    raster_path: Path,
    *,
    dpi: int,
) -> None:
    _configure_pv()
    import pyvista as pv

    offsets, rib_ids, colours, faces, face_rib = _common_setup(
        mean_shape, template_faces, rib_offsets, rib_ids,
    )
    n_ribs = len(offsets) - 1

    width_mm = S.FIG_WIDTH_HALF_MM * 2
    w_px = int(round(width_mm / 25.4 * dpi))
    h_px = int(round(w_px / 3 * 1.05))
    plotter = pv.Plotter(off_screen=True, shape=(1, 3),
                         window_size=(w_px, h_px), border=False)
    title_font = max(int(round(S.FONT_SIZE_TITLE_PT * dpi / S.SCREEN_DPI / 4)), 8)

    center = mean_shape.mean(axis=0).astype(float)
    extent = float(np.linalg.norm(mean_shape.max(axis=0) - mean_shape.min(axis=0)))
    radius = max(extent, 1.0) * 1.5
    parallel_scale = extent * 0.6

    for view_idx, (label, axis, up) in enumerate(_VIEWS):
        plotter.subplot(0, view_idx)
        axis = np.asarray(axis, dtype=float)
        position = tuple(center + radius * axis)
        plotter.camera_position = [tuple(position), tuple(center), tuple(up)]
        plotter.camera.parallel_projection = True
        plotter.camera.parallel_scale = parallel_scale
        for ri in range(n_ribs):
            s, e = offsets[ri], offsets[ri + 1]
            f_sub = faces[face_rib == ri] - s
            verts = mean_shape[s:e]
            if verts.size == 0 or f_sub.size == 0:
                continue
            mesh = pv.PolyData(np.ascontiguousarray(verts), _faces_to_vtk(f_sub))
            plotter.add_mesh(
                mesh,
                color=colours[ri],
                smooth_shading=True,
                ambient=0.45, diffuse=0.70,
                specular=0.15, specular_power=8.0,
            )
        plotter.reset_camera_clipping_range()
        plotter.add_text(label, position="upper_edge",
                         font_size=title_font, color="black")

    plotter.screenshot(str(raster_path), transparent_background=False,
                       return_img=False)
    plotter.close()


# (label, (horizontal_idx, vertical_idx), view_dir camera→scene)
_MPL_VIEWS: list[tuple[str, tuple[int, int], tuple[float, float, float]]] = [
    ("Coronal (anterior)", (0, 2), ( 0.0, -1.0,  0.0)),
    ("Sagittal (left)",    (1, 2), ( 1.0,  0.0,  0.0)),
    ("Axial (inferior)",   (0, 1), ( 0.0,  0.0,  1.0)),
]


def _mpl_render(
    mean_shape: np.ndarray,
    template_faces: np.ndarray,
    rib_offsets: np.ndarray,
    rib_ids: list[str],
    raster_path: Path,
    *,
    dpi: int,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from matplotlib.colors import to_rgb

    _, _, colours, faces, face_rib = _common_setup(
        mean_shape, template_faces, rib_offsets, rib_ids,
    )
    valid = face_rib >= 0
    rib_rgb = np.array([to_rgb(c) for c in colours], dtype=float)

    tri_v = mean_shape[faces]
    edge1 = tri_v[:, 1] - tri_v[:, 0]
    edge2 = tri_v[:, 2] - tri_v[:, 0]
    normals = np.cross(edge1, edge2)
    norm_len = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.where(norm_len > 0, norm_len, 1.0)
    centroids = tri_v.mean(axis=1)

    base_rgb = np.zeros((len(faces), 3), dtype=float)
    base_rgb[valid] = rib_rgb[face_rib[valid]]

    light = np.array([-0.4, -0.7, 0.6], dtype=float)
    light /= np.linalg.norm(light)
    ambient = 0.40
    diffuse = 0.60
    # Two-sided shading: |n·l| keeps colour on faces regardless of STL winding.
    lamb = np.abs(normals @ light)
    shaded = np.clip(base_rgb * (ambient + diffuse * lamb[:, None]), 0.0, 1.0)

    width_mm = S.FIG_WIDTH_HALF_MM * 2
    width_in = width_mm / 25.4
    h_in = width_in / 3 * 1.05
    fig, axes = plt.subplots(1, 3, figsize=(width_in, h_in), dpi=dpi)

    for ax, (label, (h_idx, v_idx), view_dir) in zip(axes, _MPL_VIEWS):
        view_dir_arr = np.asarray(view_dir, dtype=float)
        # Painter's algorithm: draw farthest-from-camera first.
        depth = centroids @ view_dir_arr
        order = np.argsort(-depth)
        order = order[valid[order]]

        poly = tri_v[order][:, :, (h_idx, v_idx)]
        pc = PolyCollection(poly, facecolors=shaded[order],
                            edgecolors="none", linewidths=0)
        ax.add_collection(pc)

        h_min, h_max = float(mean_shape[:, h_idx].min()), float(mean_shape[:, h_idx].max())
        v_min, v_max = float(mean_shape[:, v_idx].min()), float(mean_shape[:, v_idx].max())
        pad_h = 0.05 * max(h_max - h_min, 1.0)
        pad_v = 0.05 * max(v_max - v_min, 1.0)
        # Horizontal axis reversed to follow camera-frame handedness.
        ax.set_xlim(h_max + pad_h, h_min - pad_h)
        ax.set_ylim(v_min - pad_v, v_max + pad_v)
        ax.set_aspect("equal")
        ax.set_axis_off()
        ax.set_title(label, fontsize=S.FONT_SIZE_TITLE_PT)

    fig.tight_layout()
    fig.savefig(raster_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_mean_shape_views(
    mean_shape: np.ndarray,
    template_faces: np.ndarray,
    rib_offsets: np.ndarray,
    rib_ids: list[str],
    out_stem: Path,
    *,
    dpi: int = S.EXPORT_RASTER_DPI,
) -> None:
    """Render the GPA mean ribcage from three orthogonal views to ``out_stem``."""
    out_stem = Path(out_stem)
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    raster_path = out_stem.with_suffix(".png")

    backend = S.STATIC_3D_BACKEND
    if backend in ("auto", "pyvista"):
        try:
            _pv_render(mean_shape, template_faces, rib_offsets, rib_ids,
                       raster_path, dpi=dpi)
        except Exception as exc:  # why-broad: VTK/EGL/OSMesa raise varies
            if backend == "pyvista":
                raise
            logger.warning(
                "PyVista offscreen render failed (%s); falling back to matplotlib.",
                exc,
            )
            _mpl_render(mean_shape, template_faces, rib_offsets, rib_ids,
                        raster_path, dpi=dpi)
    else:
        _mpl_render(mean_shape, template_faces, rib_offsets, rib_ids,
                    raster_path, dpi=dpi)

    M.append_manifest(out_stem.parent, out_stem.name, dict(
        stem=str(out_stem),
        formats=["png"],
        files={"png": str(raster_path)},
        title="Mean ribcage — three orthogonal views",
        width_class="full",
    ))
    logger.info("Saved mean-shape views → %s", raster_path)
