"""Matplotlib 3D mesh rendering helpers (Poly3DCollection-based).

Pure-CPU alternative to PyVista off-screen rendering, used by every
3D figure in this project so they share a single visual style and run on
headless servers without OSMesa/EGL.
"""
from __future__ import annotations

import matplotlib.cm as cm
import numpy as np
import pyvista as pv
from matplotlib.colors import Colormap, Normalize, to_rgba
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Shared camera convention for every 3D matplotlib mesh render in this project.
CAMERA_ELEV: float = 20.0
CAMERA_AZIM: float = -60.0

# Cubic bounding-box padding (1.05 = 5 % margin around the data).
BOUNDS_PADDING: float = 1.05


def pv_to_triangles(mesh: pv.PolyData) -> tuple[np.ndarray, np.ndarray]:
    """Extract ``(verts, faces)`` numpy arrays from a PyVista triangle mesh.

    PyVista stores faces as a flat ``[3, i, j, k, 3, i, j, k, ...]`` array;
    this slices it into ``(n_faces, 3)``.
    """
    verts = np.asarray(mesh.points, dtype=np.float64)
    f = np.asarray(mesh.faces, dtype=np.int64)
    if f.size == 0:
        return verts, np.empty((0, 3), dtype=np.int64)
    return verts, f.reshape(-1, 4)[:, 1:]


def add_mesh(
    ax,
    verts: np.ndarray,
    faces: np.ndarray,
    *,
    color: str | tuple | None = None,
    scalars: np.ndarray | None = None,
    cmap: str | Colormap = "viridis",
    clim: tuple[float, float] | None = None,
    alpha: float = 1.0,
    shade: bool = True,
    edgecolors: str = "none",
    linewidths: float = 0.0,
    zsort: str = "average",
) -> Poly3DCollection | None:
    """Add one mesh as a ``Poly3DCollection`` to a 3D axis.

    Either ``color`` (uniform) or ``scalars`` (per-vertex, mapped through
    ``cmap`` to per-face colours via vertex-mean) must be supplied.

    Parameters
    ----------
    ax
        Matplotlib 3D axis.
    verts, faces
        ``(n_verts, 3)`` and ``(n_faces, 3)`` numpy arrays.
    color
        Any matplotlib colour spec (hex, name, RGBA tuple). Mutually
        exclusive with ``scalars``.
    scalars
        Per-vertex scalar field; averaged to per-face for colouring.
    cmap, clim
        Colourmap name + ``(vmin, vmax)``.  ``clim=None`` auto-fits.
    alpha
        Face opacity in ``[0, 1]``.
    shade
        Per-face directional shading using face normals.
    """
    if faces.size == 0:
        return None
    triangles = verts[faces]

    if scalars is not None:
        face_vals = scalars[faces].mean(axis=1)
        if clim is None:
            vmax = float(face_vals.max())
            clim = (float(face_vals.min()), max(vmax, 1e-6))
        cmap_obj = cm.get_cmap(cmap)
        facecolors = cmap_obj(Normalize(vmin=clim[0], vmax=clim[1])(face_vals))
        if alpha != 1.0:
            facecolors[:, 3] = alpha
    elif color is not None:
        facecolors = np.tile(to_rgba(color, alpha), (len(faces), 1))
    else:
        raise ValueError("add_mesh: either `color` or `scalars` must be set")

    # matplotlib 3.10's Poly3DCollection crashes on shade=True + edgecolors="none"
    # because the empty edge-color array breaks the per-face shading broadcast.
    # Pass facecolors as edgecolors so they receive the same shading.
    edge = facecolors if (shade and edgecolors == "none") else edgecolors
    coll = Poly3DCollection(
        triangles,
        facecolors=facecolors,
        edgecolors=edge,
        linewidths=linewidths,
        shade=shade,
        zsort=zsort,
    )
    ax.add_collection3d(coll)
    return coll


def add_meshes(
    ax,
    meshes: list[tuple[np.ndarray, np.ndarray, str | tuple]],
    *,
    alpha: float | list[float] = 1.0,
    shade: bool = True,
    edgecolors: str = "none",
    linewidths: float = 0.0,
    zsort: str = "average",
) -> Poly3DCollection | None:
    """Add multiple uniformly-coloured meshes as ONE ``Poly3DCollection``.

    matplotlib's ``mpl_toolkits.mplot3d`` only depth-sorts faces *within* a
    single ``Poly3DCollection``; across collections it sorts by each
    collection's aggregate ``sort_zpos``, which breaks when the
    collections interpenetrate (e.g. adjacent ribs in a rib cage —
    whichever was added later renders on top regardless of camera). Merging
    every mesh in one scene into one collection lets the per-face painter's
    sort cover the whole scene.

    Parameters
    ----------
    meshes
        List of ``(verts, faces, color)`` tuples. ``color`` is any
        matplotlib colour spec (hex / name / RGBA tuple).
    alpha
        Single value applied to every mesh, or a list with one alpha per mesh.
    """
    if not meshes:
        return None

    if isinstance(alpha, (int, float)):
        alphas = [float(alpha)] * len(meshes)
    else:
        alphas = list(alpha)
        if len(alphas) != len(meshes):
            raise ValueError("alpha length must match meshes length")

    tri_blocks: list[np.ndarray] = []
    color_blocks: list[np.ndarray] = []
    for (verts, faces, color), a in zip(meshes, alphas):
        if faces.size == 0:
            continue
        tri_blocks.append(verts[faces])
        color_blocks.append(np.tile(to_rgba(color, a), (len(faces), 1)))

    if not tri_blocks:
        return None

    triangles  = np.concatenate(tri_blocks, axis=0)
    facecolors = np.concatenate(color_blocks, axis=0)

    edge = facecolors if (shade and edgecolors == "none") else edgecolors
    coll = Poly3DCollection(
        triangles,
        facecolors=facecolors,
        edgecolors=edge,
        linewidths=linewidths,
        shade=shade,
        zsort=zsort,
    )
    ax.add_collection3d(coll)
    return coll


def set_cube_bounds(
    ax,
    verts_list: list[np.ndarray],
    *,
    padding: float = BOUNDS_PADDING,
) -> None:
    """Set equal-extent xlim/ylim/zlim from the union bounding box."""
    if not verts_list:
        return
    all_pts = np.concatenate(verts_list, axis=0)
    lo, hi = all_pts.min(axis=0), all_pts.max(axis=0)
    mid  = (lo + hi) / 2
    half = (hi - lo).max() / 2 * padding
    ax.set_xlim(mid[0] - half, mid[0] + half)
    ax.set_ylim(mid[1] - half, mid[1] + half)
    ax.set_zlim(mid[2] - half, mid[2] + half)


def style_axis(
    ax,
    *,
    elev: float = CAMERA_ELEV,
    azim: float = CAMERA_AZIM,
) -> None:
    """Project-standard 3D axis treatment: hidden axes, fixed view, cube box."""
    ax.set_axis_off()
    ax.view_init(elev=elev, azim=azim)
    try:
        ax.set_box_aspect((1, 1, 1))
    except AttributeError:
        pass
