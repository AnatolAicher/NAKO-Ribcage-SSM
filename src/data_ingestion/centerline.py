"""Per-rib centerline reconstruction from multi-label rib/vertebra NIfTI masks.

Algorithm (per rib, per side):

1. **P** = rib voxel with the smallest Euclidean distance to the paired
   vertebra mask. Sits on the rib surface near the costovertebral end.
2. Greedy gradient-ascent on the rib mask's Euclidean distance transform
   from **P** until a local maximum — pulls **P** off the surface to a
   medial-axis voxel.
3. Geodesic distance field from **P** through the rib volume — Dijkstra
   with unit cost inside the mask and anisotropic Euclidean step lengths
   (via ``skimage.graph.MCP_Geometric``).
4. **Q** = voxel of maximum geodesic distance (distal / anterior end).
5. Centered shortest path **P → Q** with per-voxel cost
   ``1 / (EDT + eps) ** alpha``.
6. Convert the integer voxel path to mesh-frame mm
   (``(ijk + bbox_offset) * zooms``) and arc-length-parameterized cubic
   B-spline resample to a fixed number of landmarks.

Outputs live in the **mesh frame** — ``voxel_idx_RAS * zooms`` with no
affine translation — matching the STL meshes from ``ssm.mesh_extraction``
(``skimage.measure.marching_cubes(..., spacing=zooms)``). The Scala
registration code reads landmarks in this frame; see
``environment/scalismo_ssm/.../RibRegistration.scala`` ("raw mesh frame").

Writers (``write_scalismo_landmarks``, ``write_centerline_record``) emit
JSON compatible with the existing Scala loader and diagnostic viewer.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable

import nibabel as nib
import numpy as np
from scipy.interpolate import splev, splprep
from scipy.ndimage import distance_transform_edt
from skimage.graph import MCP_Geometric

logger = logging.getLogger(__name__)


# ── Coordinate conversion ─────────────────────────────────────────────────────

def _pir_to_las(p: Iterable[float]) -> tuple[float, float, float]:
    """Convert a 3-vector from PIR (Posterior, Inferior, Right) to LAS axes.

    LAS axes: x=Left, y=Anterior, z=Superior. ``(x, y, z) → (-z, -y, -x)``
    (an involution).
    """
    return (-p[2], -p[1], -p[0])


def las_to_ras(points_las: np.ndarray) -> np.ndarray:
    """Flip x sign to convert LAS-mm points to RAS-mm."""
    out = np.asarray(points_las, dtype=np.float64).copy()
    out[:, 0] = -out[:, 0]
    return out


# ── NIfTI loading ─────────────────────────────────────────────────────────────

def load_seg_volume(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a multi-label rib/vertebra NIfTI into RAS-oriented int32 + zooms.

    After ``nib.as_closest_canonical`` the axes are R (dim 0 → right),
    A (dim 1 → anterior), S (dim 2 → superior).
    """
    img = nib.load(str(path))
    img = nib.as_closest_canonical(img)
    data = np.asarray(img.dataobj, dtype=np.int32)
    zooms = np.array(img.header.get_zooms()[:3], dtype=float)
    return data, zooms


# ── Bounding-box cropping ─────────────────────────────────────────────────────

def bbox_crop(
    masks: tuple[np.ndarray, ...], pad: int = 1,
) -> tuple[tuple[np.ndarray, ...], np.ndarray]:
    """3D bounding box over the union of ``masks`` with a per-axis voxel pad.

    Returns the cropped sub-masks (same order as input) and the
    ``(i0, j0, k0)`` lower-corner offset so callers can add it back when
    converting cropped voxel indices to the original volume's frame. The
    pad is clamped to the volume bounds.
    """
    if not masks:
        raise ValueError("bbox_crop: need at least one mask")
    shape = masks[0].shape
    for m in masks:
        if m.shape != shape:
            raise ValueError(f"bbox_crop: shape mismatch {shape} vs {m.shape}")

    union = np.zeros(shape, dtype=bool)
    for m in masks:
        union |= m.astype(bool)
    if not union.any():
        raise ValueError("bbox_crop: union of masks is empty")

    coords = np.where(union)
    lo = np.array([int(c.min()) for c in coords])
    hi = np.array([int(c.max()) for c in coords])
    lo = np.maximum(lo - pad, 0)
    hi = np.minimum(hi + pad + 1, np.array(shape))

    slc = tuple(slice(int(lo[d]), int(hi[d])) for d in range(3))
    cropped = tuple(m[slc] for m in masks)
    return cropped, lo


# ── Algorithm steps ───────────────────────────────────────────────────────────

def find_seed_voxel_P(
    rib_mask: np.ndarray, vert_mask: np.ndarray, zooms: np.ndarray,
) -> np.ndarray:
    """Rib voxel with the smallest Euclidean distance to the vertebra mask.

    Falls back to the most-posterior rib voxel (min dim-1 in RAS) when
    ``vert_mask`` is empty.
    """
    if not rib_mask.any():
        raise ValueError("find_seed_voxel_P: rib mask is empty")

    if not vert_mask.any():
        coords = np.argwhere(rib_mask)
        return coords[int(np.argmin(coords[:, 1]))]

    vert_distance = distance_transform_edt(~vert_mask.astype(bool), sampling=tuple(zooms))
    masked = np.where(rib_mask, vert_distance, np.inf)
    flat_idx = int(np.argmin(masked))
    return np.array(np.unravel_index(flat_idx, masked.shape), dtype=int)


def gradient_ascent_to_medial(
    edt: np.ndarray, start_ijk: np.ndarray, max_steps: int = 50,
) -> np.ndarray:
    """Greedy 26-neighbour hill-climb on ``edt`` from ``start_ijk``.

    Moves at each step to the neighbour with strictly the largest EDT.
    Terminates when no neighbour exceeds the current voxel's EDT, or after
    ``max_steps`` iterations. Returns the (cropped-frame) voxel where we
    stopped.
    """
    pos = np.asarray(start_ijk, dtype=int).copy()
    nx, ny, nz = edt.shape
    for _ in range(max_steps):
        best_val = edt[pos[0], pos[1], pos[2]]
        best_pos = pos
        for di in (-1, 0, 1):
            ni = pos[0] + di
            if ni < 0 or ni >= nx:
                continue
            for dj in (-1, 0, 1):
                nj = pos[1] + dj
                if nj < 0 or nj >= ny:
                    continue
                for dk in (-1, 0, 1):
                    if di == 0 and dj == 0 and dk == 0:
                        continue
                    nk = pos[2] + dk
                    if nk < 0 or nk >= nz:
                        continue
                    v = edt[ni, nj, nk]
                    if v > best_val:
                        best_val = v
                        best_pos = np.array([ni, nj, nk])
        if np.array_equal(best_pos, pos):
            return pos
        pos = best_pos
    return pos


def geodesic_distance(
    rib_mask: np.ndarray, source_ijk: np.ndarray, zooms: np.ndarray,
) -> np.ndarray:
    """Geodesic distance from ``source_ijk`` through ``rib_mask`` (mm).

    Uses ``MCP_Geometric`` with unit cost inside the mask and infinite cost
    outside, with ``sampling=tuple(zooms)`` so step lengths between voxels
    are anisotropic-Euclidean. Outside-mask voxels in the returned array
    are ``+inf``.
    """
    costs = np.where(rib_mask, 1.0, np.inf)
    mcp = MCP_Geometric(costs, sampling=tuple(zooms))
    cumulative, _ = mcp.find_costs([tuple(int(c) for c in source_ijk)])
    return np.asarray(cumulative)


def find_far_endpoint_Q(
    geodesic: np.ndarray, rib_mask: np.ndarray,
) -> np.ndarray:
    """Rib voxel with the largest finite geodesic distance.

    Raises ``ValueError`` if no rib voxel has a finite distance, which
    indicates the source's connected component is empty (a degenerate case
    handled by the orchestrator with a fallback or skip).
    """
    masked = np.where(rib_mask & np.isfinite(geodesic), geodesic, -np.inf)
    if not np.isfinite(masked).any():
        raise ValueError(
            "find_far_endpoint_Q: no rib voxel has finite geodesic distance"
        )
    flat_idx = int(np.argmax(masked))
    return np.array(np.unravel_index(flat_idx, masked.shape), dtype=int)


def two_pass_farthest_endpoints(
    rib_mask: np.ndarray, ref_ijk: np.ndarray, zooms: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Endpoints A (proximal) and B (distal) via two-pass geodesic diameter.

    ``ref_ijk`` only labels which diameter end is proximal; its exact
    position does not affect endpoint coordinates. Returns ``(A, B)`` in
    the same (cropped) frame as ``rib_mask``.
    """
    geo_from_ref = geodesic_distance(rib_mask, ref_ijk, zooms)
    B_ijk = find_far_endpoint_Q(geo_from_ref, rib_mask)
    geo_from_B = geodesic_distance(rib_mask, B_ijk, zooms)
    A_ijk = find_far_endpoint_Q(geo_from_B, rib_mask)
    return A_ijk, B_ijk


def centered_shortest_path(
    rib_mask: np.ndarray, edt: np.ndarray,
    ijk_P: np.ndarray, ijk_Q: np.ndarray,
    zooms: np.ndarray, alpha: float = 2.0, eps: float = 1.0,
) -> np.ndarray:
    """Cost-weighted shortest path P → Q with cost ``1 / (EDT + eps)**alpha``.

    Returns the integer voxel path (``(K, 3)``) in the cropped-frame
    coordinate system. The first row is ``ijk_P`` and the last row is
    ``ijk_Q``.
    """
    costs = np.where(rib_mask, 1.0 / (edt + eps) ** alpha, np.inf)
    mcp = MCP_Geometric(costs, sampling=tuple(zooms))
    mcp.find_costs([tuple(int(c) for c in ijk_P)])
    path = mcp.traceback(tuple(int(c) for c in ijk_Q))
    return np.asarray(path, dtype=int)


def ijk_to_mesh_frame(
    path_ijk: np.ndarray, offset_ijk: np.ndarray, zooms: np.ndarray,
) -> np.ndarray:
    """Convert cropped integer voxel indices to mesh-frame mm.

    Mesh frame = ``voxel_idx_RAS * zooms`` (no NIfTI affine translation);
    matches ``marching_cubes(spacing=zooms)``.
    """
    ijk = np.asarray(path_ijk, dtype=np.float64)
    off = np.asarray(offset_ijk, dtype=np.float64)
    zm = np.asarray(zooms, dtype=np.float64)
    return (ijk + off) * zm


# ── Top-level orchestrator ────────────────────────────────────────────────────

def extract_centerline_from_nifti(
    side_rib_mask: np.ndarray,
    vert_mask: np.ndarray,
    zooms: np.ndarray,
    *,
    n_points: int = 20,
    alpha: float = 2.0,
    eps: float = 1.0,
    smoothing: float = 0.0,
) -> dict | None:
    """Compute one rib's centerline from pre-split masks.

    Two-pass geodesic farthest-point endpoint detection ("tree diameter"
    pattern); the vertebra-closest voxel C is used only to tag which end
    is proximal.

    Parameters
    ----------
    side_rib_mask
        Binary mask of the rib volume for one side (left or right). Must
        already be split out of the multi-label NIfTI by the caller (e.g.
        via ``ssm.mesh_extraction.split_rib_components`` +
        ``resolve_ambiguous_sides``).
    vert_mask
        Binary mask of the paired vertebra (``data == rib_label - 32``).
        May be empty — the seed-finder then falls back to the most posterior
        rib voxel.
    zooms
        3-vector of physical voxel spacings in mm (from the NIfTI header,
        after ``as_closest_canonical``).
    n_points, alpha, eps, smoothing
        See module docstring. Defaults: ``n_points=20``, ``alpha=2.0``,
        ``eps=1.0`` mm, ``smoothing=0.0`` (interpolation, endpoints exact).

    Returns
    -------
    dict or None
        ``None`` if the rib mask is empty. Otherwise:

        - ``points_mesh_frame``: ``(n_points, 3)`` float — resampled centerline.
        - ``raw_path_mesh_frame``: ``(K, 3)`` float — voxel-stepwise path,
          pre-smoothing, in mesh-frame mm.
        - ``A_ijk``: ``(3,)`` int — proximal anchor in uncropped voxel coords.
        - ``B_ijk``: ``(3,)`` int — distal anchor in uncropped voxel coords.
        - ``C_ijk``: ``(3,)`` int — vertebra-closest reference voxel.
        - ``arc_length_mm``: float — arc length of the resampled polyline.

    Raises on geodesic disconnection or other algorithm failure so the
    caller can log + skip the rib.
    """
    if not side_rib_mask.any():
        return None

    (sub_rib, sub_vert), offset_ijk = bbox_crop(
        (side_rib_mask, vert_mask), pad=1,
    )

    # C: rib voxel closest to vertebra (used only to tag proximal end).
    C_ijk = find_seed_voxel_P(sub_rib, sub_vert, zooms)

    # Endpoints A (proximal) and B (distal) via two-pass geodesic diameter.
    A_ijk, B_ijk = two_pass_farthest_endpoints(sub_rib, C_ijk, zooms)

    # EDT for medial-axis-weighted shortest path.
    rib_edt = distance_transform_edt(sub_rib, sampling=tuple(zooms))

    # Centered shortest path A → B; cost 1/(EDT+eps)^alpha pulls the
    # interior onto the medial axis.
    path_ijk = centered_shortest_path(
        sub_rib, rib_edt, A_ijk, B_ijk, zooms, alpha=alpha, eps=eps,
    )
    if path_ijk.shape[0] < 2:
        raise ValueError(
            f"centered_shortest_path: traceback returned {path_ijk.shape[0]} "
            f"point(s); need >= 2"
        )

    # Step 5: convert to mesh-frame mm and B-spline-resample.
    path_mm = ijk_to_mesh_frame(path_ijk, offset_ijk, zooms)
    points_resampled = resample_arclength(
        path_mm, n_points=n_points, smoothing=smoothing,
    )

    arc_length = float(
        np.linalg.norm(np.diff(points_resampled, axis=0), axis=1).sum()
    )

    return {
        "points_mesh_frame":      points_resampled,
        "raw_path_mesh_frame":    path_mm,
        "A_ijk":                  (A_ijk + offset_ijk).astype(int),
        "B_ijk":                  (B_ijk + offset_ijk).astype(int),
        "C_ijk":                  (C_ijk + offset_ijk).astype(int),
        "arc_length_mm":          arc_length,
    }


# ── Spline resampling ─────────────────────────────────────────────────────────

def _cumulative_arc_length(pts: np.ndarray) -> np.ndarray:
    """Cumulative chord length along a polyline. Returns shape ``(K,)``."""
    diffs = np.diff(pts, axis=0)
    seg = np.linalg.norm(diffs, axis=1)
    return np.concatenate([[0.0], np.cumsum(seg)])


def resample_arclength(
    points: np.ndarray,
    n_points: int,
    smoothing: float = 0.0,
) -> np.ndarray:
    """Resample a polyline to ``n_points`` along (approximately) uniform arc length.

    Endpoints are preserved exactly (``u=0`` and ``u=1`` are sampled).

    - When ``K >= 4`` and points are non-degenerate: fit a cubic B-spline via
      ``scipy.interpolate.splprep`` (chord-length parameterization, the
      ``splprep`` default) and sample at ``n_points`` uniform ``u``.
    - When ``K < 4`` (short / floating ribs) fall back to piecewise-linear
      interpolation over cumulative arc length so cubic ``splprep`` doesn't
      error out.

    ``smoothing`` is passed through as ``s`` to ``splprep``. ``s=0`` is
    interpolation (endpoints exact). Values > 0 smooth but may move endpoints.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points must be (K, 3); got {pts.shape}")
    if n_points < 2:
        raise ValueError(f"n_points must be >= 2; got {n_points}")
    K = pts.shape[0]
    if K < 2:
        raise ValueError(f"need >= 2 input points for resampling; got {K}")

    arc = _cumulative_arc_length(pts)
    total = float(arc[-1])
    if total <= 0:
        raise ValueError("polyline has zero arc length (all points coincident)")

    if K < 4:
        target = np.linspace(0.0, total, n_points)
        out = np.empty((n_points, 3), dtype=np.float64)
        for d in range(3):
            out[:, d] = np.interp(target, arc, pts[:, d])
        return out

    tck, _u = splprep([pts[:, 0], pts[:, 1], pts[:, 2]], s=smoothing, k=3)
    u_new = np.linspace(0.0, 1.0, n_points)
    xn, yn, zn = splev(u_new, tck)
    out = np.column_stack([xn, yn, zn]).astype(np.float64, copy=False)
    out[0] = pts[0]
    out[-1] = pts[-1]
    return out


# ── Output writers ────────────────────────────────────────────────────────────

def write_scalismo_landmarks(
    points: np.ndarray,
    out_path: Path,
    id_prefix: str = "cl",
) -> None:
    """Emit Scalismo ``LandmarkIO.writeLandmarksJson`` format.

    JSON array of ``{"id": str, "coordinates": [x, y, z]}`` objects.
    Coordinates are written verbatim; callers emit in the correct frame.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points must be (N, 3); got {pts.shape}")
    width = max(2, len(str(len(pts) - 1)))
    items = [
        {"id": f"{id_prefix}_{i:0{width}d}", "coordinates": [float(x), float(y), float(z)]}
        for i, (x, y, z) in enumerate(pts.tolist())
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(items, indent=2))


def write_centerline_record(
    points: np.ndarray,
    raw_points: np.ndarray,
    out_path: Path,
    metadata: dict | None = None,
) -> None:
    """Write a richer per-rib centerline JSON for diagnostics.

    Schema::

        {
          "n_points":      20,
          "arc_length_mm": 123.45,
          "frame":         "RAS_mesh",
          "points":        [[x,y,z], ...],   # resampled, mesh frame
          "raw_points":    [[x,y,z], ...],   # pre-smoothing, mesh frame
          "raw_n_points":  K,
          ...                                 # any extra metadata
        }

    ``frame: "RAS_mesh"`` signals points are ``voxel_idx_RAS * zooms`` with
    no NIfTI affine translation — same frame as the per-rib STL meshes.
    """
    pts = np.asarray(points, dtype=np.float64)
    raw = np.asarray(raw_points, dtype=np.float64)
    arc = float(_cumulative_arc_length(pts)[-1])
    record: dict = {
        "n_points":      int(pts.shape[0]),
        "arc_length_mm": arc,
        "frame":         "RAS_mesh",
        "points":        pts.tolist(),
        "raw_points":    raw.tolist(),
        "raw_n_points":  int(raw.shape[0]),
    }
    if metadata:
        record.update(metadata)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, indent=2))
