"""Unit tests for ``data_ingestion.centerline``.

Covers the NIfTI-based skeleton centerline algorithm and the I/O helpers
shared with the SSM landmark pipeline.
"""
from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from data_ingestion.centerline import (
    _pir_to_las,
    bbox_crop,
    centered_shortest_path,
    extract_centerline_from_nifti,
    find_far_endpoint_Q,
    find_seed_voxel_P,
    geodesic_distance,
    gradient_ascent_to_medial,
    las_to_ras,
    load_seg_volume,
    resample_arclength,
    two_pass_farthest_endpoints,
    write_centerline_record,
    write_scalismo_landmarks,
)


# ── Retained coordinate / writer tests ────────────────────────────────────────

def test_pir_to_las_is_involution() -> None:
    """`_pir_to_las` is still consumed by `loaders.py` for radiomics."""
    for v in [(1.0, 2.0, 3.0), (-5.0, 0.0, 7.5), (12.0, -8.0, 4.0)]:
        out = _pir_to_las(_pir_to_las(v))
        assert np.allclose(out, v), f"{v} -> {out}"


def test_las_to_ras_flips_x_only() -> None:
    pts = np.array([[1.0, 2.0, 3.0], [-4.0, 5.0, 6.0]], dtype=np.float64)
    out = las_to_ras(pts)
    np.testing.assert_allclose(out[:, 0], [-1.0, 4.0])
    np.testing.assert_allclose(out[:, 1:], pts[:, 1:])
    np.testing.assert_allclose(las_to_ras(out), pts)


def test_resample_preserves_endpoints_and_uniform_arc() -> None:
    """Cubic B-spline resampling preserves endpoints and is near-uniform."""
    # Smooth curved polyline: half-circle of radius 50 mm, 15 raw samples.
    t = np.linspace(0.0, np.pi, 15)
    pts = np.column_stack([50.0 * np.cos(t), 50.0 * np.sin(t), np.zeros_like(t)])

    n = 20
    out = resample_arclength(pts, n_points=n, smoothing=0.0)
    assert out.shape == (n, 3)
    np.testing.assert_allclose(out[0], pts[0], atol=1e-9)
    np.testing.assert_allclose(out[-1], pts[-1], atol=1e-9)

    seg = np.linalg.norm(np.diff(out, axis=0), axis=1)
    target = seg.mean()
    # Cubic-spline arc-length resampling at uniform u (chord-length param)
    # is close to but not exactly uniform-arc; for a smooth curve the error
    # stays under 10%.
    assert seg.max() < target * 1.10
    assert seg.min() > target * 0.90


def test_resample_falls_back_to_linear_for_short_input() -> None:
    pts = np.array([[0, 0, 0], [10, 0, 0], [20, 0, 0]], dtype=np.float64)
    out = resample_arclength(pts, n_points=5)
    assert out.shape == (5, 3)
    np.testing.assert_allclose(out[:, 0], np.linspace(0, 20, 5), atol=1e-9)


def test_scalismo_landmark_writer_round_trips(tmp_path: Path) -> None:
    pts = np.array([[1.0, 2.0, 3.0], [4.5, 5.5, 6.5], [7.0, 8.0, 9.0]])
    out = tmp_path / "x.json"
    write_scalismo_landmarks(pts, out, id_prefix="cl")
    data = json.loads(out.read_text())
    assert isinstance(data, list) and len(data) == 3
    for i, entry in enumerate(data):
        assert "id" in entry and entry["id"].startswith("cl_")
        assert "coordinates" in entry and len(entry["coordinates"]) == 3
        np.testing.assert_allclose(entry["coordinates"], pts[i].tolist())


def test_centerline_record_writer_emits_mesh_frame(tmp_path: Path) -> None:
    pts = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float64)
    raw = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float64)
    out = tmp_path / "rib40_L.json"
    write_centerline_record(pts, raw, out, metadata={"rib_id": "rib40_L"})
    rec = json.loads(out.read_text())
    assert rec["n_points"] == 3
    # Frame name marks "no affine translation applied", matching STL meshes.
    assert rec["frame"] == "RAS_mesh"
    assert rec["raw_n_points"] == 2
    assert rec["rib_id"] == "rib40_L"
    assert rec["arc_length_mm"] == pytest.approx(2.0)
    assert "raw_points" in rec


# ── Skeleton algorithm tests ──────────────────────────────────────────────────

def test_bbox_crop_round_trip() -> None:
    """Voxel positions survive crop + offset addition unchanged."""
    mask = np.zeros((10, 10, 10), dtype=bool)
    mask[3, 4, 5] = True
    mask[7, 6, 2] = True

    (sub,), offset = bbox_crop((mask,), pad=1)
    # Cropped should contain exactly the 2 True voxels.
    assert sub.sum() == 2

    coords_sub = np.argwhere(sub)
    coords_orig = coords_sub + offset
    # Recovered originals must match the input positions (order-agnostic).
    expected = {(3, 4, 5), (7, 6, 2)}
    got = {tuple(c) for c in coords_orig}
    assert got == expected


def test_bbox_crop_handles_multi_mask() -> None:
    """Bounding box covers the union of all input masks."""
    rib = np.zeros((10, 10, 10), dtype=bool)
    vert = np.zeros((10, 10, 10), dtype=bool)
    rib[5, 5, 5] = True
    vert[0, 0, 0] = True  # forces bbox to span the full volume

    (sub_rib, sub_vert), offset = bbox_crop((rib, vert), pad=0)
    assert sub_rib.shape == sub_vert.shape
    assert tuple(offset) == (0, 0, 0)
    # Bbox lower corner is (0,0,0); upper is (5,5,5) inclusive → shape (6,6,6).
    assert sub_rib.shape == (6, 6, 6)


def test_find_seed_voxel_P_picks_costovertebral_end() -> None:
    """P must be the rib voxel nearest to the paired vertebra."""
    rib = np.zeros((20, 5, 5), dtype=bool)
    rib[5:15, 2, 2] = True  # 10-voxel rib line at i in [5..14]

    vert = np.zeros_like(rib)
    vert[2, 2, 2] = True  # adjacent to the i=5 end (3 voxels away)

    P = find_seed_voxel_P(rib, vert, zooms=np.array([1.0, 1.0, 1.0]))
    assert tuple(P) == (5, 2, 2), (
        f"P should be the rib voxel nearest the vertebra, got {tuple(P)}"
    )


def test_find_seed_voxel_P_falls_back_when_vert_missing() -> None:
    """Without a vertebra mask, P is the most posterior rib voxel (min dim-1)."""
    rib = np.zeros((10, 10, 10), dtype=bool)
    rib[5, 7, 3] = True
    rib[5, 2, 3] = True  # smaller dim-1 — anatomical "most posterior"
    rib[5, 9, 3] = True
    vert = np.zeros_like(rib)

    P = find_seed_voxel_P(rib, vert, zooms=np.array([1.0, 1.0, 1.0]))
    assert P[1] == 2


def test_gradient_ascent_reaches_local_max() -> None:
    """Greedy 26-neighbour climb terminates at the EDT peak."""
    # Build a quadratic bowl-shaped EDT with a single peak at (2, 2, 2).
    ii, jj, kk = np.indices((5, 5, 5))
    edt = -((ii - 2) ** 2 + (jj - 2) ** 2 + (kk - 2) ** 2).astype(float)

    end = gradient_ascent_to_medial(edt, np.array([0, 0, 0]), max_steps=20)
    assert tuple(end) == (2, 2, 2)


def test_gradient_ascent_already_at_max_returns_start() -> None:
    """If we start at the local max, ascent returns the start position."""
    ii, jj, kk = np.indices((5, 5, 5))
    edt = -((ii - 2) ** 2 + (jj - 2) ** 2 + (kk - 2) ** 2).astype(float)

    end = gradient_ascent_to_medial(edt, np.array([2, 2, 2]), max_steps=20)
    assert tuple(end) == (2, 2, 2)


def test_geodesic_and_far_endpoint_on_l_shape() -> None:
    """L-shaped 1-voxel-wide path: Q lands at the far end with arc-length distance."""
    mask = np.zeros((20, 20, 5), dtype=bool)
    mask[0:10, 0, 2] = True   # 10 voxels along x at (0..9, 0, 2)
    mask[9, 0:10, 2] = True   # 10 voxels along y at (9, 0..9, 2)
    # Total 19 unique voxels; (9, 0, 2) is the corner.

    zooms = np.array([1.0, 1.0, 1.0])
    source = np.array([0, 0, 2])
    geo = geodesic_distance(mask, source, zooms)

    Q = find_far_endpoint_Q(geo, mask)
    assert tuple(Q) == (9, 9, 2)

    # 18 cardinal steps from (0,0,2) via (9,0,2) to (9,9,2). MCP_Geometric
    # may include or exclude the source's own cost — accept either.
    assert 17.0 <= geo[9, 9, 2] <= 19.5


def test_find_far_endpoint_Q_raises_when_disconnected() -> None:
    """Isolated source (no in-mask neighbours) yields no finite geodesic."""
    mask = np.zeros((5, 5, 5), dtype=bool)
    mask[2, 2, 2] = True  # single isolated voxel
    geo = np.full(mask.shape, np.inf)
    geo[2, 2, 2] = 0.0

    # Isolated source: masked argmax returns the source itself with distance 0.
    # find_far_endpoint_Q only raises when no rib voxel has any finite geodesic.
    Q = find_far_endpoint_Q(geo, mask)
    assert tuple(Q) == (2, 2, 2)  # source itself

    # Make geodesic genuinely empty (everywhere inf, no finite values).
    geo_empty = np.full(mask.shape, np.inf)
    with pytest.raises(ValueError):
        find_far_endpoint_Q(geo_empty, mask)


def test_centered_path_stays_inside_thick_tube() -> None:
    """The cost-weighted path hugs medial / thick voxels in a fat tube."""
    # 5x5 cross-section, 20 voxels long.
    mask = np.zeros((30, 11, 11), dtype=bool)
    mask[5:25, 3:8, 3:8] = True  # 20 x 5 x 5 = 500 voxels

    from scipy.ndimage import distance_transform_edt
    zooms = np.array([1.0, 1.0, 1.0])
    edt = distance_transform_edt(mask, sampling=tuple(zooms))

    # Anchor at both ends along the centerline.
    P = np.array([5, 5, 5])
    Q = np.array([24, 5, 5])
    path = centered_shortest_path(mask, edt, P, Q, zooms, alpha=2.0, eps=1.0)

    # Every path voxel inside the mask.
    for i, j, k in path:
        assert mask[i, j, k], f"path voxel ({i},{j},{k}) outside mask"

    # Every path voxel sits in a thick region — EDT well above the mean.
    edt_along_path = np.array([edt[i, j, k] for i, j, k in path])
    mean_edt_in_mask = edt[mask].mean()
    assert edt_along_path.min() >= 0.5 * mean_edt_in_mask, (
        f"path strays into thin regions: min EDT on path = "
        f"{edt_along_path.min():.2f}, mean EDT in mask = {mean_edt_in_mask:.2f}"
    )


def test_two_pass_endpoints_robust_to_C_shift() -> None:
    """A,B endpoints are stable across small shifts of the reference C.

    The graph-diameter trick (two geodesic passes) means a noisy C only
    changes which end gets labelled proximal vs distal — never the
    geometric positions of A, B.
    """
    # 1-voxel-wide L-shaped rib (19 voxels total), like the geodesic test.
    mask = np.zeros((20, 20, 5), dtype=bool)
    mask[0:10, 0, 2] = True
    mask[9, 0:10, 2] = True
    zooms = np.array([1.0, 1.0, 1.0])

    # Try several plausible C positions (all somewhere near one end).
    candidate_Cs = [
        np.array([0, 0, 2]),  # the proximal endpoint itself
        np.array([1, 0, 2]),  # one voxel along the proximal arm
        np.array([2, 0, 2]),  # two voxels along
    ]
    results = [two_pass_farthest_endpoints(mask, c, zooms) for c in candidate_Cs]

    # All three runs should converge on the same diameter endpoints:
    # one at (0, 0, 2), the other at (9, 9, 2). Labelling (A vs B) may
    # flip depending on which end C sits closer to.
    ends = [tuple(sorted([tuple(a), tuple(b)])) for a, b in results]
    assert len(set(ends)) == 1, f"endpoints unstable across C: {ends}"
    expected = tuple(sorted([(0, 0, 2), (9, 9, 2)]))
    assert ends[0] == expected


def test_extract_centerline_from_nifti_synthetic() -> None:
    """End-to-end: synthetic 50³ volume with a straight rib + adjacent vertebra."""
    data = np.zeros((50, 50, 50), dtype=np.int32)
    # Rib: 40-voxel straight line at j=5..7, k=5..7 (3x3 cross-section).
    data[5:45, 5:8, 5:8] = 40
    # Vertebra: adjacent block at i in [0, 5).
    data[0:5, 5:8, 5:8] = 8

    rib_mask = data == 40
    vert_mask = data == 8
    zooms = np.array([1.0, 1.0, 1.0])

    result = extract_centerline_from_nifti(
        side_rib_mask=rib_mask, vert_mask=vert_mask, zooms=zooms,
        n_points=20,
    )
    assert result is not None
    pts = result["points_mesh_frame"]
    assert pts.shape == (20, 3)

    # First point near (5, 6, 6) — closest medial voxel to the vertebra.
    # Last point near (44, 6, 6) — far end of the tube.
    assert abs(pts[0, 0] - 5) <= 1.5
    assert abs(pts[-1, 0] - 44) <= 1.5
    np.testing.assert_allclose(pts[0, 1:], [6, 6], atol=1.5)
    np.testing.assert_allclose(pts[-1, 1:], [6, 6], atol=1.5)

    # Arc length ≈ 39 mm. Allow ±15% for spline / endpoint slack.
    assert 33.0 <= result["arc_length_mm"] <= 45.0


def test_mesh_frame_has_no_translation_offset(tmp_path: Path) -> None:
    """Regression: nonzero NIfTI affine translation must NOT leak into output."""
    data = np.zeros((50, 50, 50), dtype=np.int32)
    data[5:45, 5:8, 5:8] = 40
    data[0:5, 5:8, 5:8] = 8

    # Build an in-memory NIfTI with a deliberately large affine translation.
    # If the translation leaked, output coordinates would be in [100, 150]
    # rather than [0, 50).
    affine = np.eye(4)
    affine[:3, 3] = [100.0, 200.0, 50.0]
    img = nib.Nifti1Image(data, affine)
    nifti_path = tmp_path / "translated.nii.gz"
    nib.save(img, str(nifti_path))

    loaded, zooms = load_seg_volume(nifti_path)
    rib_mask = loaded == 40
    vert_mask = loaded == 8

    result = extract_centerline_from_nifti(
        side_rib_mask=rib_mask, vert_mask=vert_mask, zooms=zooms,
        n_points=20,
    )
    assert result is not None
    pts = result["points_mesh_frame"]

    # All coordinates must live inside the volume's mesh frame (~[0, 50)),
    # NOT shifted by the affine translation.
    assert pts.max() < 50.0, (
        f"affine translation leaked: max coord = {pts.max():.2f}, "
        f"would be > 100 if translation was applied"
    )
    assert pts.min() >= 0.0
