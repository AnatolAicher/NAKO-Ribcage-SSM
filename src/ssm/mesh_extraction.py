"""Per-rib triangle mesh extraction from NIfTI segmentations.

Pipeline per rib
----------------
1. Load multi-label NIfTI; reorient to RAS via ``nib.as_closest_canonical``.
2. Isolate one rib-level label.
3. Connected-component analysis (``scipy.ndimage.label``) to assign left/right
   by dim-0 centroid (lower = anatomical left in RAS).
4. Marching cubes at ``level=0.5`` with voxel spacing → vertices in mm.
5. Taubin smoothing via PyVista.
6. Quadric decimation to ``MESH_TARGET_FACES_PER_RIB`` triangles.
7. Save as binary STL.

Output: one STL per rib identity, ``{patient_id}_rib{label}_{L|R}.stl``,
written to ``<stl_dir>/<pid // 1000>/<pid>/`` (see
``utils.run_dir.patient_stl_dir``).
"""
from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import nibabel as nib
import numpy as np
import pyvista as pv
from scipy.ndimage import label as ndlabel
from skimage.measure import marching_cubes

from settings import (
    MESH_N_WORKERS,
    MESH_TARGET_FACES_PER_RIB,
    MESH_TAUBIN_ITER,
    MESH_TAUBIN_PASS_BAND,
)
from ssm.pca_surface import RIB_LABELS
from utils.rib_labels import seg_to_anatomical
from utils.run_dir import patient_stl_dir

logger = logging.getLogger(__name__)


# ── Connected-component splitting ────────────────────────────────────────────

def split_rib_components(data: np.ndarray, rib_label: int) -> dict:
    """Split a rib-level label into left and right connected components.

    Sides assigned by dim-0 centroid in RAS (lower = anatomical left).

    Returns
    -------
    dict with keys ``n_components``, ``left_mask``, ``right_mask``,
    ``left_voxels``, ``right_voxels``, ``left_centroid_dim0``,
    ``right_centroid_dim0``, ``warning``.
    """
    mask = (data == rib_label).astype(np.int32)
    labeled, n_components = ndlabel(mask)

    result: dict = {"n_components": n_components, "warning": None}

    if n_components == 0:
        result["warning"] = (
            f"Rib {seg_to_anatomical(rib_label)} (label {rib_label}): "
            f"no voxels found"
        )
        result["left_mask"] = np.zeros_like(mask, dtype=bool)
        result["right_mask"] = np.zeros_like(mask, dtype=bool)
        result["left_voxels"] = 0
        result["right_voxels"] = 0
        result["left_centroid_dim0"] = float("nan")
        result["right_centroid_dim0"] = float("nan")
        return result

    comp_info = []
    for c in range(1, n_components + 1):
        comp_mask = labeled == c
        dim0_coords = np.where(comp_mask)[0]
        centroid_dim0 = float(dim0_coords.mean())
        comp_info.append((c, centroid_dim0, int(comp_mask.sum()), comp_mask))

    # Sort by dim-0 centroid: lowest = left (in RAS), highest = right.
    comp_info.sort(key=lambda x: x[1])

    if n_components == 1:
        # Side ambiguous; resolved later by ``resolve_ambiguous_sides``.
        _, centroid, nvox, cmask = comp_info[0]
        result["ambiguous"] = True
        result["ambiguous_mask"] = cmask
        result["ambiguous_centroid_dim0"] = centroid
        result["ambiguous_voxels"] = nvox
        result["right_mask"] = cmask
        result["left_mask"] = np.zeros_like(mask, dtype=bool)
        result["right_voxels"] = nvox
        result["left_voxels"] = 0
        result["right_centroid_dim0"] = centroid
        result["left_centroid_dim0"] = float("nan")
        return result

    if n_components > 2:
        # Keep the two largest; the audit table records the warning.
        comp_info.sort(key=lambda x: x[2], reverse=True)
        two_largest = sorted(comp_info[:2], key=lambda x: x[1])
        comp_info = two_largest

    _, l_centroid, l_nvox, l_mask = comp_info[0]
    _, r_centroid, r_nvox, r_mask = comp_info[1]

    result["right_mask"] = r_mask
    result["left_mask"] = l_mask
    result["right_voxels"] = r_nvox
    result["left_voxels"] = l_nvox
    result["right_centroid_dim0"] = r_centroid
    result["left_centroid_dim0"] = l_centroid
    return result


def resolve_ambiguous_sides(splits: dict[int, dict]) -> None:
    """Resolve single-component levels by comparing to other levels' centroids.

    Compares each ambiguous component's dim-0 centroid to the midpoint of the
    median right / median left centroids from cleanly split levels. Mutates
    ``splits`` in place.
    """
    right_centroids = []
    left_centroids = []
    for split in splits.values():
        if split.get("ambiguous"):
            continue
        if not np.isnan(split["right_centroid_dim0"]):
            right_centroids.append(split["right_centroid_dim0"])
        if not np.isnan(split["left_centroid_dim0"]):
            left_centroids.append(split["left_centroid_dim0"])

    if not right_centroids or not left_centroids:
        # No reference available; ambiguous ribs default to right.
        logger.warning(
            "No cleanly split levels to use as reference; "
            "ambiguous ribs default to right"
        )
        return

    median_right = float(np.median(right_centroids))
    median_left = float(np.median(left_centroids))
    midpoint = (median_right + median_left) / 2.0

    for label, split in splits.items():
        if not split.get("ambiguous"):
            continue

        centroid = split["ambiguous_centroid_dim0"]
        mask = split["ambiguous_mask"]
        nvox = split["ambiguous_voxels"]
        empty = np.zeros_like(mask, dtype=bool)

        if centroid <= midpoint:
            assigned_side = "L"
            split["left_mask"] = mask
            split["right_mask"] = empty
            split["left_voxels"] = nvox
            split["right_voxels"] = 0
            split["left_centroid_dim0"] = centroid
            split["right_centroid_dim0"] = float("nan")
        else:
            assigned_side = "R"
            split["right_mask"] = mask
            split["left_mask"] = empty
            split["right_voxels"] = nvox
            split["left_voxels"] = 0
            split["right_centroid_dim0"] = centroid
            split["left_centroid_dim0"] = float("nan")

        split["warning"] = (
            f"Rib {seg_to_anatomical(label)} (label {label}): single "
            f"component assigned to {assigned_side} "
            f"(centroid={centroid:.1f}, midpoint={midpoint:.1f})"
        )
        logger.debug(
            "Rib %d (label %d): single component → %s (centroid=%.1f, midpoint=%.1f)",
            seg_to_anatomical(label), label, assigned_side, centroid, midpoint,
        )


# ── Single-rib mesh extraction ───────────────────────────────────────────────

def extract_single_rib_mesh(
    mask: np.ndarray,
    zooms: np.ndarray,
    target_faces: int = MESH_TARGET_FACES_PER_RIB,
) -> pv.PolyData | None:
    """Marching cubes + Taubin smoothing + decimation on a binary mask for one rib.

    Returns ``None`` if the mask has too few voxels for a valid surface.
    """
    if mask.sum() < 10:
        return None

    verts, faces, _, _ = marching_cubes(mask.astype(np.float32), level=0.5, spacing=zooms)

    faces_pv = np.column_stack([np.full(len(faces), 3, dtype=np.int32), faces])
    mesh = pv.PolyData(verts.astype(np.float32), faces_pv.ravel())

    # Taubin alternates shrink/unshrink passes, so it de-staircases the surface
    # without the volume loss of Laplacian smoothing.
    mesh = mesh.smooth_taubin(
        n_iter=MESH_TAUBIN_ITER,
        pass_band=MESH_TAUBIN_PASS_BAND,
        feature_smoothing=False,
        boundary_smoothing=False,
    )

    if mesh.n_cells > target_faces:
        ratio = 1.0 - target_faces / mesh.n_cells
        mesh = mesh.decimate(ratio, progress_bar=False)

    mesh = mesh.triangulate()
    return mesh


def extract_per_rib(
    nifti_path: str | Path,
    rib_labels: list[int] | None = None,
    target_faces: int = MESH_TARGET_FACES_PER_RIB,
) -> dict:
    """Extract individual rib meshes from a per-rib-level NIfTI segmentation.

    Parameters
    ----------
    nifti_path
        Path to the seg-vert-rib NIfTI file.
    rib_labels
        Rib-level label values (default: ``RIB_LABELS`` = 40–51).

    Returns
    -------
    dict with keys:
      ``meshes`` — dict mapping ``(label, side)`` → ``pv.PolyData``;
                   ``side`` is ``"L"`` (left) or ``"R"`` (right).
      ``audit``  — list of dicts, one per rib label, with component counts,
                   voxel counts, centroid positions, per-side mesh volume
                   against the source voxel volume, and any warnings.
    """
    if rib_labels is None:
        rib_labels = RIB_LABELS

    img = nib.load(str(nifti_path))
    # Reorient to RAS so the voxel frame is consistent across patients.
    img = nib.as_closest_canonical(img)
    data = np.asarray(img.dataobj, dtype=np.int32)
    zooms = np.array(img.header.get_zooms()[:3], dtype=float)
    voxel_mm3 = float(np.prod(zooms))

    meshes: dict[tuple[int, str], pv.PolyData] = {}
    audit: list[dict] = []

    # Pass 1: split all labels into components.
    splits: dict[int, dict] = {label: split_rib_components(data, label) for label in rib_labels}

    # Pass 2: resolve single-component levels using neighbours as reference.
    resolve_ambiguous_sides(splits)

    for label in rib_labels:
        split = splits[label]

        audit_entry = {
            "rib_label": label,
            "n_components": split["n_components"],
            "left_voxels": split["left_voxels"],
            "right_voxels": split["right_voxels"],
            "left_centroid_dim0": split["left_centroid_dim0"],
            "right_centroid_dim0": split["right_centroid_dim0"],
            "warning": split["warning"],
        }

        for side, side_mask in [("R", split["right_mask"]), ("L", split["left_mask"])]:
            audit_entry[f"{side}_voxel_mm3"] = float(side_mask.sum()) * voxel_mm3
            audit_entry[f"{side}_mesh_mm3"] = float("nan")
            audit_entry[f"{side}_vol_pct"] = float("nan")
            if side_mask.any():
                mesh = extract_single_rib_mesh(side_mask, zooms, target_faces=target_faces)
                if mesh is not None:
                    meshes[(label, side)] = mesh
                    audit_entry[f"{side}_faces"] = mesh.n_cells
                    audit_entry[f"{side}_points"] = mesh.n_points
                    audit_entry[f"{side}_mesh_mm3"] = float(mesh.volume)
                    audit_entry[f"{side}_vol_pct"] = (
                        100.0 * float(mesh.volume) / audit_entry[f"{side}_voxel_mm3"]
                    )
                else:
                    audit_entry[f"{side}_faces"] = 0
                    audit_entry[f"{side}_points"] = 0
                    if audit_entry["warning"] is None:
                        audit_entry["warning"] = (
                            f"Rib {seg_to_anatomical(label)} {side} "
                            f"(label {label}): mesh extraction failed "
                            f"(too few voxels?)"
                        )
            else:
                audit_entry[f"{side}_faces"] = 0
                audit_entry[f"{side}_points"] = 0

        audit.append(audit_entry)

    return {"meshes": meshes, "audit": audit}


# ── Worker / batch ───────────────────────────────────────────────────────────

def extract_per_rib_and_save(
    args: tuple[int, str, str, list[int] | None, int],
) -> tuple[int, bool, str, list[dict]]:
    """Worker function: extract per-rib meshes for one patient and save to STL."""
    pid, nifti_path, stl_dir, rib_labels, target_faces = args
    audit: list[dict] = []
    try:
        stl_dir_p = Path(stl_dir)
        pdir = patient_stl_dir(stl_dir_p, pid)
        if rib_labels is None:
            rib_labels = RIB_LABELS
        # Path-based skip: STLs decimated at a different ``target_faces`` are
        # reused. Clear the dir to force re-extraction.
        expected_files = [
            pdir / f"{pid}_rib{lab}_{side}.stl"
            for lab in rib_labels
            for side in ("L", "R")
        ]
        if all(f.exists() for f in expected_files):
            return pid, True, "skipped", []

        result = extract_per_rib(nifti_path, rib_labels=rib_labels, target_faces=target_faces)
        audit = result["audit"]

        pdir.mkdir(parents=True, exist_ok=True)
        for (label, side), mesh in result["meshes"].items():
            out_path = pdir / f"{pid}_rib{label}_{side}.stl"
            mesh.save(str(out_path), binary=True)

        warnings = [e["warning"] for e in audit if e["warning"] is not None]
        if warnings:
            return pid, True, f"ok (warnings: {'; '.join(warnings)})", audit
        return pid, True, "ok", audit
    except (ValueError, RuntimeError, OSError) as exc:
        return pid, False, str(exc), audit


def batch_extract_per_rib(
    patient_ids: list[int],
    nifti_pattern: str,
    stl_dir: str | Path,
    n_workers: int = MESH_N_WORKERS,
    rib_labels: list[int] | None = None,
    target_faces: int = MESH_TARGET_FACES_PER_RIB,
) -> dict:
    """Extract per-rib meshes for all patients in parallel.

    Parameters
    ----------
    patient_ids
        List of patient IDs.
    nifti_pattern
        Format string with ``{patient_id}`` and ``{block}`` placeholders
        pointing to the seg-vert-rib NIfTI files.
    stl_dir
        Output directory for per-rib STL files.
    n_workers
        Parallel workers.
    rib_labels
        Rib-level label values (default: ``RIB_LABELS`` = 40–51).
    target_faces
        Quadric decimation target per rib mesh.

    Returns
    -------
    dict with keys ``n_ok``, ``n_skipped``, ``n_failed``, ``failed_ids``,
    ``audit`` (per-patient audit records).
    """
    stl_dir = Path(stl_dir)
    stl_dir.mkdir(parents=True, exist_ok=True)

    jobs = []
    for pid in patient_ids:
        block = pid // 1000
        npath = nifti_pattern.format(patient_id=pid, block=block)
        jobs.append((pid, npath, str(stl_dir), rib_labels, target_faces))

    n_ok = n_skipped = n_failed = 0
    failed_ids: list[int] = []
    failed_msgs: list[str] = []
    all_audit: dict[int, list[dict]] = {}
    total = len(jobs)

    logger.info(f"Starting per-rib mesh extraction: {total:,} patients, {n_workers} workers")

    from utils.logging import progress

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(extract_per_rib_and_save, j): j[0] for j in jobs}
        bar = progress(as_completed(futures), total=total,
                       desc="mesh extraction", unit="patient")
        for fut in bar:
            pid, ok, msg, audit = fut.result()
            if audit:
                all_audit[pid] = audit
            if not ok:
                n_failed += 1
                failed_ids.append(pid)
                failed_msgs.append(f"  {pid}: {msg}")
            elif msg == "skipped":
                n_skipped += 1
            else:
                n_ok += 1
                if "warnings" in msg:
                    logger.warning(f"  {pid}: {msg}")
            bar.set_postfix(ok=n_ok, skip=n_skipped, fail=n_failed)

    logger.info(
        f"Per-rib batch extract complete: ok={n_ok} skipped={n_skipped} failed={n_failed}"
    )
    if failed_msgs:
        shown = failed_msgs[:20]
        for m in shown:
            logger.warning(m)
        if len(failed_msgs) > 20:
            logger.warning(f"  … and {len(failed_msgs) - 20} more failures (see log JSON)")

    warning_counts: dict[str, int] = {}
    for pid_audit in all_audit.values():
        for entry in pid_audit:
            if entry["warning"] is not None:
                key = entry["warning"].split(":")[0]
                warning_counts[key] = warning_counts.get(key, 0) + 1
    if warning_counts:
        logger.warning(f"Audit warning summary: {warning_counts}")

    return dict(
        n_ok=n_ok, n_skipped=n_skipped, n_failed=n_failed,
        failed_ids=failed_ids, audit=all_audit,
    )
