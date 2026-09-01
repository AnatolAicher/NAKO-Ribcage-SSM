"""Load registered per-rib meshes, GPA-align, run PCA, build score DataFrame.

Each patient has 24 STL files (one per rib identity, ``{pid}_rib{label}_{L|R}.stl``)
sharing the same per-rib template topology – so GPA reduces to translation +
rotation alignment without any resampling.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyvista as pv
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.decomposition import PCA

from settings import SSM_LOAD_N_WORKERS
from ssm.gpa import gpa as _gpa
from utils.run_dir import patient_stl_dir

logger = logging.getLogger(__name__)


# Canonical rib ordering: 12 vertebral levels × 2 sides = 24 identities.
# RIB_LABELS = NIfTI segmentation values 40–51 ≡ anatomical ribs 1–12.
RIB_LABELS = list(range(40, 52))
RIB_SIDES  = ["L", "R"]
RIB_ORDER  = [(lab, side) for lab in RIB_LABELS for side in RIB_SIDES]


# ── Load per-rib registered meshes ───────────────────────────────────────────

def _load_one_patient(
    pid: int,
    reg_stl_dir: Path,
    rib_order: list[tuple[int, str]],
) -> tuple[int, list[np.ndarray] | None]:
    """Load all rib STLs for one patient → ``(pid, [pts_array, ...]) | (pid, None)``."""
    rib_pts: list[np.ndarray] = []
    pdir = patient_stl_dir(reg_stl_dir, pid)
    for lab, side in rib_order:
        stl_path = pdir / f"{pid}_rib{lab}_{side}.stl"
        if not stl_path.exists():
            return pid, None
        try:
            mesh = pv.read(str(stl_path))
            pts = np.array(mesh.points, dtype=np.float32)
        except Exception as exc:  # VTK/PyVista readers raise varied types
            logger.warning(f"Could not load {stl_path.name}: {exc}")
            return pid, None
        rib_pts.append(pts)
    return pid, rib_pts


def load_registered_meshes_per_rib(
    reg_stl_dir: str | Path,
    meta_df: pd.DataFrame,
    patient_ids: list[int] | None = None,
    rib_labels: list[int] | None = None,
    sides: list[str] | None = None,
    n_workers: int | None = None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, list[int]]:
    """Load per-rib registered STLs and concatenate into one shape per patient.

    Each patient has up to 24 STL files named ``{pid}_rib{label}_{L|R}.stl``.
    Only patients with a complete set of ribs (and matching metadata) are
    included.  Vertices from each rib are concatenated in a fixed order
    (label 40 L, 40 R, 41 L, 41 R, …, 51 L, 51 R) to produce a single
    ``(n_pts_total, 3)`` shape per patient.

    Parameters
    ----------
    reg_stl_dir
        Directory containing the registered per-rib STL files.
    meta_df
        Metadata DataFrame (must contain a ``patient_id`` column).
    patient_ids
        Optional patient-ID filter.
    rib_labels
        NIfTI segmentation labels to include (default: 40–51, the
        on-disk encoding of anatomical ribs 1–12).
    sides
        Sides to include (default: ``["L", "R"]``).
    n_workers
        Parallel STL-read threads (default: ``SSM_LOAD_N_WORKERS``).

    Returns
    -------
    shapes      : ``(N, n_pts_total, 3)`` float32.
    patient_ids : ``(N,)`` int.
    meta_sub    : DataFrame aligned to ``shapes``.
    rib_offsets : list of vertex offsets per rib identity (for splitting the
                  concatenated vector back into individual ribs).
    """
    reg_stl_dir = Path(reg_stl_dir)
    meta_pids   = set(meta_df["patient_id"].values)
    pid_filter  = set(patient_ids) if patient_ids is not None else None

    _labels = rib_labels if rib_labels is not None else RIB_LABELS
    _sides  = sides if sides is not None else RIB_SIDES
    rib_order = [(lab, side) for lab in _labels for side in _sides]
    n_ribs = len(rib_order)

    # Discover candidate pids from the sharded ``<reg_stl_dir>/<block>/<pid>/``
    # layout; pid is the leaf directory name.
    candidate_pids: set[int] = set()
    for block_dir in reg_stl_dir.iterdir():
        if not block_dir.is_dir() or block_dir.name.startswith("._"):
            continue
        for pid_dir in block_dir.iterdir():
            if pid_dir.is_dir() and pid_dir.name.isdigit():
                candidate_pids.add(int(pid_dir.name))

    if pid_filter is not None:
        candidate_pids &= pid_filter
    candidate_pids &= meta_pids

    logger.info(
        f"Per-rib loading: {len(candidate_pids):,} candidate patients, "
        f"{n_ribs} rib identities each"
    )

    shapes_list: list[np.ndarray] = []
    pid_list:    list[int] = []
    rib_offsets: list[int] = []
    ref_n_pts: dict[tuple[int, str], int] = {}
    skipped = 0

    # Parallel STL read; PyVista/VTK release the GIL during I/O + parsing,
    # so threads scale without per-patient pickling cost.
    n_workers = SSM_LOAD_N_WORKERS if n_workers is None else int(n_workers)
    sorted_pids = sorted(candidate_pids)
    n_pids = len(sorted_pids)
    t0 = time.monotonic()
    logger.info(
        f"  Reading STLs in parallel (n_workers={n_workers}, threads), "
        f"{n_pids:,} patients × {n_ribs} ribs = {n_pids * n_ribs:,} files…"
    )

    results_by_pid: dict[int, list[np.ndarray] | None] = {}
    log_every = max(1, n_pids // 50)
    last_log_t = t0
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        fut_to_pid = {
            pool.submit(_load_one_patient, pid, reg_stl_dir, rib_order): pid
            for pid in sorted_pids
        }
        for i, fut in enumerate(as_completed(fut_to_pid), start=1):
            pid_done, rib_pts = fut.result()
            results_by_pid[pid_done] = rib_pts
            now = time.monotonic()
            if (i % log_every == 0) or (i == n_pids) or (now - last_log_t >= 30.0):
                elapsed = now - t0
                rate = i / elapsed if elapsed > 0 else 0.0
                eta = (n_pids - i) / rate if rate > 0 else 0.0
                logger.info(
                    f"    read {i:,}/{n_pids:,} patients "
                    f"({i * 100 / n_pids:.1f}%) – "
                    f"{elapsed / 60:.1f}m elapsed, ETA {eta / 60:.1f}m"
                )
                last_log_t = now

    # Deterministic pid-sorted order; the validation pass below takes the
    # lowest valid pid as the vertex-count reference.
    results = [(pid, results_by_pid[pid]) for pid in sorted_pids]
    logger.info(
        f"  STL read pass done in {(time.monotonic() - t0) / 60:.1f}m"
    )

    # Validate against the first valid patient's vertex counts.
    for pid, rib_pts in results:
        if rib_pts is None:
            skipped += 1
            continue
        complete = True
        for (lab, side), pts in zip(rib_order, rib_pts):
            key = (lab, side)
            if key not in ref_n_pts:
                ref_n_pts[key] = len(pts)
            elif len(pts) != ref_n_pts[key]:
                logger.warning(
                    f"PID {pid} rib{lab}_{side}: {len(pts)} pts "
                    f"(expected {ref_n_pts[key]}) – skipping patient"
                )
                complete = False
                break

        if not complete:
            skipped += 1
            continue

        concat = np.concatenate(rib_pts, axis=0)  # (n_pts_total, 3)
        shapes_list.append(concat)
        pid_list.append(pid)

        if not rib_offsets:
            offset = 0
            for pts in rib_pts:
                rib_offsets.append(offset)
                offset += len(pts)

    if not shapes_list:
        raise RuntimeError(
            f"No patients with a complete set of {n_ribs} registered per-rib meshes "
            f"found in {reg_stl_dir}"
        )

    shapes = np.stack(shapes_list, axis=0)
    pids   = np.array(pid_list, dtype=np.int64)
    meta_sub = meta_df.set_index("patient_id").loc[pids].reset_index()

    logger.info(
        f"Loaded {len(shapes):,} patients ({skipped:,} skipped)  "
        f"shape={shapes.shape}  ({n_ribs} ribs × ~{shapes.shape[1]//n_ribs} pts/rib)"
    )
    return shapes, pids, meta_sub, rib_offsets


# ── GPA ──────────────────────────────────────────────────────────────────────

def run_gpa(shapes: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """GPA-align a stack of shapes (translation + rotation; scale retained).

    Parameters
    ----------
    shapes
        ``(N, n_pts, 3)``.

    Returns
    -------
    aligned     : ``(N, n_pts, 3)``.
    mean_shape  : ``(n_pts, 3)``.
    rms_history : ``(n_iterations,)`` – relative mean-shape change at
                  each iteration (consumed by ``plot_gpa_convergence``).
    """
    aligned, mean_shape, rms_history = _gpa(shapes)
    logger.info(f"GPA complete.  aligned shape={aligned.shape}  ({len(rms_history)} iterations)")
    return aligned, mean_shape, rms_history


# ── PCA ──────────────────────────────────────────────────────────────────────

PCA_RANDOM_SEED = 42


def fit_pca(
    shapes_aligned: np.ndarray,
    variance_threshold: float = 0.95,
) -> tuple[PCA, np.ndarray, np.ndarray, np.ndarray]:
    """Fit randomised-SVD PCA on flattened, GPA-aligned shapes.

    Fits ``min(N-1, 500)`` components, then truncates to the smallest ``k``
    covering ``variance_threshold`` of total variance.

    Returns
    -------
    pca      : sklearn PCA fitted to the truncated component set.
    scores   : ``(N, n_components)`` PC scores.
    mean_vec : ``(D,)`` mean shape vector (= ``pca.mean_``).
    X_flat   : ``(N, D)`` flattened input matrix used for the fit (float32).
    """
    N = len(shapes_aligned)
    # float32 is sufficient and avoids doubling memory at N≈30k, D≈290k.
    X = shapes_aligned.reshape(N, -1).astype(np.float32, copy=False)

    k_fit = min(N - 1, 500)
    pca = PCA(n_components=k_fit, svd_solver="randomized", random_state=PCA_RANDOM_SEED)
    scores_full = pca.fit_transform(X)

    cum_var = np.cumsum(pca.explained_variance_ratio_)
    if cum_var[-1] < variance_threshold:
        logger.warning(
            f"K_fit={k_fit} only reaches {cum_var[-1]:.1%} variance "
            f"(< threshold {variance_threshold:.0%}); raise k_fit in fit_pca."
        )
        k_keep = k_fit
    else:
        k_keep = int(np.searchsorted(cum_var, variance_threshold)) + 1

    pca.components_               = pca.components_[:k_keep]
    pca.explained_variance_       = pca.explained_variance_[:k_keep]
    pca.explained_variance_ratio_ = pca.explained_variance_ratio_[:k_keep]
    pca.singular_values_          = pca.singular_values_[:k_keep]
    pca.n_components_             = k_keep
    scores = scores_full[:, :k_keep]

    cum_after = float(np.cumsum(pca.explained_variance_ratio_)[-1])
    logger.info(
        f"PCA (randomized, K_fit={k_fit}): kept {k_keep} components explain "
        f"{cum_after:.1%} of variance (threshold={variance_threshold:.0%})"
    )

    return pca, scores, pca.mean_, X


def run_pca(
    shapes_aligned: np.ndarray, variance_threshold: float = 0.95,
) -> tuple[PCA, np.ndarray]:
    """Convenience wrapper: fit PCA, return ``(pca, X_flat)``."""
    pca, _, _, X_flat = fit_pca(shapes_aligned, variance_threshold=variance_threshold)
    logger.info(
        f"PCA: {pca.n_components_} components retain "
        f"{pca.explained_variance_ratio_.sum():.3%} variance"
    )
    return pca, X_flat


# ── Build PC-scores DataFrame ────────────────────────────────────────────────

def build_surface_scores_df(
    pca: PCA,
    X_flat: np.ndarray,
    patient_ids: np.ndarray,
    meta_df: pd.DataFrame,
) -> pd.DataFrame:
    """Project shapes onto PCA components and join with metadata.

    Returns a DataFrame with ``patient_id``, ``PC_1`` … ``PC_K``, plus the
    available metadata columns.
    """
    scores   = pca.transform(X_flat)                   # (N, K) – sklearn centers internally
    pc_cols  = [f"PC_{i+1}" for i in range(scores.shape[1])]
    score_df = pd.DataFrame(scores, columns=pc_cols)
    score_df.insert(0, "patient_id", patient_ids)

    meta_cols = ["patient_id", "sex", "age", "bmi", "body_fat_pct",
                 "smoking_status", "pack_years", "height_cm", "weight_kg"]
    available = [c for c in meta_cols if c in meta_df.columns]
    merged = score_df.merge(meta_df[available], on="patient_id", how="left")
    logger.info(f"PC score DataFrame: {merged.shape}")
    return merged
