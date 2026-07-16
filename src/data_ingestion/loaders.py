"""Data loading for the NAKO rib morphology project.

Public API
----------
load_json_analytics(cfg)   -> pd.DataFrame   one row per (patient, vert_level, side)
load_metadata(cfg)         -> pd.DataFrame   one row per patient
merge_datasets(ana, meta)  -> (pd.DataFrame, dict)
"""
from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from data_ingestion.centerline import _pir_to_las
from settings import SSM_LOAD_N_WORKERS

logger = logging.getLogger(__name__)

# ── Shape feature names (pyradiomics) ─────────────────────────────────────────

SHAPE_FEATURES = [
    "original_shape_Elongation",
    "original_shape_Flatness",
    "original_shape_LeastAxisLength",
    "original_shape_MajorAxisLength",
    "original_shape_Maximum2DDiameterColumn",
    "original_shape_Maximum2DDiameterRow",
    "original_shape_Maximum2DDiameterSlice",
    "original_shape_Maximum3DDiameter",
    "original_shape_MeshVolume",
    "original_shape_MinorAxisLength",
    "original_shape_Sphericity",
    "original_shape_SurfaceArea",
    "original_shape_SurfaceVolumeRatio",
    "original_shape_VoxelVolume",   # loaded but excluded from analysis (collinear with MeshVolume)
]

SCALAR_EXTRAS = [
    "rib_length",     # arc length in mm
    "rib_volume",     # raw voxel count (excluded from analysis — use MeshVolume)
    "sr",             # short-rib flag (bool)
    "seg_at_border",  # segmentation touches image border (bool)
    "split_start",
    "split_end",
]

# All shape-like columns saved to the parquet (includes redundant ones for archival).
ALL_SHAPE_COLS = SHAPE_FEATURES + ["rib_length", "rib_volume"]

# Subset used in the statistical analysis.
# Excluded: original_shape_VoxelVolume (r ≈ 1.0 with MeshVolume); rib_volume.
# Retained: both rib_length and original_shape_MajorAxisLength (arc length vs.
# ellipsoid linear axis — different constructs).
ANALYSIS_SHAPE_COLS = [
    c for c in SHAPE_FEATURES if c != "original_shape_VoxelVolume"
] + ["rib_length"]

# ── Internal helpers ──────────────────────────────────────────────────────────

# The rescaled isotropic frame the JSON centerlines are stored in (mm/voxel).
_RESCALED_VOXEL_MM = 0.5


def _parse_side(patient_id: int, vert_level: int, side: str, d: dict) -> dict:
    """Extract one flat row from a single rib/side dict.

    Coordinate conventions:

    * ``start_point`` / ``start_point_coord``: voxel indices in the rescaled
      isotropic frame (0.5 mm/vx).
    * ``path_points_relative_to_start``: in **mm**, axis order **(P, I, R)**.
      Converted to LAS via ``(x, y, z) -> (-z, -y, -x)`` (see
      :func:`data_ingestion.centerline._pir_to_las`).
    * ``rib_volume``: voxel count in the rescaled isotropic frame, so
      ``rib_volume * 0.5**3 ≈ original_shape_MeshVolume``.
    * ``orig_zoom``: the *original* scan voxel size; metadata only.
    """
    row: dict = {"patient_id": patient_id, "vert_level": vert_level, "side": side}

    for feat in SHAPE_FEATURES + SCALAR_EXTRAS:
        row[feat] = d.get(feat)

    # Voxel count → mm³ in the rescaled isotropic frame.
    res = d.get("resolution", _RESCALED_VOXEL_MM)
    raw_vol = d.get("rib_volume")
    row["rib_volume_mm3"] = raw_vol * (res ** 3) if raw_vol is not None else None

    # Reorder centerline path points PIR → LAS via the shared converter.
    pts_pir = d.get("path_points_relative_to_start", []) or []
    row["centerline_mm"] = [list(_pir_to_las(p)) for p in pts_pir]
    row["n_centerline_pts"] = len(pts_pir)

    # Geometry
    row["start_point"] = d.get("start_point")
    row["end_point"] = d.get("end_point")
    row["orig_zoom"] = d.get("orig_zoom")

    return row


def _parse_patient(fpath: str) -> list[dict]:
    with open(fpath) as f:
        d = json.load(f)
    pid = d["pid"]
    rows = []
    for rk in d:
        if not rk.isdigit():
            continue
        vert_level = int(rk)
        for side in ("Right", "Left"):
            if side in d[rk]:
                rows.append(_parse_side(pid, vert_level, side, d[rk][side]))
    return rows


# ── Public loaders ────────────────────────────────────────────────────────────

def _enumerate_json_files(cfg: dict, n_patients: int | None = None) -> list[str]:
    """Build the JSON file list from the two-level ``block/patient_id`` layout.

    Discovery scans ``cfg["paths"]["json_analytics_base"]`` two levels deep
    (block → patient_id); the per-file path is then formatted from
    ``cfg["paths"]["json_analytics_pattern"]``, which must contain
    ``{base}``, ``{block}``, and ``{patient_id}`` placeholders.

    If ``n_patients`` is given, the path list is sorted by integer patient_id
    and truncated to the first N.
    """
    base = Path(cfg["paths"]["json_analytics_base"])
    pattern = cfg["paths"]["json_analytics_pattern"]

    logger.info(f"Scanning {base} for block directories …")
    try:
        block_dirs = sorted(p for p in base.iterdir() if p.is_dir())
    except OSError as exc:
        raise RuntimeError(f"Cannot list base directory {base}: {exc}") from exc

    logger.info(f"  {len(block_dirs)} block directories found")
    pid_paths: list[tuple[int, str]] = []
    for bi, block_dir in enumerate(block_dirs, 1):
        try:
            # All-digit name = patient_id dir; avoids one is_dir() stat per entry.
            entries = sorted(
                e for e in block_dir.iterdir() if e.name.isdigit()
            )
            for patient_dir in entries:
                pid_str = patient_dir.name
                fpath = pattern.format(
                    base=str(base),
                    block=block_dir.name,
                    patient_id=pid_str,
                )
                pid_paths.append((int(pid_str), fpath))
            logger.info(f"  Block {bi}/{len(block_dirs)} ({block_dir.name}): {len(entries)} patients")
        except OSError as exc:
            logger.warning(f"  Cannot scan block {block_dir.name}: {exc}")

    logger.info(f"  {len(pid_paths):,} candidate JSON paths constructed")

    if n_patients is not None:
        pid_paths.sort(key=lambda t: t[0])
        kept = pid_paths[:n_patients]
        logger.info(
            f"  Limiting to first {len(kept):,} patients "
            f"(of {len(pid_paths):,} candidates, sorted by integer PID)"
        )
        return [fp for _, fp in kept]

    return [fp for _, fp in pid_paths]


def load_json_analytics(
    cfg: dict,
    n_patients: int | None = None,
    n_workers: int | None = None,
) -> pd.DataFrame:
    """Discover and parse all per-patient analytic JSON files.

    Returns a tidy DataFrame: one row per ``(patient_id, vert_level, side)``.
    The centerline is stored as a list-of-lists column (``centerline_mm``).
    """
    json_files = _enumerate_json_files(cfg, n_patients=n_patients)

    rows: list[dict] = []
    n_errors = 0
    n_missing = 0
    n_workers = SSM_LOAD_N_WORKERS if n_workers is None else int(n_workers)
    logger.info(f"Beginning parse of {len(json_files):,} paths (workers={n_workers}) …")

    def _parse_safe(fpath: str) -> tuple[list[dict], str | None]:
        try:
            return _parse_patient(fpath), None
        except FileNotFoundError:
            return [], "missing"
        except (OSError, json.JSONDecodeError, ValueError, KeyError) as exc:
            return [], str(exc)

    # Milestone-style progress: print every 1000 patients. Avoids tqdm's
    # one-line-per-update behaviour under piped subprocess (no TTY).
    PROGRESS_INTERVAL = 1000
    total = len(json_files)
    n_done = 0
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_parse_safe, fp): fp for fp in json_files}
        for fut in as_completed(futures):
            result_rows, err = fut.result()
            if err is None:
                rows.extend(result_rows)
            elif err == "missing":
                n_missing += 1
            else:
                logger.error(f"Failed {futures[fut]}: {err}")
                n_errors += 1
            n_done += 1
            if n_done % PROGRESS_INTERVAL == 0 or n_done == total:
                logger.info(
                    f"  Parsed {n_done:,}/{total:,} patients "
                    f"(rows={len(rows):,}, missing={n_missing:,}, errors={n_errors:,})"
                )

    if n_missing:
        logger.info(f"  {n_missing:,} paths had no file (patient has no JSON — expected)")
    if n_errors:
        logger.warning(f"  {n_errors:,} files failed to parse")

    df = pd.DataFrame(rows)
    logger.info(
        f"Analytics: {len(df):,} rib-side rows | "
        f"{df['patient_id'].nunique():,} patients"
    )
    return df


_SENTINEL_CODES = {7775, 7776, 7777}


def load_metadata(cfg: dict) -> pd.DataFrame:
    """Load patient metadata from Excel, applying NAKO sentinel→NaN recoding.

    NAKO sentinel codes:
      ``7775`` = not applicable   (pack_years for never-smokers → 0)
      ``7776`` = not assessable   → NaN
      ``7777`` = not collected    → NaN
    Smoking-status code 4 (Unknown) → NaN.
    """
    path = cfg["paths"]["metadata_table"]
    use_cols = list(cfg["metadata_columns"])

    df = pd.read_excel(path, usecols=use_cols)

    # Never-smokers: recode pack_years sentinel 7775 ("not applicable") to 0
    # *before* the blanket sentinel→NaN replacement, so never-smokers retain
    # pack-years = 0 while 7776/7777 (missing) become NaN.
    if "a_packyears" in df.columns and "a_smok_stat_qn" in df.columns:
        never_mask = df["a_smok_stat_qn"] == 1
        df.loc[never_mask & (df["a_packyears"] == 7775), "a_packyears"] = 0.0

    sentinel_values = list(_SENTINEL_CODES)
    df.replace(sentinel_values, np.nan, inplace=True)

    if "a_smok_stat_qn" in df.columns:
        df["a_smok_stat_qn"] = df["a_smok_stat_qn"].where(
            df["a_smok_stat_qn"] != 4, other=np.nan
        )

    df = df.rename(
        columns={
            "ID": "patient_id",
            "basis_sex": "sex",
            "basis_age": "age",
            "a_smok_stat_qn": "smoking_status",
            "a_packyears": "pack_years",
            "a_anthro_groe": "height_cm",
            "a_anthro_gew": "weight_kg",
            "a_anthro_bmi": "bmi",
            "a_anthro_fettmasse": "body_fat_pct",
        }
    )

    df["sex"] = pd.Categorical(
        df["sex"].map({1: "Male", 2: "Female"}),
        categories=["Male", "Female"],
    )
    df["smoking_status"] = pd.Categorical(
        df["smoking_status"].map({1.0: "Never", 2.0: "Ex-smoker", 3.0: "Current"}),
        categories=["Never", "Ex-smoker", "Current"],
        ordered=True,
    )

    logger.info(f"Metadata: {len(df):,} patients")
    return df


def merge_datasets(
    analytics: pd.DataFrame, metadata: pd.DataFrame
) -> tuple[pd.DataFrame, dict]:
    """Inner-join analytics and metadata on ``patient_id``.

    Returns ``(merged_df, join_stats_dict)``.
    """
    a_ids = set(analytics["patient_id"].unique())
    m_ids = set(metadata["patient_id"].unique())
    both = a_ids & m_ids

    stats = {
        "n_analytics_patients": len(a_ids),
        "n_metadata_patients": len(m_ids),
        "n_inner_join_patients": len(both),
        "n_analytics_only": len(a_ids - m_ids),
        "n_metadata_only": len(m_ids - a_ids),
    }

    for k, v in stats.items():
        logger.info(f"  {k}: {v:,}")

    merged = analytics.merge(metadata, on="patient_id", how="inner")
    logger.info(
        f"Merged: {len(merged):,} rows | "
        f"{merged['patient_id'].nunique():,} patients"
    )
    return merged, stats
