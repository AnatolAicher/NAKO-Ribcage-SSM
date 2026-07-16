"""Per-rib connected-components audit on the segmentation NIfTIs.

For each patient, loads the rib/vertebra segmentation volume, runs
``scipy.ndimage.label`` on each rib label (40–51), and records the
component count. Patients whose NIfTI is missing or unreadable surface
with a sentinel row (``rib_label = -1, n_components = -1``).

Public API
----------
audit_seg_components(cfg, patient_ids) -> pd.DataFrame
    Long table: ``patient_id, rib_label, n_components``.
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import label as ndlabel

from settings import SSM_LOAD_N_WORKERS

logger = logging.getLogger(__name__)

_MISSING_SENTINEL_LABEL: int = -1
_MISSING_SENTINEL_COUNT: int = -1


def _components_for_patient(
    pid: int,
    nifti_pattern: str,
    rib_labels: list[int],
) -> tuple[int, list[tuple[int, int]], str | None]:
    """Return ``(pid, [(rib_label, n_components), ...], error_msg)``.

    On missing / unreadable NIfTI, returns a single sentinel row and an error
    message. The caller funnels sentinels into the exclusion table.
    """
    block = pid // 1000
    path = Path(nifti_pattern.format(block=block, patient_id=pid))
    if not path.exists():
        return pid, [(_MISSING_SENTINEL_LABEL, _MISSING_SENTINEL_COUNT)], "missing"
    try:
        img = nib.load(str(path))
        img = nib.as_closest_canonical(img)
        data = np.asarray(img.dataobj, dtype=np.int32)
    except (OSError, ValueError, EOFError) as exc:
        return pid, [(_MISSING_SENTINEL_LABEL, _MISSING_SENTINEL_COUNT)], str(exc)

    rows: list[tuple[int, int]] = []
    for label in rib_labels:
        mask = (data == label).astype(np.int32)
        _, n_comp = ndlabel(mask)
        rows.append((label, int(n_comp)))
    return pid, rows, None


def audit_seg_components(
    cfg: dict,
    patient_ids: list[int],
    *,
    n_workers: int | None = None,
) -> pd.DataFrame:
    """Per-rib connected-components audit across ``patient_ids``.

    Parameters
    ----------
    cfg :
        Config dict — uses ``paths.nifti_base``, ``paths.nifti_vert_rib_pattern``,
        ``nifti.rib_level_labels``.
    patient_ids :
        Candidate patient IDs to audit (typically the ingestion-merged list).
    n_workers :
        Parallel workers. Default: ``SSM_LOAD_N_WORKERS``.

    Returns
    -------
    DataFrame with columns ``patient_id, rib_label, n_components``. Patients
    with a missing/unreadable NIfTI appear with a single sentinel row
    (``rib_label = -1, n_components = -1``).
    """
    nifti_base = cfg["paths"]["nifti_base"]
    nifti_pattern = cfg["paths"]["nifti_vert_rib_pattern"].replace("{base}", nifti_base)
    rib_labels = list(cfg["nifti"]["rib_level_labels"])
    n_workers = SSM_LOAD_N_WORKERS if n_workers is None else int(n_workers)

    logger.info(
        f"Beginning seg-components audit of {len(patient_ids):,} patients "
        f"(workers={n_workers}) …"
    )

    rows: list[dict] = []
    n_missing = 0
    n_errors = 0

    PROGRESS_INTERVAL = 1000
    total = len(patient_ids)
    n_done = 0
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(_components_for_patient, pid, nifti_pattern, rib_labels): pid
            for pid in patient_ids
        }
        for fut in as_completed(futures):
            pid, pid_rows, err = fut.result()
            if err == "missing":
                n_missing += 1
            elif err is not None:
                logger.error(f"Failed seg-components audit for PID {pid}: {err}")
                n_errors += 1
            for rib_label, n_comp in pid_rows:
                rows.append({
                    "patient_id": pid,
                    "rib_label": rib_label,
                    "n_components": n_comp,
                })
            n_done += 1
            if n_done % PROGRESS_INTERVAL == 0 or n_done == total:
                logger.info(f"  Seg-components audit: {n_done:,}/{total:,} patients")

    if n_missing:
        logger.warning(f"  {n_missing:,} patient(s) had a missing segmentation NIfTI")
    if n_errors:
        logger.warning(f"  {n_errors:,} patient(s) failed to load")

    df = pd.DataFrame(rows, columns=["patient_id", "rib_label", "n_components"])
    n_anom = df[df["n_components"] != 2]["patient_id"].nunique()
    logger.info(
        f"Seg-components audit: {n_anom:,}/{total:,} patients have at least "
        f"one anomalous rib label (n_components != 2)"
    )
    return df
