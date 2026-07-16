"""Per-rib mesh-extraction runner.

Data source (per-rib-level NIfTI, one per patient)::

    {nifti_base}/{block}/{patient_id}/vibe/sub-{patient_id}_..._seg-vert-rib_msk.nii.gz

(configured in ``data/data_config.yaml`` → ``paths.nifti_base`` /
``paths.nifti_vert_rib_pattern``).

Outputs (alongside the raw data, not committed)::

    {stl_dir}/{block}/{patient_id}/{patient_id}_rib{label}_{L|R}.stl

where ``block = patient_id // 1000`` (mirrors the NAKO NIfTI sharding).
``--stl-dir`` is required and comes from
``preset.paths.extracted_stl_dir`` when invoked via run_pipeline.py;
per-patient sub-paths come from ``utils.run_dir.patient_stl_dir``.

Also writes to the same extraction directory:

  - ``per_rib_audit.csv``                    one row per patient × rib label
  - ``per_rib_volume_summary.csv``           mesh volume as % of the source
                                             voxel volume, aggregated per rib
  - ``patient_ids.txt``                      patient IDs processed (same as input)
  - ``template_id.txt``                      heuristic default (first patient
                                             with all 24 ribs) — read
                                             interactively by the
                                             ``template_selection`` notebook.
                                             The *load-bearing* per-registration
                                             template_id.txt lives in the
                                             registration output dir, written
                                             by the pipeline driver.
  - ``mesh_extraction_per_rib_log.json``     extraction stats summary

These extraction-level artifacts live alongside the per-rib STLs they
describe; downstream registration reads them as inputs.  This dir is a
deterministic, shared cache — repeated pipeline runs reuse the extracted
STLs across runs.  Repository ``results/`` holds per-run analysis
artifacts (one timestamped subdir per ``run_pipeline.py`` invocation).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from settings import MESH_TARGET_FACES_PER_RIB                           # noqa: E402
from ssm.mesh_extraction import RIB_LABELS, batch_extract_per_rib  # noqa: E402
from utils.config import load_config                                     # noqa: E402
from utils.logging import get_logger                                     # noqa: E402
from utils.paths import stage_dir                                        # noqa: E402
from utils.rib_labels import seg_to_anatomical                           # noqa: E402
from utils.run_dir import patient_stl_dir                                # noqa: E402

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Per-rib mesh extraction runner")
    p.add_argument(
        "--run-dir", type=Path, required=True, metavar="DIR",
        help="Per-run directory (created by run_pipeline.py). "
             "Reads patient IDs from <run-dir>/ingestion/analytic_clean.parquet.",
    )
    p.add_argument(
        "--stl-dir", type=Path, required=True, metavar="DIR",
        help="STL output root.  Each patient's 24 STLs are written under "
             "<stl-dir>/<block>/<pid>/ (block = pid // 1000).  Comes from "
             "preset.paths.extracted_stl_dir when invoked via run_pipeline.py.",
    )
    p.add_argument(
        "--n-patients", type=int, default=None, metavar="N",
        help="Limit extraction to the first N patients (sorted by ID). "
             "Useful for dev / smoke-testing. Default: all patients.",
    )
    p.add_argument(
        "--target-faces", type=int, default=MESH_TARGET_FACES_PER_RIB, metavar="N",
        help=f"Quadric decimation target per rib. Default: "
             f"{MESH_TARGET_FACES_PER_RIB} (settings.MESH_TARGET_FACES_PER_RIB). "
             f"Note: cached STLs in the output dir are reused regardless of "
             f"this value — clear the dir or point --stl-dir at a fresh "
             f"location to force re-extraction at a new face count.",
    )
    p.add_argument(
        "--workers", type=int, default=None, metavar="N",
        help="Parallel workers for per-rib extraction. "
             "Default: settings.MESH_N_WORKERS (32).",
    )
    return p.parse_args()


def _report_volumes(audit_rows: list[dict], stl_dir: Path) -> None:
    """Log mesh volume against source voxel volume, and write the per-rib table."""
    recs = [
        {
            "rib": seg_to_anatomical(r["rib_label"]),
            "side": side,
            "voxel_mm3": r.get(f"{side}_voxel_mm3"),
            "mesh_mm3": r.get(f"{side}_mesh_mm3"),
            "vol_pct": r.get(f"{side}_vol_pct"),
        }
        for r in audit_rows
        for side in ("L", "R")
    ]
    vdf = pd.DataFrame(recs).dropna(subset=["vol_pct"])
    if vdf.empty:
        logger.warning("No mesh volumes recorded — volume summary skipped")
        return

    summary = (
        vdf.groupby("rib")["vol_pct"]
        .agg(n="count", mean="mean", sd="std", min="min", max="max")
        .reset_index()
    )
    summary_path = stl_dir / "per_rib_volume_summary.csv"
    summary.to_csv(summary_path, index=False)

    logger.info("Mesh volume as %% of source voxel volume, by rib:")
    logger.info("\n" + summary.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    logger.info(
        f"Overall: mean {vdf['vol_pct'].mean():.2f}%  "
        f"(sd {vdf['vol_pct'].std():.2f}, min {vdf['vol_pct'].min():.2f}, "
        f"max {vdf['vol_pct'].max():.2f}, n={len(vdf):,})"
    )
    logger.info(f"Volume summary written to {summary_path}")


def main() -> None:
    args = parse_args()
    cfg = load_config()
    parquet = stage_dir(args.run_dir, "ingestion") / "analytic_clean.parquet"

    nifti_base    = cfg["paths"]["nifti_base"]
    nifti_pattern = cfg["paths"]["nifti_vert_rib_pattern"].replace("{base}", nifti_base)
    stl_dir       = args.stl_dir
    rib_labels    = cfg["nifti"].get("rib_level_labels")  # [40, 41, ..., 51]

    # The parquet lives on a network mount and can time out.  In dev mode
    # (--n-patients set), reuse a local cache once it covers ≥ N patients
    # so we can skip the network read entirely.
    LOCAL_CACHE = Path.home() / ".cache" / "nako_ribs" / "patient_ids_full.txt"
    pids_cached: list[int] | None = None
    if LOCAL_CACHE.exists():
        cached = [int(x) for x in LOCAL_CACHE.read_text().splitlines() if x.strip()]
        if args.n_patients is None or len(cached) >= args.n_patients:
            pids_cached = cached
            logger.info(
                f"Using local patient ID cache ({len(cached):,} IDs) — skipping parquet read"
            )

    if pids_cached is not None:
        patient_ids = pids_cached
    else:
        logger.info(f"Reading patient IDs from parquet: {parquet}")
        for attempt in range(1, 6):
            try:
                df = pd.read_parquet(parquet, columns=["patient_id"])
                break
            except (OSError, ValueError) as exc:  # Parquet/network read flake
                if attempt == 5:
                    raise
                logger.warning(
                    f"Parquet read failed (attempt {attempt}/5): {exc} — retrying in 10s"
                )
                time.sleep(10)
        patient_ids = sorted(df["patient_id"].unique().tolist())
        LOCAL_CACHE.parent.mkdir(parents=True, exist_ok=True)
        LOCAL_CACHE.write_text("\n".join(str(p) for p in patient_ids))
        logger.info(f"Patient ID cache written to {LOCAL_CACHE}")

    if args.n_patients is not None:
        patient_ids = patient_ids[: args.n_patients]
        logger.info(f"DEV MODE: limited to first {len(patient_ids)} patients")
    logger.info(f"Patients to process: {len(patient_ids):,}")
    logger.info(f"NIfTI source:        {nifti_base}")
    logger.info(f"STL output:          {stl_dir}")
    logger.info(f"Target faces/rib:    {args.target_faces:,}")
    _label_list = rib_labels if rib_labels is not None else RIB_LABELS
    logger.info(
        f"Rib labels:   {_label_list}  "
        f"(anatomical ribs {seg_to_anatomical(_label_list[0])}"
        f"–{seg_to_anatomical(_label_list[-1])})"
    )

    extract_kwargs: dict[str, int] = {}
    if args.workers is not None:
        extract_kwargs["n_workers"] = int(args.workers)
    stats = batch_extract_per_rib(
        patient_ids   = patient_ids,
        nifti_pattern = nifti_pattern,
        stl_dir       = stl_dir,
        rib_labels    = rib_labels,
        target_faces  = args.target_faces,
        **extract_kwargs,
    )

    # Per-patient × rib-label audit CSV.
    audit_rows = []
    for pid, entries in stats.get("audit", {}).items():
        for entry in entries:
            audit_rows.append({"patient_id": pid, **entry})
    if audit_rows:
        audit_df = pd.DataFrame(audit_rows)
        audit_path = stl_dir / "per_rib_audit.csv"
        audit_df.to_csv(audit_path, index=False)
        logger.info(f"Per-rib audit CSV written to {audit_path} ({len(audit_rows):,} rows)")
        _report_volumes(audit_rows, stl_dir)

    # Extraction log (without full audit so the JSON stays manageable).
    stats_log = {k: v for k, v in stats.items() if k != "audit"}
    log_path = stl_dir / "mesh_extraction_per_rib_log.json"
    log_path.write_text(json.dumps(stats_log, indent=2))
    logger.info(f"Log written to {log_path}")

    # Components-anomaly filtering is applied upstream in ingestion; the
    # cohort that arrives here is already clean.
    pids_path = stl_dir / "patient_ids.txt"
    pids_path.write_text("\n".join(str(p) for p in patient_ids))
    logger.info(f"Patient IDs written to {pids_path}")

    # Registration template = first patient with all 24 per-rib STLs.
    # Scalismo's TemplateBuilder refines this at registration time.
    if rib_labels is None:
        rib_labels = RIB_LABELS
    for pid in sorted(patient_ids):
        pdir = patient_stl_dir(stl_dir, pid)
        expected = [
            pdir / f"{pid}_rib{lab}_{side}.stl"
            for lab in rib_labels
            for side in ("L", "R")
        ]
        if all(f.exists() for f in expected):
            template_pid = pid
            logger.info(f"Template selected (first complete patient): {pid}")
            break
    else:
        logger.error("No patient with a complete set of 24 per-rib STLs found!")
        return

    tpl_path = stl_dir / "template_id.txt"
    tpl_path.write_text(str(template_pid))
    logger.info(f"Template ID saved: {tpl_path}")


if __name__ == "__main__":
    main()
