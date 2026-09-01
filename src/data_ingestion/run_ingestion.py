"""Data ingestion and QC pipeline.

Usage::

    cd /path/to/codebase
    .venv/bin/python src/data_ingestion/run_ingestion.py --run-dir <run-dir>

All outputs land under ``<run-dir>/ingestion/``.  The pipeline driver
(:mod:`scripts.run_pipeline`) creates the run dir up front and threads
it to every stage.

Pipeline steps
--------------
  1. Load JSON analytics  → one row per (patient, vert_level, side)
  2. Load metadata        → one row per patient; sentinel codes resolved
  3. Merge on patient_id  → ``ingestion/join_stats.json``
  4. Seg components audit (per rib label 40–51)
       → ``ingestion/per_rib_components_audit.csv``
       → ``ingestion/per_rib_anomalies.csv``
  5. Rib count audit + exclusions
       → ``ingestion/rib_count_audit.csv``
       → ``ingestion/exclusions.csv``
  6. Missingness audit on the pre-exclusion data
       → ``ingestion/missingness_report.csv``
  7. Table 1 on the included cohort
       → ``ingestion/table1_metadata.csv``
       → ``ingestion/table1_shape.csv``
  8. Distribution plots + normality on the included cohort
       → ``ingestion/normality_tests.csv``
       → ``ingestion/figures/distribution_hist_qq.png``
  9. Shape correlation matrix (analysis cols only)
       → ``ingestion/figures/corr_shape_params.png``
 10. Metadata correlation matrix
       → ``ingestion/figures/corr_metadata.png``
 11. Save clean parquet (included cohort only)
       → ``ingestion/analytic_clean.parquet``
"""
from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from data_ingestion.loaders import (             # noqa: E402
    ALL_SHAPE_COLS,
    ANALYSIS_SHAPE_COLS,
    load_json_analytics,
    load_metadata,
    merge_datasets,
)
from data_ingestion.qc import (                  # noqa: E402
    apply_exclusions,
    audit_missingness,
    audit_rib_counts,
    compute_table1,
    normality_summary,
)
from data_ingestion.seg_components import audit_seg_components  # noqa: E402
from data_ingestion.qc_altair import (           # noqa: E402
    plot_correlation_matrix,
    plot_distributions,
    plot_inclusion_flow,
    plot_missingness,
    plot_normality_summary,
)
from settings import META_CATEGORICAL, META_CONTINUOUS, apply_publication_style  # noqa: E402
from utils.config import load_config              # noqa: E402
from utils.logging import get_logger              # noqa: E402
from utils.paths import stage_dir                  # noqa: E402

apply_publication_style()

logger = get_logger(__name__)

META_ALL = META_CONTINUOUS + META_CATEGORICAL


@contextlib.contextmanager
def _step(label: str):
    """Log a step header and elapsed time."""
    logger.info(f"--- {label} ---")
    t0 = time.perf_counter()
    yield
    logger.info(f"    [{label}] done in {time.perf_counter() - t0:.1f}s")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Data ingestion + QC runner")
    p.add_argument(
        "--run-dir", type=Path, required=True, metavar="DIR",
        help="Per-run output directory (created by run_pipeline.py). "
             "Outputs land under <run-dir>/ingestion/.",
    )
    p.add_argument(
        "--n-patients", type=int, default=None, metavar="N",
        help="Limit ingestion to the first N patients (numeric sort by PID). "
             "Matches the selection used by run_mesh_extraction.py. "
             "Default: all patients.",
    )
    p.add_argument(
        "--workers", type=int, default=None, metavar="N",
        help="Parallel workers for JSON parsing. "
             "Default: settings.SSM_LOAD_N_WORKERS (32).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_dir  = args.run_dir
    OUT      = stage_dir(run_dir, "ingestion")
    FIG_OUT  = OUT / "figures"
    OUT.mkdir(parents=True, exist_ok=True)
    FIG_OUT.mkdir(parents=True, exist_ok=True)

    t_pipeline = time.perf_counter()
    cfg = load_config()

    if args.n_patients is not None:
        logger.info(f"DEV MODE: limiting ingestion to first {args.n_patients} patients")

    with _step("Loading JSON analytics"):
        analytics = load_json_analytics(cfg, n_patients=args.n_patients, n_workers=args.workers)

    with _step("Loading metadata"):
        metadata = load_metadata(cfg)

    with _step("Merging datasets"):
        merged, join_stats = merge_datasets(analytics, metadata)
        with open(OUT / "join_stats.json", "w") as f:
            json.dump(join_stats, f, indent=2)
        logger.info(f"  Join stats → {OUT / 'join_stats.json'}")

    with _step("Seg components audit"):
        candidate_pids = sorted(merged["patient_id"].unique().tolist())
        seg_audit = audit_seg_components(cfg, candidate_pids, n_workers=args.workers)
        seg_audit_path = OUT / "per_rib_components_audit.csv"
        seg_audit.to_csv(seg_audit_path, index=False)
        logger.info(f"  Per-rib components audit → {seg_audit_path}")

        anomalies = seg_audit[seg_audit["n_components"] != 2].copy()
        anomalies_path = OUT / "per_rib_anomalies.csv"
        if anomalies.empty:
            pd.DataFrame(columns=["patient_id", "rib_label", "n_components"]).to_csv(
                anomalies_path, index=False
            )
            logger.info(f"  No anomalies – empty summary written to {anomalies_path}")
        else:
            anomalies.to_csv(anomalies_path, index=False)
            logger.info(
                f"  Anomaly summary → {anomalies_path} "
                f"({anomalies['patient_id'].nunique():,} patients)"
            )

    with _step("Rib count audit + exclusions"):
        rib_audit = audit_rib_counts(merged)
        rib_audit.to_csv(OUT / "rib_count_audit.csv", index=False)
        logger.info(f"  Rib count audit → {OUT / 'rib_count_audit.csv'}")

        included, excl_table = apply_exclusions(merged, rib_audit, seg_audit, cfg)
        excl_table.to_csv(OUT / "exclusions.csv", index=False)
        logger.info(f"  Exclusion table → {OUT / 'exclusions.csv'}")
        plot_inclusion_flow(excl_table, FIG_OUT / "inclusion_flow")

    with _step("Missingness audit (pre-exclusion)"):
        miss_report = audit_missingness(merged, ALL_SHAPE_COLS, META_ALL)
        miss_report.to_csv(OUT / "missingness_report.csv")
        logger.info(f"  Missingness report → {OUT / 'missingness_report.csv'}")
        plot_missingness(miss_report, FIG_OUT / "missingness")

    with _step("Table 1"):
        meta_only = included.drop_duplicates("patient_id")[META_ALL]
        t1_meta = compute_table1(meta_only, META_CONTINUOUS, META_CATEGORICAL, strat_col="sex")
        t1_meta.to_csv(OUT / "table1_metadata.csv", index=False)
        logger.info(f"  Table 1 (metadata) → {OUT / 'table1_metadata.csv'}")

        shape_means = included.groupby("patient_id")[ANALYSIS_SHAPE_COLS].mean().reset_index()
        shape_means = shape_means.merge(
            included[["patient_id", "sex"]].drop_duplicates(), on="patient_id"
        )
        shape_means["sex"] = pd.Categorical(shape_means["sex"], categories=["Male", "Female"])
        t1_shape = compute_table1(shape_means, ANALYSIS_SHAPE_COLS, [], strat_col="sex")
        t1_shape.to_csv(OUT / "table1_shape.csv", index=False)
        logger.info(f"  Table 1 (shape, patient means) → {OUT / 'table1_shape.csv'}")

    with _step("Distribution plots + normality"):
        norm_tests = normality_summary(included, ANALYSIS_SHAPE_COLS)
        norm_tests.to_csv(OUT / "normality_tests.csv")
        non_normal = norm_tests[~norm_tests["likely_normal"]].index.tolist()
        logger.info(
            f"  Likely non-normal ({len(non_normal)}): "
            + ", ".join(non_normal[:10])
            + (" ..." if len(non_normal) > 10 else "")
        )
        logger.info(f"  Normality tests → {OUT / 'normality_tests.csv'}")
        plot_distributions(included, ANALYSIS_SHAPE_COLS, FIG_OUT)
        plot_normality_summary(norm_tests, FIG_OUT / "normality")

    with _step("Shape parameter correlation matrix"):
        plot_correlation_matrix(
            included,
            ANALYSIS_SHAPE_COLS,
            out_stem=FIG_OUT / "corr_shape_params",
            title="Shape parameter correlations",
            subtitle="Rib level · Pearson r · lower triangle",
            method="pearson",
            flag_threshold=0.7,
        )

    with _step("Metadata correlation matrix"):
        meta_corr = included.drop_duplicates("patient_id")[META_CONTINUOUS + ["sex"]].copy()
        meta_corr["sex_numeric"] = meta_corr["sex"].map({"Male": 0, "Female": 1})
        plot_correlation_matrix(
            meta_corr,
            META_CONTINUOUS + ["sex_numeric"],
            out_stem=FIG_OUT / "corr_metadata",
            title="Correlation matrix",
            subtitle="Patient level · Spearman ρ · lower triangle",
            method="spearman",
            flag_threshold=0.7,
        )

    with _step("Saving clean dataset"):
        drop_cols = ["rib_volume", "orig_zoom", "start_point", "end_point"]
        clean = included.drop(columns=[c for c in drop_cols if c in included.columns])
        clean.to_parquet(OUT / "analytic_clean.parquet", index=False)
        logger.info(f"  Clean parquet → {OUT / 'analytic_clean.parquet'}")
        logger.info(
            f"  {clean.shape[0]:,} rows × {clean.shape[1]} cols | "
            f"{clean['patient_id'].nunique():,} patients"
        )

    elapsed = time.perf_counter() - t_pipeline
    logger.info(f"Ingestion pipeline complete – total {elapsed:.1f}s")


if __name__ == "__main__":
    main()
