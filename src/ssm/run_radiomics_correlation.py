"""Correlate SSM PC scores with per-rib radiomics features.

Reads ``<run>/ssm_pca/pc_scores_surface.csv`` and
``<run>/ingestion/analytic_clean.parquet``, joins on ``patient_id``, runs a
univariate OLS per (PC, vert_level, side, feature) (z-scored PC predictor,
raw-unit feature response), and writes::

    <run>/radiomics_correlation/
    ├── effects.csv                       tidy K × 24 × N_features effect tensor
    ├── effects_top_by_abs_std.csv        top-200 |beta_std|
    ├── metadata.json                     n / K / FDR convention / git rev
    └── figures/
        ├── per_pc/pc{NN}_std.{html,svg,png}          + index.html browser
        ├── per_pc_histo/pc{NN}_std.{html,svg,png}    + index.html browser
        ├── per_feature/<short>_native.{html,svg,png} + index.html browser
        └── per_feature_histo/<short>_native.{html,svg,png}
                                                      + index.html browser

CLI::

    python src/ssm/run_radiomics_correlation.py --run-dir DIR
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from settings import FDR_DISPLAY_ALPHA, apply_publication_style       # noqa: E402
from ssm.plots_radiomics_correlation_altair import emit_all          # noqa: E402
from ssm.radiomics_correlation import (                                # noqa: E402
    compute_effects,
    join_pc_with_ribs,
    load_inputs,
    top_effects,
)
from utils.logging import get_logger                                  # noqa: E402
from utils.paths import stage_dir                                     # noqa: E402

apply_publication_style()

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Per-(PC, rib, feature) regressions linking SSM PCs to radiomics."
    )
    p.add_argument(
        "--run-dir", type=Path, required=True, metavar="DIR",
        help="Top-level per-run directory. "
             "Reads <DIR>/ssm_pca/pc_scores_surface.csv and "
             "<DIR>/ingestion/analytic_clean.parquet; "
             "writes to <DIR>/radiomics_correlation/.",
    )
    p.add_argument(
        "--no-mask-nonsig", dest="mask_nonsig", action="store_false",
        help="Show all heatmap cells regardless of FDR significance "
             "(default: hide cells with q ≥ FDR_DISPLAY_ALPHA).",
    )
    p.set_defaults(mask_nonsig=True)
    return p.parse_args()


def _git_rev() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT, stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def main() -> None:
    args = parse_args()
    t0 = time.monotonic()

    run_dir = args.run_dir.resolve()
    out_dir = stage_dir(run_dir, "radiomics_correlation")
    out_dir.mkdir(parents=True, exist_ok=True)
    figs_dir = out_dir / "figures"
    figs_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Run dir:  {run_dir}")
    logger.info(f"Output:   {out_dir}")

    pc_df, radiomics_df, pc_cols = load_inputs(run_dir)
    joined = join_pc_with_ribs(pc_df, radiomics_df)

    feature_cols = [c for c in radiomics_df.columns
                    if c not in ("patient_id", "vert_level", "side")]
    logger.info(f"Features ({len(feature_cols)}): {feature_cols}")

    effects = compute_effects(joined, pc_cols, feature_cols)
    effects_path = out_dir / "effects.csv"
    effects.to_csv(effects_path, index=False)
    logger.info(f"  effects.csv → {effects_path}  ({len(effects):,} rows)")

    top = top_effects(effects, n=200)
    top_path = out_dir / "effects_top_by_abs_std.csv"
    top.to_csv(top_path, index=False)
    logger.info(f"  top_by_abs_std → {top_path}  ({len(top):,} rows)")

    emit_all(
        effects, figs_dir,
        feature_cols=feature_cols,
        mask_nonsig=args.mask_nonsig,
    )

    n_subjects = int(joined["patient_id"].nunique())
    n_sig = int((effects["q_value"] < FDR_DISPLAY_ALPHA).sum())
    metadata = {
        "n_subjects_joined":  n_subjects,
        "n_pcs":              len(pc_cols),
        "n_features":         len(feature_cols),
        "n_rib_positions":    int(joined[["vert_level", "side"]].drop_duplicates().shape[0]),
        "n_regressions":      int(len(effects)),
        "n_fdr_significant":  n_sig,
        "fdr_method":         "BH within PC",
        "fdr_display_alpha":  FDR_DISPLAY_ALPHA,
        "mask_nonsig":        bool(args.mask_nonsig),
        "feature_cols":       feature_cols,
        "pc_cols":            pc_cols,
        "git_rev":            _git_rev(),
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    logger.info(f"  metadata.json → {out_dir / 'metadata.json'}")

    elapsed = time.monotonic() - t0
    m, s = divmod(elapsed, 60)
    logger.info(
        f"radiomics_correlation complete – {m:.0f}m{s:.0f}s  "
        f"({n_sig:,}/{len(effects):,} effects FDR-significant at q<{FDR_DISPLAY_ALPHA})"
    )


if __name__ == "__main__":
    main()
