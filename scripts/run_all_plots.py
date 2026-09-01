"""Regenerate all figures for an existing pipeline run, no heavy recompute.

Re-renders every plotting stage from the cached stage outputs already on disk
(``.npz`` / ``.csv`` / ``.parquet``): GPA and PCA are skipped via the
``ssm_pca`` caches, only the cheap PC-regression / geometry fits rerun before
plotting. STL-dependent figures (3D PC-deformation renders, per-vertex residual
diagnostics) are rendered only when the registered/extracted STL dirs recorded
in ``metadata.json`` are reachable, and skipped otherwise. The Styner QA triad
is never recomputed; its triptych is re-rendered only if a prior
``eval_styner.json`` is present.

CLI
---
    python scripts/run_all_plots.py --run-dir <pipeline-run-dir> [--workers N]

Stages (each guarded; a single failure does not abort the rest):
    ingestion (replot from cached CSV/parquet) · adjusted · ssm_pca ·
    radiomics_correlation · ssm_viewer · visualizations ·
    ssm_qa_residuals (STL-gated) · ssm_qa_metrics (replot-only, if JSON present)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from settings import apply_publication_style          # noqa: E402
from utils.logging import get_logger                  # noqa: E402
from utils.paths import stage_dir                     # noqa: E402

apply_publication_style()

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Re-render all figures for an existing run")
    p.add_argument(
        "--run-dir", type=Path, required=True, metavar="DIR",
        help="Completed pipeline run dir (the one holding ingestion/, ssm_pca/, …).",
    )
    p.add_argument(
        "--workers", type=int, default=None, metavar="N",
        help="Worker count forwarded to STL-loading stages.",
    )
    return p.parse_args()


def _stream(script: Path, extra_args: list[str]) -> int:
    """Run a stage script as a subprocess, streaming raw bytes (preserves \\r)."""
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    proc = subprocess.Popen(
        [sys.executable, str(script), *extra_args],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env,
    )
    assert proc.stdout is not None
    out = sys.stdout.buffer
    while chunk := proc.stdout.read1(4096):
        out.write(chunk)
        out.flush()
    proc.wait()
    return proc.returncode


def _read_metadata(run_dir: Path) -> dict[str, str]:
    meta_path = run_dir / "metadata.json"
    if not meta_path.exists():
        return {}
    try:
        return dict(json.loads(meta_path.read_text()).get("paths", {}))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning(f"Could not read {meta_path}: {exc}")
        return {}


def _cached_variance_threshold(run_dir: Path) -> float | None:
    pca_path = stage_dir(run_dir, "ssm_pca") / "pca_surface.npz"
    if not pca_path.exists():
        return None
    try:
        with np.load(pca_path) as d:
            if "variance_threshold" in d.files:
                return float(d["variance_threshold"])
    except (OSError, ValueError) as exc:
        logger.warning(f"Could not read variance_threshold from {pca_path}: {exc}")
    return None


# ── Stage drivers ────────────────────────────────────────────────────────────

def _ingestion(run_dir: Path) -> int:
    """Replot the ingestion QC figures from cached CSV/parquet (no re-ingest)."""
    from data_ingestion.loaders import ANALYSIS_SHAPE_COLS
    from data_ingestion.qc_altair import (
        plot_correlation_matrix,
        plot_distributions,
        plot_inclusion_flow,
        plot_missingness,
        plot_normality_summary,
    )
    from settings import META_CONTINUOUS

    ing  = stage_dir(run_dir, "ingestion")
    figs = ing / "figures"
    figs.mkdir(parents=True, exist_ok=True)

    included = pd.read_parquet(ing / "analytic_clean.parquet")

    # missingness_report.csv / normality_tests.csv carry the variable on the
    # index (written with index); the plotters reset_index() it back.
    plot_inclusion_flow(pd.read_csv(ing / "exclusions.csv"), figs / "inclusion_flow")
    plot_missingness(pd.read_csv(ing / "missingness_report.csv", index_col=0),
                     figs / "missingness")
    plot_normality_summary(pd.read_csv(ing / "normality_tests.csv", index_col=0),
                           figs / "normality")
    plot_distributions(included, ANALYSIS_SHAPE_COLS, figs)
    plot_correlation_matrix(
        included, ANALYSIS_SHAPE_COLS, out_stem=figs / "corr_shape_params",
        title="Shape parameter correlations",
        subtitle="Rib level · Pearson r · lower triangle",
        method="pearson", flag_threshold=0.7,
    )
    meta_corr = included.drop_duplicates("patient_id")[META_CONTINUOUS + ["sex"]].copy()
    meta_corr["sex_numeric"] = meta_corr["sex"].map({"Male": 0, "Female": 1})
    plot_correlation_matrix(
        meta_corr, META_CONTINUOUS + ["sex_numeric"], out_stem=figs / "corr_metadata",
        title="Correlation matrix",
        subtitle="Patient level · Spearman ρ · lower triangle",
        method="spearman", flag_threshold=0.7,
    )
    return 0


def _adjusted(run_dir: Path) -> int:
    return _stream(REPO_ROOT / "src" / "adjusted" / "run_adjusted.py",
                   ["--run-dir", str(run_dir)])


def _ssm_pca(run_dir: Path, paths: dict[str, str], workers: int | None) -> int:
    parquet = stage_dir(run_dir, "ingestion") / "analytic_clean.parquet"
    args = [
        "--run-dir",          str(run_dir),
        "--parquet",          str(parquet),
        "--extraction-dir",   paths.get("extracted_stl_dir", ""),
        "--registration-dir", paths.get("registered_stl_dir", ""),
    ]
    vt = _cached_variance_threshold(run_dir)
    if vt is not None:
        args += ["--variance-threshold", str(vt)]
    if workers is not None:
        args += ["--workers", str(workers)]
    return _stream(REPO_ROOT / "src" / "ssm" / "run_ssm.py", args)


def _radiomics(run_dir: Path) -> int:
    return _stream(REPO_ROOT / "src" / "ssm" / "run_radiomics_correlation.py",
                   ["--run-dir", str(run_dir)])


def _viewer(run_dir: Path) -> int:
    from ssm.viewer import export_both
    out_dir = stage_dir(run_dir, "ssm_viewer")
    out_dir.mkdir(parents=True, exist_ok=True)
    internal, public = export_both(
        results_dir=stage_dir(run_dir, "ssm_pca"), output_dir=out_dir,
    )
    logger.info(f"Viewer HTML (internal): {internal}")
    logger.info(f"Viewer HTML (public):   {public}")
    return 0


def _visualizations(run_dir: Path) -> int:
    return _stream(REPO_ROOT / "scripts" / "run_visualizations.py",
                   ["--run-dir", str(run_dir)])


def _residuals(run_dir: Path, paths: dict[str, str], workers: int | None) -> int:
    args = [
        "--run-dir",        str(run_dir),
        "--target-dir",     paths["extracted_stl_dir"],
        "--registered-dir", paths["registered_stl_dir"],
        "--scope",          "whole-cage",
    ]
    if workers is not None:
        args += ["--workers", str(workers)]
    return _stream(REPO_ROOT / "src" / "ssm" / "eval_residuals.py", args)


def _styner_replot(run_dir: Path) -> int:
    from ssm.eval_metrics import load_styner_json, plot_styner_triptych
    qa_dir = stage_dir(run_dir, "ssm_qa_metrics")
    json_path = qa_dir / "eval_styner.json"
    per_rib, whole_cage = load_styner_json(json_path)
    plot_styner_triptych(qa_dir / "figures" / "eval_styner.png",
                         per_rib_results=per_rib, whole_cage=whole_cage)
    logger.info(f"Re-rendered Styner triptych from {json_path}")
    return 0


def main() -> None:
    args = parse_args()
    run_dir: Path = args.run_dir
    if not run_dir.is_dir():
        raise FileNotFoundError(f"run-dir not found: {run_dir}")

    paths = _read_metadata(run_dir)
    reg = Path(paths["registered_stl_dir"]) if paths.get("registered_stl_dir") else None
    ext = Path(paths["extracted_stl_dir"]) if paths.get("extracted_stl_dir") else None
    stls_reachable = bool(reg and ext and reg.is_dir() and ext.is_dir())
    if not stls_reachable:
        logger.warning(
            "STL dirs from metadata.json are not reachable on this machine – "
            "3D STL-dependent renders and the residual diagnostics will be skipped."
        )

    styner_json = stage_dir(run_dir, "ssm_qa_metrics") / "eval_styner.json"

    results: dict[str, str] = {}

    def run(name: str, fn) -> None:
        logger.info(f"════ {name} ════")
        try:
            rc = fn()
            results[name] = "ok" if rc == 0 else f"FAILED (rc={rc})"
        except Exception as exc:  # why-broad: stage drivers fan in subprocess + renderer errors
            logger.error(f"{name} raised: {exc}")
            results[name] = f"ERROR ({type(exc).__name__})"

    run("ingestion",             lambda: _ingestion(run_dir))
    run("adjusted",              lambda: _adjusted(run_dir))
    run("ssm_pca",               lambda: _ssm_pca(run_dir, paths, args.workers))
    run("radiomics_correlation", lambda: _radiomics(run_dir))
    run("ssm_viewer",            lambda: _viewer(run_dir))
    run("visualizations",        lambda: _visualizations(run_dir))

    if stls_reachable:
        run("ssm_qa_residuals", lambda: _residuals(run_dir, paths, args.workers))
    else:
        results["ssm_qa_residuals"] = "skipped (STLs unreachable)"

    if styner_json.exists():
        run("ssm_qa_metrics", lambda: _styner_replot(run_dir))
    else:
        results["ssm_qa_metrics"] = "skipped (no eval_styner.json; not recomputed)"

    logger.info("──── summary ────")
    for name, status in results.items():
        logger.info(f"  {name:24s} {status}")
    if any(s.startswith(("FAILED", "ERROR")) for s in results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
