"""Re-emit every pipeline figure from a previous run dir into a 1:1 copy.

Copies a source run dir (compute artifacts only — figures/ subtrees are
omitted) to a fresh target and re-runs every figure-emitting code path
against the cached parquet / npz / csv / json artifacts. The source is
never modified.

Usage::

    python scripts/rerender_figures.py SOURCE_RUN_DIR \\
        [--target-dir PATH] [--force] [--preset PATH] \\
        [--methodology-figure-json JSON] \\
        [--extracted-stl-dir PATH] [--registered-stl-dir PATH] \\
        [--stages STAGE,STAGE,...]

Default target: ``<source-parent>/<source-name>_replot_<UTC>/``.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from data_ingestion.loaders import ALL_SHAPE_COLS, ANALYSIS_SHAPE_COLS  # noqa: E402
from settings import (                                                   # noqa: E402
    META_CATEGORICAL,
    META_CONTINUOUS,
    N_PCS_DISPLAY,
    apply_publication_style,
)
from ssm.pca_surface import RIB_ORDER                                    # noqa: E402
from utils.config import load_config, read_parquet                       # noqa: E402
from utils.logging import get_logger, log_step                           # noqa: E402
from utils.paths import stage_dir                                        # noqa: E402
from utils.rib_labels import display_from_seg                            # noqa: E402
from utils.run_dir import copy_run_dir, read_metadata                    # noqa: E402

apply_publication_style()

logger = get_logger("rerender_figures")

# All stages that emit figures, in canonical pipeline order. Stages with
# no figures (`mesh_extraction`, `ssm_registration`) are excluded. Update
# this list when a new figure-emitting stage is added.
FIGURE_STAGES: tuple[str, ...] = (
    "ingestion",
    "adjusted",
    "ssm_pca",
    "radiomics_correlation",
    "ssm_viewer",
    "ssm_qa_metrics",
    "ssm_qa_residuals",
    "visualizations",
)

META_ALL = META_CONTINUOUS + META_CATEGORICAL


# ── External-resource resolution ─────────────────────────────────────────────

def _resolve_external_dirs(
    target_run_dir: Path,
    *,
    extracted_override: Path | None,
    registered_override: Path | None,
) -> tuple[Path | None, Path | None]:
    """Resolve extracted / registered STL roots from CLI overrides or metadata."""
    extracted: Path | None = None
    registered: Path | None = None
    try:
        meta = read_metadata(target_run_dir)
    except FileNotFoundError:
        meta = {}
    paths_block: dict[str, str] = (meta.get("paths") or {}) if isinstance(meta, dict) else {}

    if extracted_override is not None:
        extracted = extracted_override.expanduser().resolve()
    elif paths_block.get("extracted_stl_dir"):
        extracted = Path(paths_block["extracted_stl_dir"])
    if registered_override is not None:
        registered = registered_override.expanduser().resolve()
    elif paths_block.get("registered_stl_dir"):
        registered = Path(paths_block["registered_stl_dir"])

    return (
        extracted if (extracted and extracted.is_dir()) else None,
        registered if (registered and registered.is_dir()) else None,
    )


# ── Stage: ingestion ─────────────────────────────────────────────────────────

def _replay_ingestion(run_dir: Path) -> None:
    from data_ingestion import qc_altair as qm

    stage = stage_dir(run_dir, "ingestion")
    figs  = stage / "figures"
    figs.mkdir(parents=True, exist_ok=True)

    clean_path = stage / "analytic_clean.parquet"
    if not clean_path.exists():
        raise FileNotFoundError(f"{clean_path} missing — cannot replay ingestion figures.")
    included = read_parquet(clean_path)

    excl_path = stage / "exclusions.csv"
    if excl_path.exists():
        qm.plot_inclusion_flow(pd.read_csv(excl_path), figs / "inclusion_flow")
    else:
        logger.warning(f"{excl_path} missing — skipping inclusion-flow figure.")

    miss_path = stage / "missingness_report.csv"
    if miss_path.exists():
        qm.plot_missingness(pd.read_csv(miss_path, index_col=0), figs / "missingness")
    else:
        logger.warning(f"{miss_path} missing — skipping missingness figure.")

    norm_path = stage / "normality_tests.csv"
    if norm_path.exists():
        qm.plot_normality_summary(pd.read_csv(norm_path, index_col=0), figs / "normality")
    else:
        logger.warning(f"{norm_path} missing — skipping normality summary.")

    qm.plot_distributions(included, ANALYSIS_SHAPE_COLS, figs)

    qm.plot_correlation_matrix(
        included,
        ANALYSIS_SHAPE_COLS,
        out_stem=figs / "corr_shape_params",
        title="Shape parameter correlations",
        subtitle="Rib level · Pearson r · lower triangle",
        method="pearson",
        flag_threshold=0.7,
    )

    meta_corr = included.drop_duplicates("patient_id")[META_CONTINUOUS + ["sex"]].copy()
    meta_corr["sex_numeric"] = meta_corr["sex"].map({"Male": 0, "Female": 1})
    qm.plot_correlation_matrix(
        meta_corr,
        META_CONTINUOUS + ["sex_numeric"],
        out_stem=figs / "corr_metadata",
        title="Correlation matrix",
        subtitle="Patient level · Spearman ρ · lower triangle",
        method="spearman",
        flag_threshold=0.7,
    )


# ── Stage: adjusted ──────────────────────────────────────────────────────────

def _replay_adjusted(run_dir: Path) -> None:
    from adjusted import plots_adjusted_altair as am

    stage = stage_dir(run_dir, "adjusted")
    figs  = stage / "figures"
    figs.mkdir(parents=True, exist_ok=True)

    unadj = pd.read_csv(stage / "descriptor_unadjusted.csv")
    adj   = pd.read_csv(stage / "descriptor_adjusted.csv")

    am.plot_adj_heatmap(adj, figs / "adj_heatmap_beta")
    am.plot_partial_r2_heatmap(adj, figs / "adj_heatmap_partial_r2")
    am.plot_forest_plots(adj, ANALYSIS_SHAPE_COLS, figs / "adj_forest_plots")
    am.plot_adj_heatmap(unadj, figs / "unadj_heatmap_beta")


# ── Stage: ssm_pca ───────────────────────────────────────────────────────────

def _rebuild_pca_shim(pca_npz: Path) -> Any:
    """Reconstruct a sklearn-PCA-shaped object from cached arrays."""
    from sklearn.decomposition import PCA as _PCA
    d = np.load(pca_npz)
    pca = _PCA()
    pca.components_               = d["components"]
    pca.explained_variance_       = d["explained_variance"]
    pca.explained_variance_ratio_ = d["explained_variance_ratio"]
    pca.mean_                     = d["mean"]
    pca.n_components_             = len(d["components"])
    return pca


def _ensure_template_faces(
    pca_stage: Path,
    extracted_dir: Path | None,
    registered_dir: Path | None,
) -> np.ndarray | None:
    """Prefer cached ``template_faces.npy``; fall back to extracted STLs."""
    cached = pca_stage / "template_faces.npy"
    if cached.exists():
        return np.load(cached)

    if extracted_dir is None or registered_dir is None:
        return None

    template_id_path = registered_dir / "template_id.txt"
    if not template_id_path.exists():
        logger.warning(
            f"{template_id_path} not found — skipping deformation + gallery plots."
        )
        return None

    import pyvista as pv
    from utils.run_dir import patient_stl_dir
    tpid = template_id_path.read_text().strip()
    tpid_pdir = patient_stl_dir(extracted_dir, tpid)
    all_faces: list[np.ndarray] = []
    vertex_offset = 0
    for lab, side in RIB_ORDER:
        stl_path = tpid_pdir / f"{tpid}_rib{lab}_{side}.stl"
        if not stl_path.exists():
            logger.warning(f"Template rib STL missing: {stl_path.name} — skipping.")
            return None
        try:
            tmpl = pv.read(str(stl_path))
        except Exception as exc:  # PyVista/VTK readers raise a wide variety
            logger.warning(f"PyVista failed to read {stl_path.name}: {exc}")
            return None
        faces_raw = tmpl.faces.reshape(-1, 4)[:, 1:].astype(np.int32)
        all_faces.append(faces_raw + vertex_offset)
        vertex_offset += tmpl.n_points
    faces_np = np.concatenate(all_faces, axis=0)
    np.save(cached, faces_np)
    logger.info(f"Wrote {cached} (regenerated from extracted template STLs).")
    return faces_np


def _replay_ssm_pca(
    run_dir: Path,
    extracted_dir: Path | None,
    registered_dir: Path | None,
) -> None:
    # 3D figures have no Altair port; they stay in plots_ssm.
    from ssm import plots_ssm_altair as sm
    from ssm.plots_mean_shape import plot_mean_shape_views
    from ssm.plots_ssm import plot_pc_deformations

    stage = stage_dir(run_dir, "ssm_pca")
    figs  = stage / "figures"
    figs.mkdir(parents=True, exist_ok=True)

    pca_npz = stage / "pca_surface.npz"
    if not pca_npz.exists():
        raise FileNotFoundError(f"{pca_npz} missing — cannot replay ssm_pca figures.")
    pca = _rebuild_pca_shim(pca_npz)

    mean_shape  = np.load(stage / "mean_shape_surface.npy")
    rib_offsets = np.load(stage / "rib_offsets.npy")

    gpa_hist_path = stage / "gpa_rms_history.npy"
    gpa_history = (
        np.load(gpa_hist_path) if gpa_hist_path.exists() else np.empty(0)
    )

    rib_label_strs = [display_from_seg(lab, side) for (lab, side) in RIB_ORDER]

    if gpa_history.size > 0:
        sm.plot_gpa_convergence(gpa_history, figs / "gpa_convergence")

    sm.plot_scree_surface(pca, figs / "scree_surface")
    sm.plot_pc_loadings_per_rib(pca, rib_offsets, rib_label_strs,
                                figs / "pc_loadings_per_rib")
    # Mirrored-bar companion (Altair only).
    if hasattr(sm, "plot_pc_loadings_per_rib_histo"):
        sm.plot_pc_loadings_per_rib_histo(
            pca, rib_offsets, rib_label_strs,
            figs / "pc_loadings_per_rib_histo",
        )

    faces_np = _ensure_template_faces(stage, extracted_dir, registered_dir)
    if faces_np is not None:
        plot_mean_shape_views(mean_shape, faces_np, rib_offsets,
                              rib_label_strs, figs / "mean_shape_views")
        plot_pc_deformations(mean_shape, pca, faces_np,
                             out_stem=figs / "pc_deformations_surface",
                             n_pcs=N_PCS_DISPLAY)
    else:
        logger.warning(
            "Skipping mean_shape_views + pc_deformations_surface "
            "(template faces unavailable)."
        )

    pc_disp = [f"PC_{k}" for k in range(1, N_PCS_DISPLAY + 1)]
    ev_disp = pca.explained_variance_
    adj_path   = stage / "pc_adjusted.csv"
    unadj_path = stage / "pc_unadjusted.csv"
    tgt_path   = stage / "pc_targeted.csv"
    adj   = pd.read_csv(adj_path)   if adj_path.exists()   else None
    unadj = pd.read_csv(unadj_path) if unadj_path.exists() else None
    tgt   = pd.read_csv(tgt_path)   if tgt_path.exists()   else None
    if adj is not None and not adj.empty:
        sm.plot_regression_heatmap_surface(adj, figs / "pc_regression_surface")

    scores_df = pd.read_csv(stage / "pc_scores_surface.csv")
    arrow_sex_unadj = (sm.pc_arrow_data(unadj, "is_female", ev_disp, pc_disp)
                       if unadj is not None else None)
    for color_by in ("sex", "smoking_status"):
        try:
            sm.plot_pc_scores_pairs(
                scores_df, figs / f"pc_scores_pairs_{color_by}",
                n_pcs=N_PCS_DISPLAY, color_by=color_by,
                arrow=(arrow_sex_unadj if color_by == "sex" else None),
            )
        except Exception as exc:  # vl-convert renderer varies
            logger.warning(f"PC pair-plot ({color_by}) failed: {exc}")

    if adj is not None and unadj is not None:
        try:
            sm.plot_pc_scores_pairs_adjusted(
                scores_df, figs / "pc_scores_pairs_adj_sex", color_by="sex",
                n_pcs=N_PCS_DISPLAY,
                arrow=sm.pc_arrow_data(adj, "is_female", ev_disp, pc_disp),
            )
            if tgt is not None:
                sm.plot_pc_scores_pairs_targeted(
                    scores_df, figs / "pc_scores_pairs_dag_sex", color_by="sex",
                    n_pcs=N_PCS_DISPLAY,
                    arrow=sm.pc_arrow_data(tgt, "is_female", ev_disp, pc_disp),
                )
            for pred in ("age", "height_cm", "weight_kg", "body_fat_pct"):
                sm.plot_pc_scores_pairs(
                    scores_df, figs / f"pc_scores_pairs_{pred}",
                    n_pcs=N_PCS_DISPLAY, color_by=pred,
                    arrow=sm.pc_arrow_data(unadj, pred, ev_disp, pc_disp),
                )
                sm.plot_pc_scores_pairs_adjusted(
                    scores_df, figs / f"pc_scores_pairs_adj_{pred}",
                    color_by=pred, n_pcs=N_PCS_DISPLAY,
                    arrow=sm.pc_arrow_data(adj, pred, ev_disp, pc_disp),
                )
                if tgt is not None:
                    sm.plot_pc_scores_pairs_targeted(
                        scores_df, figs / f"pc_scores_pairs_dag_{pred}",
                        color_by=pred, n_pcs=N_PCS_DISPLAY,
                        arrow=sm.pc_arrow_data(tgt, pred, ev_disp, pc_disp),
                    )
            sm.plot_pc_beta_vectors(unadj, adj, figs / "pc_beta_vectors",
                                    tgt_df=tgt, n_pcs=N_PCS_DISPLAY)
        except Exception as exc:  # vl-convert renderer varies
            logger.warning(f"PC three-model viz failed: {exc}")


# ── Stage: radiomics_correlation ─────────────────────────────────────────────

def _replay_radiomics_correlation(run_dir: Path) -> None:
    from ssm.plots_radiomics_correlation_altair import emit_all

    stage = stage_dir(run_dir, "radiomics_correlation")
    figs  = stage / "figures"
    figs.mkdir(parents=True, exist_ok=True)

    effects_path = stage / "effects.csv"
    meta_path    = stage / "metadata.json"
    if not effects_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            f"Missing {effects_path} or {meta_path} — cannot replay "
            f"radiomics_correlation figures."
        )
    effects = pd.read_csv(effects_path)
    meta = json.loads(meta_path.read_text())
    feature_cols = list(meta.get("feature_cols") or [])
    mask_nonsig  = bool(meta.get("mask_nonsig", True))

    emit_all(effects, figs, feature_cols=feature_cols, mask_nonsig=mask_nonsig)


# ── Stage: ssm_viewer ────────────────────────────────────────────────────────

def _replay_ssm_viewer(run_dir: Path) -> None:
    from ssm.viewer import export_both

    pca_dir = stage_dir(run_dir, "ssm_pca")
    out_dir = stage_dir(run_dir, "ssm_viewer")
    out_dir.mkdir(parents=True, exist_ok=True)
    internal_path, public_path = export_both(
        results_dir=pca_dir, output_dir=out_dir,
    )
    logger.info(f"Viewer HTML (internal): {internal_path}")
    logger.info(f"Viewer HTML (public):   {public_path}")


# ── Stage: ssm_qa_metrics ────────────────────────────────────────────────────

def _replay_ssm_qa_metrics(run_dir: Path) -> None:
    from ssm.eval_metrics import load_styner_json, plot_styner_triptych

    stage = stage_dir(run_dir, "ssm_qa_metrics")
    figs  = stage / "figures"
    figs.mkdir(parents=True, exist_ok=True)

    styner_path = stage / "eval_styner.json"
    if not styner_path.exists():
        raise FileNotFoundError(
            f"{styner_path} missing — cannot replay Styner triad without "
            f"the cached compute output."
        )
    per_rib, whole_cage = load_styner_json(styner_path)
    plot_styner_triptych(figs / "eval_styner.png", per_rib, whole_cage)


# ── Stage: ssm_qa_residuals ──────────────────────────────────────────────────

def _replay_ssm_qa_residuals(
    run_dir: Path,
    extracted_dir: Path | None,
    registered_dir: Path | None,
    worst_n: int,
    preset_scope: str | None,
    preset_rib_id: str | None,
) -> None:
    from ssm.eval_residuals import (
        _harvest_patient,
        load_residuals_npz,
        render_mosaic,
    )
    from ssm.plots_ssm_altair import plot_residual_distribution

    stage = stage_dir(run_dir, "ssm_qa_residuals")
    figs  = stage / "figures"
    figs.mkdir(parents=True, exist_ok=True)

    # Discover scope from filename. Preset-derived scope (when provided)
    # picks the matching npz.
    if preset_scope == "rib" and preset_rib_id:
        npz_path = stage / f"residuals_per_patient_{preset_rib_id}.npz"
        suffix   = f"_{preset_rib_id}"
    else:
        # Whole-cage default; if only a rib-scoped npz exists, fall back to it.
        whole = stage / "residuals_per_patient.npz"
        if whole.exists():
            npz_path = whole
            suffix   = ""
        else:
            candidates = sorted(stage.glob("residuals_per_patient_*.npz"))
            if not candidates:
                raise FileNotFoundError(
                    f"No residuals_per_patient[*].npz under {stage} — "
                    f"cannot replay residual figures."
                )
            npz_path = candidates[0]
            suffix   = "_" + npz_path.stem.removeprefix("residuals_per_patient_")
            logger.info(f"Auto-detected rib-scope residuals: {npz_path.name}")

    if not npz_path.exists():
        raise FileNotFoundError(f"{npz_path} missing — cannot replay residual figures.")

    per_patient, p95_by_pid, rib_ids = load_residuals_npz(npz_path)
    logger.info(
        f"Loaded {len(per_patient):,} patients × {len(rib_ids)} ribs from {npz_path.name}"
    )

    for direction in ("forward", "reverse"):
        try:
            plot_residual_distribution(
                per_patient, rib_ids,
                figs / f"residuals_distribution_{direction}{suffix}",
                direction=direction,
            )
        except Exception as exc:  # Plotly/Kaleido renderer varies
            logger.error(f"{direction} residual distribution plot failed: {exc}")

    if extracted_dir is None or registered_dir is None:
        logger.warning(
            "Extracted / registered STL dir unavailable — skipping worst-N "
            "mosaic (residual violins emitted above)."
        )
        return

    # Reharvest PolyData only for the worst-N patients per direction — seconds.
    for direction in ("forward", "reverse"):
        ranked = sorted(per_patient.keys(),
                        key=lambda p, dr=direction: -p95_by_pid[dr][p])
        worst_pids = ranked[:worst_n]
        panels: list[tuple[int, dict, float]] = []
        for pid in worst_pids:
            harvested = _harvest_patient(pid, registered_dir, extracted_dir, rib_ids)
            if harvested is None:
                logger.warning(
                    f"Could not harvest STLs for pid={pid} — skipping mosaic panel."
                )
                continue
            panels.append((pid, harvested, p95_by_pid[direction][pid]))
        if not panels:
            logger.warning(
                f"No reharvestable patients for {direction} mosaic — skipping."
            )
            continue
        out_png = figs / f"worst_patients_residuals_{direction}{suffix}"
        try:
            render_mosaic(out_png, panels, rib_ids, direction=direction)
        except Exception as exc:  # Plotly/Kaleido renderer varies
            logger.error(f"{direction} mosaic rendering failed: {exc}")


# ── Stage: visualizations ────────────────────────────────────────────────────

def _replay_visualizations(
    run_dir: Path,
    methodology_cfg: dict[str, Any] | None,
) -> None:
    from visualizations.vis_altair import (
        plot_metadata_by_sex,
        plot_rib_length_by_level,
        plot_shape_by_rib,
        plot_smoking_by_sex,
    )

    stage = stage_dir(run_dir, "visualizations")
    figs  = stage / "figures"
    figs.mkdir(parents=True, exist_ok=True)

    parquet = stage_dir(run_dir, "ingestion") / "analytic_clean.parquet"
    if not parquet.exists():
        raise FileNotFoundError(
            f"{parquet} missing — cannot replay supplemental visualisations."
        )
    df = read_parquet(parquet)

    with log_step(logger, "metadata + smoking + rib-length plots"):
        plot_metadata_by_sex(df, figs)
        plot_smoking_by_sex(df, figs)
        plot_rib_length_by_level(df, figs)
    with log_step(logger, "shape-by-rib ridgelines"):
        plot_shape_by_rib(df, figs)

    if methodology_cfg is not None:
        from visualizations.methodology_figure import render as mf_render
        try:
            data_cfg = load_config()
        except FileNotFoundError as exc:
            logger.warning(
                f"Methodology figure skipped: {exc} "
                f"(data_config.yaml is required for NIfTI volume paths)."
            )
            return
        with log_step(logger, f"methodology figure (patient {methodology_cfg['display_patient']})"):
            try:
                mf_render(
                    run_dir=run_dir,
                    data_config=data_cfg,
                    preset_cfg=methodology_cfg,
                    out_stem=figs / "methodology",
                )
            except FileNotFoundError as exc:
                logger.warning(f"Methodology figure skipped: {exc}")


# ── Preset / methodology helpers ─────────────────────────────────────────────

def _load_preset(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: top-level YAML must be a mapping.")
    return raw


def _methodology_from_preset(preset: dict[str, Any]) -> dict[str, Any] | None:
    mf = preset.get("methodology_figure")
    if not isinstance(mf, dict):
        return None
    if mf.get("display_patient") is None:
        return None
    out = dict(mf)
    out["display_patient"] = int(out["display_patient"])
    return out


# ── Main ─────────────────────────────────────────────────────────────────────

def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "source_run_dir", type=Path, metavar="SOURCE_RUN_DIR",
        help="Run dir to copy + replay figures from.",
    )
    p.add_argument(
        "--target-dir", type=Path, default=None,
        help="Target run dir. Default: "
             "<source-parent>/<source-name>_replot_<UTC>/.",
    )
    p.add_argument(
        "--force", action="store_true",
        help="Overwrite --target-dir if it already exists.",
    )
    p.add_argument(
        "--preset", type=Path, default=None,
        help="Preset YAML used to recover methodology_figure config and "
             "qa.residual_* knobs. Optional.",
    )
    p.add_argument(
        "--methodology-figure-json", type=str, default=None,
        help="Inline JSON for the methodology_figure block; alternative to --preset.",
    )
    p.add_argument(
        "--extracted-stl-dir", type=Path, default=None,
        help="Override metadata.json::paths.extracted_stl_dir.",
    )
    p.add_argument(
        "--registered-stl-dir", type=Path, default=None,
        help="Override metadata.json::paths.registered_stl_dir.",
    )
    p.add_argument(
        "--stages", type=str, default=None,
        help="Comma-separated subset of figure-emitting stages "
             f"(default: all of {','.join(FIGURE_STAGES)}).",
    )
    p.add_argument(
        "--skip-copy", action="store_true",
        help="Skip the source→target copy. Use when --target-dir already "
             "contains the compute artifacts (e.g. an earlier replot dir) "
             "and you only want to write new figures into it.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    source = args.source_run_dir.expanduser().resolve()
    if not source.is_dir():
        logger.error(f"Source run dir does not exist: {source}")
        return 2

    if args.target_dir is not None:
        target = args.target_dir.expanduser().resolve()
    else:
        target = source.parent / f"{source.name}_replot_{_utc_stamp()}"

    selected: tuple[str, ...]
    if args.stages:
        wanted = [s.strip() for s in args.stages.split(",") if s.strip()]
        unknown = [s for s in wanted if s not in FIGURE_STAGES]
        if unknown:
            logger.error(
                f"Unknown stages: {unknown}; valid: {list(FIGURE_STAGES)}"
            )
            return 2
        selected = tuple(s for s in FIGURE_STAGES if s in wanted)
    else:
        selected = FIGURE_STAGES

    # Methodology config: --preset first, then --methodology-figure-json, else off.
    preset_dict: dict[str, Any] | None = None
    methodology_cfg: dict[str, Any] | None = None
    if args.preset:
        preset_dict = _load_preset(args.preset)
        methodology_cfg = _methodology_from_preset(preset_dict)
    if methodology_cfg is None and args.methodology_figure_json:
        methodology_cfg = json.loads(args.methodology_figure_json)
        if methodology_cfg.get("display_patient") is not None:
            methodology_cfg["display_patient"] = int(methodology_cfg["display_patient"])

    # QA residual knobs from preset (if any).
    qa = (preset_dict or {}).get("qa") or {}
    preset_scope    = qa.get("residual_scope")
    preset_rib_id   = qa.get("residual_rib_id")
    preset_worst_n  = int(qa.get("residual_worst_n", 6))

    logger.info("─" * 70)
    logger.info(f"Source run dir:  {source}")
    logger.info(f"Target run dir:  {target}")
    logger.info(f"Stages selected: {list(selected)}")
    logger.info(f"Methodology:     {'on' if methodology_cfg else 'off'}")
    logger.info("─" * 70)

    t0 = time.monotonic()
    if args.skip_copy:
        if not target.is_dir():
            logger.error(
                f"--skip-copy requires --target-dir to exist; "
                f"{target} not found."
            )
            return 2
        logger.info(f"--skip-copy: reusing existing target {target}")
    else:
        with log_step(logger, "Copy source → target (excluding figures/)"):
            copy_run_dir(source, target, force=args.force)

    extracted, registered = _resolve_external_dirs(
        target,
        extracted_override=args.extracted_stl_dir,
        registered_override=args.registered_stl_dir,
    )
    logger.info(f"extracted_stl_dir resolved to:  {extracted}")
    logger.info(f"registered_stl_dir resolved to: {registered}")

    ran:     list[str] = []
    skipped: list[str] = []
    for stage_name in selected:
        logger.info(f"=== STAGE {stage_name} — START ===")
        t_stage = time.monotonic()
        try:
            if stage_name == "ingestion":
                _replay_ingestion(target)
            elif stage_name == "adjusted":
                _replay_adjusted(target)
            elif stage_name == "ssm_pca":
                _replay_ssm_pca(target, extracted, registered)
            elif stage_name == "radiomics_correlation":
                _replay_radiomics_correlation(target)
            elif stage_name == "ssm_viewer":
                _replay_ssm_viewer(target)
            elif stage_name == "ssm_qa_metrics":
                _replay_ssm_qa_metrics(target)
            elif stage_name == "ssm_qa_residuals":
                _replay_ssm_qa_residuals(
                    target, extracted, registered,
                    worst_n=preset_worst_n,
                    preset_scope=preset_scope,
                    preset_rib_id=preset_rib_id,
                )
            elif stage_name == "visualizations":
                _replay_visualizations(target, methodology_cfg)
            else:  # pragma: no cover — guarded by selected validation above
                raise ValueError(f"Unhandled stage: {stage_name}")
        except FileNotFoundError as exc:
            logger.error(f"STAGE {stage_name} — FAILED: {exc}")
            skipped.append(stage_name)
            continue
        dt = time.monotonic() - t_stage
        logger.info(f"=== STAGE {stage_name} — DONE ({dt:.0f}s) ===\n")
        ran.append(stage_name)

    total = time.monotonic() - t0
    m, s = divmod(total, 60)
    logger.info("─" * 70)
    logger.info(f"Rerender complete — {m:.0f}m{s:.0f}s")
    logger.info(f"  Ran:      {ran}")
    logger.info(f"  Skipped:  {skipped}")
    logger.info(f"  Target:   {target}")
    logger.info("─" * 70)
    return 0 if not skipped else 1


if __name__ == "__main__":
    sys.exit(main())
