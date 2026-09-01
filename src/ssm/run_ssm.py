"""Surface SSM Python runner: GPA + PCA + PC regression.

Assumes Scalismo has already written per-rib registered STL files to the
directory passed via ``--registration-dir`` (preset.paths.registered_stl_dir).
The template patient's STLs in ``--extraction-dir``
(preset.paths.extracted_stl_dir) are read for face connectivity.

Each patient has 24 STL files (``{pid}_rib{label}_{L|R}.stl``).  The loader
concatenates all 24 ribs per patient into one shape vector for PCA.

All derived outputs are written under ``--run-dir`` (the SSM subdir of the
top-level pipeline run dir, e.g. ``results/<presetname>_<UTC>/ssm_pca/``).
The pipeline driver creates this dir up front; for standalone runs, pass
``--run-dir`` explicitly.

Outputs (under the run dir)
---------------------------
  shapes_registered.npz             (N, n_pts_total, 3) aligned shape matrix + patient_ids
  mean_shape_surface.npy            (n_pts_total, 3) mean shape after GPA
  gpa_rms_history.npy               per-iteration GPA RMS convergence trace
  meta_sub_surface.parquet          metadata subset aligned to the shape matrix
  rib_offsets.npy                   vertex offsets per rib identity (for splitting)
  template_faces.npy                template face connectivity (needs template STLs)
  pca_surface.npz                   PCA components, variance, mean
  pc_scores_surface.csv             PC scores + metadata
  pc_unadjusted.csv                 per-predictor marginal OLS (HC3)
  pc_adjusted.csv                   multivariable OLS + FWL partial R² (HC3)
  pc_targeted.csv                   per-exposure DAG back-door OLS (HC3)
  pc_ttest_surface.csv              Welch t-test (sex Cohen's d)
  geometry_generator.npz            demographics → PC-score coefficients
  geometry_generator_holdout.csv    k-fold held-out per-vertex error
  figures/                          scree, PC deformations, regression heatmap
                                    pair-plots (+ β-arrow), β-vector field
"""
from __future__ import annotations

import argparse
import contextlib
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyvista as pv

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from settings import N_PCS_DISPLAY, apply_publication_style          # noqa: E402
from ssm.pc_regression import run_pc_regression                       # noqa: E402
from ssm.geometry_generator import fit_and_save                       # noqa: E402
from ssm.pca_surface import (                                          # noqa: E402
    RIB_ORDER,
    build_surface_scores_df,
    load_registered_meshes_per_rib,
    run_gpa,
    run_pca,
)
from ssm import plots_ssm_altair as sm                                # noqa: E402
from ssm.plots_mean_shape import plot_mean_shape_views                # noqa: E402
from ssm.plots_ssm import plot_pc_deformations                        # noqa: E402
from utils.logging import get_logger                                   # noqa: E402
from utils.paths import stage_dir                                      # noqa: E402
from utils.rib_labels import display_from_seg                          # noqa: E402
from utils.run_dir import patient_stl_dir                              # noqa: E402

apply_publication_style()

logger = get_logger(__name__)

FORCE_RECOMPUTE = False


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Surface SSM GPA / PCA / regression runner")
    p.add_argument(
        "--run-dir", type=Path, required=True, metavar="DIR",
        help="Top-level pipeline run dir.  Outputs land under "
             "<run-dir>/ssm_pca/.",
    )
    p.add_argument(
        "--parquet", type=Path, required=True, metavar="FILE",
        help="Path to analytic_clean.parquet "
             "(typically <run-dir>/ingestion/analytic_clean.parquet).",
    )
    p.add_argument(
        "--patient-ids", type=str, default=None, metavar="IDS",
        help="Comma-separated patient IDs to load (dev mode). Default: all registered.",
    )
    p.add_argument(
        "--patient-ids-file", type=str, default=None, metavar="FILE",
        help="File with one patient ID per line. Avoids ARG_MAX limits for large cohorts.",
    )
    p.add_argument(
        "--registration-dir", type=str, required=True, metavar="DIR",
        help="Registered per-rib STL root (preset.paths.registered_stl_dir).",
    )
    p.add_argument(
        "--extraction-dir", type=str, required=True, metavar="DIR",
        help="Extracted per-rib STL root (preset.paths.extracted_stl_dir). "
             "Used to read the template patient's STLs for the face "
             "connectivity passed to the mean-shape gallery.",
    )
    p.add_argument(
        "--variance-threshold", type=float, default=0.95, metavar="X",
        help="Cumulative-variance cutoff for PCA component count "
             "(0 < x <= 1). Default 0.95. Cached pca_surface.npz is "
             "invalidated if this differs from the value stored in it.",
    )
    p.add_argument(
        "--workers", type=int, default=None, metavar="N",
        help="Parallel workers for STL loading. "
             "Default: settings.SSM_LOAD_N_WORKERS (32).",
    )
    return p.parse_args()


@contextlib.contextmanager
def _step(label: str):
    logger.info(f"--- {label} – START ---")
    t = time.monotonic()
    yield
    elapsed = time.monotonic() - t
    m, s = divmod(elapsed, 60)
    logger.info(f"--- {label} – DONE  ({m:.0f}m{s:.0f}s) ---")


def main() -> None:
    args = parse_args()
    if args.patient_ids_file:
        patient_ids = [
            int(x) for x in Path(args.patient_ids_file).read_text().splitlines()
            if x.strip()
        ]
    elif args.patient_ids:
        patient_ids = [int(x) for x in args.patient_ids.split(",")]
    else:
        patient_ids = None
    if patient_ids is not None:
        logger.info(f"DEV MODE: restricted to {len(patient_ids)} patient IDs")

    pipeline_t0 = time.monotonic()

    reg_dir        = Path(args.registration_dir)
    extraction_dir = Path(args.extraction_dir)

    # This stage writes under <args.run_dir>/ssm_pca/.
    top_run_dir = args.run_dir
    run_dir = stage_dir(top_run_dir, "ssm_pca")
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"SSM PCA out dir: {run_dir}")
    figs_dir = run_dir / "figures"
    figs_dir.mkdir(parents=True, exist_ok=True)

    parquet = args.parquet

    logger.info(f"Registered per-rib mesh source: {reg_dir}")
    logger.info(f"Extraction (per-rib STL) source: {extraction_dir}")

    # ── Load metadata ─────────────────────────────────────────────────────────
    logger.info(f"Reading metadata from parquet: {parquet}")
    for attempt in range(1, 6):
        try:
            meta_df = (
                pd.read_parquet(parquet)
                .drop_duplicates("patient_id")[
                    ["patient_id", "sex", "age", "bmi", "body_fat_pct",
                     "smoking_status", "pack_years", "height_cm", "weight_kg"]
                ]
            )
            break
        except (OSError, ValueError) as exc:  # Parquet/network read flake
            if attempt == 5:
                raise
            logger.warning(f"Parquet read failed (attempt {attempt}/5): {exc} – retrying in 10s")
            time.sleep(10)
    logger.info(f"Metadata loaded: {len(meta_df):,} patients")

    # ── Load registered meshes + GPA ─────────────────────────────────────────
    shapes_path = run_dir / "shapes_registered.npz"
    mean_path   = run_dir / "mean_shape_surface.npy"

    # Cascading recompute: if shapes are recomputed, PCA must be too.
    recompute_shapes = FORCE_RECOMPUTE or not shapes_path.exists()

    if recompute_shapes:
        with _step("Load per-rib registered meshes"):
            shapes, pids, meta_sub, rib_offsets = load_registered_meshes_per_rib(
                reg_dir, meta_df, patient_ids=patient_ids, n_workers=args.workers
            )

        with _step(f"GPA alignment ({len(shapes):,} shapes, {shapes.shape[1]:,} pts each)"):
            aligned, mean_shape, gpa_history = run_gpa(shapes)

        np.savez_compressed(shapes_path, shapes=aligned, patient_ids=pids)
        np.save(mean_path, mean_shape)
        np.save(run_dir / "rib_offsets.npy", np.array(rib_offsets, dtype=np.int64))
        np.save(run_dir / "gpa_rms_history.npy", gpa_history)
        meta_sub.to_parquet(run_dir / "meta_sub_surface.parquet", index=False)
        logger.info(f"Saved shapes → {shapes_path}")
    else:
        logger.info(f"Loading cached shapes from {shapes_path}")
        data       = np.load(shapes_path)
        aligned    = data["shapes"]
        pids       = data["patient_ids"]
        mean_shape = np.load(mean_path)
        meta_sub   = pd.read_parquet(run_dir / "meta_sub_surface.parquet")
        gpa_hist_path = run_dir / "gpa_rms_history.npy"
        gpa_history = (np.load(gpa_hist_path)
                       if gpa_hist_path.exists() else np.empty(0))
        logger.info(f"  shapes={aligned.shape}  patients={len(pids):,}")

    # ── PCA ──────────────────────────────────────────────────────────────────
    pca_path = run_dir / "pca_surface.npz"
    recompute_pca = recompute_shapes or not pca_path.exists()
    if not recompute_pca:
        cached = np.load(pca_path)
        cached_vt = float(cached["variance_threshold"]) if "variance_threshold" in cached.files else None
        if cached_vt is None or not np.isclose(cached_vt, args.variance_threshold):
            logger.info(
                f"PCA cache variance_threshold={cached_vt} differs from "
                f"requested {args.variance_threshold} – recomputing."
            )
            recompute_pca = True

    if recompute_pca:
        with _step(f"PCA (input {aligned.shape}, variance_threshold={args.variance_threshold})"):
            pca, X_flat = run_pca(aligned, variance_threshold=args.variance_threshold)
        np.savez_compressed(
            pca_path,
            components=pca.components_,
            explained_variance=pca.explained_variance_,
            explained_variance_ratio=pca.explained_variance_ratio_,
            mean=pca.mean_,
            variance_threshold=np.float32(args.variance_threshold),
        )
        logger.info(
            f"PCA: {pca.n_components_} components, "
            f"{pca.explained_variance_ratio_.sum():.3%} variance explained  → {pca_path}"
        )
    else:
        logger.info(f"Loading cached PCA from {pca_path}")
        from sklearn.decomposition import PCA as _PCA
        d = np.load(pca_path)
        pca = _PCA()
        pca.components_              = d["components"]
        pca.explained_variance_      = d["explained_variance"]
        pca.explained_variance_ratio_= d["explained_variance_ratio"]
        pca.mean_                    = d["mean"]
        pca.n_components_            = len(d["components"])
        X_flat = aligned.reshape(len(aligned), -1).astype(np.float32, copy=False)
        logger.info(f"  {pca.n_components_} components")

    # ── GPA convergence diagnostic ───────────────────────────────────────────
    if gpa_history.size > 0:
        with _step("GPA convergence diagnostic"):
            sm.plot_gpa_convergence(gpa_history, figs_dir / "gpa_convergence")

    # ── Visualise PCA ────────────────────────────────────────────────────────
    with _step("Scree plot"):
        sm.plot_scree_surface(pca, figs_dir / "scree_surface")

    # Per-rib loading concentration heatmap.
    rib_offsets_arr = np.load(run_dir / "rib_offsets.npy")
    rib_label_strs = [display_from_seg(lab, side) for (lab, side) in RIB_ORDER]
    with _step("PC loadings per rib"):
        sm.plot_pc_loadings_per_rib(
            pca, rib_offsets_arr, rib_label_strs,
            figs_dir / "pc_loadings_per_rib",
        )
        if hasattr(sm, "plot_pc_loadings_per_rib_histo"):
            sm.plot_pc_loadings_per_rib_histo(
                pca, rib_offsets_arr, rib_label_strs,
                figs_dir / "pc_loadings_per_rib_histo",
            )

    with _step(f"PC deformation renders (first {N_PCS_DISPLAY} PCs, ±2 SD)"):
        # Face connectivity comes from the *extracted* template STLs (the
        # template patient is not written to the registered dir; registration
        # preserves triangulation). template_id.txt is written by the
        # pipeline driver into the registration output dir, per-preset.
        template_pid_path = reg_dir / "template_id.txt"
        if template_pid_path.exists():
            tpid = template_pid_path.read_text().strip()
            tpid_pdir = patient_stl_dir(extraction_dir, tpid)
            all_faces: list[np.ndarray] = []
            vertex_offset = 0
            missing_ribs = False
            for lab, side in RIB_ORDER:
                stl_path = tpid_pdir / f"{tpid}_rib{lab}_{side}.stl"
                if not stl_path.exists():
                    logger.warning(f"Template rib STL missing: {stl_path.name}")
                    missing_ribs = True
                    break
                tmpl = pv.read(str(stl_path))
                faces_raw = tmpl.faces.reshape(-1, 4)[:, 1:].astype(np.int32)
                all_faces.append(faces_raw + vertex_offset)
                vertex_offset += tmpl.n_points
            if not missing_ribs and all_faces:
                faces_np = np.concatenate(all_faces, axis=0)
                np.save(run_dir / "template_faces.npy", faces_np)

                plot_mean_shape_views(
                    mean_shape, faces_np, rib_offsets_arr,
                    rib_label_strs,
                    figs_dir / "mean_shape_views",
                )
                plot_pc_deformations(
                    mean_shape, pca, faces_np,
                    out_stem=figs_dir / "pc_deformations_surface",
                    n_pcs=N_PCS_DISPLAY,
                )
            else:
                logger.warning("Incomplete template ribs – skipping deformation renders")
        else:
            logger.warning("template_id.txt not found – skipping deformation renders")

    # ── PC scores ────────────────────────────────────────────────────────────
    with _step("PC scores"):
        scores_path = run_dir / "pc_scores_surface.csv"
        scores_df   = build_surface_scores_df(pca, X_flat, pids, meta_sub)
        scores_df.to_csv(scores_path, index=False)
    logger.info(f"PC scores → {scores_path}  ({scores_df.shape})")

    # ── Regression ───────────────────────────────────────────────────────────
    pc_cols = [c for c in scores_df.columns if c.startswith("PC_")]

    # ── Geometry prediction model (demographics → shape) ─────────────────────
    with _step("Geometry generator (demographics → shape)"):
        fit_and_save(scores_df, pc_cols, run_dir, pca.components_)

    with _step(f"PC regression ({len(pc_cols)} PCs, unadjusted + adjusted)"):
        results = run_pc_regression(scores_df, pc_cols)

    out_map = {
        "unadjusted": "pc_unadjusted.csv",
        "adjusted":   "pc_adjusted.csv",
        "targeted":   "pc_targeted.csv",
        "ttest":      "pc_ttest_surface.csv",
    }
    for key, fname in out_map.items():
        df = results.get(key)
        if df is not None and not df.empty:
            df.to_csv(run_dir / fname, index=False)
            logger.info(f"  {fname}: {len(df):,} rows")

    # Adjusted-β heatmap (sex + smoking now included as predictors).
    if not results["adjusted"].empty:
        with _step("Regression heatmap (adjusted)"):
            sm.plot_regression_heatmap_surface(results["adjusted"], figs_dir / "pc_regression_surface")

    # DAG-based per-exposure β heatmap (set 3): each column is that exposure's
    # back-door-adjusted estimate, not the omnibus partial coefficient.
    if not results["targeted"].empty:
        with _step("Regression heatmap (DAG-based)"):
            sm.plot_regression_heatmap_surface(
                results["targeted"], figs_dir / "pc_regression_targeted",
                title="Surface SSM – DAG-based per-exposure PC regression (standardised β)",
                subtitle="Per-exposure back-door adjustment set · HC3-robust SE",
            )

    # PC scores pair-plot (sex- and smoking-coloured variants), with the
    # unadjusted β-vector arrow overlaid on the sex panel.
    pc_disp = [f"PC_{k}" for k in range(1, N_PCS_DISPLAY + 1)]
    ev_disp = pca.explained_variance_
    arrow_sex_unadj = sm.pc_arrow_data(results["unadjusted"], "is_female", ev_disp, pc_disp)
    with _step("PC score pair-plots"):
        for color_by in ("sex", "smoking_status"):
            try:
                sm.plot_pc_scores_pairs(
                    scores_df, figs_dir / f"pc_scores_pairs_{color_by}",
                    n_pcs=N_PCS_DISPLAY, color_by=color_by,
                    arrow=(arrow_sex_unadj if color_by == "sex" else None),
                )
            except Exception as exc:  # Plotly/Kaleido renderer varies
                logger.warning(f"PC pair-plot ({color_by}) failed: {exc}")

    # Three-model viz: adjusted and DAG back-door partial-residual pair-plots
    # (marginal↔adjusted mediation), continuous binned-mean pair-plots, and the
    # β-vector field.
    with _step("PC pair-plots (adjusted + DAG) + β-vector field"):
        try:
            sm.plot_pc_scores_pairs_adjusted(
                scores_df, figs_dir / "pc_scores_pairs_adj_sex",
                color_by="sex", n_pcs=N_PCS_DISPLAY,
                arrow=sm.pc_arrow_data(results["adjusted"], "is_female", ev_disp, pc_disp),
            )
            sm.plot_pc_scores_pairs_targeted(
                scores_df, figs_dir / "pc_scores_pairs_dag_sex",
                color_by="sex", n_pcs=N_PCS_DISPLAY,
                arrow=sm.pc_arrow_data(results["targeted"], "is_female", ev_disp, pc_disp),
            )
            for pred in ("age", "height_cm", "weight_kg", "body_fat_pct"):
                sm.plot_pc_scores_pairs(
                    scores_df, figs_dir / f"pc_scores_pairs_{pred}",
                    n_pcs=N_PCS_DISPLAY, color_by=pred,
                    arrow=sm.pc_arrow_data(results["unadjusted"], pred, ev_disp, pc_disp),
                )
                sm.plot_pc_scores_pairs_adjusted(
                    scores_df, figs_dir / f"pc_scores_pairs_adj_{pred}",
                    color_by=pred, n_pcs=N_PCS_DISPLAY,
                    arrow=sm.pc_arrow_data(results["adjusted"], pred, ev_disp, pc_disp),
                )
                sm.plot_pc_scores_pairs_targeted(
                    scores_df, figs_dir / f"pc_scores_pairs_dag_{pred}",
                    color_by=pred, n_pcs=N_PCS_DISPLAY,
                    arrow=sm.pc_arrow_data(results["targeted"], pred, ev_disp, pc_disp),
                )
            sm.plot_pc_beta_vectors(
                results["unadjusted"], results["adjusted"],
                figs_dir / "pc_beta_vectors", tgt_df=results["targeted"],
                n_pcs=N_PCS_DISPLAY,
            )
        except Exception as exc:  # Altair/vl-convert renderer varies
            logger.warning(f"PC three-model viz failed: {exc}")

    total_elapsed = time.monotonic() - pipeline_t0
    m, s = divmod(total_elapsed, 60)
    logger.info(f"SSM Python pipeline complete – total wall time {m:.0f}m{s:.0f}s")


if __name__ == "__main__":
    main()
