"""Demographics→rib-cage geometry prediction model (the SSM generator).

Fits one OLS per PC of the GPA/PCA shape space on a demographic design
(`is_female`, age, height, weight, body-fat %, `ever_smoker`, pack-years), so a
demographic vector reconstructs a subject-specific mean rib-cage surface
(``mean + scores @ components``). This is the *prediction* counterpart to the
descriptive association models: it generates geometry, it does not test effects.

Continuous predictors are centred at cohort means (stored as ``offsets``) so the
saved intercept is the prediction at the mean profile; binary predictors are not
centred. Held-out accuracy is the per-vertex distance (mm) of the
demographics-predicted surface from each patient's own full-rank reconstruction,
estimated by k-fold cross-validation over patients.

Outputs:
  geometry_generator.npz   B, intercept, offsets, predictors, in-sample R², n
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from settings import PRED_PREDICTORS
from utils.design import DESIGN_BINARY, add_design_columns

logger = logging.getLogger(__name__)


def _design(
    df: pd.DataFrame,
    predictors: list[str],
    offsets: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the centred design matrix ``[1, x_1-off_1, …]``; compute offsets if absent."""
    cols = [np.ones(len(df))]
    out_off = np.empty(len(predictors)) if offsets is None else offsets
    for j, p in enumerate(predictors):
        x = df[p].to_numpy(dtype=float)
        if offsets is None:
            out_off[j] = 0.0 if p in DESIGN_BINARY else float(np.nanmean(x))
        cols.append(x - out_off[j])
    return np.column_stack(cols), out_off


def fit_generator(
    scores_df: pd.DataFrame,
    pc_cols: list[str],
    predictors: list[str] = PRED_PREDICTORS,
) -> dict[str, np.ndarray | list[str] | int]:
    """Fit one OLS per PC; return coefficients, intercepts, offsets and in-sample R²."""
    df = add_design_columns(scores_df)
    predictors = [p for p in predictors if p in df.columns]
    needed = [c for c in (*predictors, *pc_cols) if c in df.columns]
    sub = df.dropna(subset=needed)

    X, offsets = _design(sub, predictors)
    Y = sub[pc_cols].to_numpy(dtype=float)
    coef, *_ = np.linalg.lstsq(X, Y, rcond=None)          # (1 + n_pred, n_pc)

    resid = Y - X @ coef
    ss_res = (resid ** 2).sum(axis=0)
    ss_tot = ((Y - Y.mean(axis=0)) ** 2).sum(axis=0)
    r2 = np.where(ss_tot > 0, 1.0 - ss_res / ss_tot, np.nan)

    return {
        "B": coef[1:].T.astype(np.float64),             # (n_pc, n_pred)
        "intercept": coef[0].astype(np.float64),         # (n_pc,)
        "offsets": offsets.astype(np.float64),           # (n_pred,)
        "predictors": list(predictors),
        "pc_cols": list(pc_cols),
        "r2": r2.astype(np.float64),
        "n": int(len(sub)),
    }


def save_generator(path: Path, gen: dict) -> None:
    """Write the generator to ``geometry_generator.npz``."""
    np.savez_compressed(
        path,
        B=gen["B"],
        intercept=gen["intercept"],
        offsets=gen["offsets"],
        predictors=np.array(gen["predictors"], dtype=object),
        pc_cols=np.array(gen["pc_cols"], dtype=object),
        r2=gen["r2"],
        n=np.int64(gen["n"]),
    )


def predict_scores(gen: dict, demographics: dict[str, float]) -> np.ndarray:
    """Predict the full PC-score vector for one demographic profile.

    ``demographics`` keys are the design predictors (``is_female``, ``age``,
    ``height_cm``, ``weight_kg``, ``body_fat_pct``, ``ever_smoker``,
    ``pack_years``); for an "I-don't-care" axis pass the cohort-marginal value.
    """
    predictors = list(gen["predictors"])
    x = np.array([float(demographics[p]) for p in predictors])
    centred = x - np.asarray(gen["offsets"], dtype=float)
    return np.asarray(gen["intercept"], dtype=float) + gen["B"] @ centred


def predict_geometry(
    gen: dict,
    pca_mean: np.ndarray,
    pca_components: np.ndarray,
    demographics: dict[str, float],
    n_modes: int | None = None,
) -> np.ndarray:
    """Predicted (n_points, 3) rib-cage surface for one demographic profile."""
    scores = predict_scores(gen, demographics)
    k = len(scores) if n_modes is None else n_modes
    flat = pca_mean + scores[:k] @ pca_components[:k]
    return flat.reshape(-1, 3)


def holdout_error(
    scores_df: pd.DataFrame,
    pc_cols: list[str],
    pca_components: np.ndarray,
    predictors: list[str] = PRED_PREDICTORS,
    k_modes: tuple[int, ...] = (3, 7, 25, 113),
    n_folds: int = 10,
    seed: int = 0,
) -> pd.DataFrame:
    """K-fold per-vertex geometry-prediction error (mm) vs each patient's own reconstruction.

    Error is measured against the full-rank (all-PC) reconstruction of the
    patient's true scores — the basis's best representation of their cage — so it
    isolates demographic-prediction error from the <1% basis-truncation residual.
    """
    df = add_design_columns(scores_df)
    predictors = [p for p in predictors if p in df.columns]
    needed = [c for c in (*predictors, *pc_cols) if c in df.columns]
    sub = df.dropna(subset=needed).reset_index(drop=True)

    Y = sub[pc_cols].to_numpy(dtype=np.float32)
    comp = pca_components.astype(np.float32)
    is_female = sub["is_female"].to_numpy(dtype=float)
    n = len(sub)

    folds = np.array_split(np.random.default_rng(seed).permutation(n), n_folds)
    per_patient: dict[int, np.ndarray] = {k: np.empty(n) for k in k_modes}
    sex_of = np.empty(n)

    for test_idx in folds:
        train_idx = np.setdiff1d(np.arange(n), test_idx, assume_unique=False)
        X_tr, off = _design(sub.iloc[train_idx], predictors)
        X_te, _ = _design(sub.iloc[test_idx], predictors, offsets=off)
        coef, *_ = np.linalg.lstsq(X_tr.astype(np.float64),
                                   Y[train_idx].astype(np.float64), rcond=None)
        Yhat = (X_te @ coef).astype(np.float32)                 # (n_te, n_pc)
        Ytrue = Y[test_idx]
        sex_of[test_idx] = is_female[test_idx]

        for k in k_modes:
            diff = -Ytrue.copy()
            diff[:, :k] = Yhat[:, :k] - Ytrue[:, :k]            # truncate prediction at k
            recon = diff @ comp                                  # (n_te, n_pts*3)
            dist = np.linalg.norm(recon.reshape(len(test_idx), -1, 3), axis=2)
            per_patient[k][test_idx] = dist.mean(axis=1)         # per-patient mean per-vertex mm

    rows = []
    for k in k_modes:
        e = per_patient[k]
        rows.append({
            "n_modes": k,
            "mean_mm": float(e.mean()),
            "p95_mm": float(np.percentile(e, 95)),
            "mean_mm_male": float(e[sex_of == 0].mean()),
            "mean_mm_female": float(e[sex_of == 1].mean()),
        })
    return pd.DataFrame(rows)


def fit_and_save(
    scores_df: pd.DataFrame,
    pc_cols: list[str],
    out_dir: Path,
    pca_components: np.ndarray | None = None,
) -> dict:
    """Fit the generator, write ``geometry_generator.npz``, and log held-out error."""
    gen = fit_generator(scores_df, pc_cols)
    save_generator(out_dir / "geometry_generator.npz", gen)
    logger.info(
        f"geometry generator: n={gen['n']:,}, {len(pc_cols)} PCs × "
        f"{len(gen['predictors'])} predictors  → geometry_generator.npz"
    )
    if pca_components is not None:
        err = holdout_error(scores_df, pc_cols, pca_components)
        err.to_csv(out_dir / "geometry_generator_holdout.csv", index=False)
        full = err[err["n_modes"] == err["n_modes"].max()].iloc[0]
        logger.info(
            f"  held-out per-vertex error @ k={int(full['n_modes'])}: "
            f"mean {full['mean_mm']:.2f} mm, p95 {full['p95_mm']:.2f} mm"
        )
    return gen
