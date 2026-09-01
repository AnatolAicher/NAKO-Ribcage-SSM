"""Styner-style shape model evaluation metrics.

Three pipeline-agnostic metrics, all computed in shape space (after GPA and
flattening to 1-D vectors):

    compactness        cumulative variance vs. mode count
    generalisation     leave-one-out reconstruction error vs. mode count
    specificity        random-sample-from-model NN distance to training set

The implementations take a stack of corresponded shapes ``(N, n_pts, 3)`` or
``(N, D)`` and return arrays indexed by mode count.

References
----------
Davies et al. (2002); Cates et al. (2017).
"""
from __future__ import annotations

import logging

import numpy as np
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits

logger = logging.getLogger(__name__)


def _gen_fold(
    train_idx: np.ndarray,
    test_idx:  np.ndarray,
    X:         np.ndarray,
    max_k:     int,
    mode_counts: np.ndarray,
    n_pts:     int,
) -> tuple[np.ndarray, np.ndarray]:
    """One K-fold / LOO step. Returns ``(test_idx, errs)`` where ``errs`` has
    shape ``(|test|, len(mode_counts))``. BLAS threads pinned to 1 so that
    joblib parallelism across folds doesn't oversubscribe cores."""
    with threadpool_limits(limits=1):
        pca = PCA(n_components=max_k)
        pca.fit(X[train_idx])
        test_scores = pca.transform(X[test_idx])
        out = np.empty((len(test_idx), len(mode_counts)), dtype=np.float64)
        for j, k in enumerate(mode_counts):
            recons = pca.mean_ + test_scores[:, :k] @ pca.components_[:k]
            diffs  = (X[test_idx] - recons).reshape(len(test_idx), n_pts, 3)
            out[:, j] = np.sqrt(np.mean(np.sum(diffs ** 2, axis=2), axis=1))
    return test_idx, out


def _flatten(shapes: np.ndarray) -> np.ndarray:
    """``(N, n_pts, 3) → (N, D)`` where ``D = 3 * n_pts``; pass through if 2-D."""
    if shapes.ndim == 3:
        n, p, d = shapes.shape
        return shapes.reshape(n, p * d)
    if shapes.ndim == 2:
        return shapes
    raise ValueError(f"shapes must be 2-D or 3-D, got shape {shapes.shape}")


def compactness(shapes: np.ndarray, max_modes: int | None = None) -> np.ndarray:
    """Cumulative explained-variance ratio vs. mode count.

    Returns an array of length ``min(N-1, D, max_modes)`` where entry ``k`` is
    the cumulative variance explained by the first ``k+1`` PCA modes.
    """
    X = _flatten(shapes)
    n, D = X.shape
    rank = min(n - 1, D)
    k = min(rank, max_modes) if max_modes is not None else rank
    pca = PCA(n_components=k)
    pca.fit(X)
    return np.cumsum(pca.explained_variance_ratio_)


def generalisation(
    shapes: np.ndarray,
    mode_counts: list[int] | np.ndarray | None = None,
    n_folds: int | None = None,
    n_workers: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Leave-one-out (default) or K-fold reconstruction error vs. mode count.

    Fit PCA on each training partition, project + reconstruct every held-out
    sample with the first ``k`` modes, and report the per-vertex RMS error
    averaged across all held-out samples.

    Parameters
    ----------
    shapes
        ``(N, n_pts, 3)`` or ``(N, D)``.
    mode_counts
        Modes to evaluate; default = ``1..rank``.
    n_folds
        ``None`` or ``0``: true leave-one-out (``N`` PCA fits, ``O(N^2)``).
        ``>= 2``: K-fold CV (shuffled, seed 42) with ``n_folds`` fits. Falls
        back to LOO with a warning when ``n_folds > N``.
    n_workers
        Process-based fold parallelism via joblib. ``1`` (default) runs
        sequentially. Each worker memmaps ``shapes`` (zero-copy) but
        materialises a fresh ``(|train|, D)`` array (~7 GB at full cohort),
        so peak RAM ≈ n_workers × |train| × D × 8 bytes.

    Returns
    -------
    mode_counts : (K,) int
    errors_mean : (K,) float – mean per-vertex RMS error across held-out samples.
    """
    X = _flatten(shapes)
    n, D = X.shape

    if n_folds is None or n_folds == 0:
        splits: list[tuple[np.ndarray, np.ndarray]] = [
            (np.delete(np.arange(n), i), np.array([i])) for i in range(n)
        ]
        min_train = n - 1
    elif n_folds < 2:
        raise ValueError(f"n_folds must be 0 (LOO) or >= 2; got {n_folds}")
    elif n_folds > n:
        logger.warning(
            f"n_folds={n_folds} > n_samples={n}; falling back to leave-one-out."
        )
        splits = [(np.delete(np.arange(n), i), np.array([i])) for i in range(n)]
        min_train = n - 1
    else:
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        splits = list(kf.split(X))
        min_train = min(len(tr) for tr, _ in splits)

    rank = min(min_train - 1, D)
    if mode_counts is None:
        mode_counts = list(range(1, rank + 1))
    mode_counts = np.asarray(mode_counts, dtype=int)
    mode_counts = mode_counts[mode_counts <= rank]
    if mode_counts.size == 0:
        raise ValueError(f"all requested mode counts exceed rank ceiling {rank}")
    max_k = int(mode_counts.max())

    n_pts = X.shape[1] // 3
    errs = np.zeros((n, len(mode_counts)), dtype=np.float64)

    eff_workers = max(1, min(int(n_workers), len(splits)))
    results = Parallel(n_jobs=eff_workers, prefer="processes")(
        delayed(_gen_fold)(tr, te, X, max_k, mode_counts, n_pts)
        for tr, te in splits
    )
    for test_idx, fold_errs in results:
        errs[test_idx, :] = fold_errs

    return mode_counts, errs.mean(axis=0)


def specificity(
    shapes: np.ndarray,
    n_samples: int = 1000,
    mode_counts: list[int] | np.ndarray | None = None,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Random-sample-from-model NN distance to training set vs. mode count.

    For each ``k`` in ``mode_counts``, sample ``n_samples`` shapes from the PCA
    model (Gaussian in PC space with empirical variances), find the nearest
    training shape per sample, and report the mean NN per-vertex RMS distance.

    Returns
    -------
    mode_counts      : (K,) int
    specificity_mean : (K,) float – lower is better (samples close to data).
    """
    X = _flatten(shapes)
    n, D = X.shape
    rank = min(n - 1, D)
    if mode_counts is None:
        mode_counts = list(range(1, rank + 1))
    mode_counts = np.asarray(mode_counts, dtype=int)
    mode_counts = mode_counts[mode_counts <= rank]
    if mode_counts.size == 0:
        raise ValueError(f"all requested mode counts exceed rank ceiling {rank}")
    max_k = int(mode_counts.max())
    n_pts = X.shape[1] // 3

    pca = PCA(n_components=max_k)
    train_scores = pca.fit_transform(X)
    rng = np.random.default_rng(seed)

    spec = np.zeros(len(mode_counts), dtype=np.float64)
    X_pts = X.reshape(n, n_pts, 3)
    for j, k in enumerate(mode_counts):
        std_k = train_scores[:, :k].std(axis=0)
        samples_scores = rng.normal(size=(n_samples, k)) * std_k
        samples = pca.mean_ + samples_scores @ pca.components_[:k]
        samples_pts = samples.reshape(n_samples, n_pts, 3)

        # Sample-by-sample NN search to avoid materialising the full
        # (S, N, D) broadcast (tens of GB for full-rib-cage shapes).
        per_sample_min = np.empty(n_samples, dtype=np.float64)
        for s in range(n_samples):
            diff = X_pts - samples_pts[s]                                # (N, n_pts, 3)
            d = np.sqrt(np.mean(np.sum(diff * diff, axis=2), axis=1))    # (N,) RMS
            per_sample_min[s] = d.min()
        spec[j] = per_sample_min.mean()

    return mode_counts, spec
