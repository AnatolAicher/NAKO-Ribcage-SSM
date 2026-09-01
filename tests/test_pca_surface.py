"""Tests for ``ssm.pca_surface.fit_pca``."""
from __future__ import annotations

import numpy as np

from ssm.pca_surface import fit_pca


def _low_rank_synthetic(
    n: int = 40, n_pts: int = 50, true_rank: int = 3, noise: float = 1e-6, seed: int = 0,
) -> np.ndarray:
    """Generate ``(n, n_pts, 3)`` shapes lying on a ``true_rank``-dim affine
    subspace plus small isotropic noise.  Mode variances are spaced
    geometrically (3.0, 2.4, 1.92, …) so every mode contributes a
    similar share of total variance – useful for tests that need PCA
    to actually retain *all* ``true_rank`` modes."""
    rng = np.random.default_rng(seed)
    d = 3 * n_pts
    basis = rng.standard_normal((true_rank, d))
    basis, _ = np.linalg.qr(basis.T)
    basis = basis.T                                                   # (rank, d)
    sds = 3.0 * (0.8 ** np.arange(true_rank))
    coeffs = rng.standard_normal((n, true_rank)) * sds
    mean = rng.standard_normal(d) * 10.0
    noise_mat = rng.standard_normal((n, d)) * noise
    flat = mean + coeffs @ basis + noise_mat
    return flat.reshape(n, n_pts, 3)


def test_pca_recovers_low_rank_signal():
    shapes = _low_rank_synthetic(n=40, n_pts=50, true_rank=3, noise=1e-6, seed=42)
    pca, scores, mean_vec, _ = fit_pca(shapes, variance_threshold=0.95)

    # With geometrically-spaced mode variances and threshold = 0.95, all
    # three signal modes are needed to cross the threshold.
    assert pca.n_components_ >= 3
    # First three modes should explain ≥ 99 % of total variance (noise tiny).
    assert pca.explained_variance_ratio_[:3].sum() > 0.99
    # Mean vector matches numpy's column mean of the flattened data.
    flat = shapes.reshape(40, -1)
    assert np.allclose(mean_vec, flat.mean(0), atol=1e-8)


def test_pca_scores_round_trip():
    """``mean + scores @ components`` must reconstruct the input within
    the truncation residual."""
    shapes = _low_rank_synthetic(n=30, n_pts=40, true_rank=2, noise=1e-8, seed=7)
    pca, scores, mean_vec, _ = fit_pca(shapes, variance_threshold=0.99)

    flat = shapes.reshape(30, -1)
    recon = mean_vec + scores @ pca.components_
    # With true_rank=2 and noise=1e-8, reconstruction error should be tiny.
    err = np.linalg.norm(flat - recon) / np.linalg.norm(flat - mean_vec)
    assert err < 1e-3


def test_pca_keeps_minimum_components_for_threshold():
    """``fit_pca`` should keep the smallest k whose cumulative variance
    crosses the requested threshold, not more."""
    shapes = _low_rank_synthetic(n=50, n_pts=30, true_rank=5, noise=1e-3, seed=11)
    pca, _, _, _ = fit_pca(shapes, variance_threshold=0.90)

    # Removing the last component must drop us below the threshold.
    cum = np.cumsum(pca.explained_variance_ratio_)
    assert cum[-1] >= 0.90
    if pca.n_components_ > 1:
        assert cum[-2] < 0.90


def test_pca_is_deterministic():
    """Two fits with the same input must produce identical components
    (PCA_RANDOM_SEED pinned in pca_surface)."""
    shapes = _low_rank_synthetic(n=30, n_pts=25, true_rank=3, seed=99)
    pca1, scores1, _, _ = fit_pca(shapes, variance_threshold=0.95)
    pca2, scores2, _, _ = fit_pca(shapes, variance_threshold=0.95)
    assert pca1.n_components_ == pca2.n_components_
    assert np.allclose(np.abs(pca1.components_), np.abs(pca2.components_))
    assert np.allclose(np.abs(scores1), np.abs(scores2))
