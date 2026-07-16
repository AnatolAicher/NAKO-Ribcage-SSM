"""Tests for the Styner triad implementations in ``utils.shape_evaluation``."""
from __future__ import annotations

import numpy as np

from utils.shape_evaluation import compactness, generalisation, specificity


def _synthetic_shapes(n: int = 25, n_pts: int = 30, rank: int = 3, seed: int = 0):
    rng = np.random.default_rng(seed)
    d = 3 * n_pts
    basis, _ = np.linalg.qr(rng.standard_normal((d, rank)))
    coeffs = rng.standard_normal((n, rank)) * np.array([3.0, 1.0, 0.3])[:rank]
    flat = coeffs @ basis.T + rng.standard_normal((n, d)) * 0.05
    return flat.reshape(n, n_pts, 3)


def test_compactness_monotone_and_in_unit_interval():
    shapes = _synthetic_shapes(n=20, n_pts=15, rank=3, seed=1)
    cmp = compactness(shapes, max_modes=10)

    # Cumulative variance ratios are non-decreasing and lie in [0, 1].
    assert (cmp >= 0).all() and (cmp <= 1.0 + 1e-9).all()
    assert np.all(np.diff(cmp) >= -1e-12)
    # All-modes total ≈ 1 for non-degenerate data.
    assert cmp[-1] > 0.99


def test_generalisation_decreases_then_plateaus():
    """Adding more modes should not *increase* the LOO error on average."""
    shapes = _synthetic_shapes(n=20, n_pts=15, rank=3, seed=2)
    modes = np.arange(1, 8)
    out_modes, errs = generalisation(shapes, mode_counts=modes)

    assert (out_modes == modes).all()
    assert (errs >= 0).all()
    # First mode should give the largest error; later modes do not exceed it.
    assert errs[0] >= errs[-1] - 1e-9


def test_specificity_non_negative_and_seeded():
    shapes = _synthetic_shapes(n=20, n_pts=15, rank=3, seed=3)
    modes = np.arange(1, 5)

    _, spec_a = specificity(shapes, n_samples=50, mode_counts=modes, seed=42)
    _, spec_b = specificity(shapes, n_samples=50, mode_counts=modes, seed=42)
    # Same seed → identical results (sanity for the rng pinning).
    assert np.allclose(spec_a, spec_b)
    assert (spec_a >= 0).all()


def test_compactness_handles_2d_input():
    """Both ``(N, n_pts, 3)`` and ``(N, D)`` should be accepted."""
    shapes_3d = _synthetic_shapes(n=15, n_pts=12, rank=2, seed=4)
    shapes_2d = shapes_3d.reshape(15, -1)
    c3 = compactness(shapes_3d, max_modes=5)
    c2 = compactness(shapes_2d, max_modes=5)
    assert np.allclose(c3, c2)
