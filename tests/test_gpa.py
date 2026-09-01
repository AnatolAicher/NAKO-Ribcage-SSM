"""Tests for ``ssm.gpa`` – Generalised Procrustes Analysis."""
from __future__ import annotations

import numpy as np
import pytest

from ssm.gpa import align_to_target, gpa


def _random_rotation(rng: np.random.Generator) -> np.ndarray:
    """Sample a uniformly-random proper rotation in SO(3) via QR of a Gaussian."""
    A = rng.standard_normal((3, 3))
    Q, R = np.linalg.qr(A)
    Q = Q * np.sign(np.diag(R))     # make QR unique
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q


def test_align_to_target_recovers_rotation():
    rng = np.random.default_rng(0)
    target = rng.standard_normal((40, 3))
    R = _random_rotation(rng)
    t = rng.standard_normal(3) * 5.0
    source = (target - target.mean(0)) @ R.T + target.mean(0) + t

    aligned = align_to_target(source, target)

    # After alignment the source should match the target up to a tiny
    # numerical residual.
    assert np.allclose(aligned, target, atol=1e-9)


def test_align_to_target_corrects_reflection():
    """With a reflected source the Kabsch det-correction must still return
    a proper rotation that minimises the Procrustes distance."""
    rng = np.random.default_rng(1)
    target = rng.standard_normal((50, 3))
    # Apply a reflection (improper rotation).
    F = np.diag([1.0, 1.0, -1.0])
    source = (target - target.mean(0)) @ F + target.mean(0)

    aligned = align_to_target(source, target)
    # The optimal proper-rotation alignment cannot fully recover a
    # reflection, but the Kabsch trick must keep the rotation proper
    # (det = +1) – ascertain the resulting alignment is no worse than
    # leaving the reflected source untouched.
    assert np.linalg.norm(aligned - target) <= np.linalg.norm(source - target) + 1e-9


def test_gpa_idempotent_on_already_aligned_shapes():
    """If all input shapes are identical (already aligned), GPA must return
    them unchanged and the mean must equal each input."""
    rng = np.random.default_rng(2)
    template = rng.standard_normal((30, 3))
    shapes = np.stack([template] * 5, axis=0)

    aligned, mean, history = gpa(shapes, max_iter=20, tol=1e-8)

    assert aligned.shape == shapes.shape
    assert mean.shape == template.shape
    assert history.ndim == 1 and history.size >= 1
    # Centring is part of GPA, so each shape should match the mean.
    centred = template - template.mean(0)
    for s in aligned:
        assert np.allclose(s, centred, atol=1e-5)
    assert np.allclose(mean, centred, atol=1e-5)


def test_gpa_recovers_known_rotations():
    """GPA on rigidly-rotated copies of one template must recover the
    template up to a global frame choice."""
    rng = np.random.default_rng(3)
    template = rng.standard_normal((60, 3))
    rotations = [_random_rotation(rng) for _ in range(8)]
    translations = rng.standard_normal((8, 3)) * 3.0
    shapes = np.stack(
        [(template - template.mean(0)) @ R.T + t for R, t in zip(rotations, translations)],
        axis=0,
    )

    aligned, mean, history = gpa(shapes, max_iter=100, tol=1e-9)
    # History is monotonically non-increasing once close to convergence.
    assert history.size >= 1

    # The GPA mean must be a rigid rotation of the template (centred).
    centred = template - template.mean(0)
    R_to_template = align_to_target(mean, centred)
    assert np.linalg.norm(R_to_template - centred) < 1e-3

    # All aligned shapes should now collapse onto the same point cloud.
    for i in range(len(aligned)):
        for j in range(i + 1, len(aligned)):
            assert np.allclose(aligned[i], aligned[j], atol=1e-3)


def test_gpa_proper_rotations_only():
    """Every per-iteration rotation produced by ``align_to_target`` must
    have det = +1 (no improper rotations / reflections)."""
    rng = np.random.default_rng(4)
    target = rng.standard_normal((25, 3))
    for _ in range(20):
        source = rng.standard_normal((25, 3))
        aligned = align_to_target(source, target)
        # Recover the rotation matrix used: aligned ≈ (source - mean) R + tgt_mean.
        src_c = source - source.mean(0)
        ali_c = aligned - target.mean(0)
        R = np.linalg.lstsq(src_c, ali_c, rcond=None)[0]
        assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-6)
