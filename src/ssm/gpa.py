"""Generalised Procrustes Analysis (translation + rotation, scale retained).

Iteratively aligns every shape to the running mean via orthogonal Procrustes
(3×3 SVD per shape, with reflection correction); stops when the relative
Frobenius change in the mean falls below ``tol``. Scale is retained because
body size is a study variable.
"""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def align_to_target(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Rigid alignment (translation + rotation, no scale) of ``source`` onto ``target``.

    Parameters
    ----------
    source, target
        ``(N, 3)`` landmark arrays.

    Returns
    -------
    ``(N, 3)`` — aligned source in the target's coordinate frame.
    """
    src_c = source - source.mean(axis=0)
    tgt_c = target - target.mean(axis=0)
    tgt_centroid = target.mean(axis=0)

    # Optimal rotation: minimise ``||tgt_c − src_c R||_F``.
    M = tgt_c.T @ src_c
    U, _, Vt = np.linalg.svd(M)

    # Reflection correction (det = +1 enforced).
    d = np.linalg.det(Vt.T @ U.T)
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T

    return src_c @ R + tgt_centroid


def gpa(
    shapes: np.ndarray,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generalised Procrustes Analysis over a stack of shapes.

    Parameters
    ----------
    shapes
        ``(N_patients, N_landmarks, 3)``.
    max_iter
        Maximum number of iterations.
    tol
        Convergence threshold on relative mean-shape change.

    Returns
    -------
    aligned    : ``(N_patients, N_landmarks, 3)`` — GPA-aligned shapes.
    mean_shape : ``(N_landmarks, 3)`` — Procrustes mean shape.
    rms_history: ``(n_iterations,)`` — relative mean-shape change at
                 each iteration; useful for the convergence diagnostic
                 figure (:func:`ssm.plots_ssm.plot_gpa_convergence`).
    """
    n_subjects, n_landmarks, _ = shapes.shape
    if n_subjects < 2:
        raise ValueError(
            f"gpa() needs at least 2 shapes; got {n_subjects}. "
            f"Procrustes alignment of a single shape is the identity."
        )
    aligned = shapes.copy().astype(np.float64)

    aligned -= aligned.mean(axis=1, keepdims=True)
    mean = aligned[0].copy()

    history: list[float] = []
    for iteration in range(1, max_iter + 1):
        # Batched orthogonal Procrustes onto the current mean.
        aligned -= aligned.mean(axis=1, keepdims=True)         # re-centre
        tgt_centroid = mean.mean(axis=0)                       # (3,)
        tgt_c = mean - tgt_centroid                            # (K, 3)

        # Cross-covariance: M[n] = tgt_c.T @ aligned[n]   (3×3 each).
        M = np.einsum("ki,nkj->nij", tgt_c, aligned)           # (N, 3, 3)

        # Batched 3×3 SVD with reflection correction enforced per shape.
        U, _, Vt = np.linalg.svd(M)
        V  = np.transpose(Vt, (0, 2, 1))
        Ut = np.transpose(U,  (0, 2, 1))
        d  = np.linalg.det(V @ Ut)                             # (N,)
        D  = np.zeros((n_subjects, 3, 3), dtype=aligned.dtype)
        D[:, 0, 0] = 1.0
        D[:, 1, 1] = 1.0
        D[:, 2, 2] = d
        R  = V @ D @ Ut                                        # (N, 3, 3)

        # Apply rotation, then place at the mean's centroid.
        aligned = aligned @ R                                  # (N, K, 3)
        aligned += tgt_centroid

        new_mean = aligned.mean(axis=0)
        change = np.linalg.norm(new_mean - mean) / (np.linalg.norm(mean) + 1e-12)
        mean = new_mean
        history.append(float(change))

        if iteration % 5 == 0 or iteration == 1:
            logger.info(f"  GPA iter {iteration:3d}: relative change = {change:.2e}")

        if change < tol:
            logger.info(f"  GPA converged at iteration {iteration} (change={change:.2e})")
            break
    else:
        logger.warning(f"  GPA did not converge after {max_iter} iterations")

    return (
        aligned.astype(np.float32),
        mean.astype(np.float32),
        np.asarray(history, dtype=np.float64),
    )
