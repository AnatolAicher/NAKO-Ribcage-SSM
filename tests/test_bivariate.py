"""Sanity tests for the surviving ``bivariate.bivariate`` helpers."""
from __future__ import annotations

import numpy as np
import pandas as pd

from bivariate.bivariate import _bh_correct, run_welch_ttest


# ── BH-FDR ──────────────────────────────────────────────────────────────────

def test_bh_correct_monotone_and_q_geq_p():
    """BH-corrected q-values must be monotone non-decreasing in raw p-values
    when p is sorted, and each q must be ≥ p."""
    rng = np.random.default_rng(0)
    p = pd.Series(np.sort(rng.uniform(size=20)))
    q = _bh_correct(p)
    assert (q >= p.values - 1e-12).all()
    assert np.all(np.diff(q) >= -1e-12)


def test_bh_correct_preserves_nan():
    p = pd.Series([0.01, np.nan, 0.04, 0.5])
    q = _bh_correct(p)
    assert np.isnan(q[1])
    assert not np.isnan(q[0])
    assert not np.isnan(q[2])
    assert not np.isnan(q[3])


# ── Welch t ──────────────────────────────────────────────────────────────────

def test_run_welch_ttest_separates_known_means():
    """Two groups with means 0 and 1 (sd 1) at n=400 each should yield
    Cohen's d ≈ 1 and a vanishing p-value."""
    rng = np.random.default_rng(3)
    a = rng.normal(loc=0, scale=1.0, size=400)
    b = rng.normal(loc=1, scale=1.0, size=400)
    df = pd.DataFrame({
        "shape": np.concatenate([a, b]),
        "sex":   ["Male"] * 400 + ["Female"] * 400,
    })
    res = run_welch_ttest(df, ["shape"])
    assert len(res) == 1
    row = res.iloc[0]
    assert row["p_value"] < 1e-20
    assert abs(row["cohen_d"] - 1.0) < 0.15


# ── Empty-result schema invariant ─────────────────────────────────────────────
# An all-filtered result must still carry the documented columns; pc_regression
# indexes p_value_fdr / shape_param unconditionally.

def test_run_welch_ttest_empty_keeps_schema():
    df = pd.DataFrame({"shape": [1.0, 2.0], "sex": ["Male", "Male"]})  # one group
    res = run_welch_ttest(df, ["shape"])
    assert res.empty
    for col in ("shape_param", "p_value", "p_value_fdr", "cohen_d", "n_male", "n_female"):
        assert col in res.columns
