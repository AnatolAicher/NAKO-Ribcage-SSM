"""Tests for the shared OLS engine (utils.regression).

The empty-result-schema tests guard the invariant that downstream runners
(run_adjusted, run_ssm) index result columns unconditionally – an all-filtered
cohort must still return the documented columns, not a column-less frame.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from utils.regression import adjusted_models, marginal_models, targeted_models


def _synthetic(n: int) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    x = rng.normal(size=n)
    return pd.DataFrame({
        "PC_1":           2.0 * x + rng.normal(size=n) * 0.1,
        "age":            x,
        "sex":            rng.choice(["Male", "Female"], size=n),
        "smoking_status": rng.choice(["Never", "Current"], size=n),
        "pack_years":     rng.uniform(0, 30, size=n),
        "patient_id":     np.arange(n),
    })


def test_marginal_recovers_known_slope():
    df = _synthetic(500)
    res = marginal_models(df, ["PC_1"], ["age"], se_mode="hc3")
    row = res[res["predictor"] == "age"].iloc[0]
    assert abs(row["beta"] - 2.0) < 0.05
    r = float(np.corrcoef(df["age"], df["PC_1"])[0, 1])
    assert abs(row["beta_std"] - r) < 1e-2


def test_marginal_empty_keeps_schema():
    # n=3 < min_n=30 → every pair filtered out, but the frame must keep columns.
    res = marginal_models(_synthetic(3), ["PC_1"], ["age"], se_mode="hc3")
    assert res.empty
    for col in ("outcome", "predictor", "beta_std", "p_value", "p_value_fdr"):
        assert col in res.columns


def test_adjusted_empty_keeps_schema():
    res = adjusted_models(_synthetic(10), ["PC_1"], ["age", "is_female"], se_mode="hc3")
    assert res.empty
    for col in ("outcome", "predictor", "beta_std", "partial_r2", "p_value", "p_value_fdr"):
        assert col in res.columns


def test_targeted_recovers_focal_and_tags_estimand():
    df = _synthetic(500)
    sets = {"age": [], "is_female": ["age"]}
    res = targeted_models(df, ["PC_1"], sets, total_effect=frozenset({"is_female"}))
    age_row = res[res["predictor"] == "age"].iloc[0]
    assert abs(age_row["beta"] - 2.0) < 0.1            # PC_1 ≈ 2·age
    assert age_row["estimand"] == "adjusted_association"
    assert age_row["adjusted_for"] == "(none)"
    assert 0.0 <= age_row["partial_r2"] <= 1.0         # FWL partial R² of focal exposure
    fem_row = res[res["predictor"] == "is_female"].iloc[0]
    assert fem_row["estimand"] == "total_effect"       # in the total_effect set
    assert fem_row["adjusted_for"] == "age"            # its DAG adjustment set


def test_targeted_empty_keeps_schema():
    res = targeted_models(_synthetic(3), ["PC_1"], {"age": []}, min_n=30)
    assert res.empty
    for col in ("outcome", "predictor", "beta_std", "partial_r2", "p_value",
                "p_value_fdr", "estimand", "adjusted_for"):
        assert col in res.columns
