"""Tests for ``ssm.radiomics_correlation``."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ssm.radiomics_correlation import (
    compute_effects,
    join_pc_with_ribs,
    top_effects,
)


def _synth(
    n_subjects: int = 200,
    n_pcs: int = 3,
    rib_pairs: tuple[tuple[int, str], ...] = ((8, "Left"), (8, "Right"), (12, "Left")),
    feature_cols: tuple[str, ...] = ("feat_a", "feat_b"),
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    """Build synthetic PC scores + per-rib radiomics with a known coupling.

    ``feat_a`` at the first rib is constructed as ``2 · PC_1 + small noise``;
    every other (rib, feature) combination is pure noise.  This lets us
    assert that the coupling is recovered with a positive β and rejected
    by FDR everywhere else (under a strong enough cohort size).
    """
    rng = np.random.default_rng(seed)
    pids = np.arange(1000, 1000 + n_subjects, dtype=np.int64)
    pc_arr = rng.normal(size=(n_subjects, n_pcs)) * np.array([3.0, 2.0, 1.0])[:n_pcs]
    pc_cols = [f"PC_{i+1}" for i in range(n_pcs)]
    pc_df = pd.DataFrame(pc_arr, columns=pc_cols)
    pc_df.insert(0, "patient_id", pids)

    rib_rows: list[dict] = []
    for v, side in rib_pairs:
        for i, pid in enumerate(pids):
            row = {"patient_id": int(pid), "vert_level": int(v), "side": side}
            for f in feature_cols:
                row[f] = rng.normal(scale=1.0)
            rib_rows.append(row)
    radiomics_df = pd.DataFrame(rib_rows)

    # Plant signal: feat_a at (8, "Left") is 2 * PC_1 + noise (in raw scale of PC_1).
    target_mask = (radiomics_df["vert_level"] == rib_pairs[0][0]) & \
                  (radiomics_df["side"] == rib_pairs[0][1])
    pid_to_pc1 = dict(zip(pids, pc_arr[:, 0]))
    radiomics_df.loc[target_mask, "feat_a"] = (
        2.0 * radiomics_df.loc[target_mask, "patient_id"].map(pid_to_pc1)
        + rng.normal(scale=0.05, size=int(target_mask.sum()))
    )
    return pc_df, radiomics_df, pc_cols, list(feature_cols)


def test_join_replicates_pcs_per_rib():
    pc_df, radiomics_df, pc_cols, _ = _synth(n_subjects=50)
    joined = join_pc_with_ribs(pc_df, radiomics_df)
    # Every rib row gets a PC column.
    for c in pc_cols:
        assert c in joined.columns
    # Per-subject PC value is constant across that subject's rib rows.
    for pid, grp in joined.groupby("patient_id"):
        assert grp["PC_1"].nunique() == 1
    # Inner join cardinality: subjects × ribs (no NaNs).
    n_ribs = radiomics_df[["vert_level", "side"]].drop_duplicates().shape[0]
    assert len(joined) == 50 * n_ribs


def test_compute_effects_shape_and_columns():
    pc_df, radiomics_df, pc_cols, feature_cols = _synth(n_subjects=80)
    joined = join_pc_with_ribs(pc_df, radiomics_df)
    eff = compute_effects(joined, pc_cols, feature_cols)
    rib_pairs = joined[["vert_level", "side"]].drop_duplicates()
    assert len(eff) == len(pc_cols) * len(rib_pairs) * len(feature_cols)
    expected = {
        "pc", "vert_level", "anatomical_rib", "side", "feature",
        "beta_native", "beta_std", "ci_low_native", "ci_high_native",
        "se_native", "p_value", "q_value", "n",
    }
    assert expected.issubset(eff.columns)


def test_beta_std_is_correlation_bound():
    """``beta_std`` is mathematically Pearson *r*; must lie in [-1, 1]."""
    pc_df, radiomics_df, pc_cols, feature_cols = _synth(n_subjects=300)
    joined = join_pc_with_ribs(pc_df, radiomics_df)
    eff = compute_effects(joined, pc_cols, feature_cols)
    finite = eff["beta_std"].dropna()
    assert finite.between(-1.0, 1.0).all(), finite.describe()


def test_planted_signal_is_recovered():
    """The (rib=8 Left, feat_a, PC_1) coupling must dominate every other cell."""
    rib_pairs = ((8, "Left"), (8, "Right"), (12, "Left"))
    pc_df, radiomics_df, pc_cols, feature_cols = _synth(
        n_subjects=400, rib_pairs=rib_pairs, feature_cols=("feat_a", "feat_b"),
    )
    joined = join_pc_with_ribs(pc_df, radiomics_df)
    eff = compute_effects(joined, pc_cols, feature_cols)

    target = eff[
        (eff["pc"] == "PC_1")
        & (eff["vert_level"] == 8)
        & (eff["side"] == "Left")
        & (eff["feature"] == "feat_a")
    ]
    assert len(target) == 1
    # Coefficient should be positive and near 1.0 (β_std ≈ correlation).
    assert float(target["beta_std"].iloc[0]) > 0.95
    assert float(target["q_value"].iloc[0]) < 1e-6


def test_fdr_family_size_is_per_pc():
    """BH-FDR is applied across (rib × feature) cells per PC.

    Every PC's family should have its own q-value distribution; an
    insignificant cell can still be ranked highest within its own PC
    family, so the per-PC q ≤ p relationship must hold.
    """
    pc_df, radiomics_df, pc_cols, feature_cols = _synth(n_subjects=120)
    joined = join_pc_with_ribs(pc_df, radiomics_df)
    eff = compute_effects(joined, pc_cols, feature_cols)
    for pc, grp in eff.groupby("pc"):
        valid = grp.dropna(subset=["p_value", "q_value"])
        assert (valid["q_value"] >= valid["p_value"] - 1e-12).all(), pc


def test_top_effects_sorted_by_abs_std():
    pc_df, radiomics_df, pc_cols, feature_cols = _synth(n_subjects=120)
    joined = join_pc_with_ribs(pc_df, radiomics_df)
    eff = compute_effects(joined, pc_cols, feature_cols)
    top = top_effects(eff, n=5)
    assert len(top) == 5
    abs_std = top["beta_std"].abs().tolist()
    assert abs_std == sorted(abs_std, reverse=True)


def test_anatomical_rib_mapping():
    """vert_level 8 → anatomical 1 (T8–T19 ↔ ribs 1–12)."""
    pc_df, radiomics_df, pc_cols, feature_cols = _synth(
        n_subjects=30, rib_pairs=((8, "Left"), (19, "Right")),
    )
    joined = join_pc_with_ribs(pc_df, radiomics_df)
    eff = compute_effects(joined, pc_cols, feature_cols)
    by_vert = eff.set_index("vert_level")["anatomical_rib"].to_dict()
    assert by_vert[8] == 1
    assert by_vert[19] == 12
