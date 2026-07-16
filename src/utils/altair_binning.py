"""Python-side 2D binning for Altair bubble plots.

``bin_xy(df, x, y, ...)`` aggregates a per-row DataFrame to bin counts so
the Altair spec contains O(bins²) rows instead of O(N). Keeps vl-convert's
V8 heap small for large cohorts and removes per-patient data from the
public-facing JSON spec.

Pure pandas/numpy — no Altair import; the module is altair-only by
convention (lives next to ``altair_theme`` / ``figure_export_altair``).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def bin_xy(
    df: pd.DataFrame,
    x: str,
    y: str,
    *,
    bins: int = 50,
    group: Sequence[str] = (),
    agg_col: str | None = None,
    agg_fn: str = "mean",
    min_count: int = 5,
) -> pd.DataFrame:
    """Aggregate ``df`` to 2D bin counts.

    Parameters
    ----------
    df
        Per-row data.
    x, y
        Quantitative column names to bin on.
    bins
        Equal-width bins per axis.
    group
        Columns to split counts on (e.g. ``("sex",)``). One row per
        ``(x_bin, y_bin, *group)`` cell.
    agg_col, agg_fn
        Optional in-bin aggregation of a continuous column (e.g.
        ``agg_col="age", agg_fn="mean"``). Adds a column named ``agg_col``.
    min_count
        Bins with ``n < min_count`` are dropped (k-anonymity).

    Returns
    -------
    DataFrame with columns ``x``, ``y`` (bin centers, float), ``n``,
    the ``group`` columns, and ``agg_col`` if given.
    """
    group = list(group)
    needed = [x, y, *group] + ([agg_col] if agg_col else [])
    sub = df[needed].dropna(subset=[x, y])
    if sub.empty:
        cols = [x, y, "n", *group] + ([agg_col] if agg_col else [])
        return pd.DataFrame(columns=cols)

    x_edges = np.linspace(sub[x].min(), sub[x].max(), bins + 1)
    y_edges = np.linspace(sub[y].min(), sub[y].max(), bins + 1)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])

    x_idx = np.clip(np.digitize(sub[x].to_numpy(), x_edges) - 1, 0, bins - 1)
    y_idx = np.clip(np.digitize(sub[y].to_numpy(), y_edges) - 1, 0, bins - 1)

    work = sub.copy()
    work[x] = x_centers[x_idx]
    work[y] = y_centers[y_idx]

    keys = [x, y, *group]
    agg: dict[str, tuple[str, str]] = {"n": (x, "size")}
    if agg_col:
        agg[agg_col] = (agg_col, agg_fn)

    out = work.groupby(keys, observed=True, sort=False).agg(**agg).reset_index()
    out = out[out["n"] >= min_count].reset_index(drop=True)
    return out
