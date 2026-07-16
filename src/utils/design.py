"""Design-matrix encodings shared by the unadjusted, adjusted, and prediction models.

`sex`→`is_female` (Male=0, Female=1) and `smoking_status`→`ever_smoker`
(Never=0, Ex/Current=1). Smoking is ever/never everywhere; `smoking_status`
survives only as the source column.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Binary (0/1) design columns; every other predictor is treated as continuous.
DESIGN_BINARY: frozenset[str] = frozenset({"is_female", "ever_smoker"})


def add_design_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with ``is_female`` and ``ever_smoker`` derived."""
    out = df.copy()
    if "sex" in out.columns:
        is_female = (out["sex"].astype(str).str.lower() == "female").astype(float)
        is_female[out["sex"].isna()] = np.nan
        out["is_female"] = is_female
    if "smoking_status" in out.columns:
        ever = (out["smoking_status"].astype(str).str.lower() != "never").astype(float)
        ever[out["smoking_status"].isna()] = np.nan
        out["ever_smoker"] = ever
    return out
