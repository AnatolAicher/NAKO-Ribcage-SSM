"""Path conventions for the rib SSM pipeline.

Single source of truth for two layouts:

  1. **Per-preset root**: every preset artifact lives under ``<root>/<name>/``.
  2. **Per-stage output**: within a run dir, each stage writes to a subdir
     named after its key in ``scripts/run_pipeline.py::_STAGE_KEYS``.

Cross-stage reads (e.g. adjusted reading ingestion's parquet) flow through
:func:`stage_dir`; no path fragment should be hard-coded by importers.
"""
from __future__ import annotations

from pathlib import Path


# Stage key → subdirectory name; keys and values are identical so a typo
# raises immediately rather than silently writing to the wrong place.
_STAGE_DIRS = {
    "ingestion":             "ingestion",
    "adjusted":              "adjusted",
    "mesh_extraction":       "mesh_extraction",
    "ssm_registration":      "ssm_registration",
    "ssm_pca":               "ssm_pca",
    "radiomics_correlation": "radiomics_correlation",
    "ssm_viewer":            "ssm_viewer",
    "ssm_qa_metrics":        "ssm_qa_metrics",
    "ssm_qa_residuals":      "ssm_qa_residuals",
    "visualizations":        "visualizations",
}


def stage_dir(run_dir: Path | str, stage: str) -> Path:
    """Return ``<run_dir>/<stage>/`` for a known stage key (raises on unknown)."""
    if stage not in _STAGE_DIRS:
        raise KeyError(
            f"unknown pipeline stage {stage!r}; expected one of "
            f"{sorted(_STAGE_DIRS)}"
        )
    return Path(run_dir) / _STAGE_DIRS[stage]


def preset_root(root: Path | str, name: str) -> Path:
    """Return ``<root>/<name>/`` – the parent of every per-preset artifact."""
    return Path(root) / name


def extracted_stl_dir(root: Path | str, name: str) -> Path:
    """Per-rib mesh-extraction cache: ``<root>/<name>/extracted_stl/``."""
    return preset_root(root, name) / "extracted_stl"


def registered_stl_dir(root: Path | str, name: str) -> Path:
    """Per-rib Scalismo-registered STLs: ``<root>/<name>/registered_stl/``."""
    return preset_root(root, name) / "registered_stl"


def results_base(root: Path | str, name: str) -> Path:
    """Parent of all run-dirs for this preset: ``<root>/<name>/results/``."""
    return preset_root(root, name) / "results"
