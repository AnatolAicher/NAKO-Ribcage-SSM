"""Per-run figure manifest.

Records every figure written via :mod:`utils.figure_export` to
``figures_manifest.json`` alongside the figure, capturing palette version
and git SHA for provenance.
"""
from __future__ import annotations

import datetime as _dt
import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Any

import settings as S

logger = logging.getLogger(__name__)


# ── Git revision (cached; read once per process) ─────────────────────────────

_GIT_REV: str | None = None


def _git_rev() -> str:
    """Return the current git SHA, or ``"unknown"`` if not in a git checkout."""
    global _GIT_REV
    if _GIT_REV is not None:
        return _GIT_REV
    try:
        rev = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=False,
            cwd=os.fspath(Path(__file__).resolve().parents[2]),
        )
        _GIT_REV = rev.stdout.strip() or "unknown"
    except (subprocess.CalledProcessError, OSError, FileNotFoundError):
        _GIT_REV = "unknown"
    return _GIT_REV


# ── Manifest I/O ─────────────────────────────────────────────────────────────

MANIFEST_NAME = "figures_manifest.json"
SCHEMA_VERSION = "nako-figures-manifest/2"

# Caller-supplied record fields persisted to disk. Absolute paths (`stem`,
# `files`) are deliberately excluded – they leak host-local layout and are
# reconstructable from the manifest directory + record key + `formats`.
_PERSISTED_RECORD_FIELDS = ("formats", "title", "width_class")


def _empty_payload() -> dict[str, Any]:
    return {
        "schema":          SCHEMA_VERSION,
        "git_rev":         _git_rev(),
        "palette_version": S.PALETTE_VERSION,
        "figures":         {},
    }


def _read_manifest(dirpath: Path) -> dict[str, Any]:
    p = dirpath / MANIFEST_NAME
    if not p.exists():
        return _empty_payload()
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        logger.warning("Manifest %s unreadable (%s); rewriting from scratch.", p, exc)
        return _empty_payload()


def _write_manifest(dirpath: Path, payload: dict[str, Any]) -> None:
    p = dirpath / MANIFEST_NAME
    p.write_text(json.dumps(payload, indent=2, sort_keys=True))


def append_manifest(dirpath: str | Path, name: str, record: dict) -> None:
    """Add / replace ``record`` keyed by ``name`` in the manifest at ``dirpath``."""
    dirpath = Path(dirpath)
    dirpath.mkdir(parents=True, exist_ok=True)
    payload = _read_manifest(dirpath)

    persisted = {k: record[k] for k in _PERSISTED_RECORD_FIELDS if k in record}
    persisted["written_at_utc"]  = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
    persisted["git_rev"]         = _git_rev()
    persisted["palette_version"] = S.PALETTE_VERSION

    payload["schema"]          = SCHEMA_VERSION
    payload["figures"][name]   = persisted
    payload["git_rev"]         = _git_rev()
    payload["palette_version"] = S.PALETTE_VERSION
    _write_manifest(dirpath, payload)


def read_manifest(dirpath: str | Path) -> dict[str, Any]:
    """Public read helper – returns the parsed manifest dict (empty if missing)."""
    return _read_manifest(Path(dirpath))
