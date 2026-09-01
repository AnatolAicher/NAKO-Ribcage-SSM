"""Run-directory layout helpers (stdlib-only).

Layout::

    <root>/<name>/extracted_stl/<block>/<pid>/        per-rib extracted STLs
    <root>/<name>/registered_stl/<block>/<pid>/       per-rib registered STLs
    <root>/<name>/results/<name>_<UTC>/               per-invocation run dir
    <root>/<name>/results/<name>_latest               → most recent run dir

``block = pid // 1000``. Per-stage subdirectories under the run dir follow
the canonical mapping in :mod:`utils.paths`.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from .config import repo_root
from .paths import (
    extracted_stl_dir as _extracted_stl_dir,
    registered_stl_dir as _registered_stl_dir,
    results_base,
)


def patient_stl_dir(root: Path | str, pid: int | str) -> Path:
    """Return ``<root>/<pid // 1000>/<pid>/`` (does not create the directory)."""
    pid_i = int(pid)
    return Path(root) / str(pid_i // 1000) / str(pid_i)


def _utc_timestamp() -> str:
    """Return the current UTC time as ``YYYYmmddTHHMMSSZ`` – filename-safe."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _git_rev(repo: Path) -> str | None:
    """Best-effort ``git rev-parse HEAD``; ``None`` if not a git repo."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo, stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def make_run_dir(root: Path | str, name: str) -> Path:
    """Create ``<root>/<name>/results/<name>_<UTC>/`` + ``..._latest`` symlink.

    Writes ``metadata.json`` with the canonical path bundle so downstream
    consumers can locate the per-rib STL caches without re-passing them.
    """
    root_p = Path(root)
    results_p = results_base(root_p, name)
    results_p.mkdir(parents=True, exist_ok=True)

    ts = _utc_timestamp()
    run_name = f"{name}_{ts}"
    run_dir = results_p / run_name
    run_dir.mkdir(parents=True, exist_ok=False)

    metadata = {
        "name": name,
        "timestamp_utc": ts,
        "git_rev": _git_rev(repo_root()),
        "paths": {
            "root":                str(root_p),
            "extracted_stl_dir":   str(_extracted_stl_dir(root_p, name)),
            "registered_stl_dir":  str(_registered_stl_dir(root_p, name)),
            "results_dir":         str(results_p),
            "run_dir":             str(run_dir),
        },
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    # Relative symlink target so cross-machine moves stay clean.
    latest = results_p / f"{name}_latest"
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    os.symlink(run_name, latest)

    return run_dir


def read_run_dir(
    root: Path | str,
    name: str,
    override: Path | str | None = None,
) -> Path:
    """Resolve the run directory for ``<root>/<name>/results/``.

    Resolution order:
      1. ``override`` – explicit path; if given, must exist.
      2. ``<root>/<name>/results/<name>_latest`` – the symlink updated by
         :func:`make_run_dir`.
      3. ``FileNotFoundError`` with a clear message if neither resolves.
    """
    if override is not None:
        p = Path(override)
        if not p.exists():
            raise FileNotFoundError(f"Override run dir does not exist: {p}")
        return p.resolve()

    results_p = results_base(root, name)
    latest = results_p / f"{name}_latest"
    if not latest.exists():
        raise FileNotFoundError(
            f"{latest} not found. Run `python scripts/run_pipeline.py "
            f"<preset.yaml>` for a preset with name={name!r} and root={root!r} "
            f"first, or pass `override=<existing run dir>`."
        )
    return latest.resolve()


def read_metadata(run_dir: Path | str) -> dict:
    """Read ``<run_dir>/metadata.json`` written by :func:`make_run_dir`."""
    meta_path = Path(run_dir) / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"{meta_path} not found")
    return json.loads(meta_path.read_text())


def _copy_ignore(_dirpath: str, names: list[str]) -> list[str]:
    skip: list[str] = []
    for n in names:
        if n == "figures" or n == "__pycache__":
            skip.append(n)
        elif n.startswith("._"):  # macOS resource forks
            skip.append(n)
    return skip


def copy_run_dir(src: Path | str, dst: Path | str, *, force: bool = False) -> Path:
    """Copy a run dir to ``dst`` excluding every ``figures/`` subtree.

    Compute artifacts (parquet, npz, npy, csv, json) and ``metadata.json``
    are duplicated; figure subdirectories are intentionally omitted so the
    target starts with no stale figures. ``metadata.json`` at the target
    is rewritten with ``paths.run_dir`` / ``paths.results_dir`` updated to
    the new location and a ``rerendered_from`` field recording the source.
    The original ``git_rev`` is preserved so compute provenance is intact.

    Raises ``FileExistsError`` if ``dst`` exists unless ``force=True``.
    Returns the resolved target path.
    """
    src_p = Path(src).resolve()
    dst_p = Path(dst).resolve()
    if not src_p.is_dir():
        raise FileNotFoundError(f"Source run dir does not exist: {src_p}")
    if dst_p.exists():
        if not force:
            raise FileExistsError(
                f"Target {dst_p} already exists. Use force=True to overwrite."
            )
        shutil.rmtree(dst_p)

    shutil.copytree(src_p, dst_p, ignore=_copy_ignore, symlinks=False)

    meta_path = dst_p / "metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        paths_block = meta.get("paths") or {}
        paths_block["run_dir"]     = str(dst_p)
        paths_block["results_dir"] = str(dst_p.parent)
        meta["paths"] = paths_block
        meta["rerendered_from"] = str(src_p)
        meta_path.write_text(json.dumps(meta, indent=2))

    return dst_p
