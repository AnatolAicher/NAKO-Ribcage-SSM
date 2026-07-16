"""Unit tests for ``src/utils/run_dir.py``.

Run from the repo root::

    python -m pytest tests/test_run_dir.py -v
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from utils.paths import (
    extracted_stl_dir,
    registered_stl_dir,
    results_base,
)
from utils.run_dir import (
    make_run_dir,
    read_metadata,
    read_run_dir,
)


def test_make_run_dir_creates_layout(tmp_path: Path):
    rd = make_run_dir(tmp_path, "preset_a")
    # Layout: <root>/<name>/results/<name>_<UTC>/
    assert rd.is_dir()
    assert rd.parent == results_base(tmp_path, "preset_a")
    assert rd.name.startswith("preset_a_")
    # metadata.json records the canonical path bundle.
    meta = json.loads((rd / "metadata.json").read_text())
    assert meta["name"] == "preset_a"
    assert "timestamp_utc" in meta
    paths = meta["paths"]
    assert paths["root"]               == str(tmp_path)
    assert paths["extracted_stl_dir"]  == str(extracted_stl_dir(tmp_path, "preset_a"))
    assert paths["registered_stl_dir"] == str(registered_stl_dir(tmp_path, "preset_a"))
    assert paths["results_dir"]        == str(results_base(tmp_path, "preset_a"))
    assert paths["run_dir"]            == str(rd)


def test_make_run_dir_updates_latest_symlink(tmp_path: Path):
    rd1 = make_run_dir(tmp_path, "preset_a")
    latest = results_base(tmp_path, "preset_a") / "preset_a_latest"
    assert latest.is_symlink()
    # macOS resolves /var → /private/var; compare resolved targets.
    assert latest.resolve() == rd1.resolve()
    # Symlink target is relative so it stays valid across moves.
    assert not os.path.isabs(os.readlink(latest))

    # A second run advances the symlink.
    time.sleep(1.1)  # 1-s timestamp resolution
    rd2 = make_run_dir(tmp_path, "preset_a")
    assert rd2 != rd1
    assert latest.resolve() == rd2.resolve()


def test_read_run_dir_uses_latest_by_default(tmp_path: Path):
    rd = make_run_dir(tmp_path, "preset_a")
    assert read_run_dir(tmp_path, "preset_a") == rd.resolve()


def test_read_run_dir_honours_override(tmp_path: Path):
    rd1 = make_run_dir(tmp_path, "preset_a")
    time.sleep(1.1)
    rd2 = make_run_dir(tmp_path, "preset_a")
    # default returns rd2 (latest)
    assert read_run_dir(tmp_path, "preset_a") == rd2.resolve()
    # explicit override returns rd1
    assert read_run_dir(tmp_path, "preset_a", override=rd1) == rd1.resolve()


def test_read_run_dir_raises_when_no_latest(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        read_run_dir(tmp_path, "never_run")


def test_read_run_dir_raises_when_override_missing(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        read_run_dir(tmp_path, "preset_a", override=tmp_path / "no_such_dir")


def test_separate_presets_dont_collide(tmp_path: Path):
    rd_a = make_run_dir(tmp_path, "preset_a")
    rd_b = make_run_dir(tmp_path, "preset_b")
    # Different parent dirs, different symlinks.
    assert rd_a.parent != rd_b.parent
    assert (results_base(tmp_path, "preset_a") / "preset_a_latest").resolve() == rd_a.resolve()
    assert (results_base(tmp_path, "preset_b") / "preset_b_latest").resolve() == rd_b.resolve()


def test_read_metadata_round_trips(tmp_path: Path):
    rd = make_run_dir(tmp_path, "preset_a")
    meta = read_metadata(rd)
    assert meta["name"] == "preset_a"
    assert meta["paths"]["run_dir"] == str(rd)
