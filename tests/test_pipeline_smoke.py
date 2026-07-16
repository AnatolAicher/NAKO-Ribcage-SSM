"""Sanity tests for ``scripts/run_pipeline.py``.

Verifies (without needing NIfTI data or the Scala JVM) that:

- the driver module imports cleanly;
- the shipped example/smoke presets parse against the schema;
- removed preset path keys (``extracted_stl_dir`` / ``registered_stl_dir`` /
  ``landmark_dir`` / ``centerline_dir``) and the removed ``centerlines`` stage
  are rejected;
- ``utils.paths.stage_dir`` returns a unique subdir per stage.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


def _load_driver_module():
    """Import scripts/run_pipeline.py without invoking it as __main__."""
    spec = importlib.util.spec_from_file_location(
        "run_pipeline", REPO_ROOT / "scripts" / "run_pipeline.py",
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_pipeline"] = mod
    spec.loader.exec_module(mod)
    return mod


def _materialise_preset(tmp_path: Path, src_yaml: Path) -> Path:
    """Copy a shipped preset to tmp_path and patch the placeholders so
    ``load_preset`` doesn't bail on the obvious set-me-please values."""
    raw = yaml.safe_load(src_yaml.read_text())
    raw["template_id"] = "999999"
    raw["paths"] = {"root": str(tmp_path / "stl_root")}
    # The loader checks that paths.root exists and is a directory.
    (tmp_path / "stl_root").mkdir(parents=True, exist_ok=True)
    out = tmp_path / src_yaml.name
    out.write_text(yaml.safe_dump(raw))
    return out


def test_driver_imports_cleanly():
    mod = _load_driver_module()
    assert hasattr(mod, "load_preset")
    assert hasattr(mod, "main")


@pytest.mark.parametrize("preset_name", ["example.yaml", "smoke.yaml"])
def test_shipped_preset_parses(tmp_path: Path, preset_name: str):
    mod = _load_driver_module()
    patched = _materialise_preset(tmp_path, REPO_ROOT / "presets" / preset_name)
    preset = mod.load_preset(patched)
    assert preset.template_id == "999999"
    assert preset.stages.get("ssm_registration") is True
    # The new schema collapses paths to a single `root`.
    assert "root" in preset.paths
    assert str(preset.paths["root"]).endswith("stl_root")


@pytest.mark.parametrize("removed_path_key", ["extracted_stl_dir", "registered_stl_dir",
                                               "landmark_dir", "centerline_dir"])
def test_removed_path_keys_rejected(tmp_path: Path, removed_path_key: str):
    """Per-component path keys are not in the schema and must raise."""
    mod = _load_driver_module()
    raw = yaml.safe_load((REPO_ROOT / "presets" / "smoke.yaml").read_text())
    raw["template_id"] = "999999"
    raw["paths"] = {
        "root": str(tmp_path / "stl_root"),
        removed_path_key: "/tmp/x",
    }
    (tmp_path / "stl_root").mkdir(parents=True, exist_ok=True)
    patched = tmp_path / "bad.yaml"
    patched.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match=rf"unknown paths\.{removed_path_key}"):
        mod.load_preset(patched)


def test_unknown_stage_rejected(tmp_path: Path):
    """``stages.centerlines`` is not a valid stage key."""
    mod = _load_driver_module()
    raw = yaml.safe_load((REPO_ROOT / "presets" / "smoke.yaml").read_text())
    raw["template_id"] = "999999"
    raw["paths"] = {"root": str(tmp_path / "stl_root")}
    (tmp_path / "stl_root").mkdir(parents=True, exist_ok=True)
    raw.setdefault("stages", {})["centerlines"] = True
    patched = tmp_path / "stale_stage.yaml"
    patched.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="unknown stage `centerlines`"):
        mod.load_preset(patched)


def test_stage_dir_unique_per_stage():
    """Each pipeline stage key resolves to a distinct subdirectory."""
    from utils.paths import _STAGE_DIRS, stage_dir
    run_dir = Path("/fake/run_dir")
    seen = {stage_dir(run_dir, s) for s in _STAGE_DIRS}
    assert len(seen) == len(_STAGE_DIRS), "stage_dir mapping has duplicates"


def test_stage_dir_rejects_unknown_stage():
    from utils.paths import stage_dir
    with pytest.raises(KeyError, match="unknown pipeline stage"):
        stage_dir(Path("/fake"), "not_a_stage")
