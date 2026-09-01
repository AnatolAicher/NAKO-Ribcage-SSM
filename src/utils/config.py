"""Loads and resolves the project YAML configuration."""
from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path

import pandas as pd
import yaml

_REPO_ROOT = Path(__file__).parent.parent.parent
_DEFAULT_CONFIG = _REPO_ROOT / "data" / "data_config.yaml"

logger = logging.getLogger(__name__)


def load_config(config_path: Path | str | None = None) -> dict:
    """Load the project YAML config, defaulting to ``data/data_config.yaml``."""
    path = Path(config_path) if config_path else _DEFAULT_CONFIG
    with open(path) as f:
        return yaml.safe_load(f)


def repo_root() -> Path:
    """Return the repository root directory."""
    return _REPO_ROOT


def _on_network_mount(path: Path) -> bool:
    parts = path.resolve().parts
    return "CloudStorage" in parts or "mnt" in parts


def read_parquet(path: Path | str, **kwargs) -> pd.DataFrame:
    """Read a parquet, copying to a local temp file first if on a network mount.

    FUSE-based cloud mounts (Nextcloud, iCloud, Google Drive) can stall under
    pyarrow's memory-mapped reads. The copy goes through ``/bin/cp`` so the
    OS handles the transfer.
    """
    path = Path(path)
    if _on_network_mount(path):
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=True) as tmp:
            logger.info(f"Copying {path.name} to local temp file via cp …")
            try:
                subprocess.run(
                    ["cp", str(path), tmp.name],
                    check=True, timeout=300,
                )
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
                raise OSError(
                    f"Failed to copy {path} from network mount to local temp file. "
                    f"Check that the mount is accessible: {exc}"
                ) from exc
            logger.info(f"  Copied – reading parquet from {tmp.name}")
            return pd.read_parquet(tmp.name, **kwargs)
    return pd.read_parquet(path, **kwargs)
