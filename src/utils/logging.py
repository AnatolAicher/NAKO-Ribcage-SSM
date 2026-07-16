"""Shared logger configuration, step timing, and tqdm progress helpers."""
from __future__ import annotations

import logging
import sys
import time
from contextlib import contextmanager
from typing import Any, Iterable, Iterator

from tqdm.auto import tqdm

_FMT     = "%(asctime)s  %(levelname)-8s  %(name)-28.28s  %(message)s"
_DATEFMT = "%H:%M:%S"

_ROOT_INSTALLED = False


def _install_root_handler(level: int = logging.INFO) -> None:
    global _ROOT_INSTALLED
    if _ROOT_INSTALLED:
        return
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(_FMT, datefmt=_DATEFMT))
    root.addHandler(handler)
    root.setLevel(level)
    _ROOT_INSTALLED = True


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a logger that shares the project-wide handler."""
    _install_root_handler(level)
    return logging.getLogger(name)


@contextmanager
def log_step(logger: logging.Logger, label: str) -> Iterator[None]:
    """Emit ``BEGIN <label>`` / ``END <label> (elapsed=…)`` around a block."""
    logger.info(f"BEGIN  {label}")
    t0 = time.monotonic()
    try:
        yield
    finally:
        elapsed = time.monotonic() - t0
        if elapsed >= 60.0:
            m, s = divmod(elapsed, 60)
            human = f"{int(m)}m{s:.1f}s"
        else:
            human = f"{elapsed:.2f}s"
        logger.info(f"END    {label}  (elapsed={human})")


def log_stage_config(
    logger: logging.Logger,
    stage: str,
    settings: dict[str, Any],
) -> None:
    """Log a one-line summary of the settings that shape this stage's run."""
    if not settings:
        logger.info(f"STAGE  {stage}  (no overrides)")
        return
    kv = "  ".join(f"{k}={v}" for k, v in settings.items())
    logger.info(f"STAGE  {stage}  {kv}")


def progress(
    iterable: Iterable | None = None,
    *,
    total: int | None = None,
    desc: str | None = None,
    unit: str = "it",
    leave: bool = True,
) -> tqdm:
    """Project-wide tqdm wrapper. Writes to stderr; redraws ≤ 2× per second."""
    return tqdm(
        iterable,
        total=total,
        desc=desc,
        unit=unit,
        leave=leave,
        file=sys.stderr,
        dynamic_ncols=True,
        mininterval=0.5,
        miniters=1,
    )
