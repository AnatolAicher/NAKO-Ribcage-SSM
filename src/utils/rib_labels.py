"""Rib label conversions between data conventions and human display.

The 12 anatomical ribs are encoded two different ways in this project:

- Segmentation labels **40–51** (NIfTI mask integer values) — used by
  the SSM track (mesh extraction, registration, PCA, residuals).
- Vertebral levels **8–19** (T8–T19) — used by the data ingestion /
  bivariate / adjusted analyses.

Both encode anatomical ribs **1–12** (seg 40 = vert T8 = rib 1; seg 51 =
vert T19 = rib 12). On-disk data, dict / npz / JSON keys, and
DataFrame column values stay in their native convention; everything a
user sees (plot titles, hover text, axis labels, legend entries, log
lines, CLI input/output) is mapped to the anatomical 1–12 form via
the helpers below.

Three string forms are produced by this module:

- **internal token**  ``rib40_L``   — file paths, npz / dict keys
                                       (this module does not produce
                                       these, callers keep them).
- **cli token**       ``rib7_R``    — CLI args, output filenames.
- **display**         ``Rib 7 R``   — plot titles, hovers, logs.
"""
from __future__ import annotations

import re

SEG_TO_ANATOMICAL_OFFSET: int = 39   # seg 40 → rib 1, seg 51 → rib 12
VERT_TO_ANATOMICAL_OFFSET: int = 7   # vert 8 → rib 1, vert 19 → rib 12

_RIB_MIN: int = 1
_RIB_MAX: int = 12
_SIDE_SHORT: dict[str, str] = {"L": "L", "R": "R", "Left": "L", "Right": "R"}
_SIDE_LONG:  dict[str, str] = {"L": "Left", "R": "Right", "Left": "Left", "Right": "Right"}


def seg_to_anatomical(seg_label: int) -> int:
    """``40 → 1``, ``51 → 12``."""
    return int(seg_label) - SEG_TO_ANATOMICAL_OFFSET


def vert_to_anatomical(vert_level: int) -> int:
    """``8 → 1``, ``19 → 12``."""
    return int(vert_level) - VERT_TO_ANATOMICAL_OFFSET


def anatomical_to_seg(rib: int) -> int:
    """``1 → 40``, ``12 → 51``."""
    return int(rib) + SEG_TO_ANATOMICAL_OFFSET


def anatomical_to_vert(rib: int) -> int:
    """``1 → 8``, ``12 → 19``."""
    return int(rib) + VERT_TO_ANATOMICAL_OFFSET


def display_rib(rib: int, side: str | None = None,
                *, side_long: bool = False) -> str:
    """Human-readable rib label.

    ``display_rib(7)``                → ``'Rib 7'``
    ``display_rib(7, 'R')``           → ``'Rib 7 R'``
    ``display_rib(7, 'R', side_long=True)`` → ``'Rib 7 Right'``
    """
    if side is None:
        return f"Rib {int(rib)}"
    s = _SIDE_LONG[side] if side_long else _SIDE_SHORT[side]
    return f"Rib {int(rib)} {s}"


def display_from_seg(seg_label: int, side: str | None = None,
                     *, side_long: bool = False) -> str:
    """Display string from a segmentation label (40–51)."""
    return display_rib(seg_to_anatomical(seg_label), side, side_long=side_long)


def display_from_vert(vert_level: int, side: str | None = None,
                      *, side_long: bool = False) -> str:
    """Display string from a vertebral level (8–19)."""
    return display_rib(vert_to_anatomical(vert_level), side, side_long=side_long)


def cli_token_from_seg(seg_label: int, side: str) -> str:
    """``(40, 'L') → 'rib1_L'``  — 1-based, no spaces, parseable."""
    return f"rib{seg_to_anatomical(seg_label)}_{_SIDE_SHORT[side]}"


_CLI_TOKEN_RE = re.compile(r"^rib(\d{1,2})_([LR])$")


def parse_cli_token(s: str) -> tuple[int, str]:
    """Parse the 1-based CLI form ``'rib7_R'`` to internal ``(seg, side)``.

    Raises ``ValueError`` on the old 40-based form or any other shape so
    the caller can produce a clean error.  ``'rib46_R'`` is rejected
    even though it parses as an integer — the rib number must be in
    [1, 12].
    """
    m = _CLI_TOKEN_RE.match(s.strip())
    if m is None:
        raise ValueError(
            f"--rib-id={s!r}: expected form 'rib<N>_<L|R>' with N in 1..12"
        )
    rib = int(m.group(1))
    if not (_RIB_MIN <= rib <= _RIB_MAX):
        raise ValueError(
            f"--rib-id={s!r}: rib number must be in 1..12 "
            f"(N.B. internal NIfTI labels run 40..51 — use the anatomical form)"
        )
    side = m.group(2)
    return anatomical_to_seg(rib), side
