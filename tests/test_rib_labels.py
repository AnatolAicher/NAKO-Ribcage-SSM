"""Unit tests for utils.rib_labels – display / CLI / data conversions."""
from __future__ import annotations

import pytest

from utils.rib_labels import (
    anatomical_to_seg,
    anatomical_to_vert,
    cli_token_from_seg,
    display_from_seg,
    display_from_vert,
    display_rib,
    parse_cli_token,
    seg_to_anatomical,
    vert_to_anatomical,
)


def test_seg_round_trip():
    for seg in range(40, 52):
        assert anatomical_to_seg(seg_to_anatomical(seg)) == seg
    for rib in range(1, 13):
        assert seg_to_anatomical(anatomical_to_seg(rib)) == rib


def test_vert_round_trip():
    for vert in range(8, 20):
        assert anatomical_to_vert(vert_to_anatomical(vert)) == vert
    for rib in range(1, 13):
        assert vert_to_anatomical(anatomical_to_vert(rib)) == rib


def test_seg_anatomical_corners():
    assert seg_to_anatomical(40) == 1
    assert seg_to_anatomical(51) == 12
    assert anatomical_to_seg(1) == 40
    assert anatomical_to_seg(12) == 51


def test_vert_anatomical_corners():
    assert vert_to_anatomical(8) == 1
    assert vert_to_anatomical(19) == 12


def test_display_rib_shapes():
    assert display_rib(1) == "Rib 1"
    assert display_rib(12) == "Rib 12"
    assert display_rib(7, "L") == "Rib 7 L"
    assert display_rib(7, "R") == "Rib 7 R"
    assert display_rib(7, "Right", side_long=True) == "Rib 7 Right"
    assert display_rib(7, "L", side_long=True) == "Rib 7 Left"


def test_display_from_seg_and_vert_agree():
    # seg 40 and vert 8 both encode rib 1 → same display string.
    for seg, vert in zip(range(40, 52), range(8, 20)):
        for side in ("L", "R"):
            assert display_from_seg(seg, side) == display_from_vert(vert, side)


def test_cli_token_round_trip():
    for seg in range(40, 52):
        for side in ("L", "R"):
            tok = cli_token_from_seg(seg, side)
            assert parse_cli_token(tok) == (seg, side)


def test_cli_token_format():
    assert cli_token_from_seg(40, "L") == "rib1_L"
    assert cli_token_from_seg(46, "R") == "rib7_R"
    assert cli_token_from_seg(51, "R") == "rib12_R"


def test_parse_cli_token_rejects_seg_label_form():
    # CLI tokens use anatomical 1..12; seg labels (40..51) must raise.
    with pytest.raises(ValueError, match="must be in 1..12"):
        parse_cli_token("rib46_R")
    with pytest.raises(ValueError, match="must be in 1..12"):
        parse_cli_token("rib40_L")


def test_parse_cli_token_rejects_garbage():
    for bad in ("", "rib7", "rib_R", "ribz_R", "Rib 7 R", "rib7-R", "rib0_L", "rib13_R"):
        with pytest.raises(ValueError):
            parse_cli_token(bad)
