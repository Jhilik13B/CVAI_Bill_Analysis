"""Phase 14: pure HTML helpers in app.py (made-up inputs)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import (NOT_FOUND, _bars, _card, _dots, _mark_pill, _pill, _status_guide,  # noqa: E402
                 _step_card)


def test_pill_colour_class_per_status():
    assert "pill-valid" in _pill("VALID")
    assert "pill-repaired" in _pill("REPAIRED")
    assert "pill-check" in _pill("CHECK_REQUIRED") and "⚠" in _pill("CHECK_REQUIRED")
    assert "pill-insufficient" in _pill("INSUFFICIENT_EVIDENCE")


def test_dots_counts():
    d = _dots(0.4)
    assert d.count("dot on") == 4 and d.count("dot off") == 6


def test_card_escapes_html():
    c = _card("Merchant", "<b>x</b>")
    assert "&lt;b&gt;" in c and "<b>x" not in c


def test_not_found_unchanged():
    assert NOT_FOUND in _card("Date", NOT_FOUND)
    assert _mark_pill(NOT_FOUND) == NOT_FOUND
    assert _bars(NOT_FOUND, "10.00") == ""


def test_step_card_number_and_escaped_text():
    c = _step_card("01", "UPLOAD", "<b>x</b>", "lime")
    assert "01" in c and "&lt;b&gt;" in c and "<b>x" not in c
    assert "c-lime" in _step_card("02", "READ", "t", "red;x")


def test_status_guide_all_statuses():
    g = _status_guide()
    assert all(s in g for s in ("VALID", "REPAIRED", "CHECK_REQUIRED", "INSUFFICIENT_EVIDENCE"))
