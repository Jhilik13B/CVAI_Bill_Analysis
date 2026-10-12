import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.rows import build_rows  # noqa: E402


def _line(text: str, x0: float, y0: float, x1: float, y1: float, conf: float = 0.9) -> dict:
    return {"box": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]], "text": text, "conf": conf}


def test_jitter_one_row_sorted_by_x() -> None:
    lines = [_line("C", 200, 104, 210, 124), _line("A", 0, 96, 10, 116), _line("B", 100, 100, 110, 120)]
    rows = build_rows(lines)
    assert len(rows) == 1
    assert [t["text"] for t in rows[0]["tokens"]] == ["A", "B", "C"]


def test_two_lines_two_rows() -> None:
    rows = build_rows([_line("Tea 10.00", 0, 130, 90, 150), _line("Rice 50.00", 0, 100, 100, 120)])
    assert len(rows) == 2
    assert rows[0]["text"] == "Rice 50.00" and rows[1]["text"] == "Tea 10.00"


def test_wide_gap_separate_cells() -> None:
    rows = build_rows([_line("Rice", 0, 0, 40, 20), _line("50.00", 300, 0, 350, 20)])
    assert rows[0]["cells"] == ["Rice", "50.00"]
    assert rows[0]["amounts"][0]["cell"] == 1


def test_tall_header_overlap_same_row() -> None:
    # small box h=20, overlap 12 px = 60% of the smaller height
    rows = build_rows([_line("AMOUNT", 200, 0, 260, 40), _line("QTY", 0, 28, 30, 48)])
    assert len(rows) == 1
