import sys
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.extract import extract  # noqa: E402
from src.validate import validate  # noqa: E402
from test_extract import _rows  # noqa: E402

ITEMS = ("Tea 10.00", "Bun 5.00")


def test_loyalty_between_subtotals_is_discount() -> None:
    f = extract(_rows(*ITEMS, "Subtotal 15.00", "LOYALTY -3.00", "Subtotal 12.00", "Total 12.00"))
    assert f["discount"]["value"] == Decimal("-3.00") and f["discount"]["source"] == "negative_adjustment"
    assert f["total"]["value"] == Decimal("12.00")
    assert all("LOYALTY" not in i["name"].upper() for i in f["items"])
    assert validate(f)["status"] == "VALID"


def test_unknown_label_between_subtotals_is_discount() -> None:
    f = extract(_rows(*ITEMS, "Subtotal 15.00", "Adjust (3.00)", "Subtotal 12.00", "Total 12.00"))
    assert f["discount"]["value"] == Decimal("-3.00") and f["discount"]["source"] == "negative_adjustment"
    assert f["total"]["value"] == Decimal("12.00")
    assert all("ADJUST" not in i["name"].upper() for i in f["items"])
    assert validate(f)["status"] == "VALID"


def test_without_adjustment_unchanged() -> None:
    f = extract(_rows(*ITEMS, "Subtotal 15.00", "Total 15.00"))
    assert f["discount"]["value"] is None
    assert f["total"]["value"] == Decimal("15.00")
    assert validate(f)["status"] == "VALID"


def test_positive_discount_needs_disc_word() -> None:
    # OCR 'DlSCOUNT' fuzzy-matches the DISCOUNT label but has no 'DISC' in its text: dropped
    rows = _rows(*ITEMS, "Subtotal 15.00", "DlSCOUNT 3.00", "Total 12.00")
    assert "DISCOUNT" in {l for l, _ in rows[3]["labels"]}
    f = extract(rows)
    assert f["discount"]["value"] is None and f["discount"]["reason"] == "positive_without_disc_word"
    assert f["total"]["value"] == Decimal("12.00")
    # the same row spelled with DISC is kept
    f = extract(_rows(*ITEMS, "Subtotal 15.00", "Discount 3.00", "Total 12.00"))
    assert f["discount"]["value"] == Decimal("3.00")
    assert f["total"]["value"] == Decimal("12.00")


def test_negative_row_not_between_subtotals() -> None:
    f = extract(_rows(*ITEMS, "Total 15.00", "Adjust -3.00"))
    assert f["discount"]["value"] is None
    f = extract(_rows(*ITEMS, "Adjust -3.00", "Subtotal 12.00", "Total 12.00"))
    assert f["discount"]["value"] is None
