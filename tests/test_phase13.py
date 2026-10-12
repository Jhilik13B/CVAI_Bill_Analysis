import sys
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.extract import extract  # noqa: E402
from src.validate import validate  # noqa: E402
from test_extract import _rows  # noqa: E402

LINES = ("Tea 100.00", "Bun 50.00", "CGST @ 2.50% ON 150.00 3.75", "SGST @ 2.50% ON 150.00 3.75",
         "TOTAL GST 7.50")


def test_tax_lines_with_base_and_cash_witness() -> None:
    f = extract(_rows(*LINES, "CASH 157.50"))
    assert [(t["label"], t["base"], t["amount"]) for t in f["tax_lines"]] == [
        ("CGST", Decimal("150.00"), Decimal("3.75")), ("SGST", Decimal("150.00"), Decimal("3.75"))]
    assert f["tax_sum"] == Decimal("7.50")
    assert f["taxable"]["value"] == Decimal("150.00")
    assert f["total"]["value"] == Decimal("157.50")
    assert len(f["items"]) == 2
    v = validate(f)
    assert v["status"] == "VALID"
    assert "tax_total_matches_lines" in v["witnesses"]


def test_cash_equal_to_taxable_unchanged() -> None:
    f = extract(_rows(*LINES, "CASH 150.00"))
    assert f["total"]["value"] == Decimal("150.00")
    assert not f["total"]["breakdown"].get("cash_witness")


def test_wrong_tax_amount_not_a_tax_line() -> None:
    f = extract(_rows("Tea 100.00", "Bun 50.00", "CGST @ 2.50% ON 150.00 3.80", "Total 150.00"))
    assert not any(t["amount"] == Decimal("3.80") for t in f["tax_lines"])
    assert len(f["items"]) == 2
