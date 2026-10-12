import sys
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app import NOT_FOUND, arithmetic_panel  # noqa: E402
from src.extract import extract  # noqa: E402
from src.normalize import parse_amount  # noqa: E402
from src.validate import validate  # noqa: E402
from test_extract import _rows  # noqa: E402


def test_flag_letter_amounts() -> None:
    a = parse_amount("3.45Y")
    assert a.value == Decimal("3.45") and a.flag == "Y" and a.strength == "strong"
    assert parse_amount("12.5Q") is None  # flag only after a two-decimal price


def test_columnar_block_one_item_per_row() -> None:
    f = extract(_rows("PAPER CUPS 041234567890 3.45X", "BLUE INK 009876543210 1.10 T",
                      "DISH SOAP 012345678905 6.20N", "SUBTOTAL 10.75",
                      "TAX 1 5.50% 0.59", "TOTAL 11.34"))
    items = f["items"]
    assert [i["name"] for i in items] == ["PAPER CUPS", "BLUE INK", "DISH SOAP"]
    assert [str(i["amount"]) for i in items] == ["3.45", "1.10", "6.20"]
    assert [i["code"] for i in items] == ["041234567890", "009876543210", "012345678905"]
    assert [i["tax_flag"] for i in items] == ["X", "T", "N"]
    assert all(i["qty"] is None for i in items)


def test_numbered_tax_line_with_rate() -> None:
    f = extract(_rows("BREAD 20.00", "SUBTOTAL 20.00", "TAX 1 | 5.50% | 1.10", "TOTAL 21.10"))
    assert [(t["rate"], t["amount"]) for t in f["tax_lines"]] == [(Decimal("5.50"), Decimal("1.10"))]
    # VAT is not a GST label: 7.5% is allowed (0 to 30), 45% is not
    f = extract(_rows("BREAD 20.00", "SUBTOTAL 20.00", "VAT 7.5% 1.50", "TOTAL 21.50"))
    assert [t["rate"] for t in f["tax_lines"]] == [Decimal("7.5")]
    f = extract(_rows("BREAD 20.00", "SUBTOTAL 20.00", "VAT 45% 1.50", "TOTAL 21.50"))
    assert f["tax_lines"] == []


def _fields(total: str, cash: str) -> dict:
    """Made-up bill: one item read (4.00) of a 18.00 subtotal, two taxes, cash = total."""
    none = {"value": None}
    return {"items": [{"name": "Soap", "row": 1, "kind": "item", "qty": None, "rate": None,
                       "amount": "4.00", "status": "unverified", "numbers": [], "min_conf": 0.99}],
            "tax_lines": [{"label": "TAX", "rate": "5", "amount": "0.90", "row": 6, "min_conf": 0.99},
                          {"label": "TAX", "rate": "2.5", "amount": "0.45", "row": 7,
                           "min_conf": 0.99}],
            "tax_source": "tax_rows",
            "total": {"value": total, "alternatives": [], "min_conf": 0.99, "label": "total",
                      "rows": [8], "breakdown": {"label": 2}},
            "subtotal": {"value": "18.00", "rows": [5]}, "taxable": none, "discount": none,
            "round_off": none, "service_charge": none, "taxes": {"table": None},
            "payments": [{"label": "CASH", "amount": cash, "row": 9}]}


def test_subtotal_chain_with_payment_is_valid_items_unverified() -> None:
    v = validate(_fields("19.35", "19.35"))
    assert v["status"] == "VALID" and "items_unverified" in v["reasons"]
    assert "subtotal_chain" in v["witnesses"] and "cash" in v["witnesses"]
    v_full = validate({**_fields("19.35", "19.35"), "items": [
        {**_fields("19.35", "19.35")["items"][0], "amount": "18.00"}]})
    assert v_full["status"] == "VALID" and v["confidence"] < v_full["confidence"]


def test_subtotal_chain_wrong_total_stays_check_required() -> None:
    v = validate(_fields("21.35", "21.35"))
    assert v["status"] == "CHECK_REQUIRED" and "subtotal_chain" not in v["witnesses"]


def test_arithmetic_panel() -> None:
    f = _fields("19.35", "19.35")
    p = arithmetic_panel(f, validate(f))
    rows = {r["line"]: r["value"] for r in p["table"]}
    assert rows["items sum"] == "4.00" and rows["printed subtotal"] == "18.00"
    assert rows["taxes sum"] == "1.35" and rows["discount"] == NOT_FOUND
    assert rows["printed total"] == "19.35" and rows["match"] == "✗"
    assert p["chain"]["computed"] == "19.35" and p["chain"]["match"] == "✓"
    assert "item extraction incomplete, item check inconclusive" in p["notes"]
    assert p["verdict"] == "balanced via subtotal chain"  # never 'wrong' from items alone
    f = _fields("21.35", "21.35")
    p = arithmetic_panel(f, validate(f))
    assert p["chain"]["match"] == "✗" and p["verdict"] == "does not balance"
    p = arithmetic_panel({}, {})
    assert {r["value"] for r in p["table"]} >= {NOT_FOUND} and p["chain"]["match"] == NOT_FOUND
