import sys
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.extract import extract  # noqa: E402
from src.validate import validate  # noqa: E402
from test_extract import _rows  # noqa: E402


def _it(name: str, amount: str, qty=None, rate=None, numbers=(), conf: float = 0.99) -> dict:
    """Made-up 04_fields item (values as strings, like the JSON dump)."""
    return {"name": name, "row": 0, "kind": "item", "qty": qty, "rate": rate, "amount": amount,
            "status": "verified" if qty else "unverified", "numbers": list(numbers),
            "min_conf": conf}


def _tax(label: str, amount: str, rate=None) -> dict:
    return {"label": label, "rate": rate, "amount": amount, "row": 9, "min_conf": 0.99}


def _bill(items: list[dict], total, taxes=(), alts=(), subtotal=None, label=None,
          cash=None) -> dict:
    """label: total label name ('total'...); cash: a CASH row amount (a total witness)."""
    none = {"value": None}
    pays = [{"label": "CASH", "amount": cash, "row": 20}] if cash else []
    return {"items": items, "tax_lines": list(taxes), "tax_source": "pairs" if taxes else "none",
            "total": {"value": total, "alternatives": [{"value": a} for a in alts],
                      "min_conf": 0.99, "label": label},
            "subtotal": {"value": subtotal}, "taxable": none, "discount": none,
            "round_off": none, "service_charge": none, "payments": pays, "taxes": {"table": None}}


def test_inclusive_bill_balances() -> None:
    v = validate(_bill([_it("Tea", "100.00", "1", "100.00"), _it("Bun", "50.00", "2", "25.00")],
                       "150.00", [_tax("CGST", "3.57"), _tax("SGST", "3.57")]))
    assert v["status"] == "VALID" and v["tax_mode"] == "inclusive" and v["balance_diff"] == 0


def test_exclusive_bill_balances() -> None:
    v = validate(_bill([_it("Tea", "100.00", "1", "100.00"), _it("Bun", "50.00", "2", "25.00")],
                       "177.00", [_tax("CGST", "13.50", "9"), _tax("SGST", "13.50", "9")]))
    assert v["status"] == "VALID" and v["tax_mode"] == "exclusive"


def test_wrong_amount_repaired_by_qty_rate() -> None:
    # bill_10 pattern: 2 x 32.00 printed as 84.00
    v = validate(_bill([_it("Tea", "20.00", "1", "20.00"),
                        _it("Samosa", "84.00", numbers=["2", "32.00"])], "84.00",
                       label="total", cash="84.00"))
    assert v["status"] == "REPAIRED"
    assert [(r["old"], r["new"]) for r in v["repairs"]] == [(Decimal("84.00"), Decimal("64.00"))]


def test_two_wrong_amounts_both_repaired() -> None:
    v = validate(_bill([_it("Samosa", "84.00", numbers=["2", "32.00"]),
                        _it("Vada", "46.00", numbers=["3", "15.00"]),
                        _it("Tea", "10.00", "1", "10.00")], "119.00", label="total"))
    assert v["status"] == "REPAIRED" and len(v["repairs"]) == 2
    assert {r["new"] for r in v["repairs"]} == {Decimal("64.00"), Decimal("45.00")}


def test_weighed_item_exempt() -> None:
    v = validate(_bill([_it("Apples", "64.30", "0.535", "120.00"), _it("Tea", "10.00", "1", "10.00")],
                       "74.30"))
    assert v["status"] == "VALID" and not v["repairs"]


def test_ambiguous_repair_rejected() -> None:
    v = validate(_bill([_it("Tea", "20.00", "1", "20.00"),
                        _it("Samosa", "84.00", numbers=["2", "3", "32.00"])], "100.00",
                       label="total"))
    assert v["status"] == "CHECK_REQUIRED" and not v["repairs"]
    assert any(r.startswith("ambiguous_repair") for r in v["reasons"])


def test_cgst_ne_sgst_flagged() -> None:
    v = validate(_bill([_it("Tea", "100.00", "1", "100.00")], "118.50",
                       [_tax("CGST", "9.00"), _tax("SGST", "9.50")]))
    assert any(r.startswith("cgst_ne_sgst") for r in v["reasons"])


def test_tetal_totals_row_excluded() -> None:
    d = extract(_rows("Item Qty Rate Amount", "Tea 2 10.00 20.00", "Coffee 1 30.00 30.00",
                      "Tetal 3 50.00"))
    assert [i["name"] for i in d["items"]] == ["Tea", "Coffee"]
    assert d["excluded_rows"][0]["row"] == 3


def test_total_switched_to_supported_candidate() -> None:
    v = validate(_bill([_it("Tea", "20.00", "2", "10.00"), _it("Coffee", "30.00", "1", "30.00")],
                       "80.00", alts=["50.00", "100.00"], label="total", cash="50.00"))
    assert v["status"] == "REPAIRED" and v["total"] == Decimal("50.00")
    assert v["repairs"][0]["field"] == "total"


def test_insufficient_without_total() -> None:
    v = validate(_bill([_it("Tea", "20.00", "2", "10.00")], None))
    assert v["status"] == "INSUFFICIENT_EVIDENCE" and v["confidence"] == 0.0


def test_row_local_repair_unique() -> None:
    # bill_10 pattern: row reads [rate 65.00, amount 85.00], the total needs 65.00
    v = validate(_bill([_it("Tea", "20.00", "1", "20.00"),
                        _it("Samosa", "85.00", numbers=["65.00"])], "85.00", cash="85.00"))
    assert v["status"] == "REPAIRED"
    assert [(r["old"], r["new"], r["reason"]) for r in v["repairs"]] == [
        (Decimal("85.00"), Decimal("65.00"), "row_local (unique fix)")]


def test_row_local_repair_ambiguous() -> None:
    v = validate(_bill([_it("Tea", "20.00", "1", "20.00"),
                        _it("Samosa", "90.00", numbers=["70.00"]),
                        _it("Vada", "45.00", numbers=["25.00"])], "135.00", label="total"))
    assert v["status"] == "CHECK_REQUIRED" and not v["repairs"]
    assert any(r.startswith("ambiguous_row_local") for r in v["reasons"])


# ---------- phase 6 (made-up data) ----------

def test_total_derived_written_when_not_valid() -> None:
    v = validate(_bill([_it("Tea", "20.00", "1", "20.00"), _it("Bun", "30.00", "1", "30.00")],
                       "75.00", label="total"))
    assert v["status"] == "CHECK_REQUIRED" and v["total_derived"] == Decimal("50.00")
    assert "total_derived_from_items" in v["reasons"]
    ok = validate(_bill([_it("Tea", "20.00", "1", "20.00")], "20.00", label="total"))
    assert ok["status"] == "VALID" and ok["total_derived"] is None


def test_unsupported_repair_downgraded() -> None:
    # labeled 34.00 is digit-repaired to 84.00; only the repair supports 84.00
    v = validate(_bill([_it("Tea", "20.00", "1", "20.00"), _it("Samosa", "64.00", "2", "32.00")],
                       "34.00", label="total"))
    assert v["repairs"] and v["repairs"][0]["field"] == "total"
    assert v["status"] == "CHECK_REQUIRED" and "repair_unsupported" in v["reasons"]


def test_implied_roundoff_valid() -> None:
    v = validate(_bill([_it("Rice", "100.00", "1", "100.00"), _it("Dal", "37.64", "1", "37.64")],
                       "137.00", label="total"))
    assert v["status"] == "VALID" and not v["repairs"]
    assert "implied_roundoff: 0.64" in v["reasons"]


def test_weak_total_never_edits_items() -> None:
    # a row-local fix (100.00 -> 86.36) would fit the unlabeled 200.36: not allowed
    v = validate(_bill([_it("Shirt", "114.00", "1", "114.00"),
                        _it("Cap", "100.00", numbers=["86.36"])], "200.36"))
    assert not v["repairs"] and v["status"] == "CHECK_REQUIRED"
    assert "total_unsupported" in v["reasons"] and v["total_derived"] == Decimal("214.00")


def test_tax_matches_total_witness() -> None:
    # 118.00 inclusive of 18% GST: tax 18.00
    v = validate(_bill([_it("Lamp", "118.00", "1", "118.00")], "118.00",
                       [_tax("CGST", "9.00", "9"), _tax("SGST", "9.00", "9")]))
    assert "tax_matches_total" in v["support"] and "tax_matches_total" in v["witnesses"]


def test_cash_tie_does_not_switch_total() -> None:
    b = _bill([_it("Tea", "20.00", "2", "10.00"), _it("Coffee", "30.00", "1", "30.00")], "70.00")
    b["total"].update(breakdown={"label": 0, "position": 1, "repeat": 2},
                      alternatives=[{"value": "50.00", "label": "cash", "score": 3,
                                     "breakdown": {"label": 0}}])
    v = validate(b)
    assert v["total"] == Decimal("70.00") and v["status"] == "CHECK_REQUIRED"
    assert v["needs_review"] and "check_required" in v["review_reasons"]


# ---------- phase 8 (made-up data) ----------

def test_repair_contradicting_labeled_total_cancelled() -> None:
    # items already sum to the labeled 214.00; a row-local fix (100.00 -> 86.36) would fit 200.36
    b = _bill([_it("Shirt", "114.00", "1", "114.00"), _it("Cap", "100.00", numbers=["86.36"])],
              "200.36", cash="200.36")
    b["total"]["alternatives"] = [{"value": "214.00", "label": "total", "score": 3,
                                   "breakdown": {"label": 2}}]
    v = validate(b)
    assert not v["repairs"] and v["status"] == "CHECK_REQUIRED"
    assert "repair_contradicts_labeled_total" in v["reasons"]
    assert v["items"][1]["amount"] == Decimal("100.00")
