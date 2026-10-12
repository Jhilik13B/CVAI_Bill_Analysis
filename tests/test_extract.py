import sys
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.extract import _load, _money, _taxes, _total, extract  # noqa: E402
from src.keywords import match_labels  # noqa: E402
from src.normalize import parse_amount, split_glued  # noqa: E402
from src.rows import build_rows, row_amounts, row_labels  # noqa: E402
from src.validate import validate  # noqa: E402


def _row(idx: int, text: str) -> dict:
    """Made-up row dict shaped like 03_rows.json (labels and amounts as in build_rows)."""
    toks = [{"text": t, "conf": 0.95} for t in text.split()]
    labels = row_labels(text)
    return {"idx": idx, "y": idx * 20.0, "text": text, "cells": [text], "tokens": toks,
            "min_conf": 0.95, "labels": labels, "amounts": row_amounts(toks, [0] * len(toks), labels)}


def _rows(*texts: str) -> list[dict]:
    return [_row(i, t) for i, t in enumerate(texts)]


def test_money_style() -> None:
    assert extract(_rows("Tea 10.00", "Bun 5.00", "Total 15.00"))["money_style"]["value"] == "decimal"
    assert extract(_rows("Tea 10,000", "Bun 5,000", "Total 15,000"))["money_style"]["value"] == "integer"


def test_pincode_not_money_in_decimal_style() -> None:
    rows = _rows("SHREE STORES", "Andheri Mumbai 400053", "Item Qty Amount",
                 "Tea 2 10.00 20.00", "Coffee 1 30.00 30.00", "Total 50.00")
    d = extract(rows)
    assert d["money_style"]["value"] == "decimal"
    assert _money(_load(rows)[1], "decimal") == []
    vals = [d["total"]["value"]] + [a["value"] for a in d["total"]["alternatives"]]
    assert Decimal("400053") not in vals and d["total"]["value"] == Decimal("50.00")


def test_total_by_arithmetic_without_label() -> None:
    d = extract(_rows("Item Qty Rate Amount", "Tea 2 10.00 20.00", "Coffee 1 30.00 30.00",
                      "50.00", "Received 100.00"))
    assert d["total"]["value"] == Decimal("50.00")
    assert d["total"]["breakdown"]["items_sum"] == 3


def test_two_line_item_with_tax_detail() -> None:
    d = extract(_rows("Item Qty Rate Amount", "Paneer Butter Masala", "2 150.00 300.00",
                      "996331 [CGST@2.5% 7.50]", "Total 315.00"))
    assert len(d["items"]) == 1
    it = d["items"][0]
    assert it["name"] == "Paneer Butter Masala" and it["status"] == "verified"
    assert it["qty"] == Decimal("2") and it["rate"] == Decimal("150.00")
    assert Decimal("7.50") in it["tax_detail"]


def test_discount_row_is_negative_item() -> None:
    d = extract(_rows("Item Qty Rate Amount", "Pizza 1 500.00 500.00", "Discount -50.00",
                      "Total 450.00"))
    disc = [i for i in d["items"] if i["kind"] == "discount"]
    assert len(disc) == 1 and disc[0]["amount"] == Decimal("-50.00")
    assert d["items_sum"] == Decimal("450.00")


def test_tax_regex_glued() -> None:
    rows = _load(_rows("CGST@9.00%25.85[SGST@9.00%25.85]", "[CGST@6.0020.07]", "Net 600.0012.03"))
    hits = _taxes(rows, 0, "decimal")["regex"]
    assert [(h["label"], h["rate"], h["amount"]) for h in hits] == [
        ("CGST", Decimal("9.00"), Decimal("25.85")), ("SGST", Decimal("9.00"), Decimal("25.85")),
        ("CGST", Decimal("6.00"), Decimal("20.07"))]
    assert [a["value"] for a in rows[2]["amounts"]] == [Decimal("600.00"), Decimal("12.03")]


def test_single_tax_source() -> None:
    d = extract(_rows("Item Qty Rate Amount", "Tea 1 100.00 100.00", "Sub Total 100.00",
                      "CGST @9% 9.00", "SGST @9% 9.00", "Taxable CGST SGST Total",
                      "100.00 9.00 9.00 118.00", "Total 118.00"))
    assert d["tax_source"] == "pairs" and d["tax_sum"] == Decimal("18.00")
    assert d["taxes"]["table"]["tax"] == Decimal("18.00")


def test_bad_tax_rate_dropped() -> None:
    t = _taxes(_load(_rows("CGST @7% 9.00")), 0, "decimal")
    assert t["regex"] == [] and t["label"] == [] and t["rejected"][0]["rate"] == Decimal("7")


def test_bogus_97_percent_rejected() -> None:
    t = _taxes(_load(_rows("GST @97% 97.00")), 0, "decimal")
    assert t["regex"] == [] and t["label"] == [] and t["rejected"][0]["rate"] == Decimal("97")
    # a TAX_TOTAL of 97% of the items sum is rejected too
    d = extract(_rows("Item Qty Rate Amount", "Tea 1 100.00 100.00", "Total Tax 97.00",
                      "Total 197.00"))
    assert d["tax_sum"] == 0 and d["tax_source"] == "none"
    assert d["taxes"]["rejected"][0]["reason"] == "above_35pct_of_items"


def test_inclusive_tax_per_rate_row() -> None:
    d = extract(_rows("Item Qty Rate Amount", "Paneer 1 339.00 339.00", "Total 339.00",
                      "GST/Tax @ 18.00 | 339.00 | 51.71"))
    assert d["tax_source"] == "rate_rows" and d["tax_sum"] == Decimal("51.71")
    rr = d["taxes"]["rate_rows"][0]
    assert (rr["rate"], rr["base"], rr["amount"]) == (Decimal("18.00"), Decimal("339.00"),
                                                      Decimal("51.71"))
    v = validate(d)
    assert v["tax_checks"][0]["mode"] == "inclusive" and v["status"] == "VALID"
    assert not any(r.startswith(("rate_row_mismatch", "tax_vs_table")) for r in v["reasons"])
    assert "rate_bases_eq_items" in v["support"]


def test_tax_sum_from_tax_total_row() -> None:
    d = extract(_rows("Item Qty Rate Amount", "Laptop bag 1 3000.00 3000.00",
                      "Shoes 1 2400.00 2400.00", "Total Tax 972.00", "Taxable CGST SGST Total",
                      "5400.00 1.00 2.00 3.00", "Total 6372.00"))
    assert d["tax_source"] == "tax_total" and d["tax_sum"] == Decimal("972.00")


def test_merchant_prefers_confident_row() -> None:
    rows = _rows("GreBsn Cafe", "GREEN CAFE", "Tea 1 10.00 10.00", "Total 10.00")
    for t in rows[0]["tokens"]:
        t["conf"] = 0.5
    rows[0]["min_conf"] = 0.5
    for t in rows[1]["tokens"]:
        t["conf"] = 0.98
    assert extract(rows)["merchant"]["value"] == "GREEN CAFE"


def test_totals_row_excluded() -> None:
    d = extract(_rows("Item Qty Rate Amount", "Tea 2 10.00 20.00", "Coffee 1 30.00 30.00",
                      "Items 3 50.00"))
    assert len(d["items"]) == 2 and d["excluded_rows"][0]["reason"] == "totals_row"


def test_gstin_confusables_and_bill_no() -> None:
    d = extract(_rows("SHREE STORES", "GSTIN 27AAP5UO939F1ZV", "Bill No 27AAPFU0939F1ZV",
                      "Tea 10.00", "Total 10.00"))
    assert d["gstin"]["value"] == "27AAPSU0939F1ZV"
    assert d["bill_no"]["value"] is None


def test_item_name_cleanup() -> None:
    d = extract(_rows("Item Qty Rate Amount", ")Paneer CGST@2.5% 2 150.00 300.00",
                      "DalSGST@2.5%7.50 1 150.00 150.00", "Total 457.50"))
    assert [i["name"] for i in d["items"]] == ["Paneer", "Dal"]
    assert d["items"][1]["tax_detail"] == [Decimal("7.50")]


def test_new_keywords() -> None:
    assert match_labels("PB1 10% 5,000")[0][0] == "VAT"
    assert match_labels("PPN 5,000")[0][0] == "VAT"
    assert match_labels("JUMLAH 50,000")[0][0] == "TOTAL"
    assert match_labels("TOTAL HARGA 50,000")[0][0] == "TOTAL"


def test_repeated_amount_boosts_total() -> None:
    d = extract(_rows("Item Qty Amount", "Total 80.00", "Total 90.00", "Card 90.00"))
    assert d["total"]["value"] == Decimal("90.00")
    assert d["total"]["breakdown"]["repeat"] == 1


# ---------- phase 6 (made-up data) ----------

def _cand_values(d: dict) -> list:
    return [d["total"]["value"]] + [a["value"] for a in d["total"]["alternatives"]]


def test_cash_equals_items_picked() -> None:
    d = extract(_rows("Item Qty Rate Amount", "Tea 2 10.00 20.00", "Coffee 3 50.00 150.00",
                      "Cake 1 120.00 120.00", "Bread 2 100.00 200.00", "Milk 4 50.00 200.00",
                      "Jam 1 100.00 100.00", "CASH 790.00"))
    assert all(i["status"] == "verified" for i in d["items"]) and len(d["items"]) == 6
    assert d["total"]["value"] == Decimal("790.00")
    assert d["total"]["breakdown"]["payment_equals_items"] == 2


def test_item_rows_not_total_candidates() -> None:
    d = extract(_rows("Item Qty Rate Amount", "Tea 2 10.00 20.00", "Coffee 1 30.00 30.00",
                      "Cake 1 50.00 50.00", "Total 100.00"))
    item_rows = {i["row"] for i in d["items"]}
    rows = [d["total"]["rows"][0]] + [a["row"] for a in d["total"]["alternatives"]]
    assert not item_rows & set(rows) and d["total"]["value"] == Decimal("100.00")


def test_one_decimal_net_amount() -> None:
    d = extract(_rows("Item Qty Rate Amount", "Rice 2 100.00 200.00", "Dal 1 165.00 165.00",
                      "Net Amount: | 365.0"))
    assert d["total"]["value"] == Decimal("365.00")


def test_cash_minus_change() -> None:
    d = extract(_rows("Burger 25.00", "Fries 15.00", "40.00", "Cash $50.00", "Change $10."))
    assert d["total"]["value"] == Decimal("40.00")
    assert d["total"]["breakdown"]["cash_change"] == 3


def test_slab_rows_never_win() -> None:
    d = extract(_rows("Item Qty Rate Amount", "Soap 1 45.00 45.00", "Oil 1 55.00 55.00",
                      "0%: 0.00", "5%: 0.00", "12%: 0.00", "18%: 0.00", "Total 100.00"))
    assert d["total"]["value"] == Decimal("100.00")
    assert Decimal("0.00") not in _cand_values(d)
    assert [i["name"] for i in d["items"]] == ["Soap", "Oil"]


def test_net_amount_glued_outranks_total_mrp() -> None:
    d = extract(_rows("Item Qty Rate Amount", "Shirt 1 4000.00 4000.00",
                      "Jeans 1 4324.80 4324.80", "TotalMRP:8324.80", "NetAmount:3814.00"))
    assert d["total"]["value"] == Decimal("3814.00")
    assert Decimal("8324.80") not in _cand_values(d)


def test_merchant_prefix_and_penalties() -> None:
    d = extract(_rows("ORIGINAL FOR RECIPIENT", "Sold By: SHREE GANESH TRADERS",
                      "MG ROAD PUNE 411001", "Tea 10.00", "Total 10.00"))
    assert d["merchant"]["value"] == "SHREE GANESH TRADERS"


def test_merchant_split_across_lines() -> None:
    d = extract(_rows("TAX INVOICE", "SRI LAKSHMI", "MEDICAL STORES", "12 MAIN ROAD 560001",
                      "Tablet 50.00", "Total 50.00"))
    assert d["merchant"]["value"] == "SRI LAKSHMI MEDICAL STORES"


def test_restaurant_needs_another_word() -> None:
    d = extract(_rows("RESTAURANT", "BLUE LEAF RESTAURANT", "Tea 10.00", "Total 10.00"))
    assert d["merchant"]["value"] == "BLUE LEAF RESTAURANT"


# ---------- phase 7 (made-up data) ----------

def test_payment_equals_items_needs_verified_item() -> None:
    d = extract(_rows("Tea 20.00", "Coffee 30.00", "CASH 50.00"))
    assert not any(i["status"] == "verified" for i in d["items"])
    assert d["total"]["breakdown"]["payment_equals_items"] == 0


def test_item_row_equal_to_cash_minus_change_is_candidate() -> None:
    d = extract(_rows("Item Qty Rate Amount", "Tea 2 10.00 20.00", "Coffee 1 30.00 30.00",
                      "Cake 1 40.00 40.00", "Cash 100.00", "Change 60.00"))
    assert 3 in {i["row"] for i in d["items"]}
    assert Decimal("40.00") in _cand_values(d)


def test_labeled_total_on_item_row_stays_candidate() -> None:
    rows = _load(_rows("Tea 2 10.00 20.00", "Coffee 1 30.00 30.00", "Total 50.00"))
    items = [{"row": i, "kind": "item", "amount": a, "status": "unverified"}
             for i, a in enumerate((Decimal("20.00"), Decimal("30.00"), Decimal("50.00")))]
    t = _total(rows, None, "decimal", items, Decimal(0), None, None)
    assert t["value"] == Decimal("50.00") and t["label"] == "total"


def _line(text: str, x0: float, y0: float, x1: float, y1: float) -> dict:
    return {"box": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]], "text": text, "conf": 0.9}


def test_tall_name_box_beats_long_row() -> None:
    d = extract(build_rows([
        _line("Green Leaf", 0, 0, 200, 40), _line("12 MG ROAD 411001", 0, 50, 200, 64),
        _line("FRESH MART", 0, 90, 120, 104), _line("DAILY NEEDS", 140, 90, 260, 104),
        _line("Tea 10.00", 0, 200, 120, 214), _line("Total 10.00", 0, 386, 120, 400)]))
    m = d["merchant"]
    assert m["value"] == "Green Leaf" and m["merchant_source"] == "boxes"
    assert {"value": "FRESH MART DAILY NEEDS", "score": None, "source": "rows"} in m["candidates"]


def test_name_split_over_two_boxes_joined() -> None:
    d = extract(build_rows([
        _line("SRI LAKSHMI", 0, 0, 220, 40), _line("MEDICAL STORES", 0, 46, 260, 84),
        _line("12 MAIN ROAD 560001", 0, 95, 200, 109),
        _line("Tablet 50.00", 0, 200, 120, 214), _line("Total 50.00", 0, 386, 120, 400)]))
    assert d["merchant"]["value"] == "SRI LAKSHMI MEDICAL STORES"
    assert d["merchant"]["merchant_source"] == "boxes"
