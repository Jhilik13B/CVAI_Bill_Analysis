import sys
from decimal import Decimal
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.keywords import match_labels  # noqa: E402
from src.normalize import (parse_amount, parse_date, parse_percent, split_glued,  # noqa: E402
                           split_label_glued)


@pytest.mark.parametrize("text,value,strength", [
    ("1,00,000.00", "100000.00", "strong"),
    ("₹ 1,234.50", "1234.50", "strong"),
    ("Rs.99", "99", "strong"),
    ("75,000", "75000", "weak"),
    ("12,50", "12.50", "strong"),
    ("-15.00", "-15.00", "strong"),
    ("(15.00)", "-15.00", "strong"),
    ("15.00-", "-15.00", "strong"),
    ("1O4.OO", "104.00", "strong"),
    ("[2.04]", "2.04", "strong"),
    ("1,346,000", "1346000", "weak"),
])
def test_amount_ok(text: str, value: str, strength: str) -> None:
    a = parse_amount(text)
    assert a is not None and a.value == Decimal(value) and a.strength == strength


@pytest.mark.parametrize("text", ["TOTAL", "9100499956", "19052000", "36AABCR8939B1ZB",
                                  "13:14:27", "9.00%"])
def test_amount_none(text: str) -> None:
    assert parse_amount(text) is None


def test_percent() -> None:
    assert parse_percent("18.00 %") == Decimal("18.00")
    assert parse_percent("@9%") == Decimal("9")


@pytest.mark.parametrize("text,iso,amb", [
    ("13/05/2019", "2019-05-13", False),
    ("05/06/2019", "2019-06-05", True),
    ("07/07/17", "2017-07-07", False),
    ("2019-05-13", "2019-05-13", False),
    ("13 May 2019", "2019-05-13", False),
    ("May 13, 2019", "2019-05-13", False),
    ("Date: 13-May-19 10:20", "2019-05-13", False),
])
def test_date(text: str, iso: str, amb: bool) -> None:
    d = parse_date(text)
    assert d and d[0]["iso"] == iso and d[0]["ambiguous"] is amb


def test_date_impossible() -> None:
    assert parse_date("32/13/2019") == []


@pytest.mark.parametrize("text,label", [
    ("T0TAL", "TOTAL"), ("TOTA1", "TOTAL"), ("GRAND T0TAL", "GRAND_TOTAL"),
    ("Sub Total", "SUBTOTAL"), ("CGST@9.00%", "CGST"), ("ROUNDOFF", "ROUNDOFF"),
])
def test_labels_first(text: str, label: str) -> None:
    labels = match_labels(text)
    assert labels and labels[0][0] == label


def test_description_not_total() -> None:
    assert "TOTAL" not in [l for l, _ in match_labels("Description")]


@pytest.mark.parametrize("text,parts", [
    ("600.0012.03", ["600.00", "12.03"]),
    ("1,234.50", ["1,234.50"]),
    ("12.03", ["12.03"]),
])
def test_split_glued(text: str, parts: list[str]) -> None:
    assert split_glued(text) == parts


def test_split_hsn_glued() -> None:
    hsn, amt = split_glued("190520006.00")
    assert hsn == "HSN:19052000" and parse_amount(hsn) is None
    assert parse_amount(amt).value == Decimal("6.00")


def test_split_label_glued() -> None:
    label, num = split_label_glued("Tota1.1591600")
    assert label == "Tota1" and parse_amount(num).value == Decimal("1591600")
    assert split_label_glued("Inv:12345") == ["Inv:12345"]
    assert split_label_glued("Rs.500") == ["Rs.500"]


@pytest.mark.parametrize("text,label", [
    ("TOTAL TAX 45.90", "TAX_TOTAL"), ("Total GST", "TAX_TOTAL"), ("GST TOTAL", "TAX_TOTAL"),
    ("Taxable Value", "TAXABLE"), ("TAXABLE AMT", "TAXABLE"),
])
def test_tax_labels(text: str, label: str) -> None:
    assert label in [l for l, _ in match_labels(text)]


def test_fuzzy_month_date() -> None:
    assert parse_date("13-Mr-2024")[0]["iso"] == "2024-03-13"
    assert parse_date("12 Ma 2024") == []  # Mar/May tie: no unique month


def test_one_decimal_and_trailing_dot() -> None:
    assert parse_amount("365.0").strength == "weak"
    a = parse_amount("365.0", {"GRAND_TOTAL"})
    assert a.strength == "strong" and a.value == Decimal("365.00")
    assert parse_amount("$0.9").value == Decimal("0.90")
    assert parse_amount("$10.").value == Decimal("10.00")
    assert parse_amount("10.", {"CHANGE"}).value == Decimal("10.00")
    assert parse_amount("10.") is None


def test_label_glued_colon_equals() -> None:
    assert split_label_glued("NetAmount:3814.00") == ["Net Amount", "3814.00"]
    assert split_label_glued("Total=250.00") == ["Total", "250.00"]
