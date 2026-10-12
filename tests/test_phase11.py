import sys
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.extract import extract  # noqa: E402
from src.normalize import parse_qty  # noqa: E402
from test_extract import _rows  # noqa: E402


def test_parse_qty() -> None:
    assert parse_qty("0.500") == Decimal("0.500") and parse_qty("2.000") == 2
    assert parse_qty("1.25") == Decimal("1.25") and parse_qty("3") == 3
    assert parse_qty("1,000.00") is None and parse_qty("12.3456") is None


def test_hsn_note_attaches_to_previous_item() -> None:
    f = extract(_rows("Item Qty Rate Amount", "Paneer 1 200.00 200.00", "HSN 040690 GST 5%",
                      "Butter 2 50.00 100.00", "Total 300.00"))
    items = f["items"]
    assert [i["name"] for i in items] == ["Paneer", "Butter"]
    assert items[0]["hsn"] == "040690" and items[0]["gst_rate"] == Decimal("5")
    assert "hsn" not in items[1] and "gst_rate" not in items[1]
    assert f["tax_rates_seen"] == [Decimal("5")]
    assert f["tax_lines"] == [] and f["tax_sum"] == 0  # display only


def test_glued_ocr_note() -> None:
    # OCR glues the note into one token and reads GST as G8T
    f = extract(_rows("Item Qty Rate Amount", "Laddoo 1 300.00 300.00", "HSN12345678G8T12%",
                      "Barfi 2 100.00 200.00", "Total 500.00"))
    items = f["items"]
    assert [i["name"] for i in items] == ["Laddoo", "Barfi"]
    assert items[0]["hsn"] == "12345678" and items[0]["gst_rate"] == Decimal("12")
    f = extract(_rows("Item Qty Rate Amount", "Tea SACHET 2 5.00 10.00", "Total 10.00"))
    assert f["items"][0]["name"] == "Tea SACHET"


def test_weighed_qty_and_disc_column() -> None:
    f = extract(_rows("Item Qty Rate Disc Amount", "Apples 0.750 | 400.00 | 0.00 | 300.00",
                      "Rice 2.000 | 500.00 | 0.00 | 1,000.00", "Total 1,300.00"))
    a, r = f["items"]
    assert a["status"] == "verified" and a["weighed"] is True
    assert (a["qty"], a["rate"], a["discount"]) == (Decimal("0.750"), Decimal("400.00"), Decimal("0.00"))
    assert r["status"] == "verified" and "weighed" not in r
    assert (r["qty"], r["rate"], r["discount"]) == (2, Decimal("500.00"), Decimal("0.00"))
    assert all(i["rate"] != 0 and i["qty"] != 0 for i in (a, r))  # 0.00 disc is never qty/rate


def test_weighed_qty_integer_style() -> None:
    f = extract(_rows("Item Qty Rate Amount", "Rice 2.000 500 1,000", "Total 1,000"))
    it = f["items"][0]
    assert f["money_style"]["value"] == "integer"
    assert it["status"] == "verified" and (it["qty"], it["rate"]) == (2, 500)


def test_note_text_removed_from_names() -> None:
    f = extract(_rows("Item Qty Rate Amount", "Paneer HSN 040690 GST 5% 1 200.00 200.00",
                      "Total 200.00"))
    assert [i["name"] for i in f["items"]] == ["Paneer"]
