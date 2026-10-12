import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from evaluate import _money, evaluate  # noqa: E402


def test_evaluate_temporary_annotation(tmp_path: Path) -> None:
    ann, stages = tmp_path / "ann", tmp_path / "stages"
    ann.mkdir()
    (stages / "bill_x").mkdir(parents=True)
    (stages / "bill_y").mkdir(parents=True)
    # made-up annotation and predictions; bill_y has a blank total and is skipped
    (ann / "bill_x.json").write_text(json.dumps(
        {"merchant": "Green Cafe", "date": "2024-03-13", "total": "118.00", "tax_total": "18.00",
         "item_count": 2, "notes": ""}))
    (ann / "bill_y.json").write_text(json.dumps(
        {"merchant": "", "date": "YYYY-MM-DD", "total": "", "tax_total": "", "item_count": None,
         "notes": ""}))
    (stages / "bill_x" / "04_fields.json").write_text(json.dumps(
        {"date": {"value": "2024-03-12"}, "merchant": {"value": "GREEN CAFE"}}))
    (stages / "bill_x" / "05_validation.json").write_text(json.dumps(
        {"status": "VALID", "total": "118.00", "tax_sum": "18.00",
         "items": [{"kind": "item"}, {"kind": "item"}]}))
    rows, s = evaluate(ann, stages)
    assert s["n"] == 1 and s["accuracy"]["total"] == (1, 1) and s["accuracy"]["merchant"] == (1, 1)
    assert s["accuracy"]["date"] == (0, 1) and s["vr_precision"] == (1, 1)
    assert s["mismatches"] == ["bill_x: date 2024-03-12 vs 2024-03-13"]


def test_item_count_string_annotation_matches_int(tmp_path: Path) -> None:
    ann, stages = tmp_path / "ann", tmp_path / "stages"
    ann.mkdir()
    (stages / "bill_z").mkdir(parents=True)
    # annotation stores item_count as "2" (string); prediction counts 2 items (int)
    (ann / "bill_z.json").write_text(json.dumps(
        {"merchant": "", "date": "YYYY-MM-DD", "total": "50", "tax_total": "", "item_count": "2"}))
    (stages / "bill_z" / "04_fields.json").write_text(json.dumps(
        {"date": {"value": None}, "merchant": {"value": None}}))
    (stages / "bill_z" / "05_validation.json").write_text(json.dumps(
        {"status": "VALID", "total": "50.00", "items": [{"kind": "item"}, {"kind": "item"}]}))
    rows, s = evaluate(ann, stages)
    assert s["accuracy"]["item_count"] == (1, 1) and s["accuracy"]["total"] == (1, 1)
    assert rows[0]["date_ok"] is None and rows[0]["merchant_ok"] is None
    assert rows[0]["tax_total_ok"] is None and s["mismatches"] == []


def test_annotation_number_formats(tmp_path: Path) -> None:
    from decimal import Decimal
    assert _money("33000,00") == Decimal("33000.00") and _money(" 1 250.50 ") == Decimal("1250.50")
    ann, stages = tmp_path / "ann", tmp_path / "stages"
    ann.mkdir()
    for stem, total in (("bill_a", "33000,00"), ("bill_b", "190.000")):
        (stages / stem).mkdir(parents=True)
        (ann / f"{stem}.json").write_text(json.dumps(
            {"merchant": "", "date": "2018/04/23", "total": total, "tax_total": "",
             "item_count": "2"}))
        (stages / stem / "04_fields.json").write_text(json.dumps(
            {"date": {"value": "2018-04-23"}, "merchant": {"value": None}}))
        (stages / stem / "05_validation.json").write_text(json.dumps(
            {"status": "CHECK_REQUIRED", "total": "33000.00", "total_derived": "33000.00",
             "items": [{"kind": "item"}, {"kind": "item"}]}))
    rows, s = evaluate(ann, stages)
    assert s["accuracy"]["total"] == (1, 1) and s["accuracy"]["date"] == (2, 2)
    assert s["accuracy"]["item_count"] == (2, 2) and s["derived_accuracy"] == (1, 1)
    assert s["n_skipped"] == 1 and s["warnings"] == ["bill_b: total '190.000' is ambiguous"]
