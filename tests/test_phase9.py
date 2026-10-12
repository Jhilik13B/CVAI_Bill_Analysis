import json
import sys
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from app import NOT_FOUND, _fmt, explain  # noqa: E402
from evaluate import evaluate, write_csv  # noqa: E402
from make_report_assets import _read, _set_stats, _tamper_stats  # noqa: E402


def test_fresh_eval_and_csv_roundtrip(tmp_path: Path) -> None:
    ann, stages = tmp_path / "ann", tmp_path / "stages"
    ann.mkdir()
    (stages / "bill_a").mkdir(parents=True)
    # made-up bills: bill_a is wrong (CHECK_REQUIRED), bill_b has an empty total and is skipped
    (ann / "bill_a.json").write_text(json.dumps(
        {"merchant": "Blue Bakery", "date": "2024-01-05", "total": "250.00", "tax_total": "",
         "item_count": 3}))
    (ann / "bill_b.json").write_text(json.dumps({"merchant": "", "total": ""}))
    (stages / "bill_a" / "04_fields.json").write_text(json.dumps(
        {"date": {"value": "2024-01-05"}, "merchant": {"value": "BLUE BAKERY"}}))
    (stages / "bill_a" / "05_validation.json").write_text(json.dumps(
        {"status": "CHECK_REQUIRED", "total": "205.00", "items": [{"kind": "item"}]}))
    rows, s = evaluate(ann, stages)
    assert s["n"] == 1 and s["cr_wrong"] == (1, 1) and s["vr_precision"] == (0, 0)
    out = tmp_path / "fresh.csv"
    write_csv(rows, out)
    st = _set_stats(_read(out))
    assert (st["n"], st["total_ok"], st["total_n"], st["CHECK_REQUIRED"]) == (1, 0, 1, 1)


def test_set_stats_counts() -> None:
    rows = [{"status": "VALID", "total_ok": "True"}, {"status": "REPAIRED", "total_ok": "False"},
            {"status": "CHECK_REQUIRED", "total_ok": "False"},
            {"status": "INSUFFICIENT_EVIDENCE", "total_ok": ""}]
    s = _set_stats(rows)
    assert (s["total_ok"], s["total_n"]) == (1, 3)
    assert (s["vr_ok"], s["vr_n"]) == (1, 2) and (s["cr_wrong"], s["cr_n"]) == (1, 1)
    assert s["INSUFFICIENT_EVIDENCE"] == 1


def test_tamper_stats_test_split_only() -> None:
    rows = [{"split": "test", "kind": "tampered", "tamper_type": "total", "det_rules": "True",
             "det_combined": "True"},
            {"split": "test", "kind": "clean", "tamper_type": "", "det_rules": "False",
             "det_combined": "True"},
            {"split": "cal", "kind": "tampered", "tamper_type": "total", "det_rules": "False",
             "det_combined": "False"}]
    t = _tamper_stats(rows)
    assert t["rules"]["det"] == (1, 1) and t["rules"]["fa"] == (0, 1)
    assert t["rules+IF"]["fa"] == (1, 1) and t["rules"]["by_type"]["total"] == (1, 1)


def test_fmt_never_fabricates() -> None:
    assert _fmt(None) == NOT_FOUND and _fmt({"value": None}) == NOT_FOUND
    assert _fmt(Decimal("118.50")) == "118.50" and _fmt({"value": "2024-02-01"}) == "2024-02-01"


def test_explain_lines() -> None:
    ok = explain({"status": "VALID", "tax_mode": "exclusive", "balance_diff": Decimal("0.00"),
                  "repairs": [], "review_reasons": []}, {"flags": []})
    assert "balance" in ok and "No human check needed" in ok
    bad = explain({"status": "CHECK_REQUIRED", "tax_mode": "unknown", "balance_diff": Decimal("9.00"),
                   "repairs": [], "review_reasons": ["check_required"]},
                  {"flags": ["does_not_balance"]})
    assert "do not balance" in bad and "9.00" in bad and "does not balance" in bad
    none = explain({"status": "INSUFFICIENT_EVIDENCE", "balance_diff": None,
                    "review_reasons": ["insufficient_evidence"]}, {"flags": []})
    assert "Not enough" in none
