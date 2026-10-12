"""Compact text view of outputs/stages/<stem>/04_fields.json."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _v(f: dict) -> str:
    return "null" if f["value"] is None else f"{f['value']} (conf {f['confidence']}, rows {f['rows']})"


def main(stem: str) -> None:
    d = json.loads((ROOT / "outputs" / "stages" / stem / "04_fields.json").read_text(encoding="utf-8"))
    print(f"== {stem}  rows={d['n_rows']}  regions={d['regions']}")
    for k in ("merchant", "date", "gstin", "bill_no", "bill_type"):
        print(f"{k}: {_v(d[k])}")
    ms = d["money_style"]
    print(f"money_style: {ms['value']} (strong 2-dec: {ms['n_strong']})")
    t = d["total"]
    print(f"total: {_v(t)} gap={t.get('score_gap')}")
    if t["value"] is not None:
        for c in [{"value": t["value"], "row": t["rows"][0], "breakdown": t["breakdown"]}] + t["alternatives"]:
            b = " ".join(f"{k}={v}" for k, v in c["breakdown"].items() if v)
            print(f"  {c['value']:>12} row {c['row']:>3} score {sum(c['breakdown'].values()):>3} | {b}")
    for k in ("subtotal", "discount", "round_off"):
        print(f"{k}: {_v(d[k])}")
    tx = d["taxes"]
    for h in tx["regex"]:
        print(f"tax: {h['label']} @{h['rate']}% = {h['amount']} (row {h['row']})")
    for h in tx["label"]:
        print(f"tax(label): {h['label']} = {h['amount']} (row {h['row']})")
    print(f"tax_sum: {d['tax_sum']}  tax_table_rows: {len(tx['table_rows'])}")
    print(f"items: {len(d['items'])}")
    for it in d["items"][:40]:
        name = (it["name"] or "-")[:32]
        extra = f" tax_detail={it['tax_detail']}" if it["tax_detail"] else ""
        kind = "" if it["kind"] == "item" else f" [{it['kind']}]"
        print(f"  {it['row']:>3} | {name:<32} | {it['qty'] or '-'} | {it['rate'] or '-'} | "
              f"{it['amount']} | {it['status']}{kind}{extra}")
    print(f"sum of items: {d['items_sum']}")


if __name__ == "__main__":
    main(sys.argv[1])
