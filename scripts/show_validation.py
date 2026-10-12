"""Compact text view of outputs/stages/<stem>/05_validation.json (under 40 lines)."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main(stem: str) -> None:
    v = json.loads((ROOT / "outputs" / "stages" / stem / "05_validation.json").read_text(encoding="utf-8"))
    print(f"== {stem}  status={v['status']}  tax_mode={v['tax_mode']}  conf={v['confidence']}")
    print(f"balance diff: {v['balance_diff']}  (inclusive {v['diffs'].get('inclusive')}, "
          f"exclusive {v['diffs'].get('exclusive')})")
    print(f"sum of items: {v['items_sum']} ({len(v['items'])} items)  tax: {v['tax_sum']} "
          f"[{v['tax_source']}]  subtotal: {v['subtotal']}")
    print(f"total: {v['total']}" + (f"  (extracted {v['total_extracted']})"
                                    if v["total"] != v["total_extracted"] else ""))
    print(f"repairs: {len(v['repairs'])}")
    for r in v["repairs"][:10]:
        print(f"  {str(r['item'])[:28]:<28} {r['field']}: {r['old']} -> {r['new']}  ({r['reason']})")
    print(f"reasons: {len(v['reasons'])}")
    for s in v["reasons"][:15]:
        print(f"  - {s[:90]}")


if __name__ == "__main__":
    main(sys.argv[1])
