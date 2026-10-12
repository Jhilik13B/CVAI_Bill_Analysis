"""Run stage 5 (field extraction) on existing 03_rows.json; write 04_fields.json."""
import argparse
import json
import re
import sys
from collections import Counter
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from run_stage12 import IMG_EXT, _img_dir  # noqa: E402
from src.extract import extract  # noqa: E402
from src.normalize import parse_amount  # noqa: E402


def _gt_amount(s) -> Decimal | None:
    """CORD amounts are integer style: '1,591,600', '75.000', '40,000.' -> Decimal."""
    if not isinstance(s, str):
        return None
    s = s.strip().rstrip(".")
    s = re.sub(r"\.(?=\d{3}(\D|$))", ",", s)  # dot thousands -> comma thousands
    a = parse_amount(s)
    return a.value if a else None


def _cord_truth(stem: str) -> dict | None:
    f = ROOT / "data" / "cord" / "labels" / f"{stem}.json"
    if not f.exists():
        return None
    g = json.loads(f.read_text(encoding="utf-8"))["gt_parse"]
    menu = g.get("menu", [])
    menu = [menu] if isinstance(menu, dict) else menu
    sub, tot = g.get("sub_total") or {}, g.get("total") or {}
    sub = sub[0] if isinstance(sub, list) else sub
    tot = tot[0] if isinstance(tot, list) else tot
    return {"n_items": len(menu), "subtotal": _gt_amount(sub.get("subtotal_price")),
            "total": _gt_amount(tot.get("total_price"))}


def _pct(k: int, n: int) -> str:
    return f"{k}/{n} ({100 * k / n:.0f}%)" if n else "n/a"


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--set", choices=["desktop", "cord"])
    g.add_argument("--img-dir", type=Path, help="any image folder (e.g. data/synthetic/clean)")
    args = ap.parse_args()

    stems = sorted(p.stem for p in _img_dir(args).iterdir() if p.suffix.lower() in IMG_EXT)
    res = {}
    for stem in stems:
        dump = ROOT / "outputs" / "stages" / stem
        if not (dump / "03_rows.json").exists():
            continue
        d = extract(json.loads((dump / "03_rows.json").read_text(encoding="utf-8")))
        reocr = dump / "02b_date_ocr.json"  # keep a date recovered by run_date_reocr
        if d["date"]["value"] is None and reocr.exists():
            if (rd := json.loads(reocr.read_text(encoding="utf-8")).get("date")):
                d["date"] = {**d["date"], "value": rd, "reason": None, "date_source": "reocr"}
        (dump / "04_fields.json").write_text(json.dumps(d, indent=1, ensure_ascii=False, default=str),
                                             encoding="utf-8")
        res[stem] = d

    n = len(res)
    print(f"[{args.set or args.img_dir.name}] images: {n}")
    print(f"total found: {sum(d['total']['value'] is not None for d in res.values())}  "
          f"date found: {sum(d['date']['value'] is not None for d in res.values())}  "
          f"merchant found: {sum(d['merchant']['value'] is not None for d in res.values())}  "
          f"items>0: {sum(bool(d['items']) for d in res.values())}")
    print(f"money_style: {dict(Counter(d['money_style']['value'] for d in res.values()))}")
    print(f"bill_type: {dict(Counter(d['bill_type']['value'] for d in res.values()))}")
    print(f"date=null: {', '.join(s for s, d in res.items() if d['date']['value'] is None) or '-'}")
    print(f"items=0: {', '.join(s for s, d in res.items() if not d['items']) or '-'}")

    if args.set != "cord":
        return
    tuned = {k: v for k, v in res.items() if int(k) < 100}
    held = {k: v for k, v in res.items() if 100 <= int(k) < 150}
    mt, mh = _metrics(tuned), _metrics(held)
    print(f"{'metric':<18}{'tuned 0000-0099':<20}held-out 0100-0149")
    for k in mt:
        print(f"{k:<18}{mt[k]:<20}{mh[k]}")


def _metrics(res: dict) -> dict:
    """Stage-5 CORD metrics over a group of stems."""
    tot_ok = tot_n = sub_ok = sub_n = cnt_ok = cnt_n = 0
    for stem, d in res.items():
        gt = _cord_truth(stem)
        if gt is None:
            continue
        pred_t = d["total"]["value"]
        if gt["total"] is not None:
            tot_n += 1
            tot_ok += pred_t is not None and pred_t == gt["total"]
        if gt["subtotal"] is not None:
            sub_n += 1
            sub_ok += d["subtotal"]["value"] is not None and d["subtotal"]["value"] == gt["subtotal"]
        cnt_n += 1
        cnt_ok += sum(i["kind"] == "item" for i in d["items"]) == gt["n_items"]
    return {"images": str(len(res)), "total exact": _pct(tot_ok, tot_n),
            "subtotal exact": _pct(sub_ok, sub_n), "item-count match": _pct(cnt_ok, cnt_n)}

if __name__ == "__main__":
    main()
