"""Compare filled annotations with 04_fields.json / 05_validation.json; write phase5_<set>.csv."""
import argparse
import csv
import json
import re
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path

from rapidfuzz import fuzz

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ("total", "date", "merchant", "tax_total", "item_count")
MERCHANT_MIN = 80


PLACEHOLDERS = ("", "YYYY-MM-DD")


def _blank(x) -> bool:
    """Annotation value counts as unfilled: null, empty string or the date placeholder."""
    return x is None or str(x).strip() in PLACEHOLDERS


class Ambiguous(ValueError):
    """'190.000': one separator and 3 digits after it (thousands or decimals?)."""


def _money(x) -> Decimal | None:
    """Annotation money: spaces dropped, decimal comma accepted ('33000,00').
    Raises Ambiguous for '190.000' / '190,000'."""
    if _blank(x):
        return None
    s = re.sub(r"[^\d.,\-]", "", str(x).replace(" ", ""))
    if re.fullmatch(r"-?\d+[.,]\d{3}", s):
        raise Ambiguous(s)
    seps = [i for i, c in enumerate(s) if c in ".,"]
    if seps:
        last = seps[-1]
        dec = len(seps) == 1 or s[last] != s[seps[0]]  # '1,234.50' / '12,50' vs '1,23,456'
        s = s[:last].replace(",", "").replace(".", "") + ("." if dec else "") + s[last + 1:]
    try:
        return Decimal(s)
    except InvalidOperation:
        return None


def _dec(x) -> Decimal | None:
    """Prediction money (plain Decimal strings from the pipeline)."""
    try:
        return None if x is None else Decimal(str(x))
    except InvalidOperation:
        return None


def _count(x) -> int | None:
    if _blank(x):
        return None
    try:
        return int(Decimal(str(x).strip()))
    except (InvalidOperation, ValueError):
        return None


def _date(x) -> str | None:
    """'2018/04/23' or '2018-4-23' -> '2018-04-23'."""
    if _blank(x):
        return None
    m = re.fullmatch(r"(\d{4})[/-](\d{1,2})[/-](\d{1,2})", str(x).strip())
    return f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}" if m else None


def _truth(a: dict, warnings: list[str], stem: str) -> dict:
    """Filled annotation fields only (blanks, placeholders and ambiguous numbers are skipped)."""
    out = {"date": _date(a.get("date")),
           "merchant": None if _blank(a.get("merchant")) else str(a["merchant"]).strip(),
           "item_count": _count(a.get("item_count"))}
    for k in ("total", "tax_total"):
        try:
            out[k] = _money(a.get(k))
        except Ambiguous:
            out[k], out[f"{k}_ambiguous"] = None, True
            warnings.append(f"{stem}: {k} '{a.get(k)}' is ambiguous")
    return out


def _pred(dump: Path) -> dict:
    f = json.loads((dump / "04_fields.json").read_text(encoding="utf-8"))
    v = json.loads((dump / "05_validation.json").read_text(encoding="utf-8"))
    return {"status": v["status"], "total": _dec(v.get("total")), "date": f["date"]["value"],
            "merchant": f["merchant"]["value"], "tax_total": _dec(v.get("tax_sum")),
            "total_derived": _dec(v.get("total_derived")),
            "item_count": sum(i["kind"] == "item" for i in v.get("items", []))}


def _match(field: str, p, t) -> bool:
    if p is None:
        return False
    if field in ("total", "tax_total"):
        # both sides as Decimal so "118.00" and 118 compare equal
        p, t = _dec(p), _dec(t)
        if p is None or t is None:
            return False
        return abs(p - t) <= Decimal("0.01" if field == "total" else "0.10")
    if field == "item_count":
        p, t = _count(p), _count(t)
        return p is not None and p == t
    if field == "merchant":
        return fuzz.ratio(str(p).upper(), str(t).upper()) >= MERCHANT_MIN
    return p == t


def evaluate(ann_dir: Path, stages_dir: Path) -> tuple[list[dict], dict]:
    """Per-bill rows and a summary for every annotation with a total filled in."""
    rows, warnings, n_skipped = [], [], 0
    for af in sorted(Path(ann_dir).glob("*.json")):
        w: list[str] = []
        truth = _truth(json.loads(af.read_text(encoding="utf-8")), w, af.stem)
        dump = Path(stages_dir) / af.stem
        filled = truth["total"] is not None or truth.get("total_ambiguous")
        if not filled or not (dump / "05_validation.json").exists():
            continue
        warnings += w
        n_skipped += bool(w)
        pred = _pred(dump)
        row = {"stem": af.stem, "status": pred["status"]}
        for k in FIELDS:
            row[f"{k}_pred"], row[f"{k}_truth"] = pred[k], truth[k]
            row[f"{k}_ok"] = None if truth[k] is None else _match(k, pred[k], truth[k])
        # total_derived is only written for bills that are not VALID
        row["total_derived_pred"] = pred["total_derived"]
        row["total_derived_ok"] = None if truth["total"] is None or pred["total_derived"] is None \
            else _match("total", pred["total_derived"], truth["total"])
        rows.append(row)
    acc = {k: (sum(r[f"{k}_ok"] for r in rows if r[f"{k}_ok"] is not None),
               sum(r[f"{k}_ok"] is not None for r in rows)) for k in FIELDS}
    der = [r["total_derived_ok"] for r in rows if r["total_derived_ok"] is not None]
    table = Counter((r["status"], bool(r["total_ok"])) for r in rows if r["total_ok"] is not None)
    scored = [r for r in rows if r["total_ok"] is not None]
    vr = [r for r in scored if r["status"] in ("VALID", "REPAIRED")]
    cr = [r for r in scored if r["status"] == "CHECK_REQUIRED"]
    mism = [f"{r['stem']}: {k} {r[f'{k}_pred']} vs {r[f'{k}_truth']}"
            for r in rows for k in FIELDS if r[f"{k}_ok"] is False]
    summary = {"n": len(rows), "accuracy": acc, "table": table,
               "derived_accuracy": (sum(der), len(der)), "n_skipped": n_skipped,
               "warnings": warnings,
               "vr_precision": (sum(bool(r["total_ok"]) for r in vr), len(vr)),
               "cr_wrong": (sum(not r["total_ok"] for r in cr), len(cr)), "mismatches": mism}
    return rows, summary


def _pct(k: int, n: int) -> str:
    return f"{k}/{n} ({100 * k / n:.0f}%)" if n else "n/a"


def report(s: dict) -> None:
    """Print the evaluate() summary: accuracy, status table, precision, mismatches."""
    print(f"bills evaluated: {s['n']}")
    print("accuracy: " + "  ".join(f"{k} {_pct(*s['accuracy'][k])}" for k in FIELDS))
    print(f"bills with a skipped field: {s['n_skipped']}")
    print(f"{'status':<16}{'total ok':>9}{'wrong':>7}")
    for st in sorted({k[0] for k in s["table"]}):
        print(f"{st:<16}{s['table'][(st, True)]:>9}{s['table'][(st, False)]:>7}")
    print(f"VALID/REPAIRED precision: {_pct(*s['vr_precision'])}")
    print(f"CHECK_REQUIRED with wrong total: {_pct(*s['cr_wrong'])}")
    for m in s["mismatches"]:
        print(m)
    if s["warnings"]:
        print("annotation warnings:")
        for m in s["warnings"]:
            print("  " + m)


def write_csv(rows: list[dict], path: Path) -> None:
    """Per-bill evaluate() rows -> CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["stem", "status"] + [f"{k}_{x}" for k in FIELDS for x in ("pred", "truth", "ok")] \
        + ["total_derived_pred", "total_derived_ok"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["desktop"], required=True)
    args = ap.parse_args()

    rows, s = evaluate(ROOT / "data" / args.set / "annotations", ROOT / "outputs" / "stages")
    report(s)
    write_csv(rows, ROOT / "outputs" / "eval" / f"phase5_{args.set}.csv")


if __name__ == "__main__":
    main()
