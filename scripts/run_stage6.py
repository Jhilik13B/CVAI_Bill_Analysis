"""Run stage 6 (validation) on existing 04_fields.json; write 05_validation.json."""
import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from run_stage5 import _cord_truth, _pct  # noqa: E402
from run_stage12 import IMG_EXT, _img_dir  # noqa: E402
from src.validate import validate  # noqa: E402


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
        if not (dump / "04_fields.json").exists():
            continue
        res[stem] = validate(json.loads((dump / "04_fields.json").read_text(encoding="utf-8")), dump)

    n = len(res)
    print(f"[{args.set or args.img_dir.name}] images: {n}")
    print(f"status: {dict(Counter(v['status'] for v in res.values()))}")
    print(f"tax_mode: {dict(Counter(v['tax_mode'] for v in res.values()))}")
    print(f"repairs: {dict(Counter(r['reason'].split(' (')[0] for v in res.values() for r in v['repairs']))}"
          f"  bills repaired: {sum(bool(v['repairs']) for v in res.values())}")
    print(f"avg confidence: {sum(v['confidence'] for v in res.values()) / n:.3f}" if n else "")
    print(f"needs_review: {sum(v['needs_review'] for v in res.values())}/{n}  "
          f"reasons: {dict(Counter(x for v in res.values() for x in v['review_reasons']))}")

    if args.set:
        _snapshot(args.set, res)
    if args.set != "cord":
        return
    tuned = {k: v for k, v in res.items() if int(k) < 100}
    held = {k: v for k, v in res.items() if 100 <= int(k) < 150}
    mt, mh = _metrics(tuned), _metrics(held)
    print(f"{'metric':<32}{'tuned 0000-0099':<20}held-out 0100-0149")
    for k in mt:
        print(f"{k:<32}{mt[k]:<20}{mh[k]}")
    _save_metrics({"tuned": tuned, "held_out": held})


def _snapshot(name: str, res: dict) -> None:
    """Write status_snapshot_<set>.csv (old one moved to .prev); print stems that changed."""
    out = ROOT / "outputs" / "eval" / f"status_snapshot_{name}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    prev = out.with_name(out.name + ".prev")
    old: dict[str, tuple[str, str]] = {}
    if out.exists():
        out.replace(prev)
        with prev.open(encoding="utf-8") as f:
            old = {r["stem"]: (r["status"], r["total"]) for r in csv.DictReader(f)}
    new = {k: (v["status"], "" if v["total"] is None else str(v["total"])) for k, v in res.items()}
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["stem", "status", "total"])
        w.writerows([k, *v] for k, v in new.items())
    if old:
        changed = [k for k in new if old.get(k) != new[k]]
        print(f"changed status/total ({len(changed)}):")
        for k in changed:
            o = old.get(k, ("-", "-"))
            print(f"  {k}: {o[0]} {o[1]} -> {new[k][0]} {new[k][1]}")


STATUSES = ("VALID", "REPAIRED", "CHECK_REQUIRED", "INSUFFICIENT_EVIDENCE")


def _metric_counts(res: dict) -> dict[str, int]:
    """Raw stage-6 CORD counts over a group of stems."""
    ok = tot = vr_ok = vr_n = cr_wrong = cr_n = 0
    for stem, v in res.items():
        gt = _cord_truth(stem)
        if gt is None or gt["total"] is None:
            continue
        tot += 1
        right = v["total"] is not None and v["total"] == gt["total"]
        ok += right
        if v["status"] in ("VALID", "REPAIRED"):
            vr_n += 1
            vr_ok += right
        elif v["status"] == "CHECK_REQUIRED":
            cr_n += 1
            cr_wrong += not right
    st = Counter(v["status"] for v in res.values())
    return {"n": len(res), "total_ok": ok, "total_n": tot, "vr_ok": vr_ok, "vr_n": vr_n,
            "cr_wrong": cr_wrong, "cr_n": cr_n, **{s: st[s] for s in STATUSES}}


def _metrics(res: dict) -> dict:
    """Stage-6 CORD metrics over a group of stems."""
    c = _metric_counts(res)
    return {"status": str(dict(Counter(v["status"] for v in res.values()))),
            "validated total == truth": _pct(c["total_ok"], c["total_n"]),
            "VALID/REPAIRED precision": _pct(c["vr_ok"], c["vr_n"]),
            "CHECK_REQUIRED actually wrong": _pct(c["cr_wrong"], c["cr_n"])}


def _save_metrics(groups: dict[str, dict]) -> None:
    """outputs/eval/stage6_cord_metrics.csv: one row of raw counts per group."""
    out = ROOT / "outputs" / "eval" / "stage6_cord_metrics.csv"
    rows = [{"group": g, **_metric_counts(r)} for g, r in groups.items()]
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

if __name__ == "__main__":
    main()
