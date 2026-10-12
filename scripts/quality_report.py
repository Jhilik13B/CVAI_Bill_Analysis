"""Diagnostic: quality.json stats per image set. Changes no thresholds."""
import csv
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.preprocess import BLUR_MIN, BRIGHT_MIN, tag  # noqa: E402

STAGES = ROOT / "outputs" / "stages"


def _pct(vals: list[float]) -> str:
    if not vals:
        return "n/a"
    p = np.percentile(vals, [10, 50, 90])
    return " / ".join(f"{v:.1f}" for v in p)


def _mean(vals: list[float]) -> str:
    return f"{sum(vals) / len(vals):.3f}" if vals else "n/a"


def report(name: str) -> None:
    stems = {p.stem for p in (ROOT / "data" / name / "images").iterdir() if p.is_file()}
    qs = {}
    for s in sorted(stems):
        f = STAGES / s / "quality.json"
        if f.exists():
            qs[s] = json.loads(f.read_text(encoding="utf-8"))
    blur = [q["blur"] for q in qs.values()]
    bright = [q["brightness"] for q in qs.values()]

    conf: dict[str, list[float]] = {"LOW_QUALITY": [], "OK": []}
    csv_path = ROOT / "outputs" / "eval" / f"phase1_{name}.csv"
    if csv_path.exists():
        with csv_path.open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["numeric_conf_merged"] and r["tag"] in conf:
                    conf[r["tag"]].append(float(r["numeric_conf_merged"]))

    print(f"[{name}] images: {len(qs)}  LOW_QUALITY: "
          f"{sum(q['tag'] == 'LOW_QUALITY' for q in qs.values())}")
    print(f"  blur<{BLUR_MIN:.0f}: {sum(b < BLUR_MIN for b in blur)}  "
          f"brightness<{BRIGHT_MIN:.0f}: {sum(b < BRIGHT_MIN for b in bright)}")
    print(f"  blur p10/p50/p90: {_pct(blur)}")
    print(f"  brightness p10/p50/p90: {_pct(bright)}")
    print(f"  mean numeric_conf LOW_QUALITY: {_mean(conf['LOW_QUALITY'])} "
          f"(n={len(conf['LOW_QUALITY'])})  OK: {_mean(conf['OK'])} (n={len(conf['OK'])})")


def retag() -> None:
    """Rewrite 'tag' in every quality.json from stored blur/brightness (no OCR)."""
    for f in sorted(STAGES.glob("*/quality.json")):
        q = json.loads(f.read_text(encoding="utf-8"))
        q["tag"] = tag(q["blur"], q["brightness"])
        f.write_text(json.dumps(q, indent=2), encoding="utf-8")


def low_counts(name: str) -> None:
    stems = {p.stem for p in (ROOT / "data" / name / "images").iterdir() if p.is_file()}
    tags = [json.loads(f.read_text(encoding="utf-8"))["tag"]
            for s in stems if (f := STAGES / s / "quality.json").exists()]
    print(f"[{name}] retagged: {len(tags)}  LOW_QUALITY: {tags.count('LOW_QUALITY')}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--retag", action="store_true")
    args = ap.parse_args()
    if args.retag:
        retag()
        for name in ("desktop", "cord"):
            low_counts(name)
    else:
        for name in ("desktop", "cord"):
            report(name)
