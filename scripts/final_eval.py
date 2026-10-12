"""Final evaluation on fresh annotations (no tuning); write outputs/eval/final_fresh.csv."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate import evaluate, report, write_csv  # noqa: E402

BANNER = "FRESH TEST SET (no tuning)"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann-dir", type=Path, required=True)
    args = ap.parse_args()

    ann_dir = args.ann_dir if args.ann_dir.is_absolute() else ROOT / args.ann_dir
    files = sorted(ann_dir.glob("*.json")) if ann_dir.is_dir() else []
    if not files:
        print(f"{BANNER}: no annotations in {ann_dir.relative_to(ROOT) if ann_dir.is_relative_to(ROOT) else ann_dir}")
        return

    # evaluate() skips annotations with an empty total or without a stage-6 dump
    rows, s = evaluate(ann_dir, ROOT / "outputs" / "stages")
    print(f"=== {BANNER} ===")
    print(f"annotations: {len(files)}  skipped (empty total or no dump): {len(files) - s['n']}")
    report(s)
    out = ROOT / "outputs" / "eval" / "final_fresh.csv"
    write_csv(rows, out)
    print(f"saved: {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
