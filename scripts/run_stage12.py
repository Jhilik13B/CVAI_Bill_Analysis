"""Run stages 1-2 over an image set and write outputs/eval/phase1_<set>.csv."""
import argparse
import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def _img_dir(args) -> Path:
    """--img-dir, else data/<set>/images."""
    return args.img_dir or ROOT / "data" / args.set / "images"


def _mean(vals: list) -> float | None:
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def _fmt(v: float | None) -> str:
    return "n/a" if v is None else f"{v:.3f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--set", choices=["desktop", "cord"])
    g.add_argument("--img-dir", type=Path, help="any image folder (e.g. data/synthetic/clean)")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    from src.pipeline import run_stage12  # heavy (OCR) import only when run

    img_dir = _img_dir(args)
    files = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXT)
    if args.limit:
        files = files[:args.limit]
    # skip stems that already have an OCR dump
    files = [p for p in files if not (ROOT / "outputs" / "stages" / p.stem / "02_ocr.json").exists()]

    t0 = time.perf_counter()
    rows = [run_stage12(p, ROOT / "outputs" / "stages") for p in files]
    total = time.perf_counter() - t0

    eval_dir = ROOT / "outputs" / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    if rows:
        with (eval_dir / f"phase1_{args.set or args.img_dir.name}.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    def beat(r: dict) -> bool:
        m, o, e = (r["numeric_conf_merged"], r["numeric_conf_orig"], r["numeric_conf_enh"])
        return m is not None and m > (o if o is not None else -1) and m > (e if e is not None else -1)

    print(f"images processed: {len(rows)}")
    print(f"mean merged conf: {_fmt(_mean([r['mean_conf_merged'] for r in rows]))}")
    print("mean numeric_conf orig/enh/merged: "
          + " / ".join(_fmt(_mean([r[f'numeric_conf_{k}'] for r in rows]))
                       for k in ("orig", "enh", "merged")))
    print(f"LOW_QUALITY: {sum(r['tag'] == 'LOW_QUALITY' for r in rows)}")
    print(f"merged beat both on numeric_conf: {sum(beat(r) for r in rows)}")
    print(f"total seconds: {total:.1f}")
    scored = sorted((r for r in rows if r["numeric_conf_merged"] is not None),
                    key=lambda r: r["numeric_conf_merged"])
    print("lowest merged numeric_conf: "
          + ", ".join(f"{r['image']}={r['numeric_conf_merged']:.3f}" for r in scored[:5]))


if __name__ == "__main__":
    main()
