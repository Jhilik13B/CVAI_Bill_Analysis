"""Run stages 3-4 on existing 02_ocr.json dumps; write 03_rows.json / 03_rows.txt."""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from run_stage12 import IMG_EXT, _img_dir  # noqa: E402
from src.rows import build_rows  # noqa: E402

TOTAL_LABELS = {"TOTAL", "GRAND_TOTAL"}


def _txt_line(r: dict) -> str:
    labels = ",".join(l for l, _ in r["labels"])
    amts = " ".join(str(a["value"]) for a in r["amounts"])
    return f"{r['idx']:03d} | " + " | ".join(r["cells"]) + f" || LABELS: {labels} AMT: {amts}"


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--set", choices=["desktop", "cord"])
    g.add_argument("--img-dir", type=Path, help="any image folder (e.g. data/synthetic/clean)")
    args = ap.parse_args()

    stems = sorted(p.stem for p in _img_dir(args).iterdir() if p.suffix.lower() in IMG_EXT)
    t0 = time.perf_counter()
    n_rows, n_amts, no_total = [], [], []
    for stem in stems:
        dump = ROOT / "outputs" / "stages" / stem
        src = dump / "02_ocr.json"
        if not src.exists():
            continue
        rows = build_rows(json.loads(src.read_text(encoding="utf-8"))["merged"])
        (dump / "03_rows.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False, default=str),
                                           encoding="utf-8")
        (dump / "03_rows.txt").write_text("\n".join(_txt_line(r) for r in rows) + "\n",
                                          encoding="utf-8")
        n_rows.append(len(rows))
        n_amts.append(sum(len(r["amounts"]) for r in rows))
        if not any(l in TOTAL_LABELS for r in rows for l, _ in r["labels"]):
            no_total.append(stem)

    n = len(n_rows)
    print(f"images processed: {n}")
    print(f"mean rows/image: {sum(n_rows) / n:.1f}  mean amounts/image: {sum(n_amts) / n:.1f}"
          if n else "mean rows/image: n/a")
    print(f"no TOTAL/GRAND_TOTAL ({len(no_total)}): {', '.join(no_total) or '-'}")
    print(f"seconds: {time.perf_counter() - t0:.1f}")


if __name__ == "__main__":
    main()
