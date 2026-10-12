"""Date re-OCR for bills with date = null: top 45% of 01_orig.jpg, upscaled 2x, dates only."""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.keywords import match_labels  # noqa: E402
from src.normalize import parse_date  # noqa: E402
from src.ocr import ocr_image  # noqa: E402

TOP_SHARE = 0.45
SCALE = 2


def reocr_date(dump: Path) -> str | None:
    """OCR the top of the bill and return the first date found (DATE-labelled lines first)."""
    img = cv2.imdecode(np.fromfile(str(dump / "01_orig.jpg"), dtype=np.uint8), cv2.IMREAD_COLOR)
    top = img[: int(img.shape[0] * TOP_SHARE)]
    top = cv2.resize(top, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_CUBIC)
    lines = ocr_image(top, "date_reocr")
    hits = []
    for i, ln in enumerate(lines):
        labeled = any(l == "DATE" for l, _ in match_labels(ln["text"]))
        for d in parse_date(ln["text"]):
            hits.append((not labeled, d["ambiguous"], i, d["iso"]))
    date = min(hits)[3] if hits else None
    (dump / "02b_date_ocr.json").write_text(
        json.dumps({"date": date, "lines": [{"text": l["text"], "conf": l["conf"]} for l in lines]},
                   indent=1, ensure_ascii=False), encoding="utf-8")
    return date


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["desktop", "cord"], required=True)
    args = ap.parse_args()

    stems = sorted(p.stem for p in (ROOT / "data" / args.set / "images").iterdir() if p.is_file())
    tried = found = 0
    for stem in stems:
        dump = ROOT / "outputs" / "stages" / stem
        ff = dump / "04_fields.json"
        if not ff.exists():
            continue
        fields = json.loads(ff.read_text(encoding="utf-8"))
        if fields["date"]["value"] is not None:
            continue
        tried += 1
        date = reocr_date(dump)
        if date:
            found += 1
            fields["date"] = {**fields["date"], "value": date, "reason": None, "date_source": "reocr"}
            ff.write_text(json.dumps(fields, indent=1, ensure_ascii=False, default=str),
                          encoding="utf-8")
    print(f"date re-OCR recovered: {found}/{tried}")


if __name__ == "__main__":
    main()
