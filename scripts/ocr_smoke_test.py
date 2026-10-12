"""Run PaddleOCR on one image and print a short summary."""
import sys
from pathlib import Path

from paddleocr import PaddleOCR

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIRS = [ROOT / "data" / "desktop" / "images", ROOT / "data" / "cord" / "images"]


def pick_default() -> Path:
    for d in DEFAULT_DIRS:
        files = sorted(p for p in d.glob("*") if p.is_file())
        if files:
            return files[0]
    sys.exit("no image found in data/desktop/images or data/cord/images")


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else pick_default()
    ocr = PaddleOCR(use_angle_cls=True, lang="en", use_gpu=False, show_log=False)
    result = ocr.ocr(str(path), cls=True)
    # result is a list per page; page is None when no text is found
    lines = result[0] if result and result[0] else []
    confs = [float(line[1][1]) for line in lines]
    mean_conf = sum(confs) / len(confs) if confs else 0.0
    print(f"image: {path.name}")
    print(f"lines: {len(lines)}")
    print(f"mean_conf: {mean_conf:.3f}")
    for line in lines[:5]:
        print(f"  {line[1][0]}")


if __name__ == "__main__":
    main()
