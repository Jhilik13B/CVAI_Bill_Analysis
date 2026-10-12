"""Download the first 150 CORD-v2 train samples into data/cord/."""
from pathlib import Path

from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
IMG_DIR = ROOT / "data" / "cord" / "images"
LBL_DIR = ROOT / "data" / "cord" / "labels"


def main() -> None:
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    LBL_DIR.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("naver-clova-ix/cord-v2", split="train[:150]")
    saved = 0
    for i, sample in enumerate(ds):
        img_path = IMG_DIR / f"{i:04d}.jpg"
        lbl_path = LBL_DIR / f"{i:04d}.json"
        if img_path.exists() and lbl_path.exists():
            continue
        if not img_path.exists():
            sample["image"].convert("RGB").save(img_path, "JPEG")
        if not lbl_path.exists():
            # ground_truth is already a JSON string; store it verbatim
            lbl_path.write_text(sample["ground_truth"], encoding="utf-8")
        saved += 1
    print(saved)


if __name__ == "__main__":
    main()
