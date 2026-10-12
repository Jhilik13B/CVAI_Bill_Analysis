"""Write 20 blank annotation templates for random desktop bills (seed 7)."""
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXCLUDE = {"bill_01", "bill_10", "bill_37", "bill_46", "bill_47", "bill_48"}
N, SEED = 20, 7
TEMPLATE = {"merchant": "", "date": "YYYY-MM-DD", "total": "", "tax_total": "", "item_count": None,
            "notes": ""}


def main() -> None:
    stems = sorted(p.stem for p in (ROOT / "data" / "desktop" / "images").iterdir()
                   if p.is_file() and p.stem not in EXCLUDE)
    picked = sorted(random.Random(SEED).sample(stems, N))
    out = ROOT / "data" / "desktop" / "annotations"
    out.mkdir(parents=True, exist_ok=True)
    for stem in picked:
        f = out / f"{stem}.json"
        if not f.exists():  # never overwrite a filled annotation
            f.write_text(json.dumps(TEMPLATE, indent=1), encoding="utf-8")
        print(stem)


if __name__ == "__main__":
    main()
