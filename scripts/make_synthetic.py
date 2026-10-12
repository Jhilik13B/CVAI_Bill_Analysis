"""Make 60 clean synthetic Indian bills + 60 tampered copies with exact ground-truth JSON.
Pillow + numpy only, seed 11. All names and numbers are made up."""
import json
import random
from collections import Counter
from decimal import ROUND_CEILING, ROUND_HALF_UP, Decimal
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "synthetic"
SEED = 11
N = 60
FONT_SIZE = 22
MARGIN = 30
LINE_H = 32
NOISE_SIGMA = 6.0
MAX_ANGLE = 2.0
FONTS = ("C:/Windows/Fonts/cour.ttf", "C:/Windows/Fonts/consola.ttf")
LAYOUTS = ("gst_invoice", "supermarket_inclusive", "restaurant_service", "thermal")
TAMPERS = ("item_amount", "total", "tax_amount")
P2 = Decimal("0.01")

# made-up word pools (no real shop names)
NAME_A = ("Green", "Star", "City", "Royal", "Sunrise", "Lotus", "Silver", "Fresh", "Golden", "Blue")
NAME_B = ("Mart", "Traders", "Stores", "Kitchen", "Bazaar", "Corner", "Point", "Depot", "Cafe")
GOODS = ("Rice 5kg", "Toor Dal", "Sugar 1kg", "Tea Powder", "Milk 1L", "Bread", "Bath Soap",
         "Sunflower Oil", "Wheat Atta", "Biscuits", "Paneer 200g", "Curd", "Butter", "Salt",
         "Detergent", "Toothpaste", "Coffee", "Poha", "Besan", "Jaggery")
DISHES = ("Masala Dosa", "Idli Vada", "Paneer Tikka", "Veg Biryani", "Sweet Lassi", "Filter Coffee",
          "Butter Naan", "Dal Makhani", "Veg Thali", "Jeera Rice", "Gulab Jamun", "Tomato Soup")


def _q(x: Decimal, unit: Decimal = P2) -> Decimal:
    return x.quantize(unit, rounding=ROUND_HALF_UP)


def _m(x: Decimal) -> str:
    """Money as printed: 1,234.50 (values stay below one lakh)."""
    return f"{x:,.2f}"


def _font() -> ImageFont.ImageFont:
    for f in FONTS:
        if Path(f).exists():
            return ImageFont.truetype(f, FONT_SIZE)
    return ImageFont.load_default()


# ---------- bill content ----------

def _items(rng: random.Random, pool: tuple[str, ...], lo: int, hi: int) -> list[dict]:
    out = []
    for name in rng.sample(pool, rng.randint(3, 8)):
        qty = Decimal(rng.randint(1, 5))
        rate = Decimal(rng.randint(lo * 100, hi * 100)) / 100
        out.append({"name": name, "qty": qty, "rate": rate, "amount": qty * rate})
    return out


def _round_off(pre: Decimal) -> tuple[Decimal, Decimal]:
    total = _q(pre, Decimal("1"))
    return total, total - pre


def _bill(i: int, rng: random.Random) -> dict:
    layout = LAYOUTS[i % 4]
    merchant = f"{rng.choice(NAME_A)} {rng.choice(NAME_B)}"
    date = f"{rng.randint(1, 28):02d}/{rng.randint(1, 12):02d}/202{rng.randint(3, 6)}"
    b = {"layout": layout, "merchant": merchant, "date": date,
         "bill_no": f"{rng.randint(1000, 99999)}", "service_charge": None, "round_off": None,
         "payments": [], "taxes": []}
    if layout == "restaurant_service":
        items = _items(rng, DISHES, 60, 450)
    else:
        items = _items(rng, GOODS, 15, 600)
    if layout == "thermal":  # narrow paper: names are printed cut to 11 characters
        for it in items:
            it["name"] = it["name"][:11].rstrip()
    sub = sum((it["amount"] for it in items), Decimal(0))
    b.update(items=items, subtotal=sub)
    half = Decimal(rng.choice(("2.5", "6", "9")))
    if layout == "gst_invoice":
        tax = _q(sub * half / 100)
        b["taxes"] = [{"label": "CGST", "rate": half, "amount": tax},
                      {"label": "SGST", "rate": half, "amount": tax}]
        b["total"], b["round_off"] = _round_off(sub + 2 * tax)
        b["tax_mode"] = "exclusive"
    elif layout == "restaurant_service":
        svc = _q(sub * 5 / 100)
        half = Decimal("2.5")
        tax = _q((sub + svc) * half / 100)
        b["service_charge"] = svc
        b["taxes"] = [{"label": "CGST", "rate": half, "amount": tax},
                      {"label": "SGST", "rate": half, "amount": tax}]
        b["total"], b["round_off"] = _round_off(sub + svc + 2 * tax)
        b["tax_mode"] = "exclusive"
    else:  # inclusive: tax is already inside the item amounts
        if layout == "supermarket_inclusive":
            b["total"], b["round_off"] = _round_off(sub)
        else:
            b["total"] = sub
        tax = _q(sub * 2 * half / (100 + 2 * half) / 2)
        b["taxes"] = [{"label": "CGST", "rate": half, "amount": tax},
                      {"label": "SGST", "rate": half, "amount": tax}]
        b["tax_mode"] = "inclusive"
        cash = (b["total"] / 100).to_integral_value(rounding=ROUND_CEILING) * 100
        cash = cash if cash > b["total"] else cash + 100
        b["payments"] = [{"label": "CASH", "amount": cash},
                         {"label": "CHANGE", "amount": cash - b["total"]}]
    return b


# ---------- layout -> text lines ----------

def _lines(b: dict, rng: random.Random) -> tuple[list[tuple[str, str | None, int]], int]:
    """[(text, field key of the money string in it, column where that string starts)], width."""
    L: list[tuple[str, str | None, int]] = []
    thermal = b["layout"] == "thermal"
    W = 32 if thermal else 48

    def text(s: str = "", center: bool = False) -> None:
        L.append((s.center(W) if center else s, None, 0))

    def money(label: str, v: Decimal, key: str | None) -> None:
        s = _m(v)
        L.append((f"{label:<{W - len(s)}}{s}", key, W - len(s)))

    text(b["merchant"].upper(), center=True)
    text(f"Shop {rng.randint(1, 99)}, Main Road", center=True)
    if b["layout"] in ("gst_invoice", "restaurant_service"):
        text(f"GSTIN: {rng.randint(10, 37)}ABCDE{rng.randint(1000, 9999)}F1Z{rng.randint(1, 9)}")
        text("TAX INVOICE", center=True)
    text(f"Bill No: {b['bill_no']}")
    text(f"Date: {b['date']}")
    text("-" * W)
    if thermal:
        text(f"{'Item':<12}{'Qty':>3}{'Rate':>8}{'Amt':>9}")
    else:
        text(f"{'Item':<16}{'Qty':>4}{'Rate':>12}{'Amount':>16}")
    text("-" * W)
    for k, it in enumerate(b["items"]):
        amt = _m(it["amount"])
        if thermal:
            row = f"{it['name']:<12}{it['qty']:>3}{_m(it['rate']):>8}"
        else:
            row = f"{it['name']:<16}{it['qty']:>4}{_m(it['rate']):>12}"
        row += f"{amt:>{W - len(row)}}"
        L.append((row, f"items[{k}].amount", W - len(amt)))
    text("-" * W)
    tax_lbl = [f"{t['label']} @{t['rate']}%" for t in b["taxes"]]
    lay = b["layout"]
    if lay == "gst_invoice":
        money("Sub Total", b["subtotal"], None)
        for k, t in enumerate(b["taxes"]):
            money(tax_lbl[k], t["amount"], f"taxes[{k}].amount")
        money("Round Off", b["round_off"], None)
        text("-" * W)
        money("GRAND TOTAL", b["total"], "total")
    elif lay == "restaurant_service":
        money("Sub Total", b["subtotal"], None)
        money("Service Charge @5%", b["service_charge"], None)
        for k, t in enumerate(b["taxes"]):
            money(tax_lbl[k], t["amount"], f"taxes[{k}].amount")
        money("Round Off", b["round_off"], None)
        text("-" * W)
        money("GRAND TOTAL", b["total"], "total")
    elif lay == "supermarket_inclusive":
        money("Sub Total", b["subtotal"], None)
        money("Round Off", b["round_off"], None)
        text("-" * W)
        money("NET AMOUNT", b["total"], "total")
        text("-" * W)
        text("Prices inclusive of GST")
        for k, t in enumerate(b["taxes"]):
            money(f"Incl. {tax_lbl[k]}", t["amount"], f"taxes[{k}].amount")
        for p in b["payments"]:
            money(p["label"].title(), p["amount"], None)
    else:
        money("TOTAL", b["total"], "total")
        for k, t in enumerate(b["taxes"]):
            money(f"{tax_lbl[k]} incl", t["amount"], f"taxes[{k}].amount")
        for p in b["payments"]:
            money(p["label"], p["amount"], None)
    text("-" * W)
    text("Thank you, visit again", center=True)
    return L, W


# ---------- rendering ----------

def _render(lines: list[tuple[str, str | None, int]], width: int,
            font: ImageFont.ImageFont) -> tuple[Image.Image, dict[str, tuple]]:
    """Canvas with the text, and the pixel box of each keyed money string."""
    cw = font.getlength("0")
    img = Image.new("L", (int(2 * MARGIN + width * cw), 2 * MARGIN + LINE_H * len(lines)), 255)
    d = ImageDraw.Draw(img)
    boxes = {}
    for n, (s, key, col) in enumerate(lines):
        y = MARGIN + n * LINE_H
        d.text((MARGIN, y), s, font=font, fill=20)
        if key:
            x = MARGIN + col * cw
            boxes[key] = (x, y, len(s) - col)
    return img, boxes


def _distort(img: Image.Image, angle: float, nrng: np.random.Generator) -> Image.Image:
    img = img.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor=255)
    a = np.asarray(img, dtype=np.float32) + nrng.normal(0, NOISE_SIGMA, (img.height, img.width))
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def _one_digit(s: str, rng: random.Random) -> str:
    """Change one digit of the integer part (never a leading 0)."""
    pos = [j for j, c in enumerate(s.split(".")[0]) if c.isdigit()]
    j = rng.choice(pos)
    digits = [c for c in "0123456789" if c != s[j] and not (j == pos[0] and c == "0")]
    return s[:j] + rng.choice(digits) + s[j + 1:]


def _tamper(canvas: Image.Image, boxes: dict, b: dict, ttype: str, font: ImageFont.ImageFont,
            rng: random.Random) -> tuple[Image.Image, dict]:
    if ttype == "item_amount":
        key = f"items[{rng.randrange(len(b['items']))}].amount"
    elif ttype == "tax_amount":
        key = f"taxes[{rng.randrange(len(b['taxes']))}].amount"
    else:
        key = "total"
    old = _m(_get(b, key))
    new = _one_digit(old, rng)
    x, y, n = boxes[key]
    img = canvas.copy()
    d = ImageDraw.Draw(img)
    d.rectangle((x - 1, y, x + n * font.getlength("0") + 1, y + LINE_H - 4), fill=255)
    d.text((x, y), new, font=font, fill=20)
    return img, {"tamper_type": ttype, "changed_field": key, "old": old, "new": new}


def _get(b: dict, key: str) -> Decimal:
    if key == "total":
        return b["total"]
    lst, rest = key.split("[")
    return b[lst][int(rest.split("]")[0])]["amount"]


def _json(b: dict) -> dict:
    """Ground truth with every money/qty value as a string."""
    def conv(v):
        if isinstance(v, Decimal):
            return str(v)
        if isinstance(v, list):
            return [conv(x) for x in v]
        if isinstance(v, dict):
            return {k: conv(x) for k, x in v.items()}
        return v
    return conv(b)


def main() -> None:
    rng = random.Random(SEED)
    nrng = np.random.default_rng(SEED)
    font = _font()
    clean, tamp = OUT / "clean", OUT / "tampered"
    clean.mkdir(parents=True, exist_ok=True)
    tamp.mkdir(parents=True, exist_ok=True)
    types: Counter = Counter()
    for i in range(N):
        b = _bill(i, rng)
        lines, width = _lines(b, rng)
        canvas, boxes = _render(lines, width, font)
        angle = rng.uniform(-MAX_ANGLE, MAX_ANGLE)
        _distort(canvas, angle, nrng).save(clean / f"syn_c_{i:03d}.png")
        gt = _json(b)
        (clean / f"syn_c_{i:03d}.json").write_text(json.dumps(gt, indent=1), encoding="utf-8")
        ttype = TAMPERS[i % 3]
        timg, info = _tamper(canvas, boxes, b, ttype, font, rng)
        _distort(timg, angle, nrng).save(tamp / f"syn_t_{i:03d}.png")
        (tamp / f"syn_t_{i:03d}.json").write_text(
            json.dumps({**gt, **info, "pair": f"syn_c_{i:03d}"}, indent=1), encoding="utf-8")
        types[ttype] += 1
    n_c = len(list(clean.glob("*.png")))
    n_t = len(list(tamp.glob("*.png")))
    print(f"clean: {n_c}  tampered: {n_t}  by type: {dict(types)}")


if __name__ == "__main__":
    main()
