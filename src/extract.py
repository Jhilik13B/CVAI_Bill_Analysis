"""Stage 5: field extraction with candidate scoring. No validation or repair (Phase 4)."""
import re
from decimal import Decimal
from statistics import median

from rapidfuzz import fuzz

from src.keywords import LABELS, MIN_SCORE, _tokens
from src.normalize import (GST_NOTE_RE, HSN_RE, label_text, parse_amount, parse_date,
                           parse_percent, parse_qty)

TOL = Decimal("0.05")
STRONG_FOR_DECIMAL = 3

TAX_LABELS = {"CGST", "SGST", "IGST", "UTGST", "GST", "VAT", "CESS", "TAX_TOTAL"}
TOTALS_START = {"SUBTOTAL", "TOTAL", "GRAND_TOTAL", "TAX_TOTAL"}
TABLE_LABELS = {"TAXABLE", "CGST", "SGST", "IGST"}
PAY_LABELS = {"CASH", "CHANGE"}
# rows with these labels are never items
META_LABELS = {"BILL_NO", "GSTIN", "DATE", "CASH", "CHANGE", "TOTAL", "GRAND_TOTAL", "SUBTOTAL",
               "TAX_TOTAL", "TAXABLE", "ROUNDOFF", "SERVICE_CHARGE"} | TAX_LABELS

MERCHANT_BAD = re.compile(r"\b(ROAD|RD|STREET|ST|NEAR|PHONE|PH|TEL|MOB|GSTIN|FSSAI)\b")
MERCHANT_ROLE = re.compile(r"\b(ORIGINAL|RECIPIENT|DUPLICATE|MANAGER|PATIENT|CASHIER|INVOICE|BILL)\b")
MERCHANT_PREFIX = re.compile(r"^\W*(SOLD\s*BY|COLLECTED\s*BY|BILLED\s*BY|NAME)\b\s*[:\-]?\s*",
                             re.IGNORECASE)
PINCODE = re.compile(r"(?<!\d)\d{6}(?!\d)")
# Indian states / large cities: a row naming one is usually the address
PLACE_WORDS = {
    "ANDHRA", "PRADESH", "ARUNACHAL", "ASSAM", "BIHAR", "CHHATTISGARH", "GOA", "GUJARAT",
    "HARYANA", "HIMACHAL", "JHARKHAND", "KARNATAKA", "KERALA", "MADHYA", "MAHARASHTRA", "MANIPUR",
    "MEGHALAYA", "MIZORAM", "NAGALAND", "ODISHA", "PUNJAB", "RAJASTHAN", "SIKKIM", "TAMIL", "NADU",
    "TELANGANA", "TRIPURA", "UTTAR", "UTTARAKHAND", "BENGAL", "DELHI", "MUMBAI", "PUNE", "KOLKATA",
    "CHENNAI", "BENGALURU", "BANGALORE", "HYDERABAD", "AHMEDABAD", "SURAT", "JAIPUR", "LUCKNOW",
    "KANPUR", "NAGPUR", "INDORE", "THANE", "BHOPAL", "PATNA", "VADODARA", "NOIDA", "GURGAON",
    "GURUGRAM", "NASHIK", "INDIA"}
BUSINESS_WORDS = {"PVT", "LTD", "LIMITED", "STORE", "STORES", "MEDICAL", "HOSPITAL", "SWEETS",
                  "ELECTRONICS", "HOTEL", "KENDRA", "TRADERS", "ENTERPRISES", "MARKET", "CAFE",
                  "PHARMACY", "CLINIC", "SUPERMARKET"}
MERCHANT_ROWS = 8
MERGE_MIN = 2.5  # both adjacent rows must score at least this to be joined
BOX_TOP = 0.25  # merchant boxes start in the top quarter of the bill
BOX_MIN = 3.0  # below this best box score the row method is used
BOX_MERGE_MIN = 2.5  # both adjacent boxes must score at least this to be joined
BOX_H_SIMILAR = 0.25  # joined boxes differ in height by at most 25%
PHONE_RE = re.compile(r"(?:\d[\s\-]?){10,}")
# GST slab rows: '0%: 0.00', '18% 45.00' (not items, not totals)
SLAB_RE = re.compile(r"^\W*(\d{1,2}(?:\.\d+)?)\s*%")
# rows with these words are never the total, even with a TOTAL label
NOT_TOTAL_RE = re.compile(r"MRP|SAVING|YOU\s*SAVE|DISCOUNT|POINTS")
NET_RE = re.compile(r"NET\s*AMOUNT|NET\s*PAYABLE|GRAND\s*TOTAL")
TOTAL_TYPES = {"total", "grand_total", "net_total"}
TAX_WORDS = re.compile(r"\b(CGST|SGST|IGST|UTGST|GST|VAT|CESS|HSN|SAC)\b", re.IGNORECASE)
PLAIN_TAX = re.compile(r"\bTAX(?!ABLE|\s*INVOICE)")  # 'TAX 10%' rows with no GST/VAT word
AMOUNT_LIKE = re.compile(r"\b(AMOUNT|AMT|PAYABLE|DUE)\b")
TAX_RE = re.compile(r"(CGST|SGST|IGST|UTGST|GST|VAT|CESS)\W{0,3}(?:@\s*)?(\d{1,2}(?:\.\d{1,2})?)"
                    r"\s*%?\s*[\[\s]*(\d[\d,]*\.\d{2})")
# per-rate row: 'GST/Tax @ 18.00 | 339.00 | 51.71' -> (rate, base, tax)
RATE_ROW_RE = re.compile(r"(?<![A-Z])(GST(?:/TAX)?|VAT|TAX)\W{0,3}(?:@\s*)?(\d{1,2}(?:\.\d{1,2})?)\s*%?"
                         r"[\s|:\[\]]*(\d[\d,]*\.\d{2})[\s|:\[\]]+(\d[\d,]*\.\d{2})")
# tax line with a base: 'CGST @ 2.50% ON 150.00 3.75' (base and tax in either order)
BASE_TAX_RE = re.compile(r"(?<![A-Z])(CGST|SGST|IGST|UTGST|GST|VAT)\W{0,3}(?:@\s*)?(\d{1,2}(?:\.\d{1,2})?)"
                         r"\s*%\s*(?:ON|OF)[\s|:\[\]]*(\d[\d,]*\.\d{2})[\s|:\[\]]+(\d[\d,]*\.\d{2})")
BASE_TOL = Decimal("0.10")  # tax = base*r/100 within this
MAX_TAX_SHARE = Decimal("0.35")  # a tax sum above this share of the items sum is rejected
MID_CAP = re.compile(r"[a-z][A-Z]")  # 'GreBsn': capital inside a word
PAIR_LABELS = {"CGST", "SGST", "IGST", "UTGST", "VAT", "CESS"}
GST_RATES = {Decimal(x) for x in ("0", "0.25", "3", "5", "12", "18", "28")}
GST_FAMILY = {"GST", "CGST", "SGST", "IGST", "UTGST"}  # only these are gated to GST slabs
MAX_OTHER_RATE = Decimal("30")  # TAX/VAT/CESS rates accepted from 0 to 30 %
FLAG_LETTER = re.compile(r"[A-Z]")  # tax-flag letter printed after a price
CODE_RE = re.compile(r"\d{10,14}")  # UPC / SKU column
HALF_RATES = {r / 2 for r in GST_RATES}  # CGST/SGST/UTGST carry half the GST rate
GSTIN_RE = re.compile(r"[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z][0-9A-Z]Z[0-9A-Z]")
GSTIN_DIGIT_POS = (0, 1, 7, 8, 9, 10)
GSTIN_LETTER_POS = (2, 3, 4, 5, 6)
TO_DIGIT = {"O": "0", "I": "1", "S": "5", "B": "8"}
TO_LETTER = {v: k for k, v in TO_DIGIT.items()}
# tax text inside an item name: 'CGST@2.5%', '[SGST 9% 7.50]', glued 'PaneerCGST@2.5%'
NAME_TAX_RE = re.compile(r"\[?\s*(?:CGST|SGST|IGST|UTGST|GST)\s*@?\s*(\d{1,2}(?:\.\d{1,2})?\s*%)?"
                         r"\s*(\d[\d,]*\.\d{2})?\s*\]?", re.IGNORECASE)
# HSN/SAC code text inside an item name ('HSN 040690', 'SAC:'); the digits may already be gone
NAME_HSN_RE = re.compile(r"(?<![A-Z])(?:HSN|SAC)(?:\s*(?:CODE)?\s*[:#\-]?\s*\d{4,8}|\b)",
                         re.IGNORECASE)
# words that make a row 'label garbage' (OCR-mangled totals/tax/payment words)
GARBAGE_KWS = sorted({w for lab in ("TOTAL", "GRAND_TOTAL", "SUBTOTAL", "TAX_TOTAL", "TAXABLE",
                                     "CGST", "SGST", "IGST", "GST", "VAT", "ROUNDOFF", "CASH",
                                     "CHANGE", "DISCOUNT")
                      for kw in LABELS[lab] for w in kw.split() if len(w) >= 3})
GARBAGE_MIN = 80
BILL_NO_RE = re.compile(r"(?:BILL|INVOICE|INV|RECEIPT)\s*(?:NO|NUMBER|#)\.?\s*[:#.\-]?\s*"
                        r"([A-Z0-9][A-Z0-9/\-]*)", re.IGNORECASE)
DOT_THOUSANDS = re.compile(r"\d{1,3}(\.\d{3})+")
BILL_TYPES = {
    "tax_invoice": [r"TAX\s*INVOICE"],
    "restaurant": [r"\bTABLE\b", r"\bWAITER\b", r"\bKOT\b", r"SERVICE\s*CHARGE", r"\bDINE"],
    "supermarket": [r"\bMRP\b", r"\bHSN\b", r"\bITEMS\b.*\bQTY\b"],
    "thermal_receipt": [r"THANK\s*YOU", r"\bCASH\b", r"\bCHANGE\b"],
}


# ---------- row helpers ----------

def _labels(r: dict) -> set[str]:
    return {l for l, _ in r["labels"]}


def _header_hits(text: str) -> int:
    """Number of HEADER keywords present in the row (match_labels keeps one per label)."""
    toks = _tokens(text)
    return sum(any(fuzz.ratio(kw, t) >= MIN_SCORE for t in toks) for kw in LABELS["HEADER"])


def _is_percent(text: str, raw: str) -> bool:
    """Amount directly after '@' or directly before '%'."""
    core = raw.strip()
    for m in re.finditer(re.escape(core), text):
        if text[:m.start()].rstrip().endswith("@") or text[m.end():].lstrip().startswith("%"):
            return True
    return False


def _two_dec(a: dict) -> bool:
    return a["value"].as_tuple().exponent == -2


def _int_value(a: dict) -> Decimal:
    """Integer style: '75.000' is 75 thousand, not 75."""
    core = re.sub(r"^\D+|\D+$", "", a["raw"])
    if DOT_THOUSANDS.fullmatch(core):
        v = Decimal(core.replace(".", ""))
        return -v if a["value"] < 0 else v
    return a["value"]


def _money(r: dict, style: str) -> list[dict]:
    """Money amounts of a row (percentages dropped), values already style-adjusted."""
    out = []
    for a in r["amounts"]:
        if _is_percent(r["text"], a["raw"]):
            continue
        if a["strength"] == "strong" and (style == "decimal" or not DOT_THOUSANDS.fullmatch(
                re.sub(r"^\D+|\D+$", "", a["raw"]))):
            out.append({**a})
        elif style == "integer":
            v = _int_value(a)
            if abs(v) >= 100 or "," in a["raw"] or v != a["value"]:
                out.append({**a, "value": v})
    return out


def _name(r: dict, drop_tax_words: bool = False) -> str:
    """Text tokens of a row (amounts, bare numbers and qty markers removed)."""
    words = []
    is_amount = False
    for t in r["tokens"]:
        s = t["text"]
        after_amount, is_amount = is_amount, parse_amount(s) is not None
        if after_amount and FLAG_LETTER.fullmatch(s):
            continue  # tax-flag letter after a price ('1.24 T')
        if is_amount or not any(c.isalpha() for c in s):
            continue
        if re.fullmatch(r"[xX@*]|\d+[xX]|[xX]\d+|RS\.?|₹", s, re.IGNORECASE):
            continue
        words.append(s)
    name = " ".join(words)
    if drop_tax_words:
        name = TAX_WORDS.sub("", name).strip(" []-:%@")
    return name if sum(c.isalpha() for c in name) >= 2 else ""


def _clean_name(name: str) -> tuple[str, list[Decimal]]:
    """Drop CGST/SGST/GST and HSN/SAC text and leading junk from an item name;
    glued tax amounts are returned."""
    name = NAME_HSN_RE.sub(" ", name)
    taxes = [a.value for m in NAME_TAX_RE.finditer(name) if m.group(2)
             if (a := parse_amount(m.group(2)))]
    name = GST_NOTE_RE.sub(" ", NAME_TAX_RE.sub(" ", name))
    name = re.sub(r"\s+", " ", name).strip().lstrip(")(.[ ").strip()
    return (name if sum(c.isalpha() for c in name) >= 2 else ""), taxes


def _slab_row(r: dict) -> bool:
    """'5%: 12.50' style GST slab row (a slab rate followed by an amount)."""
    m = SLAB_RE.match(r["text"])
    return bool(m and Decimal(m.group(1)) in GST_RATES and r["amounts"])


def _garbage_label_row(r: dict) -> bool:
    """Every 3+ letter word fuzzy-matches a totals/tax/payment word (e.g. 'Tetal')."""
    words = [w for w in _tokens(_name(r)) if sum(c.isalpha() for c in w) >= 3]
    return bool(words) and all(
        any(fuzz.ratio(kw, w) >= GARBAGE_MIN for kw in GARBAGE_KWS) for w in words)


def _field(value, conf: float, rows: list[int], alts: list | None = None, **extra) -> dict:
    return {"value": value, "confidence": round(max(0.0, min(1.0, conf)), 3),
            "rows": rows, "alternatives": alts or [], **extra}


def _load(rows: list[dict]) -> list[dict]:
    """Accept rows from 03_rows.json (values as strings) or in-memory rows."""
    out = []
    for r in rows:
        amts = [{**a, "value": Decimal(str(a["value"]))} for a in r.get("amounts", [])]
        toks = r.get("tokens") or [{"text": t} for t in r["text"].split()]
        out.append({**r, "amounts": amts, "tokens": toks,
                    "labels": [tuple(l) for l in r.get("labels", [])]})
    return out


# ---------- fields ----------

def _money_style(rows: list[dict]) -> dict:
    n = sum(_two_dec(a) and a["strength"] == "strong" and not _is_percent(r["text"], a["raw"])
            for r in rows for a in r["amounts"])
    style = "decimal" if n >= STRONG_FOR_DECIMAL else "integer"
    return _field(style, 1.0 if n >= 5 or n == 0 else 0.6, [], n_strong=n)


def _regions(rows: list[dict]) -> tuple[int | None, int | None]:
    header = next((r["idx"] for r in rows if _header_hits(r["text"]) >= 2), None)
    start = -1 if header is None else header
    totals = next((r["idx"] for r in rows if r["idx"] > start and _labels(r) & TOTALS_START), None)
    return header, totals


def _boxes(rows: list[dict]) -> list[dict] | None:
    """OCR boxes rebuilt from row tokens (grouped on 'line'); None without box geometry."""
    by_line: dict[int, list[dict]] = {}
    for r in rows:
        for t in r["tokens"]:
            if t.get("box_h") is None or t.get("line") is None:
                return None
            by_line.setdefault(t["line"], []).append({**t, "row": r["idx"]})
    out = []
    for toks in by_line.values():
        toks.sort(key=lambda t: t["x0"])
        confs = [t["conf"] for t in toks if t.get("conf") is not None]
        out.append({"text": " ".join(t["text"] for t in toks), "y0": toks[0]["y0"],
                    "y1": toks[0]["y1"], "h": toks[0]["box_h"], "row": toks[0]["row"],
                    "conf": sum(confs) / len(confs) if confs else 0.0})
    return sorted(out, key=lambda b: b["y0"]) or None


def _merchant_boxes(rows: list[dict]) -> list[dict] | None:
    """Merchant candidates from box geometry: tall boxes near the top of the bill win."""
    boxes = _boxes(rows)
    if not boxes:
        return None
    top, bottom = min(b["y0"] for b in boxes), max(b["y1"] for b in boxes)
    height = max(bottom - top, 1e-6)
    med_h = median(b["h"] for b in boxes) or 1.0
    cands = []
    for b in boxes:
        rel_y = (b["y0"] - top) / height
        text = MERCHANT_PREFIX.sub("", b["text"].strip()).strip()
        if rel_y >= BOX_TOP or sum(c.isalpha() for c in text) < 2:
            continue
        up = text.upper()
        words = set(re.findall(r"[A-Z]+", up))
        digits = sum(c.isdigit() for c in text) / max(len(text), 1)
        business = bool(words & BUSINESS_WORDS) or ("RESTAURANT" in words and len(words) >= 2)
        score = min(3.0, 2 * b["h"] / med_h) + 1.5 * (1 - rel_y / BOX_TOP) + (2 if business else 0)
        score -= 2 * bool(MERCHANT_BAD.search(up) or words & PLACE_WORDS) + 3 * digits
        score -= 2 * bool(PHONE_RE.search(text)) + 3 * bool(_gstin_windows(text))
        score -= 2 * bool(PINCODE.search(text)) + 3 * bool(MERCHANT_ROLE.search(up))
        cands.append({"value": text, "score": round(score * b["conf"], 2), "conf": b["conf"],
                      "y0": b["y0"], "y1": b["y1"], "h": b["h"], "rows": [b["row"]]})
    if not cands:
        return None
    by_y = sorted(cands, key=lambda c: c["y0"])
    cands.sort(key=lambda c: -c["score"])
    best = cands[0]
    k = by_y.index(best)
    # name split over two boxes: join a vertically adjacent box of similar height
    for nb in [by_y[i] for i in (k - 1, k + 1) if 0 <= i < len(by_y)]:
        a, c = sorted((best, nb), key=lambda x: x["y0"])
        similar = abs(nb["h"] - best["h"]) <= BOX_H_SIMILAR * max(nb["h"], best["h"])
        close = c["y0"] - a["y1"] <= max(a["h"], c["h"])
        if similar and close and min(nb["score"], best["score"]) >= BOX_MERGE_MIN \
                and not set(nb["value"].upper().split()) <= set(best["value"].upper().split()):
            joined = {**best, "value": f"{a['value']} {c['value']}",
                      "rows": sorted(set(a["rows"] + c["rows"]))}
            cands = [joined] + [x for x in cands if x is not best and x is not nb]
            break
    return cands


def _merchant(rows: list[dict]) -> dict:
    """Box-geometry merchant unless its score is below BOX_MIN, then the row method."""
    row_f = _merchant_rows(rows)
    boxes = _merchant_boxes(rows) or []
    # top 3 of each method (the row winner has no single score: it may be a joined pair)
    by_box = [{"value": c["value"], "score": c["score"], "source": "boxes"} for c in boxes[:3]]
    by_row = ([{"value": row_f["value"], "score": None, "source": "rows"}] if row_f["value"] else []) \
        + [{"value": a["value"], "score": a["score"], "source": "rows"} for a in row_f["alternatives"]]
    if boxes and boxes[0]["score"] >= BOX_MIN:
        b = boxes[0]
        return _field(b["value"], min(1.0, b["score"] / 5) * b["conf"], b["rows"],
                      merchant_source="boxes", candidates=by_box + by_row[:1])
    return {**row_f, "merchant_source": "rows", "candidates": by_row[:3] + by_box[:1]}


def _merchant_rows(rows: list[dict]) -> dict:
    cands = []
    for r in rows[:MERCHANT_ROWS]:
        text = MERCHANT_PREFIX.sub("", r["text"].strip()).strip()
        letters = [c for c in text if c.isalpha()]
        if not letters:
            continue
        up = sum(c.isupper() for c in letters) / len(letters)
        digits = sum(c.isdigit() for c in text) / max(len(text), 1)
        score = 2 * up + (1 if len(letters) >= 4 else -1) - 3 * digits
        words = set(re.findall(r"[A-Z]+", text.upper()))
        if MERCHANT_BAD.search(text.upper()) or _labels(r) - {"HEADER"}:
            score -= 2
        if MERCHANT_ROLE.search(text.upper()):
            score -= 3
        if PINCODE.search(text) or words & PLACE_WORDS:
            score -= 2
        # RESTAURANT alone is a section title; with another word it names the shop
        business = bool(words & BUSINESS_WORDS) or ("RESTAURANT" in words and len(words) >= 2)
        if business:
            score += 2
        score -= 1.5 * sum(bool(MID_CAP.search(w)) for w in text.split())
        confs = [t["conf"] for t in r["tokens"] if t.get("conf") is not None]
        mean_conf = sum(confs) / len(confs) if confs else (r["min_conf"] or 0)
        # weight by read quality, small bonus for earlier rows
        score = score * mean_conf + 0.3 * (MERCHANT_ROWS - r["idx"]) / MERCHANT_ROWS
        cands.append({"value": text, "score": round(score, 2), "rows": [r["idx"]],
                      "business": business})
    if not cands:
        return _field(None, 0.0, [], reason="not_found")
    cands.sort(key=lambda c: -c["score"])
    best = cands[0]
    by_row = {c["rows"][0]: c for c in cands}
    # name split across two lines: join adjacent well-scored rows (one has a business word)
    for nb in (by_row.get(best["rows"][0] - 1), by_row.get(best["rows"][0] + 1)):
        if nb and nb["score"] >= MERGE_MIN and best["score"] >= MERGE_MIN \
                and (nb["business"] or best["business"]) \
                and not set(nb["value"].upper().split()) <= set(best["value"].upper().split()):
            a, b = sorted((best, nb), key=lambda c: c["rows"][0])
            best = {**best, "value": f"{a['value']} {b['value']}", "rows": a["rows"] + b["rows"]}
            break
    alts = [{"value": c["value"], "score": c["score"], "row": c["rows"][0]}
            for c in cands if c["rows"][0] not in best["rows"]][:2]
    conf = best["score"] / 3 * (rows[best["rows"][0]]["min_conf"] or 0)
    return _field(best["value"], conf, best["rows"], alts)


def _date(rows: list[dict]) -> dict:
    n = len(rows)
    cands = []
    for r in rows:
        for d in parse_date(r["text"]):
            score = (2 if _labels(r) & {"DATE", "BILL_NO"} else 0) + (1 if r["idx"] < n / 3 else 0) \
                + (0 if d["ambiguous"] else 0.5)
            cands.append({"value": d["iso"], "ambiguous": d["ambiguous"], "fmt": d["fmt"],
                          "score": score, "row": r["idx"]})
    cands.sort(key=lambda c: (-c["score"], c["row"]))
    if not cands:
        return _field(None, 0.0, [], reason="not_found")
    b = cands[0]
    return _field(b["value"], b["score"] / 3.5, [b["row"]], cands[1:3], ambiguous=b["ambiguous"])


def _gstin_windows(text: str) -> list[str]:
    """15-char windows that are GSTINs after OCR confusable fixes (O/I/S/B <-> 0/1/5/8)."""
    s = re.sub(r"\s+", "", text.upper())
    out = []
    for i in range(len(s) - 14):
        w = list(s[i:i + 15])
        for p in GSTIN_DIGIT_POS:
            w[p] = TO_DIGIT.get(w[p], w[p])
        for p in GSTIN_LETTER_POS:
            w[p] = TO_LETTER.get(w[p], w[p])
        w = "".join(w)
        if GSTIN_RE.fullmatch(w):
            out.append(w)
    return out


def _gstin(rows: list[dict]) -> dict:
    hits = []
    for r in rows:
        for w in _gstin_windows(r["text"]):
            hits.append({"value": w, "row": r["idx"], "labeled": "GSTIN" in _labels(r)})
    hits.sort(key=lambda h: (not h["labeled"], h["row"]))
    if not hits:
        return _field(None, 0.0, [], reason="not_found")
    return _field(hits[0]["value"], 1.0 if hits[0]["labeled"] else 0.8, [hits[0]["row"]], hits[1:3])


def _bill_no(rows: list[dict]) -> dict:
    for r in rows:
        if "BILL_NO" not in _labels(r):
            continue
        m = BILL_NO_RE.search(r["text"])
        if m and any(c.isdigit() for c in m.group(1)) and not _gstin_windows(m.group(1)):
            return _field(m.group(1), r["min_conf"] or 0, [r["idx"]])
        tok = next((t["text"] for t in r["tokens"] if any(c.isdigit() for c in t["text"])
                    and not parse_date(t["text"]) and not _gstin_windows(t["text"])), None)
        if tok:
            return _field(tok.strip(":#."), 0.5 * (r["min_conf"] or 0), [r["idx"]])
    return _field(None, 0.0, [], reason="not_found")


def _bill_type(rows: list[dict], gstin: dict) -> dict:
    text = "\n".join(r["text"].upper() for r in rows)
    scores = {t: sum(bool(re.search(p, text)) for p in pats) for t, pats in BILL_TYPES.items()}
    if gstin["value"]:
        scores["tax_invoice"] += 1
    order = list(BILL_TYPES)
    ranked = sorted((t for t in order if scores[t]), key=lambda t: (-scores[t], order.index(t)))
    if not ranked:
        return _field("unknown", 0.0, [])
    alts = [{"value": t, "score": scores[t]} for t in ranked[1:]]
    return _field(ranked[0], scores[ranked[0]] / 3, [], alts, score=scores[ranked[0]])


def _tax_detail_row(r: dict) -> bool:
    first = r["tokens"][0]["text"] if r["tokens"] else ""
    lone_code = bool(re.fullmatch(r"(?=.*\d)(?=.*[A-Za-z])[A-Za-z0-9]{2,}", first))
    return not _name(r, drop_tax_words=True) and (
        bool(re.fullmatch(r"\d{4,8}", first)) or first.startswith(("[", "-")) or lone_code)


def _pair(nums: list[Decimal], amount: Decimal) -> tuple[Decimal, Decimal] | None:
    """(qty, rate) with qty*rate within TOL of amount."""
    for i in range(len(nums)):
        for j in range(len(nums)):
            if i != j and abs(nums[i] * nums[j] - amount) <= TOL:
                q, r = sorted((nums[i], nums[j]))
                return q, r
    return None


def _weighed_pair(qtys: list[tuple[int, Decimal]], rates: list[tuple[int, Decimal]],
                  amount: Decimal) -> tuple[Decimal, Decimal] | None:
    """(qty, rate) with qty a quantity token ('0.750', '2.000') of the row and rate another number."""
    for i, q in qtys:
        for j, r in rates:
            if i != j and q > 0 and abs(q * r - amount) <= TOL:
                return q, r
    return None


def _note(r: dict, style: str) -> tuple[str | None, Decimal | None] | None:
    """Tax-detail note with no amount: 'HSN 040690', 'GST 5%' -> (hsn, gst rate)."""
    h, g = HSN_RE.search(r["text"]), GST_NOTE_RE.search(r["text"])
    if not (h or g):
        return None
    code = h.group(1) if h else None
    if any(re.sub(r"\D", "", a["raw"]) != code for a in _money(r, style)):
        return None  # the row has a real amount (the HSN code itself does not count)
    return code, Decimal(g.group(1)) if g else None


def _disc_offset(header: dict) -> int | None:
    """Header with a DISC column -> its position from the right among the header words."""
    words = [t["text"].upper().strip("[(|.:") for t in header["tokens"]
             if any(c.isalpha() for c in t["text"])]
    k = next((i for i, w in enumerate(words) if w.startswith("DISC")), None)
    return None if k is None else len(words) - 1 - k


def _is_totals_row(it: dict, items: list[dict]) -> bool:
    """Amount (and qty, if any) equal the column sums of the item rows above (within TOL)."""
    above = [i for i in items if i["kind"] == "item"]
    if len(above) < 2 or abs(sum(i["amount"] for i in above) - it["amount"]) > TOL:
        return False
    qtys = [i["qty"] for i in above]
    return it["qty"] is None or None in qtys or abs(sum(qtys) - it["qty"]) <= TOL


def _items(rows: list[dict], lo: int, hi: int, style: str,
           excluded: list[dict] | None = None, disc_pos: int | None = None) -> list[dict]:
    excluded = [] if excluded is None else excluded
    items: list[dict] = []
    pending: list[dict] = []  # wrapped-name rows waiting for an amount row
    for r in rows[lo:hi]:
        money = _money(r, style)
        labels = _labels(r)
        if "DISCOUNT" in labels and any(a["value"] < 0 for a in money):
            amt = [a for a in money if a["value"] < 0][-1]["value"]
            items.append({"row": r["idx"], "rows": [r["idx"]], "name": _name(r), "kind": "discount",
                          "qty": None, "rate": None, "amount": amt, "status": "unverified",
                          "numbers": [], "tax_detail": [], "min_conf": r["min_conf"]})
            pending = []
            continue
        if _slab_row(r):
            excluded.append({"row": r["idx"], "reason": "gst_slab", "text": r["text"]})
            pending = []
            continue
        if _base_tax_rows(r):
            excluded.append({"row": r["idx"], "reason": "tax_line", "text": r["text"]})
            pending = []
            continue
        if (note := _note(r, style)) is not None:
            # HSN/GST note belongs to the item above, never to the next item's name
            if items:
                it = items[-1]
                if note[0]:
                    it["hsn"] = note[0]
                if note[1] is not None:
                    it["gst_rate"] = note[1]
                it["rows"].append(r["idx"])
            continue
        if items and _tax_detail_row(r):
            it = items[-1]
            it["tax_detail"] += [a["value"] for a in r["amounts"]]
            it["rows"].append(r["idx"])
            it["min_conf"] = min(it["min_conf"] or 1, r["min_conf"] or 1)
            continue
        meta = labels & META_LABELS
        # an item row carrying tax text ('Paneer CGST@2.5% 2 150.00 300.00') stays an item
        tax_item = meta <= TAX_LABELS and len(r["amounts"]) >= 2 and _clean_name(_name(r))[0]
        if meta and not tax_item:
            pending = []
            continue
        name = _name(r)
        if not money:
            if name:
                pending = (pending + [r])[-2:]
            continue
        if not name and not pending:
            continue  # bare numbers with no name to attach to
        if _garbage_label_row(r):
            excluded.append({"row": r["idx"], "reason": "label_garbage", "text": r["text"]})
            pending = []
            continue
        line = money[-1]
        last_pos = max(i for i, a in enumerate(r["amounts"]) if a["raw"] == line["raw"])
        # a 0.00 in the DISC column is the item discount, never a qty or rate
        di = len(r["amounts"]) - 1 - disc_pos if disc_pos is not None else -1
        disc = di >= 0 and di != last_pos and r["amounts"][di]["value"] == 0
        skip = {last_pos, di} if disc else {last_pos}
        cands = [(i, _int_value(a) if style == "integer" else a["value"])
                 for i, a in enumerate(r["amounts"]) if i not in skip]
        cands = [(i, n) for i, n in cands if n > 0]
        nums = [n for _, n in cands]
        pr = _pair(nums, line["value"])
        if not pr:  # weighed quantity: '0.750' / '2.000' read as a plain qty
            qtys = [(i, q) for i, a in enumerate(r["amounts"])
                    if i not in skip and (q := parse_qty(a["raw"])) is not None]
            pr = _weighed_pair(qtys, cands, line["value"])
        src = pending + [r]
        full, glued_tax = _clean_name(" ".join(x for x in [_name(p) for p in pending] + [name] if x))
        it = {"row": r["idx"], "rows": [p["idx"] for p in src], "name": full,
              "kind": "item", "qty": pr[0] if pr else None, "rate": pr[1] if pr else None,
              "amount": line["value"], "status": "verified" if pr else "unverified",
              "numbers": nums, "tax_detail": glued_tax,
              "min_conf": min((p["min_conf"] or 0) for p in src)}
        # UPC/SKU column between the name and the price; tax-flag letter after the price
        code = next((t["text"] for t in r["tokens"][:-1] if CODE_RE.fullmatch(t["text"])), None)
        if code:
            it["code"] = code
        if line.get("tax_flag"):
            it["tax_flag"] = line["tax_flag"]
        if pr and pr[0] != pr[0].to_integral_value():
            it["weighed"] = True
        if disc:
            it["discount"] = Decimal("0.00")
        pending = []
        if _is_totals_row(it, items):
            excluded.append({"row": r["idx"], "reason": "totals_row", "text": r["text"]})
            continue
        items.append(it)
    return items


def _rate_ok(label: str, rate: Decimal) -> bool:
    """GST labels: a slab, or half a slab for split taxes (CGST/SGST/UTGST).
    Other tax labels (TAX, VAT, CESS): any rate from 0 to 30."""
    if label not in GST_FAMILY:
        return 0 <= rate <= MAX_OTHER_RATE
    return rate in GST_RATES or (label in {"CGST", "SGST", "UTGST"} and rate in HALF_RATES)


def _rate_rows(r: dict) -> list[dict]:
    """Per-rate rows (rate, base, tax) in one row."""
    out = []
    for m in RATE_ROW_RE.finditer(r["text"].upper()):
        base, tax = parse_amount(m.group(3)), parse_amount(m.group(4))
        if base and tax and tax.value < base.value:
            label = "GST" if "GST" in m.group(1) else m.group(1)
            out.append({"label": label, "rate": Decimal(m.group(2)), "base": base.value,
                        "amount": tax.value, "row": r["idx"], "min_conf": r["min_conf"]})
    return out


def _base_tax_rows(r: dict) -> list[dict]:
    """'CGST @ r% ON base tax' -> {label, rate, base, amount}; a row whose numbers do not
    satisfy tax = base*r/100 (in either order) gives a hit with reason 'base_mismatch'."""
    out = []
    for m in BASE_TAX_RE.finditer(r["text"].upper()):
        a, b = parse_amount(m.group(3)), parse_amount(m.group(4))
        if not (a and b):
            continue
        rate = Decimal(m.group(2))
        h = {"label": m.group(1), "rate": rate, "row": r["idx"], "min_conf": r["min_conf"]}
        pair = next(((x, y) for x, y in ((a.value, b.value), (b.value, a.value))
                     if abs(x * rate / 100 - y) <= BASE_TOL), None)
        if pair:
            out.append({**h, "base": pair[0], "amount": pair[1]})
        else:
            out.append({**h, "base": None, "amount": None, "reason": "base_mismatch"})
    return out


def _taxes(rows: list[dict], start: int, style: str) -> dict:
    regex_hits, label_hits, rate_rows, rejected, base_rows = [], [], [], [], []
    based = set()  # rows read (or rejected) as tax lines with a base
    for r in rows:  # strict shape: may sit above the totals region
        for h in _base_tax_rows(r):
            based.add(r["idx"])
            if "reason" in h:
                rejected.append(h)
            elif _rate_ok(h["label"], h["rate"]):
                base_rows.append(h)
            else:
                rejected.append({**h, "reason": "rate_not_allowed"})
    for r in rows[start:]:
        if r["idx"] in based:
            continue
        hits, bad = [], False
        for h in _rate_rows(r):
            if _rate_ok(h["label"], h["rate"]):
                hits.append(h)
            else:
                bad = True
                rejected.append({**h, "reason": "rate_not_allowed"})
        rate_rows += hits
        if not hits:
            for m in TAX_RE.finditer(r["text"].upper()):
                if not (a := parse_amount(m.group(3))):
                    continue
                h = {"label": m.group(1), "rate": Decimal(m.group(2)), "amount": a.value,
                     "row": r["idx"], "min_conf": r["min_conf"]}
                if _rate_ok(h["label"], h["rate"]):
                    hits.append(h)
                    regex_hits.append(h)
                else:
                    bad = True
                    rejected.append({**h, "reason": "rate_not_allowed"})
        tl = [l for l, _ in r["labels"] if l in TAX_LABELS]
        if not tl and PLAIN_TAX.search(r["text"].upper()):
            tl = ["TAX"]
        money = _money(r, style)
        if style == "decimal":
            money = [a for a in money if a["strength"] == "strong"]
        if not hits and not bad and tl and money:
            # 'TAX 1 | 5.50% | 1.10': the percent may sit anywhere in the row
            rate = parse_percent(r["text"])
            h = {"label": tl[0], "rate": rate, "amount": money[-1]["value"],
                 "row": r["idx"], "min_conf": r["min_conf"]}
            if rate is None or _rate_ok(tl[0], rate):
                label_hits.append(h)
            else:
                rejected.append({**h, "reason": "rate_not_allowed"})
    table = []
    for r in rows:
        if len(_labels(r) & TABLE_LABELS) >= 2:
            for nxt in rows[r["idx"] + 1:]:
                strong = [a["value"] for a in nxt["amounts"] if a["strength"] == "strong"]
                if len(strong) < 3:
                    break
                table.append({"row": nxt["idx"], "amounts": strong, "min_conf": nxt["min_conf"]})
            break
    return {"regex": regex_hits, "label": label_hits, "rate_rows": rate_rows, "rejected": rejected,
            "base_rows": base_rows, "table_rows": table, "table": _table_summary(table)}


def _table_summary(table: list[dict]) -> dict | None:
    """GST summary table -> {base, tax, total}. Rows: taxable, tax columns..., [row total]."""
    rows = [t["amounts"] for t in table]
    if len(rows) >= 3 and abs(sum(a[0] for a in rows[:-1]) - rows[-1][0]) <= TOL:
        rows = rows[-1:]  # last row is the column-sum row
    if not rows:
        return None
    base = tax = Decimal(0)
    total: Decimal | None = Decimal(0)
    for a in rows:
        if len(a) >= 3 and abs(a[0] + sum(a[1:-1]) - a[-1]) <= TOL:
            t, tot = sum(a[1:-1], Decimal(0)), a[-1]
        else:
            t, tot = sum(a[1:], Decimal(0)), None
        base, tax = base + a[0], tax + t
        total = None if total is None or tot is None else total + tot
    return {"base": base, "tax": tax, "total": total,
            "min_conf": min((t["min_conf"] or 0) for t in table)}


def _tax_lines(taxes: dict, items_sum: Decimal = Decimal(0)) -> tuple[list[dict], str]:
    """Tax lines from ONE source: CGST/SGST/... pairs, else per-rate rows, else the TAX_TOTAL
    row, else the GST table, else plain TAX rows. Sources are never summed together.
    A source whose sum is above 35% of the items sum is rejected."""
    hits = taxes["regex"] + taxes["label"] + taxes.get("base_rows", [])
    tb = taxes["table"]
    table = [{"label": "TABLE", "rate": None, "amount": tb["tax"], "row": taxes["table_rows"][0]["row"],
              "min_conf": tb["min_conf"]}] if tb and tb["tax"] else []
    sources = [
        ("pairs", [h for h in hits if h["label"] in PAIR_LABELS]),
        ("rate_rows", taxes.get("rate_rows", []) + [h for h in hits if h["label"] == "GST"]),
        ("tax_total", [h for h in hits if h["label"] == "TAX_TOTAL"][:1]),
        ("table", table),
        ("tax_rows", [h for h in hits if h["label"] == "TAX"]),
    ]
    for name, lines in sources:
        if not lines:
            continue
        s = sum((t["amount"] for t in lines), Decimal(0))
        if items_sum > 0 and s > MAX_TAX_SHARE * items_sum:
            taxes.setdefault("rejected", []).append(
                {"source": name, "amount": s, "reason": "above_35pct_of_items"})
            continue
        return lines, name
    return [], "none"


def _base_taxable(tax_lines: list[dict]) -> dict | None:
    """Taxable value from tax lines printed with a base: bases summed per label, all labels agree."""
    by_label: dict[str, Decimal] = {}
    for t in tax_lines:
        if t.get("base") is None:
            return None
        by_label[t["label"]] = by_label.get(t["label"], Decimal(0)) + t["base"]
    vals = list(by_label.values())
    if not vals or max(vals) - min(vals) > BASE_TOL:
        return None
    return _field(vals[0], min((t["min_conf"] or 0) for t in tax_lines),
                  sorted({t["row"] for t in tax_lines}), source="tax_line_base")


def _tax_total_line(rows: list[dict], style: str, tax_lines: list[dict], source: str) -> dict | None:
    """TOTAL GST / TOTAL TAX row with one lone amount; matches_lines when it equals CGST+SGST."""
    for r in rows:
        if "TAX_TOTAL" not in _labels(r):
            continue
        money = _money(r, style)
        if len(money) != 1:
            continue
        v = money[0]["value"]
        split = [t["amount"] for t in tax_lines if t["label"] in ("CGST", "SGST")]
        ok = source == "pairs" and bool(split) and abs(sum(split) - v) <= BASE_TOL
        return {"value": v, "row": r["idx"], "matches_lines": ok}
    return None


def _labeled(rows: list[dict], start: int, label: str, style: str) -> dict:
    cands = []
    for r in rows[start:]:
        if label in _labels(r):
            money = _money(r, style)
            if money:
                cands.append({"value": money[-1]["value"], "row": r["idx"]})
    if not cands:
        return _field(None, 0.0, [], reason="not_found")
    b = cands[0]
    return _field(b["value"], rows[b["row"]]["min_conf"] or 0, [b["row"]], cands[1:3])


def _adjustment_discount(rows: list[dict], style: str) -> dict | None:
    """Unlabelled negative rows between a SUBTOTAL and the next SUBTOTAL/TOTAL -> summed discount."""
    hits: list[tuple[dict, Decimal]] = []
    open_ = False
    for r in rows:
        ls = _labels(r)
        if ls & {"SUBTOTAL", "TOTAL"}:
            open_ = "SUBTOTAL" in ls  # closes the window; a SUBTOTAL row opens the next one
            continue
        if not open_ or "DISCOUNT" in ls or not re.search(r"[A-Za-z]", r["text"]):
            continue
        neg = [a for a in _money(r, style) if a["value"] < 0 and a["raw"].lstrip().startswith(("-", "("))]
        if neg:
            hits.append((r, neg[-1]["value"]))
    if not hits:
        return None
    return _field(sum((v for _, v in hits), Decimal(0)), min(r["min_conf"] or 0 for r, _ in hits),
                  [r["idx"] for r, _ in hits], source="negative_adjustment")


def _label_score(r: dict) -> tuple[int, str]:
    ls = _labels(r)
    if "CHANGE" in ls:
        return -3, "change"
    if "CASH" in ls:
        return -3, "cash"  # _total drops the penalty when no total-type label exists
    if "SUBTOTAL" in ls:
        return -2, "subtotal"
    if ls & TAX_LABELS:
        return 0, "tax"
    if NET_RE.search(label_text(r["text"]).upper()):
        return 4, "net_total"  # NET AMOUNT / NET PAYABLE / GRAND TOTAL outrank a plain TOTAL
    if "GRAND_TOTAL" in ls:
        return 3, "grand_total"
    if "TOTAL" in ls:
        return 2, "total"
    if AMOUNT_LIKE.search(r["text"].upper()):
        return 1, "amount_like"
    return 0, "none"


def _near(a: Decimal, b: Decimal | None) -> bool:
    return b is not None and abs(a - b) <= TOL


def _total(rows: list[dict], start: int | None, style: str, items: list[dict], tax_sum: Decimal,
           discount: Decimal | None, taxable: Decimal | None,
           taxable_value: Decimal | None = None) -> dict:
    n = len(rows)
    bottom40, bottom3 = n - max(1, round(0.4 * n)), n - max(1, round(n / 3))
    counts: dict[Decimal, int] = {}
    for r in rows:
        for a in _money(r, style):
            counts[a["value"]] = counts.get(a["value"], 0) + 1
    items_sum = sum((i["amount"] for i in items), Decimal(0)) if items else None
    with_tax = None
    if items_sum is not None and (tax_sum or discount):
        with_tax = items_sum + tax_sum - abs(discount or 0)
    taxable_tax = taxable + tax_sum if taxable is not None and tax_sum else None
    cash_rows = [r for r in rows if "CASH" in _labels(r) and "CHANGE" not in _labels(r)]
    change_rows = [r for r in rows if "CHANGE" in _labels(r)]
    cash = next((m[-1]["value"] for r in cash_rows if (m := _money(r, style))), None)
    change = next((m[-1]["value"] for r in change_rows if (m := _money(r, style))), None)
    item_rows = {i["row"] for i in items if i["kind"] == "item"}
    paid = cash - abs(change) if cash is not None and change is not None else None
    verified = any(i["status"] == "verified" for i in items)
    # cash == taxable + tax (or items + tax): the tendered amount is the total (needs a read taxable)
    cash_ok = taxable_value is not None and bool(tax_sum) and cash is not None and (
        _near(cash, taxable_value + tax_sum) or (items_sum is not None and _near(cash, items_sum + tax_sum)))

    def candidate_rows():
        for r in rows:
            near_cash = any(0 < c["idx"] - r["idx"] <= 2 for c in cash_rows)
            if not ((start is not None and r["idx"] >= start) or r["idx"] >= bottom40 or near_cash):
                continue
            # GST slab rows and MRP/savings/discount/points rows are never the total
            if _slab_row(r) or NOT_TOTAL_RE.search(label_text(r["text"]).upper()):
                continue
            money = [a for a in _money(r, style) if a["value"] > 0]
            # an item row is a candidate only with a TOTAL-type label or for its cash-change value
            if r["idx"] in item_rows and _label_score(r)[1] not in TOTAL_TYPES:
                money = [a for a in money if _near(a["value"], paid)]
            if money:
                yield r, money

    has_total_type = any(_label_score(r)[1] in TOTAL_TYPES for r, _ in candidate_rows())
    best: dict[Decimal, dict] = {}
    for r, money in candidate_rows():
        lab, why = _label_score(r)
        if why == "cash" and not has_total_type:
            lab = 0
        for a in money:
            v = a["value"]
            # cash == items needs a verified qty*rate item or a second witness (tax / subtotal)
            second = (tax_sum and _near(v, with_tax)) or _near(v, taxable_tax) or _near(v, taxable)
            pay_items = why == "cash" and not has_total_type and (verified or second) and (
                _near(v, items_sum) or _near(v, with_tax))
            if cash is not None and change is not None:
                cash_w = 3 if _near(v, cash - abs(change)) else 0
            else:  # change row unreadable: cash itself is a weak witness
                cash_w = 1 if cash is not None and change_rows and _near(v, cash) else 0
            parts = {"label": lab, "position": 1 if r["idx"] >= bottom3 else 0,
                     "repeat": min(2, counts[v] - 1),
                     # a bare item sum is only the total when no tax was read
                     "items_sum": 3 if not tax_sum and _near(v, items_sum) else 0,
                     "items_tax_disc": 3 if _near(v, with_tax) else 0,
                     "taxable_tax": 3 if _near(v, taxable_tax) else 0,
                     "cash_change": cash_w,
                     "payment_equals_items": 2 if pay_items else 0}
            if cash_ok and why == "cash" and _near(v, cash):
                parts["cash_witness"] = 3
            elif cash_ok and _near(v, taxable_value) and _near(taxable_value, items_sum):
                parts["cash_witness"] = -2  # the taxable base is not the total
            c = {"value": v, "score": sum(parts.values()), "row": r["idx"], "label": why,
                 "breakdown": parts}
            if v not in best or c["score"] > best[v]["score"]:
                best[v] = c
    # a non-cash candidate wins a tie with a CASH/TENDERED/PAID candidate
    ranked = sorted(best.values(), key=lambda c: (-c["score"], c["label"] == "cash",
                                                  -c["row"], -c["value"]))
    if not ranked:
        return _field(None, 0.0, [], reason="not_found")
    extra = {}
    if ranked[0]["label"] == "cash" and not ranked[0]["breakdown"].get("cash_witness"):
        # support = score without the position bonus (being low on the bill is not evidence)
        other = next((c for c in ranked if c["label"] != "cash"
                      and c["score"] - c["breakdown"]["position"] > 0), None)
        if other is not None:
            ranked = [other] + [c for c in ranked if c is not other]
        elif not (_near(ranked[0]["value"], items_sum) or _near(ranked[0]["value"], paid)):
            extra = {"reason": "cash_only"}  # kept only because nothing else exists
    ranked = ranked[:3]
    gap = ranked[0]["score"] - (ranked[1]["score"] if len(ranked) > 1 else 0)
    conf = 0.0 if extra else max(0, gap) / 5
    return _field(ranked[0]["value"], conf, [ranked[0]["row"]], ranked[1:],
                  score_gap=gap, breakdown=ranked[0]["breakdown"], label=ranked[0]["label"],
                  min_conf=rows[ranked[0]["row"]]["min_conf"], **extra)


def extract(rows: list[dict]) -> dict:
    """Rows (03_rows.json) -> candidate fields with scores and evidence rows."""
    rows = _load(rows)
    n = len(rows)
    style_f = _money_style(rows)
    style = style_f["value"]
    header, totals = _regions(rows)
    lo = 0 if header is None else header + 1
    hi = n if totals is None else totals
    tail = totals if totals is not None else n - max(1, round(0.4 * n)) if n else 0

    gstin = _gstin(rows)
    excluded: list[dict] = []
    disc_pos = _disc_offset(rows[header]) if header is not None else None
    items = _items(rows, lo, hi, style, excluded, disc_pos)
    # GST rates from item notes: display only, never added to tax_sum
    rates_seen = sorted({i["gst_rate"] for i in items if i.get("gst_rate") is not None})
    subtotal = _labeled(rows, tail, "SUBTOTAL", style)
    discount = _labeled(rows, tail, "DISCOUNT", style)
    round_off = _labeled(rows, tail, "ROUNDOFF", style)
    taxable = _labeled(rows, tail, "TAXABLE", style)
    service = _labeled(rows, tail, "SERVICE_CHARGE", style)
    payments = [{"label": l, "amount": m[-1]["value"], "row": r["idx"]}
                for r in rows for l in [next((l for l, _ in r["labels"] if l in PAY_LABELS), None)]
                if l and (m := _money(r, style))]
    base = taxable["value"] if taxable["value"] is not None else subtotal["value"]
    taxes = _taxes(rows, tail, style)
    items_sum = sum((i["amount"] for i in items if i["kind"] == "item"), Decimal(0))
    tax_lines, tax_source = _tax_lines(taxes, items_sum)
    tax_sum = sum((t["amount"] for t in tax_lines), Decimal(0))
    if taxable["value"] is None and (tb := _base_taxable(tax_lines)) is not None:
        taxable = tb
        base = taxable["value"] if base is None else base  # a printed subtotal stays the base
    tax_total_line = _tax_total_line(rows, style, tax_lines, tax_source)
    total = _total(rows, totals, style, items, tax_sum, discount["value"], base, taxable["value"])
    # after the total is chosen, so these never change it
    if discount["value"] is not None and discount["value"] > 0 \
            and "DISC" not in label_text(rows[discount["rows"][0]]["text"]).upper():
        discount = _field(None, 0.0, [], reason="positive_without_disc_word")
    if discount["value"] is None and (adj := _adjustment_discount(rows, style)):
        discount = adj

    return {
        "n_rows": n,
        "money_style": style_f,
        "regions": {"header": header, "totals_start": totals, "items": [lo, hi]},
        "merchant": _merchant(rows),
        "date": _date(rows),
        "gstin": gstin,
        "bill_no": _bill_no(rows),
        "bill_type": _bill_type(rows, gstin),
        "items": items,
        "excluded_rows": excluded,
        "items_sum": sum((i["amount"] for i in items), Decimal(0)),
        "taxes": taxes,
        "tax_lines": tax_lines,
        "tax_source": tax_source,
        "tax_sum": tax_sum,
        "tax_rates_seen": rates_seen,
        "subtotal": subtotal,
        "discount": discount,
        "round_off": round_off,
        "service_charge": service,
        "taxable": taxable,
        "tax_total_line": tax_total_line,
        "payments": payments,
        "total": total,
    }
