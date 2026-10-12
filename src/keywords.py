"""Fuzzy keyword -> label matching for bill rows."""
import re

from rapidfuzz import fuzz

MIN_SCORE = 85
PARTIAL_MIN_LEN = 5
LETTER_FIX = str.maketrans({"0": "O", "1": "L", "5": "S", "8": "B"})

LABELS: dict[str, list[str]] = {
    "GRAND_TOTAL": ["GRAND TOTAL", "NET TOTAL", "NET AMOUNT", "NET PAYABLE", "AMOUNT PAYABLE",
                    "TOTAL AMOUNT", "BILL AMOUNT"],
    "TOTAL": ["TOTAL", "JUMLAH", "TOTAL HARGA"],
    "SUBTOTAL": ["SUB TOTAL", "SUBTOTAL", "TAXABLE VALUE", "TAXABLE AMOUNT"],
    "TAX_TOTAL": ["TOTAL TAX", "TAX AMOUNT", "TOTAL GST", "GST TOTAL", "TOTAL VAT"],
    "TAXABLE": ["TAXABLE", "TAXABLE VALUE", "TAXABLE AMT"],
    "CGST": ["CGST"],
    "SGST": ["SGST"],
    "IGST": ["IGST"],
    "UTGST": ["UTGST"],
    "GST": ["GST", "GST/TAX"],
    "VAT": ["VAT", "PPN", "PB1"],
    "CESS": ["CESS"],
    "SERVICE_CHARGE": ["SERVICE CHARGE", "SERVICE", "SVC CHG", "SVC CHRG"],
    "DISCOUNT": ["DISCOUNT", "DISC"],
    "ROUNDOFF": ["ROUND OFF", "ROUNDOFF", "ROUNDING", "R/O"],
    "CASH": ["CASH", "TENDERED", "RECEIVED", "PAID"],
    "CHANGE": ["CHANGE", "BALANCE", "RETURN"],
    "BILL_NO": ["BILL NO", "INVOICE NO", "INV NO", "RECEIPT NO"],
    "GSTIN": ["GSTIN"],
    "DATE": ["DATE"],
    "HEADER": ["DESCRIPTION", "RATE", "QTY", "AMOUNT", "VALUE", "HSN", "MRP", "PRICE"],
}


def _tokens(text: str) -> list[str]:
    """Uppercase tokens; digit->letter fixes only in tokens with 3+ letters."""
    toks = [t for t in re.split(r"[^A-Z0-9/]+", text.upper()) if t]
    return [t.translate(LETTER_FIX) if sum(c.isalpha() for c in t) >= 3 else t for t in toks]


def match_labels(text: str) -> list[tuple[str, float]]:
    """[(label, score)] sorted by score, then matched keyword length (both desc)."""
    toks = _tokens(text)
    joined = " ".join(toks)
    best: dict[str, tuple[float, int]] = {}
    for label, kws in LABELS.items():
        for kw in kws:
            k = len(kw.split())
            score = max((fuzz.ratio(kw, " ".join(toks[i:i + k]))
                         for i in range(len(toks) - k + 1)), default=0.0)
            # partial only for long keywords, and only when text can contain the keyword
            if len(kw) >= PARTIAL_MIN_LEN and len(joined) >= len(kw):
                score = max(score, fuzz.partial_ratio(kw, joined))
            if score >= MIN_SCORE and (score, len(kw)) > best.get(label, (0.0, 0)):
                best[label] = (score, len(kw))
    ranked = sorted(best.items(), key=lambda kv: (-kv[1][0], -kv[1][1]))
    return [(label, round(score, 1)) for label, (score, _) in ranked]
