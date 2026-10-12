"""Stage 4: normalization of OCR tokens into amounts, percents and dates."""
import re
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation

from rapidfuzz import fuzz

from src.keywords import match_labels

EDGE_CHARS = "[]{}|:* \t"
CURRENCY_RE = re.compile(r"^(₹|RS\.?|INR|RM|RP|\$)\s*", re.IGNORECASE)
DATE_LIKE_RE = re.compile(r"\d{1,4}[/.-]\d{1,2}[/.-]\d{2,4}")
OCR_DIGIT_FIX = str.maketrans({"O": "0", "o": "0", "I": "1", "l": "1", "|": "1", "S": "5", "B": "8"})
MAX_INT_DIGITS = 6
# row labels that make a one-decimal amount ('365.0') money
ONE_DEC_LABELS = {"TOTAL", "GRAND_TOTAL", "TAX_TOTAL", "CASH", "CHANGE", "SUBTOTAL", "DISCOUNT",
                  "ROUNDOFF"}
# row labels that make a trailing-dot amount ('10.') money
TRAILING_DOT_LABELS = {"CASH", "CHANGE"}
CENTS = Decimal("0.01")
MONTH_FUZZ_MIN = 75
# price with a tax-flag letter: '2.97X', '1.24 T' (the letter is a tax code, not text)
FLAGGED_RE = re.compile(r"(\d[\d,]*\.\d{2})\s?([A-Z])")

MONTHS = ["JANUARY", "FEBRUARY", "MARCH", "APRIL", "MAY", "JUNE", "JULY", "AUGUST",
          "SEPTEMBER", "OCTOBER", "NOVEMBER", "DECEMBER"]


@dataclass
class Amount:
    value: Decimal
    strength: str  # "strong" | "weak"
    raw: str
    flag: str | None = None  # tax-flag letter printed after the price


def is_currency_marker(text: str) -> bool:
    """True when the whole token is just a currency marker (e.g. 'Rs.' or '₹')."""
    t = text.strip(EDGE_CHARS)
    m = CURRENCY_RE.match(t)
    return bool(t) and m is not None and m.end() == len(t)


def _ungroup(s: str) -> str | None:
    """Resolve group/decimal commas. Returns a plain 'digits[.digits]' string or None."""
    if "," not in s:
        return s
    if "." in s:
        int_part, _, dec = s.rpartition(".")
        if "." in int_part or "," in dec or not re.fullmatch(r"\d{1,3}(,\d{2,3})+", int_part):
            return None
        return int_part.replace(",", "") + "." + dec
    if s.count(",") == 1:
        left, right = s.split(",")
        if not left:
            return None
        if len(right) == 3:
            return left + right  # 75,000 -> thousands
        if len(right) == 2:
            return left + "." + right  # 12,50 -> decimal comma
        return None
    if re.fullmatch(r"\d{1,3}(,\d{2,3})*,\d{3}", s):
        return s.replace(",", "")
    return None


def parse_amount(text: str, labels: set[str] | frozenset = frozenset()) -> Amount | None:
    """Parse a money token into an Amount, or None when it is not money.
    labels: the row's labels, used for one-decimal ('365.0') and trailing-dot ('10.') amounts."""
    raw = text
    s = text.strip(EDGE_CHARS)
    m = CURRENCY_RE.match(s)
    has_currency = m is not None
    if m:
        s = s[m.end():].strip(EDGE_CHARS)
    if not s or s.endswith("%") or ":" in s or DATE_LIKE_RE.search(s):
        return None

    neg = False
    if s.startswith("(") and s.endswith(")"):
        neg, s = True, s[1:-1].strip()
    if s.startswith("-"):
        neg, s = True, s[1:].strip()
    if s.endswith("-"):
        neg, s = True, s[:-1].strip()

    flag = None
    if m := FLAGGED_RE.fullmatch(s):
        s, flag = m.group(1), m.group(2)

    # OCR letter fixes only on tokens that are already mostly numeric
    if not re.fullmatch(r"[0-9.,]+", s):
        if sum(c.isdigit() for c in s) < 2:
            return None
        s = s.translate(OCR_DIGIT_FIX)
        if not re.fullmatch(r"[0-9.,]+", s):
            return None

    # '10.' / '$10.' -> 10.00 only with a currency marker or a CASH/CHANGE row
    trailing_dot = bool(re.fullmatch(r"\d+\.", s)) and (
        has_currency or bool(labels & TRAILING_DOT_LABELS))
    if trailing_dot:
        s = s[:-1]
    grouped = "," in s
    s = _ungroup(s)
    if s is None or not re.fullmatch(r"\d+(\.\d+)?", s):
        return None
    int_part, _, dec = s.partition(".")
    if not dec and not grouped and len(int_part) > MAX_INT_DIGITS:
        return None  # phone, HSN, bill no. (grouped numbers like 1,346,000 are money)
    try:
        value = Decimal(s)
    except InvalidOperation:
        return None
    one_dec = len(dec) == 1 and (has_currency or bool(labels & ONE_DEC_LABELS))
    if trailing_dot or one_dec:
        value = value.quantize(CENTS)
    strength = "strong" if len(dec) == 2 or has_currency or trailing_dot or one_dec else "weak"
    return Amount(-value if neg else value, strength, raw, flag)


GLUED_RE = re.compile(r"(?:\d+\.\d{2}){2,}")
GLUED_PART_RE = re.compile(r"\d+?\.\d{2}")


HSN_GLUED_RE = re.compile(r"(\d{8,})(\.\d{2})")
NON_MONEY_LABELS = {"BILL_NO", "GSTIN", "DATE", "HEADER"}
LABEL_GLUED_RE = re.compile(r"([A-Za-z][A-Za-z0-9]*)[.:=](\d[\d.,]*)")
CAMEL_RE = re.compile(r"(?<=[a-z])(?=[A-Z])")  # 'NetAmount' -> 'Net Amount'



def split_glued(token: str) -> list[str]:
    """'600.0012.03' -> ['600.00', '12.03']; '190520006.00' -> ['HSN:19052000', '6.00'];
    anything else -> [token]. The 'HSN:' prefix makes parse_amount reject the code."""
    t = token.strip(EDGE_CHARS)
    if GLUED_RE.fullmatch(t):
        return GLUED_PART_RE.findall(t)
    m = HSN_GLUED_RE.fullmatch(t)
    if m:
        digits = m.group(1)
        for n in (8, 7, 6):  # longest HSN that leaves a clean amount
            rest = digits[n:]
            if rest and (rest == "0" or not rest.startswith("0")) and len(rest) <= MAX_INT_DIGITS:
                return ["HSN:" + digits[:n], rest + m.group(2)]
    return [token]


def split_label_glued(token: str) -> list[str]:
    """'Tota1.1591600' -> ['Tota1', '1591600']; 'NetAmount:3814.00' -> ['Net Amount', '3814.00']:
    a label (3+ letters) glued to a number by '.', ':' or '='."""
    t = token.strip(EDGE_CHARS)
    m = LABEL_GLUED_RE.fullmatch(t)
    if m and sum(c.isalpha() for c in m.group(1)) >= 3 and not CURRENCY_RE.match(t):
        label = CAMEL_RE.sub(" ", m.group(1))
        # only money labels (total, tax, cash...), never 'Inv:123' or 'Date.12'
        if {l for l, _ in match_labels(label)} - NON_MONEY_LABELS:
            num = m.group(2)
            if num.isdigit() and len(num) > MAX_INT_DIGITS:
                num = f"{int(num):,}"  # the label says money: group it so parse_amount accepts it
            return [label, num]
    return [token]


def label_text(text: str) -> str:
    """Row text with label:number glue split, for label matching ('NetAmount:3814.00')."""
    return " ".join(p for t in text.split() for p in split_label_glued(t))


QTY_RE = re.compile(r"\d+(?:\.\d{1,3})?")
# item note lines: 'HSN 04069000', 'SAC: 996331', 'GST 5%' (not CGST/SGST); OCR may glue
# them into one token ('HSN21069099GST5%') and read the S of GST as 5 or 8
HSN_RE = re.compile(r"(?<![A-Z])(?:HSN|SAC)\W{0,3}(\d{6,8})(?!\d)", re.IGNORECASE)
GST_NOTE_RE = re.compile(r"(?<![A-Z])G[S58$]T\s*@?\s*(\d{1,2}(?:\.\d{1,2})?)\s*%", re.IGNORECASE)


def parse_qty(text: str) -> Decimal | None:
    """Quantity token: a whole number or 1-3 decimals ('0.500', '2.000', '1.25') -> plain value."""
    s = text.strip(EDGE_CHARS)
    return Decimal(s) if QTY_RE.fullmatch(s) else None


def parse_percent(text: str) -> Decimal | None:
    """'18.00 %', '@18%', '9.00%' -> Decimal."""
    m = re.search(r"(\d+(?:\.\d+)?)\s*%", text)
    return Decimal(m.group(1)) if m else None


def _num(s: str) -> int:
    return int(s.replace("O", "0").replace("o", "0"))


def _month(word: str) -> int | None:
    w = word.upper().rstrip(".")
    if len(w) < 2:
        return None
    if len(w) >= 3:
        for i, full in enumerate(MONTHS, 1):
            if full.startswith(w) or (w == "SEPT" and i == 9):
                return i
    # fuzzy: best of full name and 3-letter name per month; unique best only ('Mr' -> Mar)
    scores = [max(fuzz.ratio(w, full), fuzz.ratio(w, full[:3])) for full in MONTHS]
    best = max(scores)
    if best < MONTH_FUZZ_MIN or scores.count(best) > 1:
        return None
    return scores.index(best) + 1


def _year(y: int, digits: int) -> int:
    if digits == 2:
        y = 2000 + y if 2000 + y <= date.today().year + 1 else 1900 + y
    return y


def _make(y: int, m: int, d: int, fmt: str, day_first: bool) -> dict | None:
    if not 1995 <= y <= date.today().year + 1:
        return None
    try:
        iso = date(y, m, d).isoformat()
    except ValueError:
        return None
    amb = day_first and d <= 12 and m <= 12 and d != m
    return {"iso": iso, "fmt": fmt, "ambiguous": amb}


D = r"[0-9Oo]"
YMD_RE = re.compile(rf"(?<!\d)([12]{D}{{3}})-({D}{{1,2}})-({D}{{1,2}})(?!\d)")
DMY_RE = re.compile(rf"(?<!\d)({D}{{1,2}})([/.-])({D}{{1,2}})\2({D}{{4}}|{D}{{2}})(?!\d)")
D_MON_Y_RE = re.compile(r"(?<!\d)(\d{1,2})([\s-]+)([A-Za-z]{2,9})\.?[\s,-]+(\d{4}|\d{2})(?!\d)")
MON_D_Y_RE = re.compile(r"\b([A-Za-z]{3,9})\.?\s+(\d{1,2}),?\s+(\d{4})(?!\d)")


def parse_date(text: str) -> list[dict]:
    """Find dates inside text -> [{iso, fmt, ambiguous}]. Day-first by default."""
    out: list[dict] = []
    for m in YMD_RE.finditer(text):
        out.append(_make(_num(m.group(1)), _num(m.group(2)), _num(m.group(3)),
                         "yyyy-mm-dd", day_first=False))
    for m in DMY_RE.finditer(text):
        sep, ylen = m.group(2), len(m.group(4))
        y =_year(_num(m.group(4)), ylen)
        fmt = sep.join(["dd", "mm", "yyyy" if ylen == 4 else "yy"])
        out.append(_make(y, _num(m.group(3)), _num(m.group(1)), fmt, day_first=True))
    for m in D_MON_Y_RE.finditer(text):
        mon = _month(m.group(3))
        if mon is None:
            continue
        ylen = len(m.group(4))
        fmt = "dd-Mon-yy" if "-" in m.group(2) else f"dd Mon {'yyyy' if ylen == 4 else 'yy'}"
        out.append(_make(_year(int(m.group(4)), ylen), mon, int(m.group(1)), fmt, day_first=False))
    for m in MON_D_Y_RE.finditer(text):
        mon = _month(m.group(1))
        if mon is None:
            continue
        out.append(_make(int(m.group(3)), mon, int(m.group(2)), "Mon dd, yyyy", day_first=False))

    seen: set[str] = set()
    result = []
    for d in out:
        if d and d["iso"] not in seen:
            seen.add(d["iso"])
            result.append(d)
    return result
