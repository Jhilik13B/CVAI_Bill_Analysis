"""Streamlit demo: bill image -> stages 1-6 + anomaly flags, shown with evidence."""
import html
import json
import sys
import tempfile
from decimal import Decimal, InvalidOperation
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src import anomaly, extract, ocr, preprocess, rows, validate  # noqa: E402
from src.pipeline import _overlay  # noqa: E402

DEMO_DIR = ROOT / "data" / "desktop" / "images"
DEMO_NUMS = range(46, 49)  # demo picks from the image folder (numeric suffix)
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
NOT_FOUND = "not found"
PILL = {"VALID": ("pill-valid", "✓"), "REPAIRED": ("pill-repaired", "🔧"),
        "CHECK_REQUIRED": ("pill-check", "⚠"), "INSUFFICIENT_EVIDENCE": ("pill-insufficient", "?")}
CSS = """<style>
@import url('https://fonts.googleapis.com/css2?family=Oswald:wght@400;600;700&display=swap');
:root { --lime:#b6f542; --orange:#ff9f1c; --red:#ff5a5f; --amber:#ffbf00; --card:#16161a;
        --line:#23232a; --muted:#8a8a93; --head:'Oswald','Arial Narrow',sans-serif;
        --white:#f5f5f5; --accent:var(--lime); }
.stApp { background:#0b0b0d; }
.card, [class*="st-key-card"] { background:var(--card); border:1px solid var(--line);
        border-top:2px solid var(--accent); border-radius:24px; padding:20px; margin-bottom:12px;
        transition:transform 150ms; }
.card:hover, [class*="st-key-card"]:hover { transform:translateY(-2px); }
/* accent alternates by column position; keyed result cards get fixed accents */
[data-testid="stColumn"]:nth-child(3n+1) .card { --accent:var(--lime); }
[data-testid="stColumn"]:nth-child(3n+2) .card { --accent:var(--orange); }
[data-testid="stColumn"]:nth-child(3n) .card { --accent:var(--white); }
.st-key-card-items, .st-key-card-repairs { --accent:var(--orange); }
.st-key-card-arith { --accent:var(--white); } .st-key-card-tax { --accent:var(--lime); }
.sec { font-family:var(--head); text-transform:uppercase; letter-spacing:.5px;
       font-size:18px; margin-bottom:10px; color:#f5f5f5; }
.stat-label { color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.5px; }
.stat-value { font-family:var(--head); font-size:40px; font-weight:700; line-height:1.1;
              color:#f5f5f5; overflow-wrap:anywhere; }
.hero { font-family:var(--head); font-size:72px; font-weight:700; text-transform:uppercase;
        letter-spacing:.5px; line-height:1; margin:0; }
.hero .lime { color:var(--lime); }
.hero-wrap { display:flex; flex-wrap:wrap; align-items:center; gap:16px; margin-bottom:8px; }
.logo { display:flex; gap:6px; }
.logo-dot { width:14px; height:14px; border-radius:50%; display:inline-block; }
.sub { color:var(--muted); text-transform:uppercase; letter-spacing:1px; font-size:13px; margin-bottom:12px; }
.st-key-pillbar { background:var(--card); border:1px dashed rgba(182,245,66,.4); border-radius:28px;
                  padding:12px 28px; transition:box-shadow 150ms; }
.st-key-pillbar:hover { box-shadow:0 0 24px rgba(182,245,66,.18); }
.step-num { font-family:var(--head); font-size:48px; font-weight:700; line-height:1; margin-bottom:6px; }
.c-lime { color:var(--lime); } .c-orange { color:var(--orange); } .c-white { color:var(--white); }
.guide-row { margin:10px 0; } .guide-row .stat-label { margin-top:4px; text-transform:none; }
.about { display:inline-block; border-bottom:2px solid var(--lime); padding-bottom:4px; }
.pill { display:inline-block; border-radius:999px; padding:4px 14px; font-weight:700;
        font-family:var(--head); letter-spacing:.5px; color:#0b0b0d; }
.pill-valid { background:var(--lime); }
.pill-repaired { background:var(--orange); }
.pill-check { background:var(--amber); box-shadow:0 0 0 3px #0b0b0d, 0 0 0 5px var(--amber); }
.pill-insufficient { background:var(--red); }
.status-pill-check { border:2px solid var(--amber); }
.status-pill-insufficient { border:2px solid var(--red); }
.mark-ok { background:var(--lime); } .mark-bad { background:var(--red); }
.chip { display:inline-block; border:1px solid var(--orange); color:var(--orange); border-radius:999px;
        padding:2px 10px; margin:2px 4px 2px 0; font-size:12px; }
.dot { display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:4px; }
.dot.off { background:#2a2a30; } .dot.on.lime { background:var(--lime); }
.dot.on.orange { background:var(--orange); }
.bar-row { display:flex; align-items:center; gap:8px; margin:4px 0; font-size:12px; }
.bar-row .lbl { width:120px; color:var(--muted); } .bar { height:12px; border-radius:999px; }
.bar.lime { background:var(--lime); } .bar.orange { background:var(--orange); }
table.tbl { width:100%; border-collapse:collapse; font-size:13px; }
table.tbl th { color:var(--muted); text-align:left; font-weight:400; text-transform:uppercase;
               font-size:11px; border-bottom:1px solid var(--line); padding:6px; }
table.tbl td { border-bottom:1px solid var(--line); padding:6px; }
.tbl-wrap { overflow-x:auto; }
@media (max-width:640px) { .stat-value { font-size:28px; } .hero { font-size:44px; }
                           .step-num { font-size:36px; } }
</style>"""
DISCLAIMER = ("This system detects arithmetic inconsistency (items, taxes and total that do not "
              "add up). It cannot tell tampering apart from OCR error: a flag means "
              "\"check this bill\", not \"this bill is forged\".")
STEPS = [("01", "UPLOAD", "Drop a bill photo", "lime"), ("02", "READ", "OCR finds every line", "orange"),
         ("03", "CHECK", "Items + tax must equal total", "white"),
         ("04", "FLAG", "Anything unproven is marked for review", "lime")]
STATUS_GUIDE = {"VALID": "Items, taxes and total balance as read.",
                "REPAIRED": "Balances only after fixing misread value(s).",
                "CHECK_REQUIRED": "Items, taxes and total do not add up: check this bill.",
                "INSUFFICIENT_EVIDENCE": "Total or line items missing: arithmetic cannot be checked."}
ACCENTS = {"lime", "orange", "white"}


def _fmt(x) -> str:
    """Display value: 'not found' for None; money stays a Decimal string (never float)."""
    if x is None or x == "":
        return NOT_FOUND
    if isinstance(x, dict):  # extracted field {"value": ...}
        return _fmt(x.get("value"))
    return str(x)


def explain(val: dict, anom: dict) -> str:
    """One plain-language line: what balanced, what did not, what needs a human check."""
    st_, mode, diff = val.get("status"), val.get("tax_mode"), val.get("balance_diff")
    reps = val.get("repairs") or []
    if st_ == "VALID":
        parts = [f"Items, taxes and total balance ({mode} tax, difference {_fmt(diff)})."]
    elif st_ == "REPAIRED":
        parts = [f"The bill balances only after {len(reps)} repair(s) of misread value(s)."]
    elif st_ == "CHECK_REQUIRED":
        parts = [f"Items, taxes and total do not balance (difference {_fmt(diff)})."]
    else:
        parts = ["Not enough was read (total or line items missing) to check the arithmetic."]
    checks = [r.replace("_", " ") for r in val.get("review_reasons") or []]
    checks += [f.replace("_", " ") for f in anom.get("flags") or []
               if f.replace("_", " ") not in checks]
    parts.append("Human check: " + ", ".join(checks) + "." if checks else "No human check needed.")
    return " ".join(parts)


@st.cache_resource(show_spinner="Loading OCR model...")
def _ocr_engine():
    """Load PaddleOCR once per server process."""
    return ocr._engine()


@st.cache_data(show_spinner="Reading the bill...")
def run_pipeline(data: bytes, suffix: str) -> dict:
    """Stages 1-6 + anomaly on image bytes; nothing is written to outputs/."""
    _ocr_engine()
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / f"bill{suffix}"
        p.write_bytes(data)
        pre = preprocess.run(p)
    res = ocr.two_pass(pre["orig"], pre["enh"])
    fields = extract.extract(rows.build_rows(res["merged"]))
    val = validate.validate(fields)
    anom = anomaly.score(fields, val)
    return {"overlay": _overlay(pre["orig"], res["merged"]), "fields": fields,
            "validation": val, "anomaly": anom}


def _demo_files() -> list[Path]:
    if not DEMO_DIR.exists():
        return []
    return sorted(p for p in DEMO_DIR.iterdir() if p.suffix.lower() in IMG_EXT
                  and p.stem.rsplit("_", 1)[-1].isdigit()
                  and int(p.stem.rsplit("_", 1)[-1]) in DEMO_NUMS)


def _dec(x) -> Decimal | None:
    if isinstance(x, dict):
        x = x.get("value")
    return None if x is None or x == "" else Decimal(str(x))


def _mark(ok: bool | None) -> str:
    return NOT_FOUND if ok is None else ("✓" if ok else "✗")


def arithmetic_panel(fields: dict, val: dict) -> dict:
    """Item check (items + taxes - discount +- round-off vs printed total) and the
    subtotal chain (subtotal + taxes +- round-off vs total). Decimal only; None -> 'not found'."""
    items = val.get("items") or []
    items_sum = _dec(val.get("items_sum")) if items else None
    sub, taxes = _dec(fields.get("subtotal")), _dec(val.get("tax_sum")) or Decimal(0)
    disc = None if any(i.get("kind") == "discount" for i in items) else _dec(fields.get("discount"))
    ro = _dec(fields.get("round_off"))
    printed = _dec(fields.get("total"))
    tol = max(Decimal("0.05"), abs(printed or 0) * Decimal("0.001"))
    add_tax = taxes if val.get("tax_mode") != "inclusive" else Decimal(0)

    def best(base: Decimal) -> Decimal:  # round-off sign that fits the printed total best
        opts = [base + r for r in ({ro, -ro} if ro else {Decimal(0)})]
        return min(opts, key=lambda v: abs(v - printed)) if printed is not None else opts[0]

    recomputed = best(items_sum + add_tax - abs(disc or 0)) if items_sum is not None else None
    diff = recomputed - printed if recomputed is not None and printed is not None else None
    item_ok = None if diff is None else abs(diff) <= tol
    chain = best(sub + taxes) if sub is not None else None
    chain_diff = chain - printed if chain is not None and printed is not None else None
    chain_ok = None if chain_diff is None else abs(chain_diff) <= tol

    notes = []
    if items_sum is not None and sub is not None and abs(items_sum - sub) > tol:
        notes.append("item extraction incomplete, item check inconclusive")
    if item_ok or chain_ok:
        verdict = "balanced" if item_ok else "balanced via subtotal chain"
    elif chain_ok is False or (chain_ok is None and item_ok is False):
        verdict = "does not balance"
    else:
        verdict = "inconclusive"
    table = [("items sum", items_sum), ("printed subtotal", sub), ("taxes sum", taxes),
             ("discount", disc), ("round-off", ro), ("recomputed total", recomputed),
             ("printed total", printed), ("difference", diff)]
    return {"table": [{"line": k, "value": _fmt(v)} for k, v in table]
            + [{"line": "match", "value": _mark(item_ok)}],
            "chain": {"subtotal": _fmt(sub), "taxes": _fmt(taxes), "round_off": _fmt(ro),
                      "computed": _fmt(chain), "total": _fmt(printed),
                      "difference": _fmt(chain_diff), "match": _mark(chain_ok)},
            "notes": notes, "verdict": verdict}


def _items_table(fields: dict, val: dict) -> list[dict]:
    by_row = {i["row"]: i for i in fields.get("items", [])}
    out = []
    for i in val.get("items", []):
        f = by_row.get(i["row"], {})
        out.append({"name": i["name"] or NOT_FOUND, "kind": i["kind"], "qty": _fmt(i["qty"]),
                    "rate": _fmt(i["rate"]), "amount": _fmt(i["amount"]),
                    "verified": f.get("status") == "verified", "hsn": _fmt(f.get("hsn")),
                    "gst_rate": _fmt(f.get("gst_rate")), "weighed": bool(f.get("weighed")),
                    "discount": _fmt(f.get("discount"))})
    return out


def _rates_line(fields: dict) -> str | None:
    """'GST rates seen: 5% (item notes; prices inclusive)' or None."""
    rates = fields.get("tax_rates_seen") or []
    if not rates:
        return None
    shown = ", ".join(f"{Decimal(str(r)).normalize():f}%" for r in rates)
    return f"GST rates seen: {shown} (item notes; prices inclusive)"


def _esc(x) -> str:
    return html.escape(str(x))


def _card(label: str, value, extra: str = "") -> str:
    """Stat card: small grey label + big value (escaped); 'not found' passes through unchanged."""
    return (f'<div class="card"><div class="stat-label">{_esc(label)}</div>'
            f'<div class="stat-value">{_esc(value)}</div>{extra}</div>')


def _pill(status) -> str:
    """Rounded status pill; colour class by status, unknown -> red (insufficient)."""
    cls, icon = PILL.get(status, ("pill-insufficient", "?"))
    return f'<span class="pill {cls}">{icon} {_esc(status)}</span>'


def _dots(value, color: str = "lime") -> str:
    """10 dots, round(value*10) filled; '' when value is not a number."""
    try:
        n = int(round(Decimal(str(value)) * 10))
    except (InvalidOperation, TypeError, ValueError):
        return ""
    n = max(0, min(10, n))
    return ('<div>' + f'<span class="dot on {color}"></span>' * n
            + '<span class="dot off"></span>' * (10 - n) + '</div>')


def _mark_pill(mark: str) -> str:
    """✓ -> lime pill, ✗ -> red pill, anything else (e.g. 'not found') as plain text."""
    if mark == "✓":
        return '<span class="pill mark-ok">✓</span>'
    if mark == "✗":
        return '<span class="pill mark-bad">✗</span>'
    return _esc(mark)


def _bars(recomputed: str, printed: str) -> str:
    """Two bars (recomputed lime, printed orange) scaled to the larger; '' if either is missing."""
    try:
        a, b = abs(Decimal(recomputed)), abs(Decimal(printed))
    except (InvalidOperation, TypeError, ValueError):
        return ""
    top = max(a, b)
    if top == 0:
        return ""
    row = ('<div class="bar-row"><span class="lbl">{l}</span>'
           '<div class="bar {c}" style="width:{w:.1f}%"></div><span>{v}</span></div>')
    # float only for the CSS width; displayed values stay the original strings
    return (row.format(l="recomputed total", c="lime", w=float(a / top) * 70, v=_esc(recomputed))
            + row.format(l="printed total", c="orange", w=float(b / top) * 70, v=_esc(printed)))


def _chips(xs) -> str:
    return "".join(f'<span class="chip">{_esc(x)}</span>' for x in xs) if xs else "none"


def _table(rows_: list[dict], bool_cols: frozenset | set = frozenset()) -> str:
    """Escaped HTML table with every column; bool columns as lime/grey dots."""
    if not rows_:
        return NOT_FOUND
    cols = list(rows_[0])
    head = "".join(f"<th>{_esc(c)}</th>" for c in cols)
    body = "".join(
        "<tr>" + "".join(
            f'<td><span class="dot {"on lime" if r[c] else "off"}"></span></td>' if c in bool_cols
            else f"<td>{_esc(r[c])}</td>" for c in cols) + "</tr>" for r in rows_)
    return f'<div class="tbl-wrap"><table class="tbl"><tr>{head}</tr>{body}</table></div>'


def _hero() -> str:
    """Logo mark (3 dots) + title with CHECK in lime + subtitle."""
    dots = "".join(f'<span class="logo-dot" style="background:var(--{c})"></span>'
                   for c in ("lime", "orange", "white"))
    return (f'<div class="hero-wrap"><div class="logo">{dots}</div><div>'
            '<p class="hero">BILL <span class="lime">CHECK</span></p>'
            '<div class="sub">arithmetic verification for any bill</div></div></div>')


def _step_card(num: str, title: str, text: str, color: str) -> str:
    """'How it works' card: big coloured number, title, grey text (escaped); unknown colour -> lime."""
    color = color if color in ACCENTS else "lime"
    return (f'<div class="card"><div class="step-num c-{color}">{_esc(num)}</div>'
            f'<div class="sec">{_esc(title)}</div><div class="stat-label">{_esc(text)}</div></div>')


def _status_guide() -> str:
    """Static card: each status pill with one grey line on what it means."""
    rows_ = "".join(f'<div class="guide-row">{_pill(s)}<div class="stat-label">{_esc(t)}</div></div>'
                    for s, t in STATUS_GUIDE.items())
    return f'<div class="card"><div class="sec">Status guide</div>{rows_}</div>'


def _meter_preview() -> str:
    """Decorative 10-dot meter (no numbers, not a result)."""
    dots = "".join(f'<span class="dot {"on lime" if i % 2 == 0 else "off"}"></span>' for i in range(10))
    return (f'<div class="card"><div class="sec">Confidence meter</div><div>{dots}</div>'
            '<div class="stat-label" style="margin-top:8px">Preview: appears with each result.</div></div>')


def _html(s: str) -> None:
    st.markdown(s, unsafe_allow_html=True)


def _sec(title: str) -> None:
    _html(f'<div class="sec">{_esc(title)}</div>')


def _show(r: dict) -> None:
    f, v, a = r["fields"], r["validation"], r["anomaly"]
    cls = PILL.get(v["status"], ("pill-insufficient",))[0]
    review = ""
    if v["needs_review"]:
        review = ('<div style="margin-top:8px"><span class="stat-label">⚑ Needs review</span><br>'
                  + _chips([x.replace("_", " ") for x in v["review_reasons"]]) + "</div>")
    _html(f'<div class="card status-{cls}">{_pill(v["status"])}'
          f'<p style="margin:12px 0 0">{_esc(explain(v, a))}</p>{review}</div>')

    left, right = st.columns([2, 3])
    with left:
        with st.container(key="card-img"):
            st.image(r["overlay"], channels="BGR",
                     caption="Merged OCR boxes (green = original, blue = enhanced)")
    with right:
        stats = [("Merchant", _fmt(f.get("merchant")), ""), ("Date", _fmt(f.get("date")), ""),
                 ("GSTIN", _fmt(f.get("gstin")), ""), ("Subtotal", _fmt(v.get("subtotal")), ""),
                 ("Round-off", _fmt(f.get("round_off")), ""), ("Total", _fmt(v.get("total")), ""),
                 ("Tax mode", _fmt(v.get("tax_mode")), ""), ("Tax sum", _fmt(v.get("tax_sum")), ""),
                 ("Confidence", _fmt(v.get("confidence")), _dots(v.get("confidence"), "lime")),
                 ("Risk score", _fmt(a.get("risk")), _dots(a.get("risk"), "orange"))]
        for i in range(0, len(stats), 3):
            for col, (label, value, extra) in zip(st.columns(3), stats[i:i + 3]):
                with col:
                    _html(_card(label, value, extra))

    with st.container(key="card-items"):
        _sec("Items")
        _html(_table(_items_table(f, v), {"verified", "weighed"}))
        if line := _rates_line(f):
            st.caption(line)

    with st.container(key="card-arith"):
        _sec("Arithmetic check")
        p = arithmetic_panel(f, v)
        vals = {t["line"]: t["value"] for t in p["table"]}
        _html(_table([t for t in p["table"] if t["line"] != "match"]))
        _html(f'<p style="margin-top:8px">match {_mark_pill(vals["match"])}</p>')
        _html(_bars(vals["recomputed total"], vals["printed total"]))
        c = p["chain"]
        _html(_esc(f"Subtotal chain: {c['subtotal']} + taxes {c['taxes']} ± round-off "
                   f"{c['round_off']} = {c['computed']} vs total {c['total']} "
                   f"(difference {c['difference']}) ") + _mark_pill(c["match"]))
        for n in p["notes"]:
            st.caption(n)
        st.caption(f"Verdict: {p['verdict']}")

    t_col, r_col = st.columns(2)
    with t_col, st.container(key="card-tax"):
        _sec("Tax lines")
        taxes = [{"label": t.get("label"), "rate %": _fmt(t.get("rate")), "amount": _fmt(t.get("amount"))}
                 for t in f.get("tax_lines", [])]
        _html(_table(taxes) if taxes else _esc(f"Taxes: {NOT_FOUND}"))
    with r_col, st.container(key="card-repairs"):
        _sec("Repairs")
        if v["repairs"]:
            for x in v["repairs"]:
                _html(f"<div>• <b>{_esc(x['item'])}</b> {_esc(x['field'])}: "
                      f"<code>{_esc(_fmt(x['old']))}</code> -> <code>{_esc(_fmt(x['new']))}</code> "
                      f"({_esc(x['reason'])})</div>")
        else:
            st.caption("No repairs.")
    rs_col, fl_col = st.columns(2)
    with rs_col:
        _html('<div class="card"><div class="sec">Reasons</div>' + _chips(v["reasons"]) + "</div>")
    with fl_col:
        _html('<div class="card"><div class="sec">Anomaly flags</div>' + _chips(a["flags"]) + "</div>")

    payload = json.dumps({k: r[k] for k in ("fields", "validation", "anomaly")},
                         indent=1, ensure_ascii=False, default=str)
    with st.sidebar:
        st.download_button("Download JSON result", payload, file_name="bill_result.json",
                           mime="application/json", icon=":material/download:")


def main() -> None:
    st.set_page_config(page_title="CVAI bill check", page_icon=":material/receipt_long:", layout="wide")
    _html(CSS)
    _html(_hero())
    with st.sidebar:
        _html(f'<div class="card"><div class="sec about">ABOUT</div><div>ⓘ {_esc(DISCLAIMER)}</div></div>')

    with st.container(key="pillbar"):
        c_up, c_demo = st.columns(2)  # stacks on narrow width
        with c_up:
            up = st.file_uploader("Upload a bill image", type=[e.lstrip(".") for e in IMG_EXT])
        demos = _demo_files()
        pick = None
        if up is None and demos:
            with c_demo:
                pick = st.selectbox("...or pick a demo image", demos, format_func=lambda p: p.name,
                                    index=None, placeholder="Choose a demo image")
    if up is not None:
        data, suffix = up.getvalue(), Path(up.name).suffix.lower()
    elif pick is not None:
        data, suffix = pick.read_bytes(), pick.suffix.lower()
    else:  # empty state: static how-it-works, status guide, meter preview
        for col, s in zip(st.columns(4), STEPS):
            with col:
                _html(_step_card(*s))
        g_col, m_col = st.columns([3, 2])
        with g_col:
            _html(_status_guide())
        with m_col:
            _html(_meter_preview())
        return
    _show(run_pipeline(data, suffix))


if __name__ == "__main__":
    main()
