"""Stage 6: arithmetic validation, tax-mode detection and logged repairs. Never invents values."""
import json
from decimal import Decimal
from pathlib import Path

ITEM_TOL = Decimal("0.05")
TAX_TOL = Decimal("0.10")
SURE_CONF = 0.95  # a number read above this confidence is only repaired if the bill then balances
MAX_QTY = 99
DIGIT_POOL = 3
# OCR-confusable digits (both directions)
CONFUSABLE = {"0": "8", "8": "036", "1": "7", "7": "12", "3": "8", "5": "6", "6": "58", "2": "7"}
GST_RATES = tuple(Decimal(x) for x in ("0.25", "3", "5", "12", "18", "28"))
TOTAL_TYPES = {"total", "grand_total", "net_total"}  # total label names written by extract
ROUNDOFF_MAX = Decimal("1.00")
MERCHANT_LOW = 0.5  # merchant confidence below this needs review


def _d(x) -> Decimal | None:
    return None if x is None else Decimal(str(x))


def _val(f: dict | None) -> Decimal | None:
    return _d(f["value"]) if f else None


def _tol(total: Decimal) -> Decimal:
    return max(Decimal("0.05"), abs(total) * Decimal("0.001"))


def _is_int(x: Decimal) -> bool:
    return x == x.to_integral_value()


# ---------- bill state ----------

def _load(fields: dict) -> dict:
    """04_fields.json (values as strings) -> working state with Decimals."""
    items = []
    for it in fields.get("items", []):
        items.append({"name": it.get("name") or "", "row": it.get("row"), "kind": it["kind"],
                      "qty": _d(it.get("qty")), "rate": _d(it.get("rate")),
                      "amount": _d(it["amount"]), "status": it.get("status"),
                      "numbers": [_d(n) for n in it.get("numbers", [])],
                      "min_conf": it.get("min_conf")})
    taxes = [{"label": t.get("printed_label", t["label"]), "rate": _d(t.get("rate")),
              "amount": _d(t["amount"]), "row": t.get("row"), "min_conf": t.get("min_conf")}
             for t in fields.get("tax_lines", [])]
    tot = fields.get("total") or {}
    tx = fields.get("taxes") or {}
    table = tx.get("table")
    rate_rows = [{"rate": _d(r["rate"]), "base": _d(r["base"]), "tax": _d(r["amount"])}
                 for r in tx.get("rate_rows", [])]
    tax_total = next((_d(h["amount"]) for h in tx.get("label", []) if h["label"] == "TAX_TOTAL"), None)
    ttl = fields.get("tax_total_line") or {}
    if tax_total is None:
        tax_total = _d(ttl.get("value"))
    return {
        "items": items, "taxes": taxes,
        "total": _val(tot), "total_conf": tot.get("min_conf"), "total_rows": tot.get("rows"),
        "total_label": (tot.get("breakdown") or {}).get("label", 0),
        "total_name": tot.get("label"),
        "total_score": sum((tot.get("breakdown") or {}).values()),
        # the total's TOTAL/GRAND_TOTAL/NET label is a witness until a repair changes the total
        "label_ok": tot.get("label") in TOTAL_TYPES,
        "alternatives": [(_d(a["value"]), (a.get("breakdown") or {}).get("label", 0),
                          a.get("label"), a.get("score", 0)) for a in tot.get("alternatives", [])],
        "subtotal": _val(fields.get("subtotal")), "taxable": _val(fields.get("taxable")),
        "sub_rows": (fields.get("subtotal") or {}).get("rows"),
        "discount": _val(fields.get("discount")), "round_off": _val(fields.get("round_off")),
        "service": _val(fields.get("service_charge")),
        "table": {k: _d(table.get(k)) for k in ("base", "tax", "total")} if table else None,
        "payments": [(p["label"], _d(p["amount"])) for p in fields.get("payments", [])],
        "tax_source": fields.get("tax_source"),
        "rate_rows": rate_rows, "tax_total": tax_total,
        "tax_total_matches": bool(ttl.get("matches_lines")),
    }


def _sums(b: dict, use_sub: bool = False) -> tuple[Decimal, Decimal, Decimal, Decimal, list[Decimal]]:
    """(items or subtotal base, taxes, other charges, discount, round-off options)."""
    if b["items"] and not use_sub:
        s = sum((i["amount"] for i in b["items"]), Decimal(0))
        disc = Decimal(0) if any(i["kind"] == "discount" for i in b["items"]) \
            else abs(b["discount"] or 0)
    else:
        s, disc = b["subtotal"] or Decimal(0), abs(b["discount"] or 0)
    t = sum((x["amount"] for x in b["taxes"]), Decimal(0))
    ro = b["round_off"] or Decimal(0)
    return s, t, b["service"] or Decimal(0), disc, [ro, -ro] if ro else [Decimal(0)]


def _diffs(b: dict, use_sub: bool = False) -> dict[str, Decimal]:
    """computed - total for both tax modes (round-off sign taken as best fit)."""
    s, t, sc, disc, ros = _sums(b, use_sub)
    excl = min((s + t + sc - disc + r - b["total"] for r in ros), key=abs)
    incl = min((s - disc + r - b["total"] for r in ros), key=abs)
    return {"inclusive": incl, "exclusive": excl}


def _sub_chain_ok(b: dict) -> bool:
    """A labeled subtotal on its own row (not the total's row) can carry the balance."""
    return b["items"] and b["subtotal"] is not None and b["sub_rows"] != b["total_rows"]


def _mode(b: dict, items_only: bool = False) -> tuple[str, Decimal, str]:
    """(tax mode or 'unknown', diff, chain used: 'items' | 'subtotal')."""
    tol = _tol(b["total"])
    d = _diffs(b)
    mode = min(d, key=lambda k: abs(d[k]))  # ties (no taxes read) -> inclusive
    if abs(d[mode]) <= tol or items_only or not _sub_chain_ok(b):
        return (mode if abs(d[mode]) <= tol else "unknown"), d[mode], "items"
    d2 = _diffs(b, use_sub=True)
    m2 = min(d2, key=lambda k: abs(d2[k]))
    if abs(d2[m2]) <= tol:
        return m2, d2[m2], "subtotal"
    return "unknown", d[mode], "items"


def _balanced(b: dict, items_only: bool = False) -> bool:
    return _mode(b, items_only)[0] != "unknown"


def _with(b: dict, changes: list[tuple[int, str, Decimal]]) -> dict:
    """Copy of the bill with item fields changed: [(item index, field, new value)]."""
    items = [dict(i) for i in b["items"]]
    for k, f, v in changes:
        items[k][f] = v
    return {**b, "items": items}


# ---------- item checks ----------

def _weighed(it: dict) -> bool:
    if it["qty"] is not None:
        return not _is_int(it["qty"])
    return it["qty"] is None and it["status"] != "verified" and any(
        n.as_tuple().exponent < -2 and not _is_int(n) for n in it["numbers"])


def _item_candidates(it: dict) -> list[tuple[str, Decimal, Decimal]]:
    """Single-number fixes [(field, old, new)] that make qty*rate == amount."""
    q, r, amt = it["qty"], it["rate"], it["amount"]
    out: list[tuple[str, Decimal, Decimal]] = []
    if q is not None and r is not None:
        out.append(("amount", amt, q * r))
        for n in it["numbers"]:
            if n != q and abs(n * r - amt) <= ITEM_TOL:
                out.append(("qty", q, n))
            if n != r and abs(q * n - amt) <= ITEM_TOL:
                out.append(("rate", r, n))
    else:
        # unverified row: qty is a small integer, rate another number of the row
        nums = it["numbers"]
        for i, a in enumerate(nums):
            for j, c in enumerate(nums):
                if i != j and _is_int(a) and 1 <= a <= MAX_QTY and c > a:
                    out.append(("amount", amt, a * c))
    seen, uniq = set(), []
    for f, o, n in out:
        if (f, n) not in seen and n != o:
            seen.add((f, n))
            uniq.append((f, o, n))
    return uniq


def _repair_items(b: dict, repairs: list[dict], reasons: list[str]) -> dict:
    unique: list[tuple[int, str, Decimal, Decimal]] = []
    for k, it in enumerate(b["items"]):
        if it["kind"] != "item" or _weighed(it):
            continue
        q, r = it["qty"], it["rate"]
        if q is not None and r is not None and abs(q * r - it["amount"]) <= ITEM_TOL:
            continue
        if q is None and r is None and not it["numbers"]:
            continue  # nothing to check against
        cands = _item_candidates(it)
        if q is not None and r is not None:
            reasons.append(f"qty_rate_mismatch: row {it['row']}")
        if len(cands) == 1:
            unique.append((k, *cands[0]))
        elif len(cands) > 1:
            reasons.append(f"ambiguous_repair: row {it['row']} ({len(cands)} candidates)")

    def ch(us):
        return [(k, f, n) for k, f, _, n in us]

    def gap(bb: dict) -> Decimal:
        return abs(_mode(bb, items_only=True)[1])

    accepted: list[tuple[tuple, str]] = []
    if unique and _balanced(_with(b, ch(unique)), items_only=True):
        accepted = [(u, "balances_bill") for u in unique]
    elif unique:
        singles = [u for u in unique if _balanced(_with(b, ch([u])), items_only=True)]
        if len(singles) == 1:
            accepted = [(singles[0], "balances_bill")]
        done = [u for u, _ in accepted]
        # low-confidence reads may be repaired, but never in a way that worsens the balance
        accepted += [(u, "low_conf") for u in unique if u not in done
                     and (b["items"][u[0]]["min_conf"] or 0) < SURE_CONF
                     and gap(_with(b, ch([u]))) < gap(b)]
    for (k, f, old, new), why in accepted:
        it = b["items"][k]
        repairs.append({"item": it["name"] or f"row {it['row']}", "field": f, "old": old,
                        "new": new, "reason": f"qty*rate ({why})"})
    b = _with(b, ch([u for u, _ in accepted]))
    for (k, f, _, _), _ in accepted:
        it = b["items"][k]
        if f == "amount" and it["qty"] is None:  # the pair that produced the amount
            nums = it["numbers"]
            it["qty"], it["rate"] = next((a, c) for a in nums for c in nums
                                         if _is_int(a) and 1 <= a <= MAX_QTY and c > a
                                         and a * c == it["amount"])
    return b


def _row_local_repair(b: dict, repairs: list[dict], reasons: list[str]) -> dict:
    """Unverified item rows: try each other number of the row as the amount.
    Accepted only when exactly one (item, number) makes the bill balance."""
    hits = []
    for k, it in enumerate(b["items"]):
        if it["kind"] != "item" or _weighed(it):
            continue
        q, r = it["qty"], it["rate"]
        if q is not None and r is not None and abs(q * r - it["amount"]) <= ITEM_TOL:
            continue
        for n in dict.fromkeys(it["numbers"]):
            if n > 0 and n != it["amount"]:
                nb = _with(b, [(k, "amount", n)])
                if _balanced(nb):
                    hits.append((k, n, nb))
    if len(hits) > 1:
        reasons.append(f"ambiguous_row_local: {len(hits)} candidates")
    if len(hits) != 1:
        return b
    k, n, nb = hits[0]
    it = b["items"][k]
    repairs.append({"item": it["name"] or f"row {it['row']}", "field": "amount",
                    "old": it["amount"], "new": n, "reason": "row_local (unique fix)"})
    return nb


# ---------- tax checks ----------

def _check_taxes(b: dict, reasons: list[str], tax_checks: list[dict], support: list[str]) -> None:
    rr = b["rate_rows"]
    for r in rr:
        # tax = base*r/(100+r) when the base includes tax, base*r/100 when it does not
        incl = r["base"] * r["rate"] / (100 + r["rate"])
        excl = r["base"] * r["rate"] / 100
        mode = "inclusive" if abs(incl - r["tax"]) <= TAX_TOL else             "exclusive" if abs(excl - r["tax"]) <= TAX_TOL else None
        tax_checks.append({"rate": r["rate"], "base": r["base"], "tax": r["tax"], "mode": mode})
        if mode is None:
            reasons.append(f"rate_row_mismatch: {r['rate']}% of {r['base']} != {r['tax']}")
    # CGST/SGST at half rate must be half the tax of the matching rate row
    for t in b["taxes"]:
        if t["label"] in ("CGST", "SGST", "UTGST") and t["rate"] is not None:
            row = next((r for r in rr if r["rate"] == 2 * t["rate"]), None)
            if row and abs(row["tax"] / 2 - t["amount"]) > TAX_TOL:
                reasons.append(f"split_vs_rate_row: {t['label']} {t['amount']} != {row['tax']}/2")
    cg = [t["amount"] for t in b["taxes"] if t["label"] == "CGST"]
    sg = [t["amount"] for t in b["taxes"] if t["label"] == "SGST"]
    if cg and sg and abs(sum(cg) - sum(sg)) > ITEM_TOL:
        reasons.append(f"cgst_ne_sgst: {sum(cg)} != {sum(sg)}")
    t_sum = sum((t["amount"] for t in b["taxes"]), Decimal(0))
    if rr and b["tax_source"] != "rate_rows":
        rr_tax = sum((r["tax"] for r in rr), Decimal(0))
        if abs(t_sum - rr_tax) > TAX_TOL * len(rr):
            reasons.append(f"tax_vs_table: {t_sum} != {rr_tax}")
    items_sum = sum((i["amount"] for i in b["items"]), Decimal(0))
    if rr and b["items"] and abs(sum(r["base"] for r in rr) - items_sum) <= ITEM_TOL:
        support.append("rate_bases_eq_items")
    if b["taxable"] is not None and b["tax_total"] is not None and b["total"] is not None             and abs(b["taxable"] + b["tax_total"] - b["total"]) <= _tol(b["total"]):
        support.append("taxable_plus_tax_total_eq_total")
    if _tax_matches_total(b):
        support.append("tax_matches_total")


def _tax_matches_total(b: dict) -> bool:
    """Inclusive bill: the read taxes equal total*r/(100+r) for a GST rate r (within 0.10)."""
    t = sum((x["amount"] for x in b["taxes"]), Decimal(0))
    return b["total"] is not None and t > 0 and any(
        abs(b["total"] * r / (100 + r) - t) <= TAX_TOL for r in GST_RATES)


def _check_subtotal(b: dict, reasons: list[str]) -> None:
    sub, total = b["subtotal"], b["total"]
    if sub is None:
        return
    if b["items"]:
        s = sum((i["amount"] for i in b["items"]), Decimal(0))
        if abs(s - sub) > _tol(sub):
            reasons.append(f"items_vs_subtotal: {s} != {sub}")
    ro = b["round_off"] or Decimal(0)
    _, t, sc, disc, _ = _sums(b)
    if not any(abs(sub + x + r - total) <= _tol(total)
               for x in (Decimal(0), t + sc - disc) for r in (ro, -ro)):
        reasons.append(f"subtotal_chain: {sub} (+tax/round_off) != {total}")


# ---------- total consensus ----------

def _expected_totals(b: dict) -> list[Decimal]:
    s, t, sc, disc, ros = _sums(b)
    exp = [s + t + sc - disc + r for r in ros]
    if not t:  # a bare item sum is only evidence for the total when no tax was read
        exp += [s - disc + r for r in ros]
    if b["subtotal"] is not None and b["items"]:
        sub = b["subtotal"]
        # never 'total = subtotal' by itself: that would support any subtotal candidate
        exp += [sub + x for r in ros for x in (t + sc - disc + r, r - disc) if x]
    pays = dict(b["payments"])
    if "CASH" in pays and "CHANGE" in pays:
        exp.append(pays["CASH"] - abs(pays["CHANGE"]))
    elif "CASH" in pays:
        exp.append(pays["CASH"])
    if b["table"] and b["table"]["total"]:
        exp.append(b["table"]["total"])
    if b["taxable"] is not None and b["tax_total"] is not None:
        exp.append(b["taxable"] + b["tax_total"])
    return exp


def _consensus(b: dict, repairs: list[dict], reasons: list[str]) -> dict:
    exp = _expected_totals(b)

    def supported(v: Decimal) -> bool:
        return any(abs(v - e) <= _tol(v) for e in exp)

    if supported(b["total"]):
        return b
    # (score, is_cash) of every candidate: a cash alternative never beats an equal non-cash one
    others = [(b["total_score"], b["total_name"] == "cash")] + \
        [(s, n == "cash") for _, _, n, s in b["alternatives"][:2]]
    for alt, label, name, score in b["alternatives"][:2]:  # top 3 = best + 2 alternatives
        if name == "cash" and any(not c and s >= score for s, c in others):
            continue
        # never switch to the subtotal, or away from a better-labeled total
        if alt is not None and alt > 0 and alt != b["subtotal"] \
                and label >= b["total_label"] and supported(alt):
            repairs.append({"item": "total", "field": "total", "old": b["total"], "new": alt,
                            "reason": "arithmetic_consensus"})
            return {**b, "total": alt, "label_ok": name in TOTAL_TYPES}
    reasons.append("total_not_supported")
    return b


def _items_match_labeled_alt(b: dict) -> Decimal | None:
    """A labeled total candidate (not the current total) that the items already equal."""
    if not b["items"]:
        return None
    s, t, sc, disc, _ = _sums(b)
    for alt, _, name, _ in b["alternatives"]:
        if alt is None or name not in TOTAL_TYPES or alt == b["total"]:
            continue
        if any(abs(x - alt) <= _tol(alt) for x in (s - disc, s + t + sc - disc)):
            return alt
    return None


# ---------- total witnesses ----------

def _witnesses(b: dict) -> list[str]:
    """Evidence for the total that does not come from the items or a repair."""
    total, tol, w = b["total"], _tol(b["total"]), []
    if b["label_ok"]:
        w.append("total_label")
    pays = dict(b["payments"])
    if "CASH" in pays:
        paid = pays["CASH"] - abs(pays["CHANGE"]) if "CHANGE" in pays else pays["CASH"]
        if abs(paid - total) <= tol:
            w.append("cash")
    if b["table"] and b["table"]["total"] is not None and abs(b["table"]["total"] - total) <= tol:
        w.append("tax_table")
    if b["taxable"] is not None and b["tax_total"] is not None \
            and abs(b["taxable"] + b["tax_total"] - total) <= tol:
        w.append("taxable_plus_tax_total")
    if _tax_matches_total(b):
        w.append("tax_matches_total")
    if b.get("tax_total_matches"):
        w.append("tax_total_matches_lines")  # TOTAL GST row == CGST + SGST lines
    # subtotal + taxes +- round-off == total; never without a printed subtotal on its own row
    if b["subtotal"] is not None and b["sub_rows"] != b["total_rows"]:
        _, t, sc, disc, ros = _sums(b, use_sub=True)
        if any(x and abs(b["subtotal"] + x - total) <= tol
               for r in ros for x in (t + sc - disc + r, r - disc)):
            w.append("subtotal_chain")
    return w


def _total_weak(b: dict) -> bool:
    """No witness, or only a label on a value that is also an item amount."""
    w = _witnesses(b)
    return not w or (w == ["total_label"] and any(i["amount"] == b["total"] for i in b["items"]))


def _implied_roundoff(b: dict) -> tuple[str, Decimal] | None:
    """Labeled total within 1.00 of the item chain: the gap is an unprinted round-off."""
    if not b["label_ok"] or not b["items"]:
        return None
    d = _diffs(b)
    mode = min(d, key=lambda k: abs(d[k]))
    return (mode, d[mode]) if abs(d[mode]) < ROUNDOFF_MAX else None


def _derived(b: dict, mode: str) -> Decimal | None:
    """Total from the items: inclusive items - discount, exclusive items + taxes - discount."""
    if not b["items"]:
        return None
    s, t, sc, disc, _ = _sums(b)
    if mode == "unknown":
        mode = "inclusive" if not t or b["total"] is None else min(
            _diffs(b).items(), key=lambda kv: abs(kv[1]))[0]
    return s - disc if mode == "inclusive" else s + t + sc - disc


# ---------- single-digit repair ----------

def _digit_variants(v: Decimal) -> list[Decimal]:
    s = str(abs(v))
    out = []
    for i, ch in enumerate(s):
        for alt in CONFUSABLE.get(ch, ""):
            if i == 0 and alt == "0":
                continue
            out.append(Decimal(s[:i] + alt + s[i + 1:]) * (-1 if v < 0 else 1))
    return out


def _digit_repair(b: dict, repairs: list[dict], items_ok: bool = True) -> dict:
    pool = []  # (conf, kind, index)
    for k, it in enumerate(b["items"]):
        if items_ok and it["kind"] == "item" and it["status"] != "verified":
            pool.append((it["min_conf"] or 0, "item", k))
    for k, t in enumerate(b["taxes"]):
        pool.append((t["min_conf"] or 0, "tax", k))
    pool.append((b["total_conf"] or 0, "total", 0))
    pool.sort(key=lambda p: p[0])
    hits = []
    for _, kind, k in pool[:DIGIT_POOL]:
        old = (b["items"][k]["amount"] if kind == "item" else
               b["taxes"][k]["amount"] if kind == "tax" else b["total"])
        for new in _digit_variants(old):
            if kind == "item":
                nb = _with(b, [(k, "amount", new)])
            elif kind == "tax":
                nb = {**b, "taxes": [{**t, "amount": new} if j == k else t
                                     for j, t in enumerate(b["taxes"])]}
            else:
                nb = {**b, "total": new, "label_ok": False}
            if _balanced(nb):
                hits.append((kind, k, old, new, nb))
    if len(hits) != 1:
        return b
    kind, k, old, new, nb = hits[0]
    name = (b["items"][k]["name"] or f"row {b['items'][k]['row']}" if kind == "item" else
            b["taxes"][k]["label"] if kind == "tax" else "total")
    repairs.append({"item": name, "field": "amount" if kind != "total" else "total",
                    "old": old, "new": new, "reason": "single_digit (unique fix)"})
    return nb


# ---------- main ----------

def validate(fields: dict, out_dir: Path | None = None) -> dict:
    """Validate 04_fields -> status, tax_mode, repairs, reasons, confidence.
    Writes out_dir/05_validation.json when out_dir is given."""
    b = _load(fields)
    repairs: list[dict] = []
    reasons: list[str] = []
    total0 = b["total"]
    tax_checks: list[dict] = []
    support: list[str] = []
    witnesses: list[str] = []
    no_parts = not b["items"] and not b["taxes"] and b["subtotal"] is None
    if b["total"] is None or no_parts:
        reasons.append("no_total" if b["total"] is None else "no_items_tax_or_subtotal")
        mode = "unknown"
        res = {"status": "INSUFFICIENT_EVIDENCE", "tax_mode": "unknown", "balance_diff": None,
               "diffs": {}, "repairs": [], "reasons": reasons, "confidence": 0.0}
    else:
        def settled(bb: dict) -> bool:
            return _balanced(bb) or _implied_roundoff(bb) is not None

        # a weak total (no witness) never pulls item amounts towards it
        weak = not settled(b) and _total_weak(b)
        b0 = b
        if weak:
            reasons.append("total_unsupported")
        else:
            b = _repair_items(b, repairs, reasons)
            if not settled(b):
                b = _row_local_repair(b, repairs, reasons)
        if not settled(b):
            b = _consensus(b, repairs, reasons)
        if not settled(b):
            b = _digit_repair(b, repairs, items_ok=not weak)
        # an item edit is cancelled when the unedited items already equal another labeled total
        guarded = any(x["amount"] != y["amount"] for x, y in zip(b0["items"], b["items"])) \
            and _items_match_labeled_alt(b0) is not None
        if guarded:
            b, repairs = b0, []
            reasons.append("repair_contradicts_labeled_total")
        _check_taxes(b, reasons, tax_checks, support)
        _check_subtotal(b, reasons)
        mode, diff, chain = _mode(b)
        implied = _implied_roundoff(b) if mode == "unknown" else None
        if implied:
            mode, diff = implied
            reasons.append(f"implied_roundoff: {diff}")
        if mode == "unknown":
            status = "CHECK_REQUIRED"
            reasons.append(f"does_not_balance: diff {diff}")
        elif guarded:
            status = "CHECK_REQUIRED"
        else:
            if chain == "subtotal":
                reasons.append("balanced_via_subtotal_chain")
            status = "REPAIRED" if repairs else "VALID"
        witnesses = _witnesses(b)
        if status == "REPAIRED" and not witnesses:
            status = "CHECK_REQUIRED"
            reasons.append("repair_unsupported")
        # balanced only through the subtotal (items_sum != subtotal): a payment row must agree
        items_unverified = False
        if status in ("VALID", "REPAIRED") and chain == "subtotal":
            if "subtotal_chain" in witnesses and "cash" in witnesses:
                items_unverified = True
                reasons.append("items_unverified")
            else:
                status = "CHECK_REQUIRED"
                reasons.append("subtotal_chain_without_payment")
        confs = [i["min_conf"] for i in b["items"] if i["min_conf"] is not None]
        confs += [b["total_conf"]] if b["total_conf"] is not None else []
        conf = (0.85 ** len(repairs)) * (0.5 + 0.5 * min(confs, default=1.0))
        if items_unverified:
            conf *= 0.7
        if status == "CHECK_REQUIRED":
            conf *= 0.4
        res = {"status": status, "tax_mode": mode, "balance_diff": diff,
               "diffs": _diffs(b, chain == "subtotal"), "repairs": repairs, "reasons": reasons,
               "confidence": round(conf, 3)}
    res.update({
        "items_sum": sum((i["amount"] for i in b["items"]), Decimal(0)),
        "tax_sum": sum((t["amount"] for t in b["taxes"]), Decimal(0)),
        "tax_source": b["tax_source"], "subtotal": b["subtotal"],
        "tax_checks": tax_checks, "support": support,
        "total": b["total"], "total_extracted": total0, "witnesses": witnesses,
        "total_derived": None,
        "items": [{"name": i["name"], "row": i["row"], "qty": i["qty"], "rate": i["rate"],
                   "amount": i["amount"], "kind": i["kind"]} for i in b["items"]],
    })
    # never used to make a bill VALID: only a fallback value for review
    if res["status"] != "VALID" and (td := _derived(b, mode)) is not None:
        res["total_derived"] = td
        reasons.append("total_derived_from_items")
    review = []
    if res["status"] in ("CHECK_REQUIRED", "INSUFFICIENT_EVIDENCE"):
        review.append(res["status"].lower())
    merchant = fields.get("merchant") or {}
    if merchant.get("value") is None or (merchant.get("confidence") or 0) < MERCHANT_LOW:
        review.append("low_merchant_confidence")
    if res["repairs"]:
        review.append("repairs_applied")
    res["needs_review"], res["review_reasons"] = bool(review), review
    if out_dir is not None:
        (Path(out_dir) / "05_validation.json").write_text(
            json.dumps(res, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    return res
