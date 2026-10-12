"""Stage 7: rule-based anomaly flags + risk score; optional IsolationForest secondary score."""
import json
from decimal import Decimal
from pathlib import Path

WEIGHTS: dict[str, float] = {
    "does_not_balance": 0.5,
    "tax_rate_mismatch": 0.3,
    "cgst_ne_sgst": 0.3,
    "total_not_supported": 0.3,
    "repaired_value": 0.25,
    "duplicate_item_rows": 0.15,
    "round_off_too_large": 0.3,
}
TAX_TOL = Decimal("0.10")
ROUNDOFF_MAX = Decimal("1.00")
SPLIT = {"CGST", "SGST", "UTGST"}  # half-rate taxes: the combined rate is twice theirs
NO_DIFF = 999.0  # balance_diff feature when the bill could not be balanced at all
SEED = 11


def _d(x) -> Decimal | None:
    return None if x is None else Decimal(str(x))


def _val(f: dict | None) -> Decimal | None:
    return _d(f.get("value")) if f else None


def _tax_rate_mismatch(fields: dict) -> bool:
    """A tax line with a printed rate whose amount fits neither base*r/100 nor base*r/(100+R)."""
    items = sum((_d(i["amount"]) for i in fields.get("items", []) if i["kind"] == "item"), Decimal(0))
    sub = _val(fields.get("subtotal"))
    base = _val(fields.get("taxable"))
    if base is None and sub is not None:
        base = sub + (_val(fields.get("service_charge")) or 0)
    if base is None:
        base = items
    if not base:
        return False
    for t in fields.get("tax_lines", []):
        r, amt = _d(t.get("rate")), _d(t["amount"])
        if r is None or r <= 0:
            continue
        full = 2 * r if t.get("label") in SPLIT else r
        excl, incl = base * r / 100, base * r / (100 + full)
        if abs(excl - amt) > TAX_TOL and abs(incl - amt) > TAX_TOL:
            return True
    return False


def _duplicates(fields: dict) -> bool:
    seen = set()
    for i in fields.get("items", []):
        if i["kind"] != "item":
            continue
        key = (" ".join((i.get("name") or "").lower().split()), _d(i["amount"]))
        if key[0] and key in seen:
            return True
        seen.add(key)
    return False


def rule_flags(fields: dict, val: dict) -> list[str]:
    reasons = val.get("reasons", [])

    def has(*prefixes: str) -> bool:
        return any(r.startswith(prefixes) for r in reasons)

    ro = _val(fields.get("round_off"))
    checks = {
        "does_not_balance": has("does_not_balance"),
        "tax_rate_mismatch": has("rate_row_mismatch", "split_vs_rate_row", "tax_vs_table")
        or _tax_rate_mismatch(fields),
        "cgst_ne_sgst": has("cgst_ne_sgst"),
        "total_not_supported": has("total_not_supported", "total_unsupported", "repair_unsupported"),
        "repaired_value": bool(val.get("repairs")),
        "duplicate_item_rows": _duplicates(fields),
        "round_off_too_large": ro is not None and abs(ro) > ROUNDOFF_MAX,
    }
    return [k for k, v in checks.items() if v]


def features(fields: dict, val: dict) -> list[float]:
    """[|balance_diff|, min OCR conf over items + total, tax_sum / items_sum] (model input only)."""
    bd = val.get("balance_diff")
    confs = [i["min_conf"] for i in fields.get("items", []) if i.get("min_conf") is not None]
    tc = (fields.get("total") or {}).get("min_conf")
    confs += [tc] if tc is not None else []
    items = _d(val.get("items_sum")) or Decimal(0)
    tax = _d(val.get("tax_sum")) or Decimal(0)
    return [NO_DIFF if bd is None else float(abs(_d(bd))), float(min(confs, default=0.0)),
            float(tax / items) if items else 0.0]


def score(fields: dict, val: dict, out_dir: Path | None = None) -> dict:
    """Flags + risk 0..1 (weighted sum, capped). Writes out_dir/06_anomaly.json when given."""
    flags = rule_flags(fields, val)
    res = {"flags": flags, "risk": round(min(1.0, sum(WEIGHTS[f] for f in flags)), 3),
           "features": features(fields, val)}
    if out_dir is not None:
        (Path(out_dir) / "06_anomaly.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    return res


# ---------- optional secondary score ----------

def fit_iforest(X: list[list[float]]):
    """IsolationForest fitted on clean bills only; None when scikit-learn is missing."""
    try:
        from sklearn.ensemble import IsolationForest
    except ImportError:
        return None
    return IsolationForest(n_estimators=200, random_state=SEED).fit(X)


def iforest_score(model, X: list[list[float]]) -> list[float | None]:
    """0..1, higher = more unusual (-score_samples is in about 0..1 already)."""
    if model is None:
        return [None] * len(X)
    return [round(min(1.0, max(0.0, float(-s))), 4) for s in model.score_samples(X)]
