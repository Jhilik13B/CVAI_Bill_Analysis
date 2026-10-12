"""Run stages 1-6 + anomaly on the synthetic clean/tampered sets; report tamper detection.
Thresholds are chosen on pairs 0-29 and reported on pairs 30-59."""
import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.anomaly import fit_iforest, iforest_score, score  # noqa: E402

SYN = ROOT / "data" / "synthetic"
STAGES = ROOT / "outputs" / "stages"
SCRIPTS = ("run_stage12.py", "run_stage34.py", "run_stage5.py", "run_stage6.py")
CAL_PAIRS = 30


def _run_stages(img_dir: Path) -> bool:
    for s in SCRIPTS:
        p = subprocess.run([sys.executable, str(ROOT / "scripts" / s), "--img-dir", str(img_dir)],
                           cwd=ROOT, capture_output=True, text=True)
        if p.returncode != 0:
            print(f"FAIL {s} {img_dir.name}: {(p.stderr.strip().splitlines() or ['?'])[-1]}")
            return False
    return True


def _load(stem: str) -> tuple[dict, dict] | None:
    d = STAGES / stem
    if not (d / "04_fields.json").exists() or not (d / "05_validation.json").exists():
        return None
    return (json.loads((d / "04_fields.json").read_text(encoding="utf-8")),
            json.loads((d / "05_validation.json").read_text(encoding="utf-8")))


def _prf(rows: list[dict], key: str) -> dict:
    tp = sum(r[key] and r["kind"] == "tampered" for r in rows)
    fp = sum(r[key] and r["kind"] == "clean" for r in rows)
    pos = sum(r["kind"] == "tampered" for r in rows)
    neg = len(rows) - pos
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / pos if pos else 0.0
    return {"recall": r, "fa": fp / neg if neg else 0.0, "precision": p,
            "f1": 2 * p * r / (p + r) if p + r else 0.0}


def _pick(rows: list[dict], cands: list[float], det) -> float:
    """Threshold with the best calibration F1 (ties -> higher threshold)."""
    best = max(sorted(set(cands)), key=lambda t: (_prf([{**r, "x": det(r, t)} for r in rows], "x")["f1"], t))
    return best


def main() -> None:
    for kind in ("clean", "tampered"):
        if not _run_stages(SYN / kind):
            return
    rows, missing = [], 0
    for kind, prefix in (("clean", "syn_c"), ("tampered", "syn_t")):
        for gt_file in sorted((SYN / kind).glob(f"{prefix}_*.json")):
            gt = json.loads(gt_file.read_text(encoding="utf-8"))
            pair = int(gt_file.stem.rsplit("_", 1)[1])
            r = {"stem": gt_file.stem, "pair": pair, "split": "cal" if pair < CAL_PAIRS else "test",
                 "kind": kind, "tamper_type": gt.get("tamper_type", ""), "status": "MISSING",
                 "risk": 1.0, "flags": "", "feat": None, "if_score": None}
            loaded = _load(gt_file.stem)
            if loaded is None:
                missing += 1
            else:
                fields, val = loaded
                a = score(fields, val, STAGES / gt_file.stem)
                r.update(status=val["status"], risk=a["risk"], flags=";".join(a["flags"]),
                         feat=a["features"])
            rows.append(r)

    cal = [r for r in rows if r["split"] == "cal"]
    test = [r for r in rows if r["split"] == "test"]
    # IsolationForest: fitted on calibration clean bills only
    model = fit_iforest([r["feat"] for r in cal if r["kind"] == "clean" and r["feat"]])
    have = [r for r in rows if r["feat"]]
    for r, s in zip(have, iforest_score(model, [r["feat"] for r in have])):
        r["if_score"] = s
    for r in rows:
        if r["feat"] is None:
            r["if_score"] = 1.0 if model is not None else None

    def det_rules(r: dict, t: float) -> bool:
        return r["risk"] >= t or r["status"] in ("CHECK_REQUIRED", "MISSING")

    t_r = _pick(cal, [r["risk"] for r in cal] + [1.01], det_rules)
    for r in rows:
        r["det_rules"] = det_rules(r, t_r)

    def det_comb(r: dict, t: float) -> bool:
        return r["det_rules"] or (r["if_score"] is not None and r["if_score"] >= t)

    t_if = _pick(cal, [r["if_score"] for r in cal if r["if_score"] is not None] + [2.0], det_comb)
    for r in rows:
        r["det_combined"] = det_comb(r, t_if)

    out = ROOT / "outputs" / "eval" / "phase8_synthetic.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["stem", "pair", "split", "kind", "tamper_type", "status", "risk", "flags", "if_score",
            "det_rules", "det_combined"]
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    n_c = sum(r["kind"] == "clean" for r in rows)
    print(f"bills: {len(rows)} (clean {n_c}, tampered {len(rows) - n_c})  missing outputs: {missing}")
    print(f"thresholds (from pairs 0-{CAL_PAIRS - 1}): risk >= {t_r}  if_score >= {t_if}"
          + ("" if model is not None else "  [IsolationForest unavailable]"))
    print(f"report on pairs {CAL_PAIRS}-59 ({len(test)} bills):")
    for name, key in (("rules", "det_rules"), ("rules+IF", "det_combined")):
        m = _prf(test, key)
        print(f"  {name:<9} detection {m['recall']:.2f}  false-alarm {m['fa']:.2f}  "
              f"P {m['precision']:.2f}  R {m['recall']:.2f}  F1 {m['f1']:.2f}")
    print("  per tamper_type (test) rules / rules+IF:")
    for tt in ("item_amount", "total", "tax_amount"):
        g = [r for r in test if r["tamper_type"] == tt]
        print(f"    {tt:<12} {sum(r['det_rules'] for r in g)}/{len(g)} / "
              f"{sum(r['det_combined'] for r in g)}/{len(g)}")
    print(f"saved: {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
