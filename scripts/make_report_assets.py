"""Report assets from existing CSVs: outputs/report/summary.md + figures/*.png."""
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "outputs" / "eval"
REPORT = ROOT / "outputs" / "report"
FIGS = REPORT / "figures"

STATUSES = ("VALID", "REPAIRED", "CHECK_REQUIRED", "INSUFFICIENT_EVIDENCE")
TAMPER_TYPES = ("item_amount", "total", "tax_amount")
DETECTORS = (("rules", "det_rules"), ("rules+IF", "det_combined"))
# reference palette: categorical slots 1-2, status colours, recessive inks
SERIES = ("#2a78d6", "#eb6834")
STATUS_COLOR = {"VALID": "#0ca30c", "REPAIRED": "#fab219", "CHECK_REQUIRED": "#ec835a",
                "INSUFFICIENT_EVIDENCE": "#d03b3b"}
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
# short two-line x-axis labels for figures (table names stay long)
CORD_SET, DESK_SET, FRESH_SET = ("CORD held-out (0100-0149)", "Desktop development set (tuned on)",
                                 "Fresh test set (no tuning)")
SHORT = {CORD_SET: "CORD\nheld-out", DESK_SET: "Desktop dev\n(tuned)", FRESH_SET: "Fresh\ntest"}
LIMITATIONS = (
    "OCR errors (misread digits, merged or split rows) remain the main source of wrong totals.",
    "A balanced forgery (item, tax and total all edited consistently) is not detected.",
    "Merchant extraction is weak (header text, logos and slogans are often picked instead).",
    "The synthetic tamper set is easier than real bills (clean renders, known layouts).",
    "Detection flags arithmetic inconsistency; it cannot separate tampering from OCR error.",
)


def _read(path: Path) -> list[dict] | None:
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _true(x: str) -> bool:
    return str(x).strip() == "True"


def _set_stats(rows: list[dict]) -> dict:
    """Per-bill eval rows (status, total_ok) -> counts used in the report."""
    scored = [r for r in rows if str(r.get("total_ok", "")).strip() in ("True", "False")]
    vr = [r for r in scored if r["status"] in ("VALID", "REPAIRED")]
    cr = [r for r in scored if r["status"] == "CHECK_REQUIRED"]
    return {"n": len(rows), "total_ok": sum(_true(r["total_ok"]) for r in scored),
            "total_n": len(scored), "vr_ok": sum(_true(r["total_ok"]) for r in vr),
            "vr_n": len(vr), "cr_wrong": sum(not _true(r["total_ok"]) for r in cr),
            "cr_n": len(cr), **{s: sum(r["status"] == s for r in rows) for s in STATUSES}}


def _cord_stats(rows: list[dict]) -> dict | None:
    held = next((r for r in rows if r["group"] == "held_out"), None)
    return None if held is None else {k: int(v) for k, v in held.items() if k != "group"}


def _tamper_stats(rows: list[dict]) -> dict:
    """Test-split detection / false-alarm per detector, and detection per tamper type."""
    test = [r for r in rows if r["split"] == "test"]
    tam = [r for r in test if r["kind"] == "tampered"]
    cln = [r for r in test if r["kind"] == "clean"]
    out = {"n": len(test), "n_tampered": len(tam), "n_clean": len(cln)}
    for name, key in DETECTORS:
        out[name] = {"det": (sum(_true(r[key]) for r in tam), len(tam)),
                     "fa": (sum(_true(r[key]) for r in cln), len(cln)),
                     "by_type": {t: (sum(_true(r[key]) for r in tam if r["tamper_type"] == t),
                                     sum(r["tamper_type"] == t for r in tam)) for t in TAMPER_TYPES}}
    return out


def _rate(k: int, n: int) -> float | None:
    return k / n if n else None


def _pct(k: int, n: int) -> str:
    return f"{k}/{n} ({100 * k / n:.0f}%)" if n else "n/a"


def _style(ax, title: str) -> None:
    ax.set_title(title, loc="left", color=INK, fontsize=12)
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2)
    ax.set_axisbelow(True)


def _save(fig, name: str) -> None:
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout()
    fig.savefig(FIGS / name, dpi=150)
    plt.close(fig)


def _rate_bars(sets: dict[str, tuple[int, int]], title: str, name: str,
               hatch: str | None = None) -> None:
    """Single-series bar chart of k/n rates per set, values labelled; `hatch` marks one set."""
    labels = list(sets)
    vals = [100 * (_rate(*sets[s]) or 0) for s in labels]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    _style(ax, title)
    bars = ax.bar([SHORT.get(s, s) for s in labels], vals, width=0.5, color=SERIES[0],
                  edgecolor=SURFACE)
    for b, s in zip(bars, labels):
        if s == hatch:
            b.set_hatch("//")
    for b, s in zip(bars, labels):
        ax.annotate(_pct(*sets[s]), (b.get_x() + b.get_width() / 2, b.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha="center", color=INK, fontsize=9)
    ax.set_ylim(0, 110)
    ax.set_ylabel("%", color=INK2)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    _save(fig, name)


def _fig_status(sets: dict[str, dict]) -> None:
    """100% stacked horizontal bars of status share per set, with counts."""
    fig, ax = plt.subplots(figsize=(8, 0.8 + 0.7 * len(sets)))
    _style(ax, "Validation status by set (share of bills)")
    names = list(sets)[::-1]
    left = [0.0] * len(names)
    for st in STATUSES:
        w = [100 * sets[n][st] / sets[n]["n"] if sets[n]["n"] else 0 for n in names]
        ax.barh(names, w, left=left, color=STATUS_COLOR[st], edgecolor=SURFACE, linewidth=2,
                label=st, height=0.55)
        for i, n in enumerate(names):
            if w[i] >= 6:
                ax.text(left[i] + w[i] / 2, i, str(sets[n][st]), ha="center", va="center",
                        color=INK, fontsize=9)
        left = [a + b for a, b in zip(left, w)]
    ax.set_xlim(0, 100)
    ax.set_xlabel("% of bills", color=INK2)
    ax.legend(ncol=4, fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.35))
    _save(fig, "status_counts.png")


def _fig_tamper_type(t: dict) -> None:
    fig, ax = plt.subplots(figsize=(7, 3.6))
    _style(ax, "Synthetic tamper detection by type (test pairs)")
    x = range(len(TAMPER_TYPES))
    w = 0.36
    for j, (name, _) in enumerate(DETECTORS):
        vals = [100 * (_rate(*t[name]["by_type"][tt]) or 0) for tt in TAMPER_TYPES]
        bars = ax.bar([i + (j - 0.5) * w for i in x], vals, width=w - 0.02, color=SERIES[j],
                      label=name)
        for b, v in zip(bars, vals):
            ax.annotate(f"{v:.0f}%", (b.get_x() + b.get_width() / 2, b.get_height()),
                        xytext=(0, 3), textcoords="offset points", ha="center", color=INK, fontsize=8)
    ax.set_xticks(list(x), TAMPER_TYPES)
    ax.set_ylim(0, 115)
    ax.set_ylabel("detected %", color=INK2)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.legend(frameon=False, fontsize=9)
    _save(fig, "tamper_detection_by_type.png")


def _fig_fa_vs_det(t: dict) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    _style(ax, "False alarm vs detection (synthetic test)")
    for j, (name, _) in enumerate(DETECTORS):
        fa, det = 100 * (_rate(*t[name]["fa"]) or 0), 100 * (_rate(*t[name]["det"]) or 0)
        ax.scatter([fa], [det], s=80, color=SERIES[j], edgecolor=SURFACE, linewidth=2,
                   label=name, zorder=3)
        ax.annotate(f"{name}\n{det:.0f}% det / {fa:.0f}% FA", (fa, det), xytext=(8, -4),
                    textcoords="offset points", color=INK, fontsize=8)
    ax.set_xlim(-2, 102)
    ax.set_ylim(-2, 105)
    ax.set_xlabel("false-alarm rate on clean bills (%)", color=INK2)
    ax.set_ylabel("detection rate on tampered bills (%)", color=INK2)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.text(0.02, 0.02, "rules and rules+IF identical", transform=ax.transAxes, color=INK2,
            fontsize=8, style="italic")
    _save(fig, "false_alarm_vs_detection.png")


def main() -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    sets: dict[str, dict] = {}
    if (c := _read(EVAL / "stage6_cord_metrics.csv")) and (cs := _cord_stats(c)):
        sets[CORD_SET] = cs
    if (d := _read(EVAL / "phase5_desktop.csv")):
        sets[DESK_SET] = _set_stats(d)
    if (fr := _read(EVAL / "final_fresh.csv")):
        sets[FRESH_SET] = _set_stats(fr)
    syn = _read(EVAL / "phase8_synthetic.csv")
    t = _tamper_stats(syn) if syn else None

    lines = ["# CVAI results summary", "",
             "All numbers are read from CSV files in outputs/eval/.", "",
             "| Set | Bills | Total accuracy | VALID/REPAIRED precision | CHECK_REQUIRED with wrong total | Tamper detection | False alarm |",
             "|---|---|---|---|---|---|---|"]
    for name, s in sets.items():
        lines.append(f"| {name} | {s['n']} | {_pct(s['total_ok'], s['total_n'])} | "
                     f"{_pct(s['vr_ok'], s['vr_n'])} | {_pct(s['cr_wrong'], s['cr_n'])} | - | - |")
    if t:
        for name, _ in DETECTORS:
            lines.append(f"| Synthetic tamper test pairs, {name} | {t['n']} | - | - | - | "
                         f"{_pct(*t[name]['det'])} | {_pct(*t[name]['fa'])} |")
        lines += ["", "## Synthetic tamper detection by type (test pairs)", "",
                  "| Tamper type | " + " | ".join(n for n, _ in DETECTORS) + " |",
                  "|---|" + "---|" * len(DETECTORS)]
        for tt in TAMPER_TYPES:
            lines.append(f"| {tt} | " + " | ".join(_pct(*t[n]["by_type"][tt]) for n, _ in DETECTORS) + " |")
    lines += ["", "## Status counts", "", "| Set | " + " | ".join(STATUSES) + " |",
              "|---|" + "---|" * len(STATUSES)]
    for name, s in sets.items():
        lines.append(f"| {name} | " + " | ".join(str(s[st]) for st in STATUSES) + " |")
    lines += ["", "## Figures", ""]
    figs = []
    if sets:
        _fig_status(sets)
        _rate_bars({n: (s["total_ok"], s["total_n"]) for n, s in sets.items()},
                   "Total accuracy by set", "total_accuracy_by_set.png")
        _rate_bars({n: (s["vr_ok"], s["vr_n"]) for n, s in sets.items()},
                   "VALID/REPAIRED precision by set", "valid_precision_by_set.png", hatch=DESK_SET)
        figs += ["status_counts.png", "total_accuracy_by_set.png", "valid_precision_by_set.png"]
    if t:
        _fig_tamper_type(t)
        _fig_fa_vs_det(t)
        figs += ["tamper_detection_by_type.png", "false_alarm_vs_detection.png"]
    lines += [f"![{f}](figures/{f})" for f in figs]
    lines += ["", "## Limitations", ""] + [f"- {x}" for x in LIMITATIONS] + [""]
    (REPORT / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"sets: {len(sets)}  synthetic: {'yes' if t else 'no'}  figures: {len(figs)}")
    print(f"saved: {(REPORT / 'summary.md').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
