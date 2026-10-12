"""Pipeline runners with per-stage dumps."""
import json
import time
from pathlib import Path

import cv2
import numpy as np

from src import ocr, preprocess


def _imwrite(path: Path, img: np.ndarray) -> None:
    # imencode + write_bytes avoids cv2.imwrite's issues with non-ASCII Windows paths
    ok, buf = cv2.imencode(".jpg", img)
    if not ok:
        raise RuntimeError(f"jpeg encode failed: {path}")
    path.write_bytes(buf.tobytes())


def _mean(vals: list[float]) -> float | None:
    return sum(vals) / len(vals) if vals else None


def _mean_conf(lines: list[dict]) -> float | None:
    return _mean([l["conf"] for l in lines])


def _numeric_conf(lines: list[dict]) -> float | None:
    return _mean([l["conf"] for l in lines if any(c.isdigit() for c in l["text"])])


def _overlay(img: np.ndarray, lines: list[dict]) -> np.ndarray:
    out = img.copy()
    colors = {"orig": (0, 200, 0), "enh": (255, 0, 0)}  # green = orig, blue = enh
    for l in lines:
        pts = np.array(l["box"], dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(out, [pts], True, colors.get(l["source"], (0, 0, 255)), 2)
    return out


def run_stage12(path: str | Path, out_dir: str | Path) -> dict:
    """Preprocess + two-pass OCR for one image; dump artefacts; return a flat summary."""
    t0 = time.perf_counter()
    path = Path(path)
    dump = Path(out_dir) / path.stem
    dump.mkdir(parents=True, exist_ok=True)

    pre = preprocess.run(path)
    res = ocr.two_pass(pre["orig"], pre["enh"])

    _imwrite(dump / "01_orig.jpg", pre["orig"])
    _imwrite(dump / "01_enh.jpg", pre["enh"])
    q = pre["quality"]
    (dump / "quality.json").write_text(json.dumps({"angle": pre["angle"], **q}, indent=2),
                                       encoding="utf-8")
    (dump / "02_ocr.json").write_text(json.dumps(res, indent=1, ensure_ascii=False),
                                      encoding="utf-8")
    _imwrite(dump / "02_overlay.jpg", _overlay(pre["orig"], res["merged"]))

    summary = {"image": path.name, "tag": q["tag"], "angle": round(pre["angle"], 2),
               "blur": round(q["blur"], 1), "n_orig": len(res["orig_lines"]),
               "n_enh": len(res["enh_lines"]), "n_merged": len(res["merged"])}
    for key, lines in (("orig", res["orig_lines"]), ("enh", res["enh_lines"]),
                       ("merged", res["merged"])):
        summary[f"mean_conf_{key}"] = _mean_conf(lines)
        summary[f"numeric_conf_{key}"] = _numeric_conf(lines)
    summary["seconds"] = round(time.perf_counter() - t0, 2)
    return summary
