"""Stage 2: two-pass PaddleOCR (original + enhanced) with box-level merging."""
import numpy as np

_OCR = None

MATCH_IOU = 0.5
NEW_BOX_IOU = 0.3


def _engine():
    """Lazy module-level PaddleOCR singleton."""
    global _OCR
    if _OCR is None:
        from paddleocr import PaddleOCR
        _OCR = PaddleOCR(use_angle_cls=True, lang="en", use_gpu=False, show_log=False)
    return _OCR


def ocr_image(img: np.ndarray, source: str = "orig") -> list[dict]:
    """OCR a BGR image -> [{box, text, conf, source}]."""
    result = _engine().ocr(img, cls=True)
    page = result[0] if result else None
    if not page:
        return []
    return [{"box": [[float(x), float(y)] for x, y in box], "text": str(text),
             "conf": float(conf), "source": source}
            for box, (text, conf) in page]


def _rect(box: list) -> tuple[float, float, float, float]:
    xs = [p[0] for p in box]
    ys = [p[1] for p in box]
    return min(xs), min(ys), max(xs), max(ys)


def _iou(a: list, b: list) -> float:
    ax0, ay0, ax1, ay1 = _rect(a)
    bx0, by0, bx1, by1 = _rect(b)
    iw = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    ih = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = iw * ih
    union = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / union if union > 0 else 0.0


def _sort_key(line: dict) -> tuple[float, float]:
    x0, y0, _, y1 = _rect(line["box"])
    return (y0 + y1) / 2, x0


def merge(a: list[dict], b: list[dict]) -> list[dict]:
    """Merge orig lines (a) with enh lines (b); matched pairs keep the higher conf."""
    used_b: set[int] = set()
    kept: list[dict] = []
    for la in a:
        best_j, best_iou = -1, 0.0
        for j, lb in enumerate(b):
            if j in used_b:
                continue
            iou = _iou(la["box"], lb["box"])
            if iou > best_iou:
                best_j, best_iou = j, iou
        if best_iou > MATCH_IOU:
            used_b.add(best_j)
            lb = b[best_j]
            win, src = (lb, "enh") if lb["conf"] > la["conf"] else (la, "orig")
            kept.append({**win, "source": src})
        else:
            kept.append(dict(la))
    for j, lb in enumerate(b):
        if j in used_b:
            continue
        if all(_iou(lb["box"], k["box"]) < NEW_BOX_IOU for k in kept):
            kept.append({**lb, "source": "enh"})
    return sorted(kept, key=_sort_key)


def two_pass(orig: np.ndarray, enh: np.ndarray) -> dict:
    """OCR both images and merge."""
    orig_lines = ocr_image(orig, "orig")
    enh_lines = ocr_image(enh, "enh")
    return {"orig_lines": orig_lines, "enh_lines": enh_lines,
            "merged": merge(orig_lines, enh_lines)}
