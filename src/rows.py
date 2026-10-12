"""Stage 3: rebuild table rows and cells from OCR line boxes."""
import re
from statistics import median

from src.keywords import match_labels
from src.normalize import (is_currency_marker, label_text, parse_amount, split_glued,
                           split_label_glued)
from src.ocr import _rect

ROW_OVERLAP = 0.5
CELL_GAP_CHARS = 2.5
FLAG_TOKEN = re.compile(r"[A-Z]")  # lone tax-flag letter after a price


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def _group_rows(rects: list[tuple]) -> list[list[int]]:
    """Group box indices into rows by vertical overlap with each row's median y-range."""
    order = sorted(range(len(rects)), key=lambda i: (rects[i][1] + rects[i][3]) / 2)
    rows: list[list[int]] = []
    for i in order:
        _, y0, _, y1 = rects[i]
        best, best_ratio = None, 0.0
        for r in rows:
            ry0 = median(rects[j][1] for j in r)
            ry1 = median(rects[j][3] for j in r)
            small = min(y1 - y0, ry1 - ry0)
            if small <= 0:
                continue
            ratio = _overlap(y0, y1, ry0, ry1) / small
            if ratio >= ROW_OVERLAP and ratio > best_ratio:
                best, best_ratio = r, ratio
        if best is None:
            rows.append([i])
        else:
            best.append(i)
    rows.sort(key=lambda r: median((rects[j][1] + rects[j][3]) / 2 for j in r))
    return [sorted(r, key=lambda j: rects[j][0]) for r in rows]


def _sub_tokens(line: dict, rect: tuple, line_idx: int = 0) -> list[dict]:
    """Whitespace sub-tokens with x0/x1 estimated from character offsets.
    y0/y1/box_h/line keep the source OCR box geometry (used for the merchant)."""
    text = line["text"]
    x0, y0, x1, y1 = rect
    n, w = max(len(text), 1), x1 - x0
    return [{"text": m.group(), "x0": x0 + w * m.start() / n, "x1": x0 + w * m.end() / n,
             "conf": float(line["conf"]), "y0": float(y0), "y1": float(y1),
             "box_h": float(y1 - y0), "line": line_idx} for m in re.finditer(r"\S+", text)]


def row_labels(text: str) -> list[tuple[str, float]]:
    """Labels of a row; label:number glue ('NetAmount:3814.00') is split first."""
    return match_labels(label_text(text))


def row_amounts(tokens: list[dict], cell_of: list[int],
                labels: list[tuple[str, float]]) -> list[dict]:
    """Amounts of a row; the row labels decide one-decimal and trailing-dot money."""
    lab = {l for l, _ in labels}
    out = []
    i = 0
    while i < len(tokens):
        text, cell = tokens[i]["text"], cell_of[i]
        # glue a bare currency marker to the following token
        if is_currency_marker(text) and i + 1 < len(tokens):
            text, cell = f"{text} {tokens[i + 1]['text']}", cell_of[i + 1]
            i += 1
        # labels glued to numbers and back-to-back decimals glued by OCR are split apart
        for part in (p for piece in split_label_glued(text) for p in split_glued(piece)):
            a = parse_amount(part, lab)
            if a is not None:
                amt = {"value": a.value, "strength": a.strength, "cell": cell, "raw": a.raw}
                # tax-flag letter as its own last token: '1.24 X'
                flag = a.flag or (tokens[i + 1]["text"] if i + 2 == len(tokens) and a.strength ==
                                  "strong" and FLAG_TOKEN.fullmatch(tokens[i + 1]["text"]) else None)
                if flag:
                    amt["tax_flag"] = flag
                out.append(amt)
        i += 1
    return out


def build_rows(lines: list[dict]) -> list[dict]:
    """OCR lines [{box, text, conf}] -> Row dicts. No field extraction here."""
    if not lines:
        return []
    rects = [_rect(l["box"]) for l in lines]
    char_w = median((r[2] - r[0]) / max(len(l["text"]), 1) for l, r in zip(lines, rects))
    gap_max = CELL_GAP_CHARS * char_w

    rows = []
    for idx, members in enumerate(_group_rows(rects)):
        tokens = [t for j in members for t in _sub_tokens(lines[j], rects[j], j)]
        cells: list[list[str]] = []
        cell_of: list[int] = []
        prev_x1 = None
        for t in tokens:
            if prev_x1 is None or t["x0"] - prev_x1 > gap_max:
                cells.append([])
            cells[-1].append(t["text"])
            cell_of.append(len(cells) - 1)
            prev_x1 = t["x1"]
        text = " ".join(lines[j]["text"] for j in members)
        labels = row_labels(text)
        rows.append({
            "idx": idx,
            "y": round(median((rects[j][1] + rects[j][3]) / 2 for j in members), 1),
            "text": text,
            "cells": [" ".join(c) for c in cells],
            "tokens": tokens,
            "min_conf": min((t["conf"] for t in tokens), default=None),
            "labels": labels,
            "amounts": row_amounts(tokens, cell_of, labels),
        })
    return rows
