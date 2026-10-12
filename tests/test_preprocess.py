import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import preprocess  # noqa: E402


def _text_like(h: int = 400, w: int = 300) -> np.ndarray:
    img = np.full((h, w, 3), 255, np.uint8)
    for y in range(40, h - 40, 30):
        cv2.rectangle(img, (30, y), (w - 30, y + 8), (0, 0, 0), -1)
    return img


def test_enhance_not_binarized() -> None:
    grad = np.tile(np.linspace(30, 220, 200, dtype=np.uint8), (150, 1))
    img = cv2.merge((grad, grad, grad))
    out = preprocess.enhance(img)
    assert out.shape == img.shape and out.dtype == img.dtype
    assert len(np.unique(cv2.cvtColor(out, cv2.COLOR_BGR2GRAY))) > 2


def test_normalize_limits() -> None:
    big = np.zeros((3000, 500, 3), np.uint8)
    small = np.zeros((500, 400, 3), np.uint8)
    mid = np.zeros((1500, 900, 3), np.uint8)
    assert max(preprocess.normalize(big).shape[:2]) == 2000
    assert max(preprocess.normalize(small).shape[:2]) == 1200
    assert preprocess.normalize(mid).shape == mid.shape


def test_deskew_straight_is_zero() -> None:
    img = _text_like()
    out, angle = preprocess.deskew(img)
    assert angle == 0.0
    assert out is img


def test_quality_blank_is_low() -> None:
    img = np.full((20, 20, 3), 128, np.uint8)
    assert preprocess.quality(img)["tag"] == "LOW_QUALITY"


def test_quality_bright_text_is_ok() -> None:
    # white paper with sharp text: brightness > 220 must not be LOW_QUALITY
    img = np.full((400, 300, 3), 255, np.uint8)
    for y in range(40, 360, 30):
        cv2.rectangle(img, (30, y), (270, y + 2), (0, 0, 0), -1)
    q = preprocess.quality(img)
    assert q["brightness"] > 220 and q["tag"] == "OK"
