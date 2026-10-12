"""Stage 1: load, normalize, deskew, quality check and enhancement of bill images."""
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

MAX_SIDE = 2000
MIN_SIDE = 1000
UPSCALE_TO = 1200

DESKEW_MIN_DEG = 0.5
DESKEW_MAX_DEG = 15.0

BLUR_MIN = 60.0
BRIGHT_MIN = 60.0

CLAHE_CLIP = 2.0
CLAHE_TILE = (8, 8)
DENOISE_H = 3


def load(path: str | Path) -> np.ndarray:
    """Open with PIL, honour EXIF orientation, return BGR uint8."""
    with Image.open(path) as im:
        rgb = ImageOps.exif_transpose(im).convert("RGB")
        return cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)


def normalize(img: np.ndarray) -> np.ndarray:
    """Clamp the long side: >MAX_SIDE -> MAX_SIDE, <MIN_SIDE -> UPSCALE_TO."""
    h, w = img.shape[:2]
    long_side = max(h, w)
    if long_side > MAX_SIDE:
        target, interp = MAX_SIDE, cv2.INTER_AREA
    elif long_side < MIN_SIDE:
        target, interp = UPSCALE_TO, cv2.INTER_CUBIC
    else:
        return img
    scale = target / long_side
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    return cv2.resize(img, size, interpolation=interp)


def _skew_angle(img: np.ndarray) -> float:
    """Skew in degrees from text pixels. Otsu is used only to locate the text."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    ys, xs = np.nonzero(mask)
    if len(xs) < 10:
        return 0.0
    pts = np.column_stack((xs, ys)).astype(np.float32)
    angle = cv2.minAreaRect(pts)[2]
    # OpenCV >= 4.5 reports angles in (0, 90]; map to (-45, 45]
    if angle > 45:
        angle -= 90
    return float(angle)


def deskew(img: np.ndarray) -> tuple[np.ndarray, float]:
    """Rotate to straighten text when DESKEW_MIN_DEG < |angle| <= DESKEW_MAX_DEG."""
    angle = _skew_angle(img)
    if not (DESKEW_MIN_DEG < abs(angle) <= DESKEW_MAX_DEG):
        return img, 0.0
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    out = cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_CUBIC,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    return out, angle


def quality(img: np.ndarray) -> dict:
    """Blur (Laplacian variance), brightness and contrast of the gray image, plus a tag."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())
    contrast = float(gray.std())
    return {"blur": blur, "brightness": brightness, "contrast": contrast,
            "tag": tag(blur, brightness)}


def tag(blur: float, brightness: float) -> str:
    """LOW_QUALITY when blurry or too dark. Bright paper is normal, so no upper limit."""
    return "LOW_QUALITY" if blur < BLUR_MIN or brightness < BRIGHT_MIN else "OK"


def enhance(img: np.ndarray) -> np.ndarray:
    """CLAHE on L of LAB, then light colour denoise. No binarization."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_TILE).apply(l)
    out = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    return cv2.fastNlMeansDenoisingColored(out, None, DENOISE_H, DENOISE_H, 7, 21)


def run(path: str | Path) -> dict:
    """Full stage 1: orig is normalized+deskewed, enh is its enhanced copy."""
    img = normalize(load(path))
    orig, angle = deskew(img)
    return {"orig": orig, "enh": enhance(orig), "angle": angle, "quality": quality(orig)}
