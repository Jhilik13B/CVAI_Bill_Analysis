# CVAI Bill Analysis

Reads a bill image, extracts merchant, date, items, taxes and total, then checks that the arithmetic balances and flags anything it cannot prove.

## Pipeline
1. Preprocess (deskew, light contrast enhancement, no binarization)
2. Two-pass PaddleOCR (PP-OCRv4, pretrained, no training)
3. Row reconstruction and number/date normalization
4. Field extraction with candidate scoring
5. Arithmetic validation, tax-mode detection, evidence-based repair
6. Rule-based anomaly flags (Isolation Forest as an optional secondary score)
7. Streamlit demo (`app.py`)

## Setup
Python 3.10 is recommended.

    python -m venv .venv
    .venv\Scripts\activate
    pip install -r requirements.txt

## Run the demo
    streamlit run app.py

## Run the tests
    python -m pytest -q

## Results (see report/summary.md)
- CORD held-out: about 87% total accuracy, 97% VALID/REPAIRED precision
- 12 fresh real bills (no tuning): about 75% total accuracy
- The 20 annotated desktop bills are a development set (tuned on)
- Synthetic tamper detection flags arithmetic inconsistency; it cannot separate tampering from OCR error

## Notes
- `Images/` holds my own bill photos. `annotations/` holds hand-written ground truth.
- The CORD dataset is not included; run `scripts/download_cord.py` to fetch it.
- The system is not error-free. Bills it cannot prove are marked CHECK_REQUIRED for human review.
