# CVAI: Bill analysis project

Goal: image of a bill -> structured JSON (merchant, date, items, taxes, total) -> arithmetic validation -> anomaly flags.
Pipeline: preprocess -> two-pass PaddleOCR -> row reconstruction -> normalization -> field extraction -> validation/repair -> anomaly flags -> JSON.

## Rules
- Work only inside this folder. Do exactly what the prompt lists: no extra files, refactors, or features.
- Never open, read, or summarize images, datasets, or large JSON files. Run scripts instead.
- Keep terminal output short: counts and pass/fail lines only.
- Do NOT spawn subagents or do broad exploration unless I ask.
- Use pathlib, type hints, short comments. Money is parsed with Decimal, never float.
- No hard binarization in preprocessing. Taxes are read from the bill, never assumed.
- If something fails, report the exact error once and stop. Max 2 retries.
- Finish with a short summary: files created, commands run, results.
- Never tune rules on data/desktop/ (it is the honest test set).