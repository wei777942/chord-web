# chord-web

Password-gated upload site for in-place chord transposition on song-sheet images/PDFs.

Runs `chord-transpose detect` then `apply` in-process (no webhook).

## Local

```bash
python3 -m venv venv && . venv/bin/activate
pip install -r requirements.txt
# needs tesseract-ocr, poppler-utils, fonts-liberation, fonts-noto-cjk
export CHORD_WEB_PASSWORD='...'   # or write config.json {"password":"..."}
python app.py
```

## Render

Docker runtime. Set `CHORD_WEB_PASSWORD`. Free web service sleeps when idle.
