# chord-transpose

In-place chord-symbol transpose for song-sheet images/PDFs (簡譜 or staff).

```bash
# 1) detect → editable JSON + red-box preview
python3 transpose.py detect sheet.png --out chords.json
# 2) (optional) edit chords.json, then apply — one uniform font size per page
python3 transpose.py apply sheet.png chords.json --semitones 5 --key G --out out.png
# optional: --font-size N  (else median chord-box height)
```

`apply` never sizes per-box; long chords extend right. Report includes `font sizes used: 1`.
Needs: tesseract (`vendor/tessusr`), pillow, numpy, poppler-utils.
