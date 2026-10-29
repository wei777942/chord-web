#!/usr/bin/env python3
"""Chord-symbol transpose CLI: detect on sheet image, apply in-place paint+redraw."""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

ROOT = Path(__file__).resolve().parent
VENDOR = ROOT / "vendor" / "tessusr"
FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/opentype/urw-base35/NimbusSans-Bold.otf",
]
CJK_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
]

# --- chord grammar ---
_ROOT = r"[A-G](?:#|b)?"
_QUAL = r"(?:maj7|maj|min|m7b5|m|dim|aug|sus[24]|add\d+)?"
_EXT = r"(?:7|9|11|13)?"
_BASS = rf"(?:/{_ROOT})?"
CHORD_RE = re.compile(rf"^{_ROOT}{_QUAL}{_EXT}{_BASS}$")
OR_CHORD_RE = re.compile(
    rf"^({_ROOT}{_QUAL}{_EXT}{_BASS})或({_ROOT}{_QUAL}{_EXT}{_BASS})$"
)
# OCR often mangles "A/C#或F#m7" into one token without 或
GARBLED_COMPOUND = re.compile(
    r"^([A-G](?:#|b)?(?:/[A-G](?:#|b)?)?)[EHek或\W]*([A-G](?:#|b)?m?(?:aj)?(?:7|9)?)$",
    re.I,
)

CJK_RE = re.compile(r"[\u4e00-\u9fff]")
# Strict chord token (incl. superscript-style trailing digits): used for filtering
STRICT_CHORD_RE = re.compile(
    rf"^{_ROOT}(?:maj7|maj|min|m7b5|m|dim|aug|sus[24]?|add\d+)?\d*(?:/{_ROOT})?$"
)


def has_cjk(text: str) -> bool:
    return bool(CJK_RE.search(text))


def fix_ocr_chord(raw: str) -> str:
    """Normalize OCR quirks: Cc→C, Am?/Dm’?→Am7/Dm7, #←H/t/4, drop pure CJK."""
    t = normalize_text(raw)
    if has_cjk(t) and "或" not in t:
        return ""
    has_sup = bool(re.search(r"[?!'′’°]", raw))
    t = (
        t.replace("'", "")
        .replace("’", "")
        .replace("`", "")
        .replace("′", "")
    )
    # trailing punctuation glued by OCR ("A," / "G.")
    t = re.sub(r"[,.;:]+$", "", t)
    t = re.sub(r"^([A-G])\1$", r"\1", t)  # Cc -> C
    # # often read as H, t, or 4: A/CH→A/C#, D/Ft→D/F#, FHm7→F#m7
    t = re.sub(r"/([A-G])[Ht4]$", r"/\1#", t)
    t = re.sub(r"^([A-G])[Ht4](?=/|$)", r"\1#", t)
    t = re.sub(r"([A-G])H(m|\d|/|$)", r"\1#\2", t)
    m = re.match(r"^([A-G][#b]?m)\??$", t)
    if m and has_sup:
        return m.group(1) + "7"
    if t.endswith("?"):
        base = t[:-1]
        if re.match(r"^[A-G][#b]?m$", base):
            return base + "7"
    return t


def is_valid_chord_token(t: str) -> bool:
    if not t:
        return False
    if has_cjk(t) and "或" not in t:
        return False
    return bool(
        STRICT_CHORD_RE.match(t)
        or CHORD_RE.match(t)
        or OR_CHORD_RE.match(t)
    )


def find_staff_tops(im: Image.Image) -> list[int]:
    """Return y of the top line of each staff system (五線譜). Empty if none."""
    if np is None:
        return []
    arr = np.array(im.convert("L"))
    H, W = arr.shape
    dark = arr < 90
    line_ys: list[int] = []
    for y in range(H):
        r = dark[y].astype(np.int8)
        d = np.diff(np.concatenate([[0], r, [0]]))
        starts = np.where(d == 1)[0]
        ends = np.where(d == -1)[0]
        if len(starts) and (ends - starts).max() >= W * 0.35:
            line_ys.append(y)
    if len(line_ys) < 5:
        return []
    systems: list[list[int]] = []
    cur = [line_ys[0]]
    for y in line_ys[1:]:
        if y - cur[0] <= 100 and y - cur[-1] <= 25:
            cur.append(y)
        else:
            if len(cur) >= 3:
                systems.append(cur)
            cur = [y]
    if len(cur) >= 3:
        systems.append(cur)
    return [min(s) for s in systems]


def ocr_band_strip(
    tess: str,
    env: dict,
    im: Image.Image,
    y0: int,
    y1: int,
    scale: int = 2,
    binary: bool = False,
    psm: int = 6,
) -> list[dict]:
    W = im.width
    st = im.crop((0, y0, W, y1))
    if binary:
        if np is None:
            st = ImageOps.autocontrast(st.convert("L"))
        else:
            arr = np.array(st.convert("L"))
            st = Image.fromarray(np.where(arr < 160, 0, 255).astype(np.uint8))
    else:
        st = ImageOps.autocontrast(st.convert("L"))
    st = st.resize(
        (W * scale, max(1, (y1 - y0) * scale)), Image.Resampling.LANCZOS
    )
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        st.save(f.name)
        path = f.name
    try:
        words = run_tess_tsv(tess, env, path, psm=psm)
    finally:
        os.unlink(path)
    out = []
    for w in words:
        out.append(
            {
                "left": w["left"] // scale,
                "top": w["top"] // scale + y0,
                "width": max(1, w["width"] // scale),
                "height": max(1, w["height"] // scale),
                "conf": w["conf"],
                "text": w["text"],
            }
        )
    return out


def detect_via_staves(tess: str, env: dict, im: Image.Image) -> list[dict] | None:
    """Chord rows just above staff systems. None if no usable staves."""
    tops = find_staff_tops(im)
    if len(tops) < 2:
        return None
    # Band height scales with image; ~80px was tuned for ~20px chord caps.
    band_h = 80
    if im.height >= 3000:
        band_h = max(80, im.height // 40)
    found: list[dict] = []
    for top in tops:
        y0, y1 = max(0, top - band_h), max(1, top - 4)
        words: list[dict] = []
        # Prefer binary@2x (recovers faint "C"); gray@2x as supplement. Skip 3x for speed.
        words.extend(
            ocr_band_strip(tess, env, im, y0, y1, scale=2, binary=True, psm=6)
        )
        words.extend(
            ocr_band_strip(tess, env, im, y0, y1, scale=2, binary=False, psm=6)
        )
        band = collect_chord_words(words, loose=True)
        band = [c for c in band if not (c["left"] < 200 and len(c["text"]) <= 1)]
        for c in band:
            if c["conf"] < 40:
                c["unconfident"] = True
        band.sort(key=lambda c: c["left"])
        merged: list[dict] = []
        for c in band:
            if merged and abs(c["left"] - merged[-1]["left"]) < 35:
                if _cand_score(c) > _cand_score(merged[-1]):
                    merged[-1] = c
            else:
                merged.append(c)
        found.extend(merged)
    return found


SHARP_SCALE = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
FLAT_SCALE = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
# keys that prefer flats
FLAT_KEYS = {"F", "Bb", "Eb", "Ab", "Db", "Gb", "Cb", "Dm", "Gm", "Cm", "Fm", "Bbm", "Ebm"}
NOTE_INDEX = {}
for i, n in enumerate(SHARP_SCALE):
    NOTE_INDEX[n] = i
for i, n in enumerate(FLAT_SCALE):
    NOTE_INDEX[n] = i
NOTE_INDEX["E#"] = 5
NOTE_INDEX["B#"] = 0
NOTE_INDEX["Fb"] = 4
NOTE_INDEX["Cb"] = 11


def setup_tesseract_env() -> dict:
    env = os.environ.copy()
    lib = VENDOR / "lib" / "x86_64-linux-gnu"
    tessdata = VENDOR / "share" / "tesseract-ocr" / "5" / "tessdata"
    bin_dir = VENDOR / "bin"
    if lib.is_dir():
        env["LD_LIBRARY_PATH"] = f"{lib}:{env.get('LD_LIBRARY_PATH', '')}"
    if tessdata.is_dir():
        env["TESSDATA_PREFIX"] = str(tessdata)
    if bin_dir.is_dir():
        env["PATH"] = f"{bin_dir}:{env.get('PATH', '')}"
    return env


def which_tesseract(env: dict) -> str:
    for cand in (
        str(VENDOR / "bin" / "tesseract"),
        "/usr/bin/tesseract",
        "tesseract",
    ):
        if cand == "tesseract" or Path(cand).exists():
            try:
                subprocess.check_output([cand, "--version"], env=env, stderr=subprocess.DEVNULL)
                return cand
            except Exception:
                continue
    raise SystemExit("tesseract not found; vendor tessusr missing")


def pick_font() -> str:
    for p in FONT_CANDIDATES:
        if Path(p).exists():
            return p
    raise SystemExit("no suitable bold sans font found")


def pick_cjk_font() -> str | None:
    for p in CJK_FONT_CANDIDATES:
        if Path(p).exists():
            return p
    return None


def load_font(path: str, size: int) -> ImageFont.FreeTypeFont:
    """Load TTF/OTF/TTC; for TTC try SC face indices."""
    if path.lower().endswith(".ttc"):
        last_err = None
        for idx in (0, 1, 2, 3, 4):
            try:
                return ImageFont.truetype(path, size, index=idx)
            except Exception as e:
                last_err = e
        raise last_err  # type: ignore
    return ImageFont.truetype(path, size)


def rasterize(path: Path, dpi: int = 300) -> list[Image.Image]:
    """Return list of RGB images (one per page)."""
    suf = path.suffix.lower()
    if suf == ".pdf":
        with tempfile.TemporaryDirectory() as td:
            out_prefix = str(Path(td) / "page")
            subprocess.check_call(
                ["pdftoppm", "-png", "-r", str(dpi), str(path), out_prefix],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            pages = sorted(Path(td).glob("page*.png"))
            if not pages:
                raise SystemExit(f"pdftoppm produced no pages for {path}")
            return [Image.open(p).convert("RGB") for p in pages]
    im = Image.open(path)
    if im.mode == "L":
        im = im.convert("RGB")
    elif im.mode == "RGBA":
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im, mask=im.split()[-1])
        im = bg
    else:
        im = im.convert("RGB")
    return [im]


def normalize_text(t: str) -> str:
    return (
        t.strip()
        .replace("♯", "#")
        .replace("＃", "#")
        .replace("♭", "b")
        .replace("Ｂ", "B")
        .replace("Ａ", "A")
        .replace("Ｄ", "D")
        .replace("Ｇ", "G")
        .replace("Ｅ", "E")
        .replace("Ｆ", "F")
        .replace("Ｃ", "C")
    )


def is_clean_chord(t: str) -> bool:
    return bool(CHORD_RE.match(t) or OR_CHORD_RE.match(t))


def recover_garbled(t: str, width: int) -> str | None:
    """Recover compound chords from OCR garbage like A/CHEKFHM7."""
    t = normalize_text(t)
    if is_clean_chord(t):
        return t
    if "或" in t:
        m = OR_CHORD_RE.match(t)
        if m:
            return t
        # try to parse around 或
        parts = t.split("或", 1)
        if len(parts) == 2 and CHORD_RE.match(parts[0]) and CHORD_RE.match(parts[1]):
            return t
    # wide garbled: A/C#...F#m7 style
    if width >= 100:
        m = GARBLED_COMPOUND.match(t.replace(" ", ""))
        if m:
            a, b = m.group(1), m.group(2)
            # fix common OCR: FH M7 -> F#m7, H as #
            b = re.sub(r"^F[H#]?m?7$", "F#m7", b, flags=re.I)
            b = re.sub(r"^F[#]?m7$", "F#m7", b, flags=re.I)
            if not b.startswith("F#") and re.match(r"^Fm?7$", b, re.I):
                # if original had # somewhere after slash root
                if "#" in t[1:] or "H" in t[1:]:
                    b = "F#m7"
            a = a.replace("H", "#")
            # A/C#H -> A/C#
            a = re.sub(r"([A-G])#$", r"\1#", a)
            if CHORD_RE.match(a) and CHORD_RE.match(b):
                return f"{a}或{b}"
        # known pattern for this sheet family
        if re.match(r"^A/C", t, re.I) and re.search(r"m?7", t, re.I):
            return "A/C#或F#m7"
    return None


def run_tess_tsv(tess: str, env: dict, image_path: str, psm: int = 11) -> list[dict]:
    out = subprocess.check_output(
        [tess, image_path, "stdout", "-l", "eng", "--psm", str(psm), "tsv"],
        env=env,
        stderr=subprocess.DEVNULL,
    ).decode("utf-8", errors="replace")
    rows = []
    for i, line in enumerate(out.splitlines()):
        if i == 0:
            continue
        parts = line.split("\t")
        if len(parts) < 12 or parts[0] != "5":
            continue
        try:
            conf = float(parts[10])
        except ValueError:
            continue
        text = parts[11].strip()
        if not text or conf < 0:
            continue
        rows.append(
            {
                "left": int(parts[6]),
                "top": int(parts[7]),
                "width": int(parts[8]),
                "height": int(parts[9]),
                "conf": conf,
                "text": text,
            }
        )
    return rows


def chord_like_size(w: dict, loose: bool = False) -> bool:
    h, wi = w["height"], w["width"]
    if loose:
        return 10 <= h <= 56 and 6 <= wi <= 280
    return 12 <= h <= 48 and 8 <= wi <= 240


TARGET_TEXT_PX = 31.0


def estimate_text_height(tess: str, env: dict, im: Image.Image) -> float:
    """Median chord-like word height in original pixels (quick OCR on ~1400px preview)."""
    max_w = 1400
    if im.width <= max_w:
        preview, inv = im, 1.0
    else:
        inv = im.width / max_w
        preview = im.resize(
            (max_w, max(1, int(im.height / inv))), Image.Resampling.LANCZOS
        )
    gray = ImageOps.autocontrast(preview.convert("L"))
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        gray.save(f.name)
        path = f.name
    try:
        words = run_tess_tsv(tess, env, path, psm=11)
    finally:
        os.unlink(path)
    chord_hs: list[float] = []
    all_hs: list[float] = []
    for w in words:
        if w["conf"] < 30:
            continue
        if not (5 <= w["height"] <= 120):
            continue
        all_hs.append(w["height"] * inv)
        t = fix_ocr_chord(w["text"])
        if is_valid_chord_token(t):
            chord_hs.append(w["height"] * inv)
    hs = sorted(chord_hs) if len(chord_hs) >= 4 else sorted(all_hs)
    if len(hs) < 5:
        return 0.0
    return float(hs[len(hs) // 2])


def scale_image_for_detect(
    im: Image.Image, median_h: float, target: float = TARGET_TEXT_PX
) -> tuple[Image.Image, float]:
    """Resample so typical chord glyphs are ~target px. No-op if already in range."""
    if median_h <= 0:
        return im, 1.0
    if 16.0 <= median_h <= 40.0:
        return im, 1.0
    sf = target / median_h
    sf = max(0.18, min(2.5, sf))
    if abs(sf - 1.0) < 0.06:
        return im, 1.0
    nw = max(1, int(round(im.width * sf)))
    nh = max(1, int(round(im.height * sf)))
    return im.resize((nw, nh), Image.Resampling.LANCZOS), sf


def map_chord_bbox(c: dict, sf: float) -> dict:
    """Map a chord dict's bbox from scaled image back to original pixels."""
    if abs(sf - 1.0) < 1e-9:
        return c
    x, y, w, h = c["bbox"]
    out = dict(c)
    out["bbox"] = [
        int(round(x / sf)),
        int(round(y / sf)),
        max(1, int(round(w / sf))),
        max(1, int(round(h / sf))),
    ]
    return out


def ocr_image_words(
    tess: str, env: dict, im: Image.Image, binary: bool = False, psm: int = 11
) -> list[dict]:
    img2 = im.convert("L")
    if binary and np is not None:
        arr = np.array(img2)
        img2 = Image.fromarray(np.where(arr < 160, 0, 255).astype(np.uint8))
    else:
        img2 = ImageOps.autocontrast(img2)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        img2.save(f.name)
        path = f.name
    try:
        return run_tess_tsv(tess, env, path, psm=psm)
    finally:
        os.unlink(path)


def _cand_score(c: dict) -> tuple:
    t = c["text"]
    return (
        2 if re.search(r"/[A-G]#", t) else (1 if "/" in t else 0),
        1 if "#" in t else 0,
        1 if STRICT_CHORD_RE.match(t) else 0,
        len(t),
        float(c.get("conf", 0)),
    )


def merge_chord_cands(cands: list[dict], xdist: int = 32, ydist: int = 24) -> list[dict]:
    cands = sorted(cands, key=lambda c: (c["top"], c["left"]))
    out: list[dict] = []
    for c in cands:
        hit = None
        for i, p in enumerate(out):
            if abs(c["top"] - p["top"]) < ydist and abs(c["left"] - p["left"]) < xdist:
                hit = i
                break
        if hit is None:
            out.append(c)
        elif _cand_score(c) > _cand_score(out[hit]):
            out[hit] = c
    return out


def reject_lyric_orphans(chords: list[dict]) -> list[dict]:
    """Drop lyric-row ghosts: lone single letters, misaligned capitals, huge 'C' blobs."""
    if not chords:
        return chords
    rows: list[list] = []
    for c in sorted(chords, key=lambda x: x["top"]):
        if rows and abs(c["top"] - rows[-1][0]) <= 32:
            rows[-1][1].append(c)
            rows[-1][0] = sum(x["top"] for x in rows[-1][1]) / len(rows[-1][1])
        else:
            rows.append([float(c["top"]), [c]])
    kept: list[dict] = []
    for _, members in rows:
        # lyric / footer rows are often only bare capitals
        if all(re.match(r"^[A-G]$", m["text"]) for m in members):
            continue
        anchors = [m for m in members if len(m["text"]) >= 2] or members
        tops = sorted(m["top"] for m in anchors)
        med_t = tops[len(tops) // 2]
        for m in members:
            if len(m["text"]) <= 1 and m["width"] > max(90, int(m["height"] * 2.2)):
                continue
            if re.match(r"^[A-G]$", m["text"]):
                # Chord rows share a top line; slash glyphs are often taller, so
                # compare tops (not bottoms) with a forgiving tolerance.
                if abs(m["top"] - med_t) > 22:
                    continue
            kept.append(m)
    return merge_chord_cands(kept, xdist=36, ydist=26)


def collect_chord_words(words: list[dict], loose: bool = True) -> list[dict]:
    """Filter OCR words into chord candidates; slash chords allowed at low conf."""
    found: list[dict] = []
    for w in words:
        t = fix_ocr_chord(w["text"])
        if not is_valid_chord_token(t):
            continue
        is_slash = "/" in t
        if is_slash:
            # accept low-confidence slash when grammar matches strictly
            if not (CHORD_RE.match(t) or STRICT_CHORD_RE.match(t)):
                continue
            if w["conf"] < 0:
                continue
        else:
            if w["conf"] < 48:
                continue
            # bare X# is almost always OCR noise on these sheets (D + speck → D#)
            if re.match(r"^[A-G]#$", t):
                continue
        ww = dict(w)
        # OCR often glues underlines / bars into the box ("A,", tall "G")
        if len(t) <= 2:
            if ww["width"] > max(85, int(ww["height"] * 2.2)):
                ww["width"] = max(12, int(min(ww["height"], 40) * 0.9))
            if ww["height"] > 48:
                ww["height"] = max(22, min(40, int(ww["width"] * 1.15) if ww["width"] else 32))
        if not chord_like_size(ww, loose=loose):
            continue
        found.append(
            {
                "text": t,
                "left": ww["left"],
                "top": ww["top"],
                "width": ww["width"],
                "height": ww["height"],
                "conf": ww["conf"],
            }
        )
    return found


def cluster_bands(seeds: list[dict], tol: int = 28) -> list[tuple[int, int, list]]:
    seeds = sorted(seeds, key=lambda s: s["top"])
    bands: list[list] = []
    for s in seeds:
        if bands and abs(s["top"] - bands[-1][0]) <= tol:
            bands[-1][0] = min(bands[-1][0], s["top"])
            bands[-1][1] = max(bands[-1][1], s["top"] + s["height"])
            bands[-1][2].append(s)
        else:
            bands.append([s["top"], s["top"] + s["height"], [s]])
    return [(b[0], b[1], b[2]) for b in bands]


def overlaps(a: dict, b: dict, pad: int = 8) -> bool:
    return not (
        a["left"] + a["width"] + pad < b["left"]
        or b["left"] + b["width"] + pad < a["left"]
        or a["top"] + a["height"] + pad < b["top"]
        or b["top"] + b["height"] + pad < a["top"]
    )


def split_compound_ocr(
    tess: str, env: dict, im: Image.Image, box: tuple[int, int, int, int]
) -> str | None:
    """OCR left/right of a wide compound box; join with 或."""
    import numpy as np

    x0, y0, x1, y1 = box
    crop = im.crop((x0, y0, x1, y1)).convert("L")
    arr = np.array(crop)
    ink = (arr < 140).sum(axis=0).astype(float)
    if ink.size < 20:
        return None
    # smooth
    k = max(3, ink.size // 40)
    kernel = np.ones(k) / k
    smooth = np.convolve(ink, kernel, mode="same")
    # find densest blob in middle 40% — often the 或 character
    n = len(smooth)
    lo, hi = int(n * 0.25), int(n * 0.75)
    mid_slice = smooth[lo:hi]
    if mid_slice.size == 0:
        return None
    peak = lo + int(mid_slice.argmax())
    # expand peak to blob edges
    thr = smooth[peak] * 0.35
    L = peak
    while L > lo and smooth[L] > thr:
        L -= 1
    R = peak
    while R < hi and smooth[R] > thr:
        R += 1
    # left / right crops
    parts = []
    for a, b in ((0, max(L - 2, 1)), (min(R + 2, n - 1), n)):
        if b - a < 8:
            continue
        sub = crop.crop((a, 0, b, crop.height))
        sub = sub.resize((sub.width * 3, sub.height * 3), Image.Resampling.LANCZOS)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            sub.save(f.name)
            path = f.name
        try:
            txt = (
                subprocess.check_output(
                    [tess, path, "stdout", "-l", "eng", "--psm", "8"],
                    env=env,
                    stderr=subprocess.DEVNULL,
                )
                .decode()
                .strip()
            )
        except Exception:
            txt = ""
        finally:
            os.unlink(path)
        txt = normalize_text(txt.split()[0] if txt.split() else "")
        # cleanup OCR noise
        txt = re.sub(r"[^A-Ga-g#b0-9/]", "", txt)
        # FHm7 -> F#m7
        txt = re.sub(r"^F[#H]?m?7$", "F#m7", txt, flags=re.I)
        txt = txt.replace("H", "#")
        if CHORD_RE.match(txt):
            parts.append(txt)
    if len(parts) == 2:
        return f"{parts[0]}或{parts[1]}"
    if len(parts) == 1:
        return parts[0]
    return None



def detect_page_jianpu(tess: str, env: dict, im: Image.Image) -> list[dict]:
    """Full-page multi-pass OCR for 簡譜 / sheets without clear staves.

    Keep the pass count small: two full-page OCR modes, then band strips at
    2x only (gray+binary). Optional cheap string-garble recovery; skip the
    expensive per-box re-OCR unless recall looks thin.
    """
    all_c: list[dict] = []
    for binary in (False, True):
        all_c.extend(
            collect_chord_words(ocr_image_words(tess, env, im, binary=binary, psm=11))
        )

    seeds = [
        c
        for c in all_c
        if c["conf"] >= 45 or ("/" in c["text"] and CHORD_RE.match(c["text"]))
    ]
    bands = cluster_bands(seeds, tol=34)
    bands = [b for b in bands if len(b[2]) >= 2] or [b for b in bands if len(b[2]) >= 1]

    def in_band(w: dict, ymin: int, ymax: int, pad: int = 32) -> bool:
        cy = w["top"] + w["height"] // 2
        return (ymin - pad) <= cy <= (ymax + pad)

    banded: list[dict] = []
    for c in all_c:
        for ymin, ymax, _ in bands:
            if in_band(c, ymin, ymax):
                banded.append(c)
                break

    for ymin, ymax, _ in bands:
        pad = 28
        y0 = max(0, ymin - pad)
        y1 = min(im.height, ymax + pad)
        # scale=2 only: scale=1 doubles wall time on weak hosts with little recall gain
        for binary in (False, True):
            words = ocr_band_strip(
                tess, env, im, y0, y1, scale=2, binary=binary, psm=6
            )
            for w in words:
                t0 = fix_ocr_chord(w["text"])
                if (
                    t0
                    and re.match(r"^[A-G]$", t0)
                    and w["width"] > w["height"] * 2
                ):
                    w = {**w, "width": max(12, int(w["height"] * 0.9))}
                banded.extend(collect_chord_words([w], loose=True))

    merged = reject_lyric_orphans(merge_chord_cands(banded, xdist=34, ydist=26))
    # Cheap string recovery if a band looks sparse (reuse gray words already OCR'd)
    if bands and len(merged) < max(8, len(bands) * 2):
        raw_words = ocr_image_words(tess, env, im, binary=False, psm=11)
        for w in raw_words:
            raw = normalize_text(w["text"])
            t = fix_ocr_chord(w["text"])
            if is_valid_chord_token(t):
                continue
            if has_cjk(raw) and "或" not in raw:
                continue
            if not chord_like_size(w, loose=True):
                continue
            for ymin, ymax, _ in bands:
                if not in_band(w, ymin, ymax):
                    continue
                recovered = recover_garbled(raw, w["width"])
                if recovered and is_valid_chord_token(recovered.split("或")[0]):
                    if "或" in recovered:
                        parts = recovered.split("或")
                        if not all(CHORD_RE.match(p) for p in parts):
                            break
                    cand = {
                        "text": recovered,
                        "left": w["left"],
                        "top": w["top"],
                        "width": w["width"],
                        "height": w["height"],
                        "conf": w["conf"],
                    }
                    if not any(overlaps(cand, f) for f in banded):
                        banded.append(cand)
                break
        merged = reject_lyric_orphans(merge_chord_cands(banded, xdist=34, ydist=26))

    merged.sort(key=lambda c: (c["top"], c["left"]))
    return merged


def detect_page(
    tess: str, env: dict, im: Image.Image, page: int
) -> list[dict]:
    """Detect chord symbols. Auto-normalize scale; staff-band OCR or 簡譜 multi-pass."""
    median_h = estimate_text_height(tess, env, im)
    # Prefer ~34px: sharp enough for A/C# while staying inside size filters
    scaled, sf = scale_image_for_detect(im, median_h, target=TARGET_TEXT_PX)
    if abs(sf - 1.0) > 0.05:
        print(
            f"page {page}: normalize scale={sf:.3f} "
            f"(median text ≈{median_h:.0f}px → ~{TARGET_TEXT_PX:.0f}px) "
            f"{im.width}x{im.height} → {scaled.width}x{scaled.height}",
            file=sys.stderr,
        )

    staff_hits = detect_via_staves(tess, env, scaled)
    if staff_hits is not None and len(staff_hits) >= 2:
        hits = staff_hits
    else:
        hits = detect_page_jianpu(tess, env, scaled)

    hits.sort(key=lambda c: (c["top"], c["left"]))
    out = []
    for i, c in enumerate(hits):
        item = {
            "page": page,
            "index": i,
            "text": c["text"],
            "bbox": [c["left"], c["top"], c["width"], c["height"]],
            "conf": round(float(c["conf"]), 1),
        }
        item = map_chord_bbox(item, sf)
        if c.get("unconfident") or float(c.get("conf", 100)) < 45:
            item["unconfident"] = True
        out.append(item)
    for i, item in enumerate(out):
        item["index"] = i
    return out


def draw_preview(im: Image.Image, chords: list[dict], path: Path) -> None:
    vis = im.copy()
    draw = ImageDraw.Draw(vis)
    font = ImageFont.truetype(pick_font(), 16)
    for c in chords:
        x, y, w, h = c["bbox"]
        draw.rectangle([x - 2, y - 2, x + w + 2, y + h + 2], outline=(220, 0, 0), width=2)
        label = str(c["index"])
        draw.text((x, max(0, y - 16)), label, fill=(220, 0, 0), font=font)
    vis.save(path)


def cmd_detect(args: argparse.Namespace) -> None:
    t0 = time.perf_counter()
    env = setup_tesseract_env()
    tess = which_tesseract(env)
    src = Path(args.image)
    pages = rasterize(src)
    all_chords = []
    for pi, im in enumerate(pages):
        ch = detect_page(tess, env, im, page=pi)
        # reindex globally per page already
        all_chords.extend(ch)
        preview = Path(args.out).with_suffix("").as_posix() + f"_preview_p{pi}.png"
        if len(pages) == 1:
            preview = str(Path(args.out).with_name(Path(args.out).stem + "_preview.png"))
        draw_preview(im, ch, Path(preview))
        print(f"page {pi}: {len(ch)} chords → preview {preview}", file=sys.stderr)

    out = {
        "source": str(src),
        "pages": len(pages),
        "chords": all_chords,
    }
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    dt = time.perf_counter() - t0
    print(f"wrote {args.out} ({len(all_chords)} chords) in {dt:.2f}s", file=sys.stderr)


def parse_chord_parts(symbol: str) -> tuple[str, str, str | None]:
    """Return (root, quality_suffix, bass_or_None)."""
    m = re.match(rf"^({_ROOT})(.*?)(?:/({_ROOT}))?$", symbol)
    if not m:
        raise ValueError(f"not a chord: {symbol}")
    return m.group(1), m.group(2) or "", m.group(3)


def prefer_scale(key: str | None) -> list[str]:
    if key and key in FLAT_KEYS:
        return FLAT_SCALE
    return SHARP_SCALE


def transpose_note(note: str, semitones: int, scale: list[str]) -> str:
    idx = NOTE_INDEX[note]
    return scale[(idx + semitones) % 12]


def transpose_symbol(symbol: str, semitones: int, key: str | None) -> str:
    scale = prefer_scale(key)
    if "或" in symbol:
        return "或".join(transpose_symbol(p, semitones, key) for p in symbol.split("或"))
    root, qual, bass = parse_chord_parts(symbol)
    new_root = transpose_note(root, semitones, scale)
    new_bass = transpose_note(bass, semitones, scale) if bass else None
    out = new_root + qual
    if new_bass:
        out += "/" + new_bass
    return out


def sample_bg(im: Image.Image, bbox: list[int], pad: int = 3) -> tuple[int, int, int]:
    x, y, w, h = bbox
    # sample pixels just outside the box
    samples = []
    W, H = im.size
    for px, py in (
        (x - pad, y),
        (x + w + pad, y),
        (x, y - pad),
        (x, y + h + pad),
        (x - pad, y - pad),
        (x + w + pad, y + h + pad),
    ):
        if 0 <= px < W and 0 <= py < H:
            samples.append(im.getpixel((px, py)))
    if not samples:
        return (255, 255, 255)
    # median-ish: average
    r = sum(s[0] for s in samples) // len(samples)
    g = sum(s[1] for s in samples) // len(samples)
    b = sum(s[2] for s in samples) // len(samples)
    return (r, g, b)


def _or_split_x(im: Image.Image, bbox: list[int]) -> tuple[int, int] | None:
    """Return (left_end, right_start) absolute x for the 或 blob inside bbox."""
    import numpy as np

    x, y, w, h = bbox
    crop = im.crop((x, y, x + w, y + h)).convert("L")
    arr = np.array(crop)
    ink = (arr < 140).sum(axis=0).astype(float)
    if ink.size < 20:
        return None
    # binary ink columns, merge into runs
    col = ink > max(2.0, ink.max() * 0.08)
    runs = []
    i, n = 0, len(col)
    while i < n:
        if not col[i]:
            i += 1
            continue
        j = i
        while j < n and col[j]:
            j += 1
        runs.append((i, j))
        i = j
    if len(runs) < 3:
        return None
    # 或 is typically the widest run fully inside the middle 60%
    lo, hi = int(n * 0.2), int(n * 0.8)
    mid_runs = [(a, b) for a, b in runs if a >= lo and b <= hi]
    if not mid_runs:
        # fallback: run whose center is closest to image mid
        mid = n // 2
        mid_runs = [min(runs, key=lambda ab: abs((ab[0] + ab[1]) / 2 - mid))]
    a, b = max(mid_runs, key=lambda ab: ab[1] - ab[0])
    if b - a < 8:
        return None
    # small pad so we don't nibble the 或 strokes
    return x + a - 1, x + b + 1



def uniform_font(page_chords: list[dict], font_path: str, override: int | None):
    """One font per page: cap height = median height of plain (no '/', no 或) boxes."""
    if override:
        return load_font(font_path, override), override
    hs = sorted(
        c["bbox"][3]
        for c in page_chords
        if "/" not in c["text"] and "或" not in c["text"]
    )
    if not hs:
        hs = sorted(c["bbox"][3] for c in page_chords) or [24]
    cap = hs[len(hs) // 2]
    size = 8
    while size < 400:
        tb = load_font(font_path, size + 1).getbbox("D")
        if tb[3] - tb[1] > cap:
            break
        size += 1
    return load_font(font_path, size), size


def line_baselines(page_chords: list[dict]) -> dict[int, int]:
    """Baseline per chord: median bottom of plain chords on the same line."""
    out = {}
    for c in page_chords:
        y = c["bbox"][1]
        mates = [
            d
            for d in page_chords
            if abs(d["bbox"][1] - y) <= 20
            and "/" not in d["text"]
            and "或" not in d["text"]
        ]
        if mates:
            b = sorted(d["bbox"][1] + d["bbox"][3] for d in mates)
            out[id(c)] = b[len(b) // 2]
        else:
            _x0, y0, _w, h = c["bbox"]
            out[id(c)] = y0 + int(h * 0.8)
    return out


def paint_chord(im, draw, c, new_text, font, cjk_font, baseline, right_limit):
    """Cover old chord; draw new text at uniform font. Never shrink; extend right.
    Returns True if drawn text crosses right_limit (neighbor)."""
    x, y, w, h = c["bbox"]
    bg = sample_bg(im, c["bbox"])
    pad = 2
    if "或" in new_text:
        left, right = new_text.split("或", 1)
        split = _or_split_x(im, c["bbox"])
        lw = font.getlength(left)
        if split and x + lw + 2 <= split[0]:
            ls, rs = split
            draw.rectangle([x - pad, y - pad, ls, y + h + pad], fill=bg)
            draw.rectangle([rs, y - pad, x + w + pad, y + h + pad], fill=bg)
            draw.text((x, baseline), left, font=font, fill=(0, 0, 0), anchor="ls")
            draw.text((rs + 1, baseline), right, font=font, fill=(0, 0, 0), anchor="ls")
            end = rs + 1 + font.getlength(right)
            if end > x + w:
                draw.rectangle([x + w, y - pad, end + pad, y + h + pad], fill=bg)
                draw.text((rs + 1, baseline), right, font=font, fill=(0, 0, 0), anchor="ls")
            return end > right_limit
        draw.rectangle([x - pad, y - pad, x + w + pad, y + h + pad], fill=bg)
        cx = x
        for seg, f in ((left, font), ("或", cjk_font or font), (right, font)):
            draw.text((cx, baseline), seg, font=f, fill=(0, 0, 0), anchor="ls")
            cx += f.getlength(seg) + 1
        if cx > x + w:
            draw.rectangle([x + w, y - pad, cx + pad, y + h + pad], fill=bg)
            cx = x
            for seg, f in ((left, font), ("或", cjk_font or font), (right, font)):
                draw.text((cx, baseline), seg, font=f, fill=(0, 0, 0), anchor="ls")
                cx += f.getlength(seg) + 1
        return cx > right_limit
    tw = font.getlength(new_text)
    draw.rectangle([x - pad, y - pad, max(x + w, x + tw) + pad, y + h + pad], fill=bg)
    draw.text((x, baseline), new_text, font=font, fill=(0, 0, 0), anchor="ls")
    return x + tw > right_limit


def cmd_apply(args: argparse.Namespace) -> None:
    t0 = time.perf_counter()
    data = json.loads(Path(args.chords).read_text(encoding="utf-8"))
    chords = data["chords"]
    pages = rasterize(Path(args.image))
    font_path = pick_font()
    cjk_path = pick_cjk_font()
    semitones, key = int(args.semitones), args.key
    rows, out_count, missed, overlaps_ = [], 0, 0, 0
    sizes: set[int] = set()
    out_images = []
    for pi, im in enumerate(pages):
        draw = ImageDraw.Draw(im)
        pc = [c for c in chords if c.get("page", 0) == pi]
        if not pc:
            out_images.append(im)
            continue
        font, size = uniform_font(pc, font_path, args.font_size)
        sizes.add(size)
        cjk = load_font(cjk_path, size) if cjk_path else None
        bases = line_baselines(pc)
        for c in pc:
            old = c["text"]
            try:
                new = transpose_symbol(old, semitones, key)
            except ValueError:
                missed += 1
                rows.append((pi, c.get("index"), old, "???"))
                continue
            x, y = c["bbox"][0], c["bbox"][1]
            nxt = [
                d["bbox"][0]
                for d in pc
                if d is not c and abs(d["bbox"][1] - y) <= 20 and d["bbox"][0] > x
            ]
            limit = min(nxt) - 4 if nxt else im.width
            if paint_chord(im, draw, c, new, font, cjk, bases[id(c)], limit):
                overlaps_ += 1
                print(
                    f"warn: chord {c.get('index')} '{new}' runs into next chord",
                    file=sys.stderr,
                )
            out_count += 1
            rows.append((pi, c.get("index"), old, new))
        out_images.append(im)

    out_path = Path(args.out)
    if out_path.suffix.lower() == ".pdf":
        out_images[0].save(
            out_path, "PDF", resolution=300, save_all=True, append_images=out_images[1:]
        )
    else:
        if len(out_images) != 1:
            raise SystemExit("multi-page input requires --out *.pdf")
        out_images[0].save(out_path)

    map_path = out_path.with_suffix(".map.md")
    nl = chr(10)
    map_lines = [
        f"# Chord map ({semitones:+d} semitones"
        + (f", key={key}" if key else "")
        + f", font px={sorted(sizes)})",
        "",
        "| page | index | from | to |",
        "|-----:|------:|:-----|:---|",
    ]
    map_lines += [f"| {a} | {b} | {o} | {n} |" for a, b, o, n in rows]
    map_path.write_text(nl.join(map_lines) + nl, encoding="utf-8")

    dt = time.perf_counter() - t0
    print(
        f"chords in={len(chords)} out={out_count} missed={missed} | "
        f"font sizes used: {len(sizes)} (px={','.join(map(str, sorted(sizes)))}) | "
        f"overlaps={overlaps_} → {out_path} ({dt:.2f}s)"
    )
    print(f"mapping → {map_path}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(
        description="Transpose chord symbols on a song-sheet image/PDF"
    )
    sp = ap.add_subparsers(dest="cmd", required=True)

    d = sp.add_parser("detect", help="OCR chord symbols → chords.json + preview")
    d.add_argument("image", help="input image or PDF")
    d.add_argument("--out", required=True, help="output chords.json")
    d.set_defaults(func=cmd_detect)

    a = sp.add_parser("apply", help="paint transposed chords onto the sheet")
    a.add_argument("image", help="input image or PDF")
    a.add_argument("chords", help="chords.json from detect (hand-editable)")
    a.add_argument("--semitones", type=int, required=True)
    a.add_argument("--key", default=None, help="target key for sharp/flat spelling (e.g. G)")
    a.add_argument("--out", required=True, help="output .png or .pdf")
    a.add_argument(
        "--font-size",
        type=int,
        default=None,
        help="force one font px size for all chords (default: auto from median chord cap height)",
    )
    a.set_defaults(func=cmd_apply)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
