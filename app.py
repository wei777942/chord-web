#!/usr/bin/env python3
"""Chord transpose upload web app — in-process detect+apply (Render-ready)."""
from __future__ import annotations

import json
import os
import re
import secrets
import subprocess
import sys
import threading
import time
import traceback
from collections import defaultdict, deque
from pathlib import Path

from flask import (
    Flask,
    abort,
    jsonify,
    redirect,
    render_template,
    request,
    send_from_directory,
    url_for,
)

BASE = Path(__file__).resolve().parent
JOBS = BASE / "jobs"
CONFIG_PATH = BASE / "config.json"
TRANSPOSE_PY = BASE / "chord_transpose" / "transpose.py"
FAILED_LOG = JOBS / "failed.log"
MAX_BYTES = 20 * 1024 * 1024
ALLOWED_EXT = {".png", ".jpg", ".jpeg", ".pdf"}
RATE_LIMIT = 10
RATE_WINDOW = 3600

KEYS = [
    "C", "C#", "Db", "D", "D#", "Eb", "E", "F",
    "F#", "Gb", "G", "G#", "Ab", "A", "A#", "Bb", "B",
]
KEY_TO_SEMI = {
    "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3,
    "E": 4, "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8,
    "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11,
}
# One spelling per pitch class for detected-key display
PREFERRED_KEY = ["C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
NOTE_TO_PC = {
    "C": 0, "B#": 0,
    "C#": 1, "Db": 1,
    "D": 2,
    "D#": 3, "Eb": 3,
    "E": 4, "Fb": 4,
    "F": 5, "E#": 5,
    "F#": 6, "Gb": 6,
    "G": 7,
    "G#": 8, "Ab": 8,
    "A": 9,
    "A#": 10, "Bb": 10,
    "B": 11, "Cb": 11,
}
# Scale degrees relative to tonic → expected quality family
# I maj, ii min, iii min, IV maj, V maj/dom, vi min
DIATONIC = {
    0: "maj",
    2: "min",
    4: "min",
    5: "maj",
    7: "maj",  # V / V7
    9: "min",
}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_BYTES

_rate: dict[str, deque] = defaultdict(deque)


def load_password() -> str:
    env = os.environ.get("CHORD_WEB_PASSWORD", "")
    if env:
        return env
    if CONFIG_PATH.is_file():
        with open(CONFIG_PATH, encoding="utf-8") as f:
            return json.load(f)["password"]
    return ""


def check_password(pw: str) -> bool:
    expected = load_password()
    if not expected:
        return False
    return secrets.compare_digest(pw or "", expected)


def client_ip() -> str:
    forwarded = request.headers.get("X-Forwarded-For", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.remote_addr or "unknown"


def rate_ok(ip: str) -> bool:
    now = time.time()
    q = _rate[ip]
    while q and now - q[0] > RATE_WINDOW:
        q.popleft()
    if len(q) >= RATE_LIMIT:
        return False
    q.append(now)
    return True


def sanitize_filename(name: str) -> str:
    name = os.path.basename(name or "upload.bin")
    name = re.sub(r"[^\w.\-]+", "_", name, flags=re.UNICODE)
    name = name.strip("._") or "upload.bin"
    if len(name) > 180:
        stem, ext = os.path.splitext(name)
        name = stem[:160] + ext
    return name


def compute_semitones(from_key: str, to_key: str) -> int:
    a = KEY_TO_SEMI[from_key]
    b = KEY_TO_SEMI[to_key]
    delta = (b - a) % 12
    if delta > 6:
        delta -= 12
    # delta == 6 stays +6 (consistent pick of +6 over -6)
    return delta


def parse_chord_symbol(text: str) -> tuple[int, str] | None:
    """Return (pitch_class, quality_family) or None. Ignores slash bass."""
    if not text:
        return None
    s = str(text).strip()
    # OCR sometimes yields "A/C#或F#m7" — take first alternative
    if "或" in s:
        s = s.split("或")[0].strip()
    if "/" in s:
        s = s.split("/", 1)[0].strip()
    s = s.replace("♯", "#").replace("♭", "b")
    m = re.match(r"^([A-Ga-g])([#b]?)\s*(.*)$", s)
    if not m:
        return None
    root = m.group(1).upper() + m.group(2)
    rest = (m.group(3) or "").strip()
    pc = NOTE_TO_PC.get(root)
    if pc is None:
        return None

    r = rest.lower().replace(" ", "")
    # quality family
    if r.startswith(("m7b5", "m7♭5", "ø", "dim", "o", "°")):
        quality = "dim"
    elif r.startswith(("aug", "+")):
        quality = "aug"
    elif r.startswith(("maj7", "maj9", "maj13", "ma7", "Δ", "△")) or r in ("maj", "ma"):
        quality = "maj"
    elif r.startswith(("mmaj", "minmaj", "mΔ")):
        quality = "min"
    elif r.startswith(("min", "mi", "m")) or (r.startswith("-") and not r.startswith("--")):
        # m, min, mi, m7, m9, -7 …
        quality = "min"
    elif r.startswith("sus"):
        quality = "sus"
    else:
        # bare, 7, 9, 11, 13, 6, add9, dom → treat as major/dominant family
        quality = "maj"
    return pc, quality


def estimate_key(chord_texts: list[str]) -> str | None:
    """Score 12 major keys by diatonic fit; return preferred spelling or None."""
    parsed: list[tuple[int, str]] = []
    for t in chord_texts:
        p = parse_chord_symbol(t)
        if p is not None:
            parsed.append(p)
    if not parsed:
        return None

    scores = [0.0] * 12
    for tonic in range(12):
        score = 0.0
        for i, (pc, quality) in enumerate(parsed):
            deg = (pc - tonic) % 12
            expected = DIATONIC.get(deg)
            if expected is None:
                continue
            hit = False
            if expected == "maj" and quality in ("maj", "sus"):
                hit = True
            elif expected == "min" and quality == "min":
                hit = True
            if not hit:
                continue
            weight = 1.0
            # V7 (dominant seventh on degree 7) gets extra weight
            if deg == 7 and quality == "maj":
                # any V / V7 counts; bump a bit more for explicit 7 later via text? keep simple
                weight = 1.5
            # first / last chord as tonic
            if deg == 0 and i == 0:
                weight += 2.0
            if deg == 0 and i == len(parsed) - 1:
                weight += 2.0
            score += weight
        scores[tonic] = score

    best = max(range(12), key=lambda t: (scores[t], -t))
    if scores[best] <= 0:
        return None
    return PREFERRED_KEY[best]


def append_failed(job: dict) -> None:
    JOBS.mkdir(parents=True, exist_ok=True)
    with open(FAILED_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(job, ensure_ascii=False) + "\n")


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def form_error(message: str, status: int = 400):
    return render_template("index.html", keys=KEYS, error=message), status


def run_transpose_job(
    job_dir: Path,
    fname: str,
    from_key: str,
    to_key: str,
) -> None:
    """detect → (optional auto key) → apply; write result.json + result image."""
    meta_path = job_dir / "meta.json"
    result_path = job_dir / "result.json"
    src = job_dir / fname
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
        meta["status"] = "processing"
        write_json(meta_path, meta)

        chords_json = job_dir / "chords.json"
        ext = src.suffix.lower()
        out_ext = ".pdf" if ext == ".pdf" else ".png"
        out_img = job_dir / f"result{out_ext}"

        cmd_detect = [
            sys.executable,
            str(TRANSPOSE_PY),
            "detect",
            str(src),
            "--out",
            str(chords_json),
        ]
        try:
            subprocess.check_output(cmd_detect, cwd=str(BASE), timeout=300, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"detect failed: {e.output[-2000:] if e.output else e}") from e

        chords_data = json.loads(chords_json.read_text(encoding="utf-8"))
        chords = chords_data.get("chords", [])
        unconfident = [
            {"index": c.get("index"), "text": c.get("text"), "conf": c.get("conf")}
            for c in chords
            if c.get("unconfident")
        ]

        detected_key = None
        resolved_from = from_key
        if from_key == "auto":
            texts = [c.get("text", "") for c in chords]
            detected_key = estimate_key(texts)
            if not detected_key:
                err = {
                    "status": "error",
                    "message": "無法自動判斷原調，請改選「原調」後再試。",
                    "detected_key": None,
                }
                write_json(result_path, err)
                meta["status"] = "error"
                write_json(meta_path, meta)
                return
            resolved_from = detected_key
            meta["detected_key"] = detected_key
            meta["from_key"] = detected_key
            write_json(meta_path, meta)

        if resolved_from not in KEY_TO_SEMI or to_key not in KEY_TO_SEMI:
            raise RuntimeError("無效的調性")

        if KEY_TO_SEMI[resolved_from] == KEY_TO_SEMI[to_key]:
            err = {
                "status": "error",
                "message": f"原調與目標調相同（{resolved_from}），無需移調。請選擇不同的目標調。",
                "detected_key": detected_key,
            }
            write_json(result_path, err)
            meta["status"] = "error"
            write_json(meta_path, meta)
            return

        semitones = compute_semitones(resolved_from, to_key)
        meta["semitones"] = semitones
        meta["key"] = f"{resolved_from}->{to_key}"
        write_json(meta_path, meta)

        spell_key = to_key
        cmd_apply = [
            sys.executable,
            str(TRANSPOSE_PY),
            "apply",
            str(src),
            str(chords_json),
            "--semitones",
            str(semitones),
            "--out",
            str(out_img),
            "--key",
            spell_key,
        ]
        try:
            subprocess.check_output(cmd_apply, cwd=str(BASE), timeout=300, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"apply failed: {e.output[-2000:] if e.output else e}") from e

        map_path = out_img.with_suffix(".map.md")
        mapping = []
        missed = 0
        if map_path.is_file():
            for line in map_path.read_text(encoding="utf-8").splitlines():
                if not line.startswith("|"):
                    continue
                parts = [p.strip() for p in line.strip("|").split("|")]
                if len(parts) >= 4 and parts[0] not in ("page", "-----:"):
                    if parts[0].startswith("-"):
                        continue
                    try:
                        mapping.append({"from": parts[2], "to": parts[3]})
                    except Exception:
                        pass

        # count missed from map header / apply side channel if present
        if map_path.is_file():
            raw = map_path.read_text(encoding="utf-8")
            m_miss = re.search(r"missed\s*=\s*(\d+)", raw, re.I)
            if m_miss:
                missed = int(m_miss.group(1))

        warnings = []
        if unconfident:
            warnings.append(
                "以下和弦信心較低，請人工核對："
                + ", ".join(f"#{u['index']} {u['text']} (conf={u['conf']})" for u in unconfident)
            )

        msg_parts = [f"已移調 {len(chords)} 個和弦（{semitones:+d} 半音，拼寫調={spell_key}）"]
        if detected_key:
            msg_parts.insert(0, f"偵測原調：{detected_key}")

        result = {
            "status": "done",
            "count": len(chords),
            "mapping": mapping,
            "message": "。".join(msg_parts) if detected_key else msg_parts[0],
            "detected_key": detected_key,
            "from_key": resolved_from,
            "to_key": to_key,
            "semitones": semitones,
            "missed": missed,
            "warnings": warnings,
            "unconfident": unconfident,
        }
        write_json(result_path, result)
        meta["status"] = "done"
        meta["count"] = len(chords)
        meta["missed"] = missed
        write_json(meta_path, meta)
    except Exception as e:
        tb = traceback.format_exc()
        err = {
            "status": "error",
            "message": f"處理失敗：{e}",
            "detail": tb[-1500:],
        }
        write_json(result_path, err)
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
            meta["status"] = "error"
            write_json(meta_path, meta)
            append_failed({**meta, "error": str(e)})
        except Exception:
            pass


@app.route("/")
def index():
    return render_template("index.html", keys=KEYS)


@app.route("/upload", methods=["POST"])
def upload():
    password = request.form.get("password", "")
    if not check_password(password):
        return form_error("密碼錯誤", 403)

    ip = client_ip()
    if not rate_ok(ip):
        return form_error("此 IP 每小時最多 10 個任務，請稍後再試", 429)

    from_key = request.form.get("from_key", "auto") or "auto"
    to_key = request.form.get("to_key", "C")
    if from_key != "auto" and from_key not in KEY_TO_SEMI:
        return form_error("無效的原調")
    if to_key not in KEY_TO_SEMI:
        return form_error("無效的目標調")
    if from_key != "auto" and KEY_TO_SEMI[from_key] == KEY_TO_SEMI[to_key]:
        return form_error("原調與目標調相同，無需移調。請選擇不同的調。")

    f = request.files.get("file")
    if not f or not f.filename:
        return form_error("請選擇或拖入檔案")

    fname = sanitize_filename(f.filename)
    ext = Path(fname).suffix.lower()
    if ext not in ALLOWED_EXT:
        return form_error("僅支援 png / jpg / pdf")

    job_id = secrets.token_hex(6)
    job_dir = JOBS / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    dest = job_dir / fname
    f.save(dest)

    meta = {
        "job_id": job_id,
        "from_key": from_key,
        "to_key": to_key,
        "spell_key": to_key,
        "key": f"{from_key}->{to_key}",
        "file": fname,
        "created": time.time(),
        "ip": ip,
        "status": "submitted",
        "detected_key": None,
    }
    write_json(job_dir / "meta.json", meta)

    t = threading.Thread(
        target=run_transpose_job,
        args=(job_dir, fname, from_key, to_key),
        daemon=True,
    )
    t.start()

    return redirect(url_for("job_page", job_id=job_id))


@app.route("/job/<job_id>")
def job_page(job_id: str):
    if not re.fullmatch(r"[0-9a-f]{12}", job_id):
        abort(404)
    if not (JOBS / job_id).is_dir():
        abort(404)
    return render_template("job.html", job_id=job_id)


@app.route("/api/job/<job_id>")
def api_job(job_id: str):
    if not re.fullmatch(r"[0-9a-f]{12}", job_id):
        abort(404)
    job_dir = JOBS / job_id
    if not job_dir.is_dir():
        abort(404)

    result_path = job_dir / "result.json"
    meta_path = job_dir / "meta.json"
    meta = {}
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            meta = {}

    if result_path.is_file():
        with open(result_path, encoding="utf-8") as f:
            result = json.load(f)
        status = result.get("status", "done")
        detected = result.get("detected_key") or meta.get("detected_key")
        out = {
            "status": status,
            "message": result.get("message", ""),
            "mapping": result.get("mapping"),
            "count": result.get("count"),
            "warnings": result.get("warnings") or [],
            "unconfident": result.get("unconfident") or [],
            "detected_key": detected,
            "from_key": result.get("from_key") or meta.get("from_key"),
            "to_key": result.get("to_key") or meta.get("to_key"),
            "semitones": result.get("semitones", meta.get("semitones")),
            "missed": result.get("missed", meta.get("missed")),
            "download": None,
        }
        if status in ("done", "ok"):
            for p in sorted(job_dir.iterdir()):
                if p.name.startswith("result.") and p.name != "result.json" and not p.name.endswith(".map.md"):
                    out["download"] = url_for("download_result", job_id=job_id, filename=p.name)
                    break
            if out["status"] == "ok":
                out["status"] = "done"
        return jsonify(out)

    return jsonify({
        "status": "processing",
        "message": "處理中（自動辨識並移調）…",
        "detected_key": meta.get("detected_key"),
    })


@app.route("/job/<job_id>/download/<path:filename>")
def download_result(job_id: str, filename: str):
    if not re.fullmatch(r"[0-9a-f]{12}", job_id):
        abort(404)
    if ".." in filename or filename.startswith("/"):
        abort(404)
    job_dir = JOBS / job_id
    path = job_dir / filename
    if not path.is_file() or not filename.startswith("result."):
        abort(404)
    return send_from_directory(job_dir, filename, as_attachment=True)


@app.route("/health")
def health():
    return jsonify({"ok": True, "transpose": TRANSPOSE_PY.is_file()})


if __name__ == "__main__":
    JOBS.mkdir(parents=True, exist_ok=True)
    host = os.environ.get("CHORD_WEB_HOST", "0.0.0.0")
    port = int(os.environ.get("CHORD_WEB_PORT", "8788"))
    app.run(host=host, port=port, threaded=True)
