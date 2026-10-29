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
    return delta


def append_failed(job: dict) -> None:
    JOBS.mkdir(parents=True, exist_ok=True)
    with open(FAILED_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(job, ensure_ascii=False) + "\n")


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def run_transpose_job(job_dir: Path, fname: str, semitones: int, spell_key: str | None) -> None:
    """detect → apply in-process; write result.json + result image."""
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
        subprocess.check_call(cmd_detect, cwd=str(BASE), timeout=120)

        chords_data = json.loads(chords_json.read_text(encoding="utf-8"))
        chords = chords_data.get("chords", [])
        unconfident = [
            {"index": c.get("index"), "text": c.get("text"), "conf": c.get("conf")}
            for c in chords
            if c.get("unconfident")
        ]

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
        ]
        if spell_key:
            cmd_apply.extend(["--key", spell_key])
        subprocess.check_call(cmd_apply, cwd=str(BASE), timeout=120)

        # parse mapping markdown if present
        map_path = out_img.with_suffix(".map.md")
        mapping = []
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

        warnings = []
        if unconfident:
            warnings.append(
                "以下和弦信心較低，請人工核對："
                + ", ".join(f"#{u['index']} {u['text']} (conf={u['conf']})" for u in unconfident)
            )

        result = {
            "status": "done",
            "count": len(chords),
            "mapping": mapping,
            "message": (
                f"已移調 {len(chords)} 個和弦（{semitones:+d} 半音"
                + (f"，拼寫調={spell_key}" if spell_key else "")
                + "）"
            ),
            "warnings": warnings,
            "unconfident": unconfident,
        }
        write_json(result_path, result)
        meta["status"] = "done"
        meta["count"] = len(chords)
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
    return render_template(
        "index.html",
        keys=KEYS,
        semitone_range=list(range(-11, 12)),
    )


@app.route("/upload", methods=["POST"])
def upload():
    password = request.form.get("password", "")
    if not check_password(password):
        return render_template(
            "index.html",
            keys=KEYS,
            semitone_range=list(range(-11, 12)),
            error="密碼錯誤",
        ), 403

    ip = client_ip()
    if not rate_ok(ip):
        return render_template(
            "index.html",
            keys=KEYS,
            semitone_range=list(range(-11, 12)),
            error="此 IP 每小時最多 10 個任務，請稍後再試",
        ), 429

    mode = request.form.get("mode", "semitones")
    spell_key = None
    if mode == "keys":
        from_key = request.form.get("from_key", "C")
        to_key = request.form.get("to_key", "C")
        if from_key not in KEY_TO_SEMI or to_key not in KEY_TO_SEMI:
            return render_template(
                "index.html",
                keys=KEYS,
                semitone_range=list(range(-11, 12)),
                error="無效的調性",
            ), 400
        semitones = compute_semitones(from_key, to_key)
        key_label = f"{from_key}->{to_key}"
        spell_key = to_key
    else:
        try:
            semitones = int(request.form.get("semitones", "0"))
        except ValueError:
            return render_template(
                "index.html",
                keys=KEYS,
                semitone_range=list(range(-11, 12)),
                error="半音數無效",
            ), 400
        if semitones < -11 or semitones > 11:
            return render_template(
                "index.html",
                keys=KEYS,
                semitone_range=list(range(-11, 12)),
                error="半音數須在 -11 到 11",
            ), 400
        # optional spelling key from to_key field
        to_key = request.form.get("to_key") or request.form.get("spell_key")
        if to_key in KEY_TO_SEMI:
            spell_key = to_key
        key_label = f"semitones:{semitones}"

    f = request.files.get("file")
    if not f or not f.filename:
        return render_template(
            "index.html",
            keys=KEYS,
            semitone_range=list(range(-11, 12)),
            error="請選擇檔案",
        ), 400

    fname = sanitize_filename(f.filename)
    ext = Path(fname).suffix.lower()
    if ext not in ALLOWED_EXT:
        return render_template(
            "index.html",
            keys=KEYS,
            semitone_range=list(range(-11, 12)),
            error="僅支援 png / jpg / pdf",
        ), 400

    job_id = secrets.token_hex(6)
    job_dir = JOBS / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    dest = job_dir / fname
    f.save(dest)

    meta = {
        "job_id": job_id,
        "semitones": semitones,
        "key": key_label,
        "spell_key": spell_key,
        "file": fname,
        "created": time.time(),
        "ip": ip,
        "status": "submitted",
    }
    write_json(job_dir / "meta.json", meta)

    t = threading.Thread(
        target=run_transpose_job,
        args=(job_dir, fname, semitones, spell_key),
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
    if result_path.is_file():
        with open(result_path, encoding="utf-8") as f:
            result = json.load(f)
        status = result.get("status", "done")
        out = {
            "status": status,
            "message": result.get("message", ""),
            "mapping": result.get("mapping"),
            "count": result.get("count"),
            "warnings": result.get("warnings") or [],
            "unconfident": result.get("unconfident") or [],
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

    return jsonify({"status": "processing", "message": "處理中（自動辨識並移調）…"})


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
