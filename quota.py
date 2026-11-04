"""Belegung von Home und Scratch (Euler: lquota; sonst df), fuer refill.py und watch_paper.py.

Hintergrund (4.11.2026): 39 GB Job-Logs haben das Home-Quota (50 GB) gefuellt; Jobs und Daemon sind mit
"OSError 122 Disk quota exceeded" abgestuerzt. refill.py reicht deshalb ab HOME_LIMIT (80 %) der Soft Quota (Platz
oder Dateien) nichts mehr ein und beendet den Daemon; fuer Scratch nur eine Warnung.

lquota (Euler) gibt eine Tabelle aus:
    | Storage location:     | Quota type: | Used:     | Soft quota: | Hard quota: |
    | /cluster/home/user    | space       |  10.04 GB |    50.00 GB |    55.00 GB |
    | /cluster/home/user    | files       |      8961 |      160000 |      200000 |
    | /cluster/scratch/user | space       |  16.38 kB |     2.50 TB |     2.70 TB |
Fehlt lquota oder ist die Ausgabe nicht lesbar: df -Pk <Pfad> (auf Euler zeigt df fuer das eigene Home die Quota;
auf einem geteilten Dateisystem ist es nur die Belegung des Dateisystems). Geht beides nicht: None (unbekannt).
"""
import os
import re
import subprocess
from pathlib import Path

HOME_LIMIT = 0.8
SCRATCH_LIMIT = 0.8
UNITS = {"": 1, "b": 1, "k": 1e3, "kb": 1e3, "m": 1e6, "mb": 1e6, "g": 1e9, "gb": 1e9, "t": 1e12, "tb": 1e12,
         "p": 1e15, "pb": 1e15, "kib": 2 ** 10, "mib": 2 ** 20, "gib": 2 ** 30, "tib": 2 ** 40, "pib": 2 ** 50}


def _run(cmd, timeout=60):
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout if result.returncode == 0 else None


def parse_size(text):
    """'10.04 GB' -> 1.004e10, '8961' -> 8961.0; None wenn nicht lesbar."""
    match = re.fullmatch(r"\s*([\d.,]+)\s*([A-Za-z]*)\s*", text or "")
    if not match or match.group(2).lower() not in UNITS:
        return None
    try:
        return float(match.group(1).replace(",", "")) * UNITS[match.group(2).lower()]
    except ValueError:
        return None


def parse_lquota(text):
    """[dict(ort, art, used, soft, hard)] aus der lquota-Tabelle; Zeilen, die nicht passen, werden uebersprungen."""
    rows = []
    for line in (text or "").splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 5 or cells[1] not in ("space", "files"):
            continue
        used, soft, hard = (parse_size(c) for c in cells[2:5])
        if used is None or not soft:
            continue
        rows.append(dict(ort=cells[0], art=cells[1], used=used, soft=soft, hard=hard))
    return rows


def parse_df(text, location):
    """df -Pk: [dict(ort, art='space', used, soft=Groesse, hard=Groesse)] (KiB in Bytes)."""
    lines = [l.split() for l in (text or "").splitlines()[1:] if l.strip()]
    if not lines or len(lines[-1]) < 6:
        return []
    try:
        size, used = int(lines[-1][1]) * 1024, int(lines[-1][2]) * 1024
    except ValueError:
        return []
    return [dict(ort=str(location), art="space", used=used, soft=size, hard=size)] if size else []


def locations():
    """{'home': Pfad, 'scratch': Pfad oder None}."""
    home = Path(os.environ.get("HOME", str(Path.home())))
    scratch = os.environ.get("SCRATCH") or (f"/cluster/scratch/{os.environ['USER']}" if os.environ.get("USER")
                                            else None)
    return {"home": home, "scratch": Path(scratch) if scratch else None}


def _belongs(row_location, path):
    """Gehoert eine lquota-Zeile zu path? (Pfade koennen per Symlink verschieden geschrieben sein.)"""
    a, b = str(row_location).rstrip("/"), str(path).rstrip("/")
    try:
        return a == b or os.path.realpath(a) == os.path.realpath(b) or Path(a).name == Path(b).name and \
            Path(a).parent.name == Path(b).parent.name
    except OSError:
        return a == b


def usage(lquota_text=False, df_texts=None):
    """{'home': {..}, 'scratch': {..}} je dict(quelle, rows=[...], anteil=max(used/soft) oder None).
    lquota_text/df_texts nur fuer Tests (False/None: Befehle selbst aufrufen)."""
    where = locations()
    text = _run(["lquota"]) if lquota_text is False else lquota_text
    rows = parse_lquota(text)
    out = {}
    for name, path in where.items():
        mine = [r for r in rows if path is not None and _belongs(r["ort"], path)]
        source = "lquota"
        if not mine and path is not None:
            df = (df_texts or {}).get(name) if df_texts is not None else (
                _run(["df", "-Pk", str(path)]) if path.exists() else None)
            mine, source = parse_df(df, path), "df"
        share = max((r["used"] / r["soft"] for r in mine), default=None)
        out[name] = dict(pfad=path, quelle=source if mine else None, rows=mine, anteil=share)
    return out


def fmt_bytes(n):
    for unit, size in (("TB", 1e12), ("GB", 1e9), ("MB", 1e6), ("kB", 1e3)):
        if n >= size:
            return f"{n / size:.1f} {unit}"
    return f"{n:.0f} B"


def describe(name, info):
    """'Home 41.0 GB / 50.0 GB (82 %), Dateien 9000 / 160000 (6 %) [lquota]' oder '... unbekannt'."""
    label = name.capitalize()
    if not info or info["anteil"] is None:
        return f"{label}: unbekannt (weder lquota noch df)"
    parts = []
    for r in info["rows"]:
        if r["art"] == "space":
            parts.append(f"{fmt_bytes(r['used'])} / {fmt_bytes(r['soft'])} ({r['used'] / r['soft']:.0%})")
        else:
            parts.append(f"Dateien {r['used']:.0f} / {r['soft']:.0f} ({r['used'] / r['soft']:.0%})")
    return f"{label}: " + ", ".join(parts) + f" [{info['quelle']}]"


def check(info=None, home_limit=HOME_LIMIT, scratch_limit=SCRATCH_LIMIT):
    """(Home in Ordnung?, Textzeilen). Unbekannte Belegung blockiert nicht (nur Hinweis)."""
    info = usage() if info is None else info
    ok, lines = True, []
    home, scratch = info.get("home"), info.get("scratch")
    if home and home["anteil"] is not None and home["anteil"] > home_limit:
        ok = False
        lines.append(f"WARNUNG: {describe('home', home)} ueber {home_limit:.0%} der Soft Quota -> nichts "
                     "einreichen, Daemon beenden. Platz schaffen (z. B. du -sh ~/HPOTreeBoosting/* ; Logs liegen in "
                     "jobs_paper/logs -> $SCRATCH), dann python refill.py --run beide --daemon")
    elif not home or home["anteil"] is None:
        lines.append("Hinweis: Home-Belegung unbekannt (weder lquota noch df) -> nicht geprueft")
    if scratch and scratch["anteil"] is not None and scratch["anteil"] > scratch_limit:
        lines.append(f"WARNUNG: {describe('scratch', scratch)} ueber {scratch_limit:.0%} der Soft Quota "
                     "(Job-Logs in jobs_paper/logs)")
    return ok, lines
