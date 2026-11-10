"""Hilfen fuer die Rechenjobs (run_ablation.py, run_default.py): kleine Logs und atomar geschriebene Ergebnisse.

quiet_logging(): Logs je Job nur Start, Fehler und Ende (Euler: 39 GB Job-Logs haben das Home-Quota gefuellt):
  optuna auf WARNING (sonst eine INFO-Zeile je Versuch),
  gpboost-Warnungen "<alias> in param dict is overridden" und "Overriding the parameters from Reference Dataset"
  (je Modell zweimal mit Quelltextzeile) per warnings.filterwarnings unterdrueckt,
  gpboost-Infos ("[GPBoost] [Info] ...", je Training drei Zeilen) verworfen, native Warnungen und Fehler bleiben.
  SMAC und das Zufallsgitter (cv-Ausgabe je Boosting-Runde) stellt run_ablation.quiet_methods() leise.
write_csv_atomic(): Ergebnis-CSV erst in <datei>.tmp schreiben (bei append: alte Datei kopieren und anhaengen),
  auf die Platte zwingen und dann umbenennen; ein Abbruch (Zeitlimit, Disk quota exceeded) hinterlaesst nie eine
  halbe Datei, hoechstens eine .tmp-Datei, die check_results.py meldet.
temp_dir(): neuer Ordner fuer Zwischendateien der Bibliotheken (DEHB-Log) in $TMPDIR, sonst $SCRATCH, sonst im
  System-Temp, nie unter $HOME (Quota); der Aufrufer loescht ihn.
"""
import logging
import os
import shutil
import tempfile
import warnings
from pathlib import Path

GPBOOST_WARNINGS = (r".* in param dict is overridden", r"Overriding the parameters from Reference Dataset")


class _GPBoostLogger(logging.Logger):
    """Wie gpboosts _DummyLogger (Warnungen ueber warnings.warn, damit die Filter greifen), aber ohne Infos.
    gpboost.register_logger verlangt eine Unterklasse von logging.Logger."""

    def __init__(self):
        super().__init__("gpboost_quiet")

    def info(self, msg, *args, **kwargs):
        if "[Info]" not in msg:          # native "[GPBoost] [Warning]"/"[Fatal]" bleiben sichtbar
            print(msg)

    def warning(self, msg, *args, **kwargs):
        warnings.warn(msg, stacklevel=3)


def quiet_logging():
    for pattern in GPBOOST_WARNINGS:
        warnings.filterwarnings("ignore", message=pattern)
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        pass
    try:
        import gpboost
        gpboost.register_logger(_GPBoostLogger())
    except (ImportError, AttributeError, TypeError):
        pass


def tmp_path(path):
    path = Path(path)
    return path.with_name(path.name + ".tmp")


def write_csv_atomic(frame, path, append=False):
    """frame nach path (ohne Index); append=True haengt ohne Kopfzeile an eine bestehende Datei an."""
    path = Path(path)
    tmp = tmp_path(path)
    try:
        append = append and path.exists()
        if append:
            shutil.copyfile(path, tmp)
        with open(tmp, "a" if append else "w", newline="") as handle:
            frame.to_csv(handle, index=False, header=not append)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def temp_dir(prefix="hpo_"):
    """Neuer temporaerer Ordner (Path) in $TMPDIR (Euler: je Job), sonst $SCRATCH, sonst tempfile.gettempdir();
    Orte unter $HOME werden uebersprungen."""
    home = os.path.realpath(os.path.expanduser("~"))
    for base in (os.environ.get("TMPDIR"), os.environ.get("SCRATCH"), tempfile.gettempdir()):
        if not base or not os.path.isdir(base) or not os.access(base, os.W_OK):
            continue
        if (os.path.realpath(base) + os.sep).startswith(home + os.sep):
            continue
        return Path(tempfile.mkdtemp(prefix=prefix, dir=base))
    raise OSError("kein temporaerer Ordner ausserhalb von $HOME ($TMPDIR, $SCRATCH, /tmp)")
