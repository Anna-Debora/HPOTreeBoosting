# Reproduktions-Testlauf (Paper Koster & Sigrist) – vorbereitet, NICHT gestartet

Ziel: stichprobenartig prüfen, ob der heutige Code die Ergebnisse aus "Selecting Hyperparameters for
Tree-Boosting" (Koster & Sigrist, arXiv:2602.05786v3) im Wesentlichen wiedergibt, bevor die grossen
Läufe starten.

## Wo die Paper-Ergebnisse liegen

- **Im Repository gibt es keine Ergebnisdateien des Papers.** `Results/` ist leer und per `.gitignore`
  ausgeschlossen. Auch das Original-Repo `github.com/fl0risk/HPOTreeBoosting` enthält nur Code.
- Das Paper selbst: `~/Documents/ETH/HS26/2602.05786v3.pdf` (lokal, nicht im Git).
  - Table 5 (S. 7): normalisierte Scores, Ränge, relative Differenzen, gemittelt über alle 59 Aufgaben.
  - Abbildungen 5–10 (S. 15–20): R², RMSE, Accuracy und Log Loss **pro Datensatz** als Funktion der
    Versuche (Suchraum num_leaves, 20 Seeds, mit 95-%-Band über die Seeds).
- `extract_paper_figures.py` liest die Kurven direkt aus den Vektorgrafiken des PDFs aus und schreibt sie
  nach `paper_reference_values.csv` (118 Felder, 7 Methoden, t = 45, 60, …, 135, mit Bandgrenzen).
  Datensätze, deren Name in zwei Suiten vorkommt (z. B. diamonds in 335 und 336), sind dort nicht
  unterscheidbar und als `doppelt` markiert. Für sie gibt es keinen Vergleich.
- Skripte, die die Experimente starten: `run_experiments.py` (Gitter, Zufall, TPE, GP-BO je Strategie),
  `run_experiment_H_SMAC.py` (SMAC, Hyperband), `run_default.py` (Default), Job-Dateien mit
  `make_euler_jobs.py` (grosse Läufe) bzw. `make_repro_jobs.py` (dieser Testlauf).

## Auswahl

| Punkt | Wahl | Grund |
|---|---|---|
| Bibliothek | nur GPBoost | Das Paper nutzt nur GPBoost (LightGBM-Kern). Kein XGBoost, kein CatBoost. |
| Suchraum | nur `num_leaves` | Hauptergebnis des Papers (Abb. 1, 5–10) |
| Methoden | Default, deterministisches Gitter, Zufallsgitter, TPE, GP-BO, SMAC | alle Methoden aus Abb. 5–10 ausser Hyperband (im Paper zweitschlechteste Methode). SMAC ist dabei, weil „SMAC ist am besten“ der Hauptbefund ist. |
| Seeds | 27225, 32244 | die ersten zwei der 20 Paper-Seeds (`utils_modelling.py`) |
| Aufgaben | 7 (siehe unten) | beide Aufgabentypen, alle vier Suiten, eindeutige Namen, im Paper deutliche Unterschiede zwischen den Methoden (Spannweite ≥ 2.6 Seed-Streuungen), kleine bis mittlere Datensätze |

| Suite | Task | Datensatz | Art | Zeilen | Trennschärfe¹ | Sek./Versuch² |
|---|---|---|---|---|---|---|
| 336 | 361076 | wine_quality | Regression | 6 497 | 6.5 | 0.38 |
| 336 | 361085 | sulfur | Regression | 10 081 | 3.1 | 0.54 |
| 335 | 361097 | Mercedes_Benz_Greener_Manufacturing | Regression | 4 209 | 5.8 | 0.85 |
| 334 | 361286 | compas-two-years | Klassifikation | 4 966 | 2.6 | 0.04 |
| 337 | 361065 | MagicTelescope | Klassifikation | 13 376 | 4.2 | 0.66 |
| 337 | 361278 | heloc | Klassifikation | 10 000 | 3.3 | 0.30 |
| 337 | 361055 | credit | Klassifikation | 16 714 | 4.3 | 0.23 |

¹ Spannweite der Methoden im Paper bei t = 135 geteilt durch die Streuung eines einzelnen Seeds. Je
grösser, desto eher lässt sich mit 1–2 Seeds erkennen, ob die Rangfolge stimmt.
² lokal gemessen (1 Thread, 27 Zufallsversuche auf Fold 0, inklusive Test-Neuanpassung).
Nicht genommen, weil zu teuer (mit Sicherheitsfaktor über 4 h pro Job): cpu_act, MiamiHousing2016,
houses. Nicht genommen, weil der Name in zwei Suiten vorkommt: z. B. abalone, diamonds, electricity.

## Jobs und Kosten

- 35 Jobs in einem Slurm-Array: 14 Hauptjobs (7 Aufgaben × 2 Seeds), 14 SMAC-Jobs, 7 Default-Jobs.
- Geschätzte Kernstunden: **≈ 50** (Hauptjobs 2–4.4 h, SMAC 0.1–0.8 h, Default wenige Minuten,
  jeweils mit Sicherheitsfaktor 2). Alle Jobs laufen parallel, die Wanduhrzeit beträgt also etwa
  **4–5 Stunden** plus Wartezeit in der Warteschlange.
- Der grösste Posten ist das GP-BO-Surrogat (scikit-optimize): 607 s für 270 Aufrufe und einen Fold,
  also ~0.9 h pro Hauptjob ohne jedes Training. Deshalb Partition `normal.24h` mit 12 h Zeitlimit.
- Günstigere Variante mit nur einem Seed: `python make_repro_jobs.py --seeds 27225` → 21 Jobs, ≈ 25
  Kernstunden. Damit ist der Vergleich pro Aufgabe weniger sicher.
- Dateien: `jobs_repro/jobs.tsv`, `jobs_repro/kosten.tsv`, `jobs_repro/run_repro_array.sbatch`.
- Ergebnisse landen auf Euler in `~/HPOTreeBoosting/Results_repro/`:
  `Results/seed_*/gpboost/<suite>_<task>__num_leaves.csv`, `Results_SMAC/seed_*/gpboost/…`,
  `Results_Default_Run/gpboost/…`. Logs in `jobs_repro/logs/`.

### Starten auf Euler (Reihenfolge)

`data/` ist nicht im Git. Der erste Versuch (Job 18448979) brach deshalb in allen 35 Teiljobs nach etwa
20 s mit `FileNotFoundError: data/336_361076/336_361076_X.csv` ab. Vor dem Start müssen die Daten der
7 Aufgaben auf Euler erzeugt werden:

```
cd ~/HPOTreeBoosting
git pull origin xgboost-catboost
bash euler_prepare_repro_data.sh                          # prüft Pakete, lädt die 7 Aufgaben, Kontrolle am Ende
sbatch --array=0 jobs_repro/run_repro_array.sbatch        # Probe: nur Teiljob 0 (Default, wine_quality, < 1 min)
sbatch jobs_repro/run_repro_array.sbatch                  # erst wenn die Probe in Ordnung ist: alle 35 Teiljobs
```

`euler_prepare_repro_data.sh` lädt nur fehlende oder unvollständige Ordner nach und darf mehrfach laufen.
Fehlen die Daten trotzdem, bricht jeder Teiljob jetzt sofort mit
`ABBRUCH: data/<suite>_<task>/<suite>_<task>_X.csv fehlt oder ist leer` ab (Exit-Code 2).
Jeder SMAC-Teiljob schreibt seine SMAC-Zwischendateien nach `smac3_output/job_<Slurm-Job-ID>/`. Vorher
hätten sich gleichzeitige SMAC-Jobs mit gleichem Seed und Suchraum denselben Ordner geteilt und ihn
gegenseitig gelöscht (`overwrite=True`).

## Vergleich mit dem Paper

`compare_with_paper.py --results Results_repro` (auf Euler oder lokal nach dem Kopieren von `Results_repro/`):

- Pro Aufgabe, Methode und Kennzahl: unser Testwert des bisher besten Versuchs nach t = 135 Versuchen
  (Mittel über 5 Folds und unsere Seeds) gegen die Paper-Kurve bei t = 135.
- Statistischer Massstab: Aus dem 95-%-Band im Paper folgt die Streuung eines einzelnen Seeds.
  z = Abweichung / erwartete Streuung. |z| ≤ 2 heisst: im Rahmen des Zufalls.
- Deterministisches Gitter: im Paper 135 Konfigurationen, bei uns 270. Verglichen wird jeweils das
  Gitterende. Default hat keine Seed-Streuung, dort zählt die relative Abweichung (< 2 %).
- Zusätzlich: gleicher Gewinner wie im Paper? Spearman-Korrelation der Rangfolge der Methoden pro Aufgabe,
  mittlere Ränge über alle Aufgaben (qualitativ vergleichbar mit Table 5).
- Ausgabe: `repro_vergleich.csv` (alle Zahlen) und `repro_vergleich.md` (Zusammenfassung).

**Wichtig beim Lesen:** Der Code ist nicht mehr genau der des Papers. Heute 270 statt 135 Versuche
(verglichen wird bei 135), `lambda_l1` zusätzlich im Suchraum, grösseres Zufallsgitter und
deterministisches Gitter, GPBoost 1.7.4. Ausserdem hat der Paper-Code `line_search_step_length` an
`cv()` übergeben, was GPBoost ≥ 1.5.8 nicht mehr kennt (siehe Übergabe-Notiz). Kleine systematische
Abweichungen sind deshalb normal. Bedenklich wären andere Gewinner auf vielen Aufgaben oder Abweichungen
weit über der Seed-Streuung.

## Vorab-Probe (lokal, 26.10.2026): compas-two-years, Seed 27225

Alle drei Job-Arten einmal lokal genau mit den Befehlen aus dem sbatch-Skript ausgeführt (Rechner stark
ausgelastet): Hauptjob 1.5 CPU-Stunden (Schätzung inkl. Faktor 2: 1.9 h), SMAC 12 min, Default 5 s.
Ergebnis von `compare_with_paper.py` (1 Aufgabe, 1 Seed, t = 135):

| Methode | Accuracy wir / Paper | Log Loss wir / Paper | Urteil |
|---|---|---|---|
| Default | 0.6702 / 0.6700 | 0.7092 / 0.7090 | ok (< 0.03 %) |
| Deterministisches Gitter | 0.6845 / 0.6845 | 0.6039 / 0.6039 | ok (identisch) |
| Zufallsgitter | 0.6859 / 0.6851 | 0.6044 / 0.6062 | ok |
| GP-BO | 0.6861 / 0.6877 | 0.6041 / 0.6041 | ok |
| SMAC | 0.6861 / 0.6870 | 0.6073 / 0.6042 | Accuracy ok, Log Loss auffällig (z = 4.9) |
| TPE | 0.6847 / 0.6873 | 0.6071 / 0.6043 | auffällig (z = −2.4 / 3.8) |

Default und Gitter stimmen exakt mit dem Paper überein. Daten, Folds und Modell sind also dieselben, und
die ausgelesenen Paper-Werte stimmen. Die adaptiven Methoden liegen innerhalb von etwa 0.4 %.
Ob die Abweichungen bei TPE/SMAC System haben (z. B. wegen `lambda_l1`), zeigt erst der volle Testlauf.
