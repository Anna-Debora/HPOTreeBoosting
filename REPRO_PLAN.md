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

---

# Grosse Läufe auf allen 59 Aufgaben (vorbereitet 30.10.2026)

Zwei GPBoost-Läufe, gleichzeitig auf Euler, Strategie `num_leaves` (wie Table 5 und Abb. 1, 5–10), 20 Seeds des
Papers, 5 Folds, alle 7 Methoden (Default, deterministisches Gitter, GP-BO, Hyperband, Zufallsgitter, SMAC, TPE),
Kategorien nativ (kein One-Hot):

| Lauf | Ordner | Versuche | lambda_l1 | deterministisches Gitter |
|---|---|---|---|---|
| `paper` | `Results_paper/` | 135 | nein (fest 0) | 135 Punkte wie im Paper-Code (`--paper_grid`) |
| `l1_270` | `Results_l1_270/` | 270 | ja, linear (0, 100) | heutiges Gitter, 270 Punkte (bagging_fraction {0.5, 1}, ohne lambda_l1) |

Beide nutzen `run_ablation.py` (Lauf `paper` = Einstellung „A“: `--trials 135 --no_lambda_l1 --paper_grid`, Lauf
`l1_270` = `--trials 270`, also der heutige Code) und für Default `run_default.py`. Hyperband hat keine Versuchszahl
(R = 2150, η = 2.8, 11 Ergebniszeilen je Fold) und ist in beiden Läufen bis auf lambda_l1 gleich; Default ist in
beiden Läufen identisch.

## Table 5: Definitionen

Quellen: Paper Abschnitt 2.5 und 3 (S. 5–7) und der Paper-Code (Commit a19b4ff: `utils_plot_maker.create_scores_dict`,
`get_statistics`, `normalize_scores`, `get_extremum`; `plot_maker.create_table`). Umgesetzt in `make_table5.py`,
geprüft mit `test_make_table5.py` (von Hand gerechnete künstliche Daten).

1. **Grundwert.** s[k,m,l,j,t] = Testwert (R², RMSE, Accuracy oder Log Loss) der bis Versuch t besten
   Konfiguration nach Validierung (`current_best_test_*`), Aufgabe k, Methode m, Seed l, Fold j.
   Dann Mittel über die 5 Folds: s[k,m,l,t]. Hyperband: eine Zeile je Rung, Endwert = letzte Zeile.
   Default: ein Wert je Aufgabe (ohne Seeds).
2. **Budget.** Table 5 nimmt den Endwert bei T = 135. Für `l1_270` zusätzlich T = 270; bei T = 135 ist das der
   Stand nach den ersten 135 von 270 Versuchen (deterministisches Gitter: Zeile 135 des 270er-Gitters).
3. **norm (ADTM, Grinsztajn et al.).** Pro Aufgabe ein Pool P_k aus allen s[k,m,l,t] mit t ≤ T der sechs HPO-Methoden
   über alle Seeds, im Paper-Code zusätzlich der Default-Wert (je HPO-Methode und Strategie T-mal).
   R², Accuracy: (s − Q10(P_k)) / (max P_k − Q10(P_k)), unten bei 0 abgeschnitten.
   RMSE, Log Loss: (s − min P_k) / (Q90(P_k) − min P_k), oben bei 1 abgeschnitten.
   Tabellenwert: Mittel über Aufgaben (36 Regression bzw. 23 Klassifikation) und Seeds des normalisierten Endwerts.
4. **Rel. diff.** Pro Aufgabe: s[k,m] = Mittel über die Seeds des Endwerts (roh, nicht normalisiert);
   best_k = bester Wert über die Methoden; δ[k,m] = |s[k,m] − best_k| / |best_k|. Tabellenwert: Mittel über Aufgaben.
5. **Rank.** Pro Aufgabe Rang 1 (bester) bis 7 von s[k,m] unter den 7 Methoden; Mittel über Aufgaben.
   Gleichstand wie im Paper-Code: stabile Sortierung in Spaltenreihenfolge, Default zuletzt.

**Warum so, wo Text und Code auseinandergehen:**

- *Rel. diff.:* `create_table` im Paper-Code rechnet (Mittel der normalisierten Werte − bestes Mittel) / bestes
  Mittel. Das ergäbe für Default bei R² etwa (0.181 − 0.898)/0.898 = −0.80, nicht 0.046; so ist die Tabelle also
  nicht entstanden. Probe mit den Werten pro Datensatz aus Abb. 5–10 (`paper_reference_values.csv`, t = 135, nur
  die Datensätze mit eindeutigem Namen und allen 7 Methoden: 20 von 36 Regression, 15 von 23 Klassifikation):

  | Default, Rel. diff. | pro Aufgabe (Text, gewählt) | aus Mittelwerten über Aufgaben | Paper |
  |---|---|---|---|
  | R² | 0.061 | 0.030 | 0.046 |
  | RMSE | 0.121 | 0.007 | 0.154 |
  | Log Loss | 0.131 | 0.105 | 0.222 |

  Mit derselben Probe kommen die Ränge nahe an Table 5 (R²: 6.95 / 4.30 / 3.75 / 5.75 / 3.00 / 1.45 / 2.80 gegen
  6.889 / 4.444 / 3.722 / 5.778 / 3.167 / 1.417 / 2.583).
- *Pool der Normalisierung:* Der Text nennt nur die HPO-Methoden („m = 1, …, 6“), der Code nimmt Default dazu und
  poolt über alle vier Strategien (num_leaves, max_depth, joint, num_iter). Wir rechnen nur num_leaves; Default im
  Pool wie im Code (`--pool code`, Standard), die Variante nach dem Text mit `--pool text`.
- *Hyperband:* Der Text spricht von 9 Rungs, Code und unsere Läufe haben 11 Ergebniszeilen je Fold
  (R = 2150, η = 2.8). Für Table 5 zählt nur die letzte Zeile, im Pool alle.

## Prüfungen vor dem Start (Mac, 30.10.2026)

- `check_native_encoding.py` auf allen 59 Aufgaben: alle kategorialen Spalten kommen als pandas `category` an,
  Merkmalszahl unverändert (kein One-Hot): 59 von 59 in Ordnung.
- Gespeicherte Kennzahlen pro Versuch: `test_score` (R² bzw. Accuracy), `test_rmse` (Regression),
  `test_log_loss` (Klassifikation), jeweils mit `current_best_*`. In allen Dateien von Results_repro und
  Results_ablation/A vorhanden und ohne Lücken. Nichts zu ergänzen.
- `test_run_ablation_resume.py`: ein abgebrochener und mit `--resume` fortgesetzter Job ergibt dieselben Zahlen wie
  ein Lauf am Stück (Zufallsgitter, TPE, SMAC, Hyperband), halb geschriebene Folds werden neu gerechnet.
- `test_make_table5.py`: 10 Tests mit künstlichen Daten (Normalisierung, Ränge, Rel. diff., Budget, unvollständige
  Folds, beide Ergebnis-Layouts).
- `make_table5.py` an Results_repro (7 Aufgaben, 2 Seeds) und Results_ablation/A, /B (wine_quality) ausgeführt.

## Jobs, Kosten und Ablauf

`python make_paper_jobs.py` erzeugt `jobs_paper/` (Job-Listen, Arrays, `run_paper_job.sbatch`, `submit.sh`,
`zusammenfassung.txt`). Ein Job = (Aufgabe, Seed, Methode) auf 1 Kern; Default einmal je Aufgabe.
**7139 Jobs je Lauf** (59 × 20 × 6 + 59), aufgeteilt in Arrays nach Partition und Speicher (höchstens 1000
Teiljobs je Array, `--max_array`). Probe: wine_quality, Seed 27225, alle 7 Methoden (`arrays/probe.idx`).

| Lauf | Jobs | geschätzte Kernstunden | Summe der Zeitlimits | teuerster Job | Jobs in normal.4h / 24h / 120h |
|---|---|---|---|---|---|
| paper | 7139 | ≈ 19'200 | ≈ 42'000 h | superconduct GP-BO, 14.5 h | 4219 / 2680 / 240 |
| l1_270 | 7139 | ≈ 37'400 | ≈ 78'000 h | superconduct GP-BO, 29.8 h | 3059 / 3080 / 1000 |

**Kostenmodell** (`make_paper_jobs.kosten`): Sekunden je Versuch inklusive Test-Refit, lokal gemessen (Mac,
1 Thread, Fold 0, je 15 Versuche Zufallsgitter und TPE, das Maximum; `jobs_paper/zeitmessung_mac.tsv`, alle 59
Aufgaben ohne Fehler), × 1.5 für Euler, × 5 Folds × Versuche. Gitter (Paper) und Hyperband × 0.6, GP-BO und SMAC
plus Surrogat je Versuch (Euler-Testlauf: GP-BO 1.6 s bei 135 bzw. 3.6 s bei 270 Aufrufen, SMAC 0.9 bzw. 1.4 s).
Abgleich: sulfur-Hauptjob des Testlaufs (Gitter, Zufall, TPE, GP-BO mit je 270 Versuchen) laut Modell ≈ 13 h für
2 Folds, auf Euler nach 12 h Zeitlimit 2 Folds fertig. Die Kosten je Versuch schwanken stark mit den gezogenen
Konfigurationen (15 Versuche sind eine kleine Stichprobe); deshalb Zeitlimit = 2 × Schätzung, und Jobs mit
Zeitlimit lassen sich fortsetzen (`--resume`, `watch_paper.py --retry` gibt ihnen die nächstlängere Partition).
Speicher: 4 GB, 8 GB für Rohdaten > 100 MB (gemessene Spitze beim Laden höchstens 2.0 GB, delays_zurich_transport).
Speicherplatz auf Euler: Ergebnisse ≈ 1.2 GB (paper) + 2.4 GB (l1_270), dazu etwa 28'000 Log-Dateien und
2'400 kleine SMAC-Ordner (≈ 110 KB je Job).

Wanduhrzeit hängt davon ab, wie viele Kerne gleichzeitig laufen dürfen: bei 500 Kernen etwa
57'000 / 500 ≈ 115 h ≈ 5 Tage für beide Läufe, bei 200 Kernen etwa 12 Tage (jeweils plus Warteschlange).

## Befehle

**Mac** (nach dem Commit, schon erledigt, wenn dieser Abschnitt im Git steht):
```
git push origin xgboost-catboost
```

**Euler** (einmal vorbereiten; nie von Euler pushen):
```
cd ~/HPOTreeBoosting
git pull origin xgboost-catboost
bash euler_prepare_repro_data.sh jobs_paper/aufgaben.tsv     # 59 Aufgaben laden + Kodierung prüfen (endet mit FERTIG)
my_share_info                                                # Anteil/Gruppe des Gastaccounts
sacctmgr show assoc user=$USER format=Account,Partition,GrpTRES,GrpJobs,MaxJobs,MaxSubmit,MaxWall -P
sacctmgr show qos format=Name,MaxWall,MaxJobsPU,MaxSubmitPU,MaxTRESPU -P
scontrol show config | grep -E "MaxArraySize|MaxJobCount"
lquota                                                       # Speicher und Dateizahl in $HOME
```
Ist `MaxArraySize` kleiner als 1001 oder `MaxSubmit` kleiner als die Zahl der Teiljobs, auf dem **Mac**
`python make_paper_jobs.py --max_array <n>` laufen lassen, committen, pushen und auf Euler `git pull`.

**Euler, Probe** (1 Aufgabe, 1 Seed, alle 7 Methoden, beide Läufe; < 1 h):
```
bash jobs_paper/submit.sh paper probe
bash jobs_paper/submit.sh l1_270 probe
squeue -u $USER
python watch_paper.py                                        # nach dem Ende: 7 von 7 Probe-Jobs je Lauf fertig?
python make_table5.py --run Probe_paper=Results_paper:135 --run Probe_l1=Results_l1_270:135,270 --out table5_probe
```

**Euler, volle Läufe in Wellen** (Welle 1 = Seeds 1–5 + Default, 2 = 6–10, 3 = 11–15, 4 = 16–20; je alle
59 Aufgaben und Methoden). Kosten je Welle: `jobs_paper/kosten_wellen.tsv` (paper ≈ 4'800 Kernstunden, ~3 Tage bei
64 Kernen; l1_270 ≈ 9'350, ~6 Tage; beide ≈ 14'150, ~9 Tage bei 64 bzw. ~5.4 Tage bei 110 Kernen).

Abschicken mit `refill.py` (seit 31.10.2026): der QOS erlaubt nur ~1000 eigene Jobs in der Schlange (jeder
Array-Teiljob zählt; sonst `QOSMaxSubmitJobPerUserLimit`). refill.py füllt die Schlange bis `--limit` (940) auf,
Welle 1 vor 2, schnelle Partitionen zuerst, je Partition zuerst `--reserve` (100) Plätze; schickt nie doppelt
(Ergebnisdateien, submitted.tsv, sacct; unbekannte hpo-Jobs in der Schlange → Abbruch) und wiederholt
beendete Jobs ohne Ergebnis (TIMEOUT → längere Partition, OUT_OF_MEMORY → nächste Speicherstufe), höchstens
`--max_versuche` (3) Mal. Die Wellen-Arrays von `submit.sh` (arrays/welle*.idx, arrays.tsv) sind eingefroren.
```
DRY=1 python refill.py --run beide          # Vorschau
python refill.py --run beide                # nachfüllen; täglich 1-2x
python refill.py --run beide --daemon       # oder: selbstfortsetzender Slurm-Job, alle 10 min
python refill.py --stop                     # Daemon beenden
```

**Speicher** (31.10.2026): MiniBooNE random_search scheiterte mit 4 GB an OUT_OF_MEMORY. Ursache: GPBoost legt je
Blatt ein Histogramm an (num_leaves bis 1024 × Bins bis 10'000 je Merkmal × 16 Byte); bei vielen stetigen
Merkmalen mehrere GB. `measure_memory.py` misst die Spitze je Aufgabe auf dem Mac (`jobs_paper/speicher_mac.tsv`;
jannis 6.1, MiniBooNE 6.1, topo_2_1 4.6, superconduct 4.2, Higgs 4.0 GB), `make_paper_jobs.py` setzt
Speicher = Stufe über 1.5 × Spitze + 1.5 GB: 12 GB für jannis, MiniBooNE, topo_2_1; 8 GB für superconduct, Higgs,
covertype, delays_zurich_transport; sonst 4 GB (Default immer nach Dateigrösse). Auf Euler:
`python watch_paper.py --speicher` (MaxRSS aus sacct je Aufgabe, schreibt `jobs_paper/speicher_euler.tsv`, das
refill.py als Untergrenze nimmt).

**Euler, beobachten:**
```
python watch_paper.py                       # Schlange, je Welle fertig/laufend/wartend/fehlgeschlagen/fehlt/nachfüllen
python watch_paper.py --run paper --aufgaben
python watch_paper.py --speicher
```

**Table 5 erzeugen** (Euler oder Mac):
```
python make_table5.py                                        # Paper=Results_paper:135, L1_270=Results_l1_270:135,270 -> table5.csv/.md/.pdf
python make_table5.py --seeds 1-5 --partial --out table5_welle1   # Zwischenstand: nur Aufgaben mit allen Seeds 1-5
```
`--partial`: nur Aufgaben, in denen jede Methode alle verlangten Seeds vollständig hat; der Kopf nennt Aufgaben,
Seeds und Methoden. Dazu immer `<out>_datensaetze.md/.csv`: Aufgabe × Methode, Test-Score (R² bzw. Accuracy)
bei T, Mittel über die vorhandenen Seeds.
**Mac**, Ergebnisse holen statt auf Euler auszuwerten:
```
rsync -av andubach@euler.ethz.ch:HPOTreeBoosting/Results_paper andubach@euler.ethz.ch:HPOTreeBoosting/Results_l1_270 .
python make_table5.py
```
