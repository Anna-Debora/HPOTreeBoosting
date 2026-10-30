# Datenauswahl – Zusammenfassung

Stand: Branch `xgboost-catboost`, 30.10.2026, nach den Änderungen in Absprache mit dem Betreuer. Erzeugt aus preprocessing_report*.csv, excluded_duplicates.csv, fraglich.csv, duplicate_rows_report.csv und task_indices/. „Vorher“ = Stand vom 26.10.2026 (Commit 04917ed).

## Was sich geändert hat

| Regel | vorher | nachher |
|---|---|---|
| Mindestzahl Zeilen | 3000 (Grinsztajn) | 100 (technische Untergrenze) |
| Mindestzahl Zeilen in der kleineren Klasse | – | 20 (technische Untergrenze) |
| Mindestens 4 Merkmale | ja | ja |
| Merkmale/Zeilen < 0.1 | ja | ja |
| Unterstichprobe auf 100'000 Zeilen | ja (methods.py) | ja (methods.py) |
| Kategoriale Merkmale mit > 20 Stufen löschen | ja | **nein** |
| Numerische Merkmale mit < 10 Werten löschen | ja | **nein** |
| Numerische Merkmale mit genau 2 Werten → kategorial | ja | ja |
| BCCO: ganzzahlig mit 3–20 Werten → kategorial | ja | ja |
| Konstante Spalten (nur ein Wert) löschen | (über die < 10-Regel) | ja, eigene Regel |
| One-Hot-Kodierung | GPBoost, XGBoost | **keine**; alle Bibliotheken nativ (GPBoost/XGBoost: pandas `category`, CatBoost: `cat_features`), auch für 334–337 |

## Pro Quelle (vorher → nachher)

### OpenML (TabArena/TabZilla, new_tasks.csv)

|  | vorher | nachher |
|---|---|---|
| Kandidaten | 72 | 72 |
| **aufgenommen** (in task_indices) | 17 | 53 |
| ausgeschieden: zu wenige Zeilen | 45 | 4 |
| ausgeschieden: zu wenige Merkmale | 1 | 0 |
| ausgeschieden: zu hochdimensional | 2 | 11 |
| ausgeschieden: nach der Bereinigung bleibt nichts übrig | 3 | 0 |
| übersprungen: Name schon vorhanden (Duplikat) | 0 | 0 |
| übersprungen: in excluded_duplicates.csv ausgeschlossen | 4 | 4 |
| Einträge in excluded_duplicates.csv (diese Quelle) | 4 | 4 |
| aufgenommen und in fraglich.csv | 2 | 7 |

### BCCO (Hugging Face)

|  | vorher | nachher |
|---|---|---|
| Kandidaten | 156 | 156 |
| **aufgenommen** (in task_indices) | 32 | 83 |
| ausgeschieden: zu wenige Zeilen | 73 | 6 |
| ausgeschieden: zu wenige Merkmale | 10 | 10 |
| ausgeschieden: zu hochdimensional | 0 | 12 |
| ausgeschieden: nach der Bereinigung bleibt nichts übrig | 0 | 0 |
| übersprungen: Name schon vorhanden (Duplikat) | 35 | 35 |
| übersprungen: in excluded_duplicates.csv ausgeschlossen | 6 | 10 |
| Einträge in excluded_duplicates.csv (diese Quelle) | 6 | 10 |
| aufgenommen und in fraglich.csv | 3 | 9 |

### TALENT (Google Drive)

|  | vorher | nachher |
|---|---|---|
| Kandidaten | 300 | 300 |
| **aufgenommen** (in task_indices) | 78 | 150 |
| ausgeschieden: zu wenige Zeilen | 119 | 3 |
| ausgeschieden: zu wenige Merkmale | 6 | 3 |
| ausgeschieden: zu hochdimensional | 0 | 21 |
| ausgeschieden: nach der Bereinigung bleibt nichts übrig | 0 | 0 |
| übersprungen: Name schon vorhanden (Duplikat) | 81 | 99 |
| übersprungen: in excluded_duplicates.csv ausgeschlossen | 16 | 24 |
| Einträge in excluded_duplicates.csv (diese Quelle) | 14 | 19 |
| aufgenommen und in fraglich.csv | 10 | 21 |

Hinweise:
- Vorher stand bei OpenML „ok 21“ inklusive 4 später als Duplikat ausgeschlossener Aufgaben; jetzt werden diese 4 im Bericht als übersprungen geführt.
- „zu hochdimensional“ ist nachher häufiger, weil viele kleine Datensätze jetzt die Zeilengrenze bestehen, aber an Merkmale/Zeilen < 0.1 scheitern (vorher schieden sie schon an den 3000 Zeilen aus).
- „nach der Bereinigung bleibt nichts übrig“ entfällt, weil keine Spalten mehr wegen Kardinalität gelöscht werden.
- Die Klassen-Untergrenze (20) hat keinen Datensatz ausgeschlossen: alle Klassifikationsdatensätze mit ≥ 100 Zeilen haben genug Zeilen in der kleineren Klasse.
- TALENT „Name schon vorhanden“ steigt von 81 auf 99, weil jetzt mehr BCCO-Datensätze aufgenommen sind und TALENT-Datensätze gleichen Namens dann übersprungen werden.
- TALENT „in excluded_duplicates.csv ausgeschlossen“ (24) umfasst die 19 TALENT-Einträge sowie 5 Namenstreffer mit anderen Einträgen (2dplanes, house_sales_reduced, Fitness_Club_c, airlines_seed_0_…, predict_students_dropout_…).

## Aufgaben pro Suite nach Quelle

| Suite | OpenML | BCCO | TALENT | neu gesamt | vorher |
|---|---|---|---|---|---|
| 1001 (Regression) | 7 | 25 | 50 | 82 | 49 |
| 1002 (Klassifikation) | 46 | 58 | 100 | 204 | 78 |
| neue gesamt | 53 | 83 | 150 | **286** | 127 |
| alte Suiten 334–337 | | | | 59 | 59 |
| **insgesamt** | | | | **345** | 186 |

Alte Suiten: 334: 7, 335: 17, 336: 19, 337: 16 (zusammen 59, unverändert; nur die One-Hot-Kodierung entfällt dort beim Training).

Größe der neuen Aufgaben (Zeilen nach Vorverarbeitung): Minimum 104, Median 2398; 156 der 286 Aufgaben haben weniger als 3000 Zeilen, 33 weniger als 500.

## Alle Einträge in excluded_duplicates.csv (33, davon 9 neu)

| suite_id | task_id | Quelle | name | entsprechende_alte_aufgabe | grund | sicherheit | neu (30.10.) |
|---|---|---|---|---|---|---|---|
| 1001 | 900033 | BCCO | house_sales_reduced | house_sales (335, 336) | gleicher Inhalt (Spalten und log-Ziel identisch), zusaetzlich Zeilennummer/id/zipcode | sicher |  |
| 1001 | 900022 | BCCO | cpu_activity | cpu_act (336) | gleicher Delve-Datensatz, gleiche Groesse 8192x21 | sicher |  |
| 1001 | 363705 | OpenML | superconductivity | superconduct (336) | gleicher Datensatz, 21263 Zeilen | sicher |  |
| 1001 | 363686 | OpenML | miami_housing | MiamiHousing2016 (336) | gleicher Datensatz (Miami Housing 2016) | sicher |  |
| 1002 | 910051 | BCCO | eye_movements_bin | eye_movements (334, 337) | gleicher Inhalt wie 337 (alle Spalten und Ziel identisch, nur Zeilenreihenfolge anders) | sicher |  |
| 1002 | 363627 | OpenML | credit_card_clients_default | default-of-credit-card-clients (334, 337) | gleicher Datensatz (UCI Taiwan), nach Ausgleich wie alt 13272 Zeilen | sicher |  |
| 1002 | 363673 | OpenML | GiveMeSomeCredit | credit (337) | gleicher Datensatz (Give Me Some Credit), nach Ausgleich wie alt 16714 Zeilen | sicher |  |
| 1002 | 910104 | BCCO | wine-quality-dataset-balanced-classification | wine_quality (336) | abgeleitet von wine_quality (336), kuenstlich vervielfachte Zeilen | sicher |  |
| 1002 | 910019 | BCCO | amazon-survey-coupon-recommendation-dataset | in_vehicle_coupon_recommendation (OpenML 363681) | gleicher Datensatz wie in_vehicle_coupon_recommendation (OpenML 363681), BCCO hat Kategorien als Zahlen | sicher |  |
| 1001 | 900001 | BCCO | 2dplanes |  | kuenstlich erzeugt, Grinsztajn schliesst synthetische Daten aus | sicher |  |
| 1001 | 920004 | TALENT | Abalone_reg | abalone (335, 361288) | TALENT, Inhaltsvergleich: 7 Spalten gleich, 100% der Zeilen gleich | sicher |  |
| 1002 | 930006 | TALENT | Bank_Customer_Churn_Dataset | Bank_Customer_Churn (1002, 363619) | TALENT, Inhaltsvergleich: 4 Spalten gleich, 100% der Zeilen gleich | sicher |  |
| 1001 | 920013 | TALENT | Brazilian_houses_reproduced | Brazilian_houses (335, 361098) | TALENT, Inhaltsvergleich: 6 Spalten gleich, 100% der Zeilen gleich | sicher |  |
| 1001 | 920038 | TALENT | Physicochemical_r | physiochemical_protein (1001, 363693) | TALENT, Inhaltsvergleich: 10 Spalten gleich, 100% der Zeilen gleich | sicher |  |
| 1001 | 920041 | TALENT | Superconductivty | superconduct (336, 361088) | TALENT, Inhaltsvergleich: 80 Spalten gleich, 100% der Zeilen gleich | sicher |  |
| 1002 | 930074 | TALENT | adult | income (1002, 910062) | TALENT, Inhaltsvergleich: 4 Spalten gleich, 100% der Zeilen gleich | sicher |  |
| 1002 | 930082 | TALENT | bank | bank-marketing (337, 361066) | TALENT, Inhaltsvergleich: 4 Spalten gleich, 100% der Zeilen gleich | sicher |  |
| 1002 | 930172 | TALENT | shrutime | Bank_Customer_Churn (1002, 363619) | TALENT, Inhaltsvergleich: 4 Spalten gleich, 100% der Zeilen gleich | sicher |  |
| 1002 | 930008 | TALENT | California-Housing-Classification | california (337, 361277) | TALENT, von Hand geprueft: gleiche 20640 Zeilen wie houses (336) und california (337), gleiche Klassifikation des Hauspreises; Merkmale als Summen statt Durchschnitte | sicher |  |
| 1001 | 920079 | TALENT | pole | pol (336, 361073; 337, 361062) | TALENT, von Hand geprueft: gleicher Datensatz (pol), alle 22 pruefbaren Spalten zugeordnet, Ziel gleich verteilt, 87 % der Zeilen exakt gleich | sicher |  |
| 1002 | 930145 | TALENT | naticusdroid+android+permissions+dataset | NATICUSdroid (1002, 363689) | TALENT, von Hand geprueft: gleicher UCI-Datensatz 722 (nur 0/1-Merkmale, daher nicht ueber Werte vergleichbar) | sicher |  |
| 1002 | 930151 | TALENT | online_shoppers | online_shoppers_intention (1002, 363691) | TALENT, von Hand geprueft: gleicher UCI-Datensatz (12330 Zeilen), TALENT-Version ohne die *_Duration-Spalten | sicher |  |
| 1002 | 930062 | TALENT | Shipping | E-CommereShippingData (1002, 363632) | TALENT, von Hand geprueft: gleicher Kaggle-Datensatz (10999 Zeilen, gleiche Spalten) | sicher |  |
| 1002 | 930095 | TALENT | dabetes_130-us_hospitals | Diabetes130US (337, 361273) | TALENT, von Hand geprueft: gleicher UCI-Datensatz 296 (gleiche Zeilen und Ziel), alte Version mit weniger Merkmalen | sicher |  |
| 1001 | 920006 | TALENT | Another-Dataset-on-used-Fiat-500-(1538-rows) | Another-Dataset-on-used-Fiat-500 (1001, 363615) | TALENT, Inhaltsvergleich (neue Kriterien 30.10.): 5 Spalten gleich, 100% der Zeilen gleich | sicher | ja |
| 1001 | 920094 | TALENT | us_crime | communities_and_crime (1001, 920056) | TALENT, Inhaltsvergleich (neue Kriterien 30.10.): 97 Spalten gleich, 100% der Zeilen gleich; der alphabetisch erste bleibt | sicher | ja |
| 1002 | 910018 | BCCO | airlines_seed_0_nrows_2000_nclasses_10_ncols_100_stratify_True | airlines (1002, 189354) | BCCO, Inhaltsvergleich (neue Kriterien 30.10.): Stichprobe von 2000 Zeilen aus airlines, 97% der Zeilen gleich | sicher | ja |
| 1002 | 930011 | TALENT | Contaminant-detection-in-packaged-cocoa-hazelnut-spread-jars-using-Microwaves-Sensing-and-Machine-Learning-10.0GHz(Urbinati) | hazelnut-spread-contaminant-detection (1002, 363674) | TALENT, Inhaltsvergleich (neue Kriterien 30.10.): 30 Spalten gleich, 100% der Zeilen gleich; OpenML-Version bleibt | sicher | ja |
| 1002 | 910003 | BCCO | Fitness_Club_c | Fitness_Club (1002, 363671) | BCCO, von Hand geprueft (neue Kriterien 30.10.): gleiche 1500 Zeilen, alle 7 Spalten mit gleicher Haeufigkeitsverteilung, Kategorien als Zahlen | sicher | ja |
| 1002 | 910094 | BCCO | simple-diabetes-dataset | diabetes (1002, 363629) | BCCO, von Hand geprueft (neue Kriterien 30.10.): Pima Indians Diabetes, 768 Zeilen, alle 9 Spalten gleich verteilt | sicher | ja |
| 1002 | 930056 | TALENT | Pima_Indians_Diabetes_Database | diabetes (1002, 363629) | TALENT, von Hand geprueft (neue Kriterien 30.10.): Pima Indians Diabetes, 768 Zeilen, alle 9 Spalten gleich verteilt | sicher | ja |
| 1002 | 930059 | TALENT | QSAR_biodegradation | qsar-biodeg (1002, 363696) | TALENT, von Hand geprueft (neue Kriterien 30.10.): 1054 Zeilen, alle 42 Spalten gleich verteilt | sicher | ja |
| 1002 | 910085 | BCCO | predict_students_dropout_and_academic_success | students_dropout_and_academic_success (1002, 363704) | BCCO, von Hand geprueft (neue Kriterien 30.10.): gleicher UCI-Datensatz 697 (4424 Zeilen), 16 von 35 Spalten gleich verteilt, Rest anders kodiert; BCCO-Version ohne 2 Spalten | wahrscheinlich | ja |

## Alle Einträge in fraglich.csv (37, davon 22 neu)

Doppelte Zeilen: Anteil der Zeilen (Merkmale und Ziel), die im vorbereiteten Datensatz schon weiter oben genau so vorkommen; Schwelle wie bisher 30 % (duplicate_rows_report.py).

| suite_id | task_id | Quelle | name | grund | aufgenommen | neu (30.10.) |
|---|---|---|---|---|---|---|
| 1002 | 910100 | BCCO | turiye_student_evaluation | Bewertungsskala 1 bis 5, ob kategorial ist Ansichtssache, Fabio fragen | ja |  |
| 1002 | 910071 | BCCO | medium-app-reviews-from-google-play-store | viele doppelte Zeilen, Fabio fragen | ja |  |
| 1002 | 9890 | OpenML | poker-hand | frueher viele doppelte Zeilen (65%, weil Spalten geloescht wurden); nach den Regeln vom 30.10. nur noch 0.2%, Fabio fragen ob streichen | ja |  |
| 1002 | 189354 | OpenML | airlines | viele doppelte Zeilen, Fabio fragen | ja |  |
| 1002 | 910069 | BCCO | malware-analysis-datasets-pe-section-headers | viele doppelte Zeilen, Fabio fragen | ja |  |
| 1001 | 920016 | TALENT | CookbookReviews | viele doppelte Zeilen (78%), Fabio fragen | ja |  |
| 1001 | 920017 | TALENT | Data_Science_Salaries | viele doppelte Zeilen (55%), Fabio fragen | ja |  |
| 1001 | 920025 | TALENT | Laptop_Prices_Dataset | viele doppelte Zeilen (44%), Fabio fragen | ja |  |
| 1001 | 920026 | TALENT | Large-scale_Wave_Energy_Farm_Perth_100 | viele doppelte Zeilen (62%), Fabio fragen | ja |  |
| 1001 | 920028 | TALENT | Large-scale_Wave_Energy_Farm_Sydney_49 | viele doppelte Zeilen (73%), Fabio fragen | ja |  |
| 1002 | 930005 | TALENT | BNG(tic-tac-toe) | viele doppelte Zeilen (56%), Fabio fragen | ja |  |
| 1002 | 930020 | TALENT | Employee | viele doppelte Zeilen (73%), Fabio fragen | ja |  |
| 1002 | 930039 | TALENT | INNHotelsGroup | frueher viele doppelte Zeilen (30%); nach den Regeln vom 30.10. 28%, Fabio fragen ob streichen | ja |  |
| 1002 | 930054 | TALENT | PhishingWebsites | viele doppelte Zeilen (43%), Fabio fragen | ja |  |
| 1002 | 930118 | TALENT | internet_firewall | viele doppelte Zeilen (72%), Fabio fragen | ja |  |
| 1002 | 930130 | TALENT | led7 | viele doppelte Zeilen (89%), Fabio fragen | ja | ja |
| 1002 | 910008 | BCCO | LED-display-domain-7digit | viele doppelte Zeilen (75%), Fabio fragen | ja | ja |
| 1002 | 363685 | OpenML | maternal_health_risk | viele doppelte Zeilen (53%), Fabio fragen | ja | ja |
| 1002 | 14964 | OpenML | artificial-characters | viele doppelte Zeilen (48%), Fabio fragen | ja | ja |
| 1001 | 920027 | TALENT | Large-scale_Wave_Energy_Farm_Sydney_100 | viele doppelte Zeilen (47%), Fabio fragen | ja | ja |
| 1002 | 363707 | OpenML | website_phishing | viele doppelte Zeilen (44%), Fabio fragen | ja | ja |
| 1002 | 930192 | TALENT | water_quality | viele doppelte Zeilen (41%), Fabio fragen | ja | ja |
| 1002 | 910049 | BCCO | employees-attrition-analysis | viele doppelte Zeilen (41%), Fabio fragen | ja | ja |
| 1002 | 910023 | BCCO | analcatdata_dmft | viele doppelte Zeilen (31%), Fabio fragen | ja | ja |
| 1001 | 920046 | TALENT | archive2 | Teilmenge von wine_quality (336): Rotwein-Zeilen, gleiches Ziel (Inhaltsvergleich: alle Zeilen in wine_quality), Fabio fragen | ja | ja |
| 1002 | 910105 | BCCO | wine-quality-red | Klassifikationsvariante von wine_quality (336), Rotwein-Zeilen, Fabio fragen | ja | ja |
| 1002 | 930196 | TALENT | wine | Klassifikationsvariante von wine_quality (336), alle Zeilen in wine_quality, Fabio fragen | ja | ja |
| 1002 | 930198 | TALENT | wine-quality-white | Klassifikationsvariante von wine_quality (336) bzw. Wine_Quality_white (1001), Fabio fragen | ja | ja |
| 1002 | 930185 | TALENT | thyroid-ann | vermutlich Teilmenge (UCI ann-thyroid, 3772 von 7200 Zeilen) von thyroid (1002, 930184), anders skaliert, Fabio fragen | ja | ja |
| 1002 | 930073 | TALENT | ada_prior | Stichprobe (4562 Zeilen) aus Adult, wie income (1002, 910062); 93% der Zeilen dort enthalten, Fabio fragen | ja | ja |
| 1002 | 930038 | TALENT | Heart-Disease-Dataset-(Comprehensive) | enthaelt heart-h (1002, 50) fast vollstaendig (96% der Zeilen), Fabio fragen | ja | ja |
| 1001 | 900000 | BCCO | 1000-Cameras-Dataset | kategoriale Spalte mit lauter verschiedenen Werten (Modellname, wie eine ID), Fabio fragen | ja | ja |
| 1001 | 920053 | TALENT | chscase_foot | kategoriale Spalte mit mehr als 50% verschiedenen Werten (wie eine ID), Fabio fragen | ja | ja |
| 1001 | 920054 | TALENT | colleges | 2 kategoriale Spalten mit mehr als 50% verschiedenen Werten (wie eine ID), Fabio fragen | ja | ja |
| 1002 | 363613 | OpenML | Amazon_employee_access | kategoriale Spalte mit mehr als 50% verschiedenen Werten (wie eine ID), Fabio fragen | ja | ja |
| 1002 | 363683 | OpenML | kddcup09_appetency | 5 kategoriale Spalten mit mehr als 50% verschiedenen Werten (wie eine ID), Fabio fragen | ja | ja |
| 1002 | 910079 | BCCO | pbcseq | kategoriale Spalte mit mehr als 50% verschiedenen Werten (wie eine ID), Fabio fragen | ja | ja |

## Euler-Testlauf mit Job 17674581

**Nicht ermittelbar.** Die Job-ID kommt in keiner lokalen Datei vor: weder im Repository (Job-Dateien, Logs), noch in `~/Downloads`, `~/Documents`, `~/Desktop`, noch in früheren Claude-Protokollen. Die Slurm-Logs (`*.out`/`*.err`) und das Job-Skript (`job.sh`) liegen laut Übergabe-Notiz nur auf Euler, und Euler ist von diesem Rechner aus nicht erreichbar (kein SSH-Schlüssel).

Lokal dokumentiert ist nur ein früherer Testjob: `sbatch test_job.sh` → Job 15752064 (Projekt_Dokumentation_Teil2.md, Schritt B.6). Einstellungen dazu sind dort nicht festgehalten.

Der hier vorbereitete Reproduktions-Testlauf (`jobs_repro/`) ist **nicht** dieser Job, er wurde nie abgeschickt.

So findest du die Angaben auf Euler (nach `ssh andubach@euler.ethz.ch`):

```
sacct -j 17674581 --format=JobID,JobName%30,Partition,State,Elapsed,TotalCPU,MaxRSS,Start,End
scontrol show job 17674581          # nur solange der Job noch bekannt ist
ls ~/HPOTreeBoosting/*17674581* ~/HPOTreeBoosting/**/logs/*17674581* 2>/dev/null
grep -l 17674581 -r ~/HPOTreeBoosting --include='*.out' --include='*.err' 2>/dev/null
```

Aufgabe, Seeds, Strategien und Bibliothek stehen in der ersten Zeile der `.out`-Datei (bei Arrays aus `make_euler_jobs.py` in der Zeile von `jobs.tsv`, die zur Array-Nummer gehört) bzw. im Befehl im Job-Skript. Die Dauer zeigt `sacct` in der Spalte Elapsed.
