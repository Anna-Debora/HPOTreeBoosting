# Suchräume und Skalen für XGBoost und CatBoost

Freigegeben am 24.11.2026 mit diesen Entscheidungen:
1. XGBoost `min_child_weight` wie bisher aus `min_data_in_leaf` umgerechnet (×0,25 bei Klassifikation); die
   Bestätigung durch Fabio steht noch aus.
2. CatBoost: `depth` in [1, 10] statt `num_leaves`.
3. CatBoost: `bootstrap_type=Bernoulli` fest.
4. CatBoost-Gitter auf 135 Punkte aufgefüllt: `subsample` ∈ {0,5; 0,75; 1} ersetzt die wegfallende Achse
   `min_data_in_leaf` (siehe Abschnitt „Gitter“).
5. Nachträglich, noch zu bestätigen: CatBoost-Early-Stopping bei Klassifikation auf Accuracy wie GPBoost/XGBoost
   (Fehlerrate) statt Logloss, siehe Hinweise.

Umgesetzt in `methods.py` (nur mit `--scales gpboost`; mit `--scales paper` bleibt alles bitgleich) und
`run_ablation.py --library xgboost|catboost`. Die Läufe heißen `xgb_alle` und `cat_alle` (`make_paper_jobs.py`).

## Wörtlicher Auszug aus GPBoost `engine.py`

Quelle: https://raw.githubusercontent.com/fabsig/GPBoost/master/python-package/gpboost/engine.py, abgerufen am
24.11.2026, letzter Commit an der Datei 23.11.2026 (17238309). Zeilen 1279–1300:

```python
_TUNING_PARAMS_FLOAT_LOG = ['learning_rate', 'shrinkage_rate']
# Parameters on a logarithmic scale for which 0 is a valid value. Since a logarithmic scale excludes 0, a lower
# bound of 0 is replaced by 1e-8, which practically has the same effect
_TUNING_PARAMS_FLOAT_LOG_ZERO = ['lambda_l2', 'reg_lambda', 'lambda',
                                 'lambda_l1', 'reg_alpha',
                                 'min_gain_to_split', 'min_split_gain',
                                 'min_sum_hessian_in_leaf', 'min_sum_hessian_per_leaf', 'min_sum_hessian',
                                 'min_hessian', 'min_child_weight']
_TUNING_PARAMS_FLOAT = ['bagging_fraction', 'sub_row', 'subsample', 'bagging',
                        'feature_fraction', 'sub_feature', 'colsample_bytree',
                        'cat_l2',
                        'cat_smooth',
                        'cov_fct_shape']
_TUNING_PARAMS_INT_LOG = ['num_leaves', 'num_leaf', 'max_leaves', 'max_leaf',
                          'max_bin']
# Integer parameters on a logarithmic scale for which 0 is a valid value. A lower bound of 0 is replaced by 1
_TUNING_PARAMS_INT_LOG_ZERO = ['min_data_in_leaf', 'min_data_per_leaf', 'min_data', 'min_child_samples']
_TUNING_PARAMS_INT = ['max_depth', 'cov_fct_order']
# Parameters whose two values can also be given as a list [value_1, value_2] instead of {'values': [...]}
_TUNING_PARAMS_TWO_VALUES = ['line_search_step_length']
# Parameters that can only be given as a set of values
_TUNING_PARAMS_VALUES_ONLY = ['cov_function']
```

`_parse_tuning_search_space` (dieselbe Datei) setzt bei `_LOG_ZERO` eine untere Grenze 0 auf `max(lower, 1e-8)`
(float) bzw. `max(lower, 1)` (int) und `log: True`. Alles, was in keiner Log-Liste steht, ist linear:
`_TUNING_PARAMS_FLOAT` als float, der Rest (`max_depth`) als int.

## Quellen, in dieser Reihenfolge

1. **Paper-Code a19b4ff:** enthält weder XGBoost noch CatBoost (`git grep -i "xgb\|catboost" a19b4ff`: keine Treffer).
   Von dort kommen nur die GPBoost-Bereiche, die in `methods.py` stehen.
2. **GPBoost `engine.py` (GitHub master, `_TUNING_PARAMS_*`, `_parse_tuning_search_space`):** legt die Skalen auch
   für die XGBoost-Aliase fest (wörtlicher Auszug oben).
   - log, float, untere Grenze 0 → 1e-8: `lambda_l2/reg_lambda`, `lambda_l1/reg_alpha`, `min_child_weight`,
     `min_gain_to_split/min_split_gain`
   - log, float: `learning_rate`
   - log, int: `num_leaves/max_leaves`, `max_bin`; log, int, 0 → 1: `min_data_in_leaf/min_child_samples`
   - linear: `bagging_fraction/subsample`, `feature_fraction/colsample_bytree`, `max_depth`
   - CatBoost kommt in engine.py nicht vor.
   - Hinweis: Die lokal installierte gpboost-Version (`.venv`) ist älter. Dort sind `lambda_l1/l2` und
     `reg_alpha/reg_lambda` linear. `methods.GPBOOST_LOG_PARAMS` (Lauf gpb) folgt der master-Version.
3. **Offizielle Dokus und übliche HPO-Benchmarks:** nur dort, wo 1. und 2. nichts sagen. Das betrifft vor allem
   CatBoost, das symmetrische Bäume hat und deshalb kein `num_leaves`.

## Gemeinsame Festlegungen (wie gpb_alle)

Ohne L1 (`reg_alpha` = 0 fest; CatBoost hat keinen L1-Parameter), T = 135, 20 Seeds, 5 Folds, alle Methoden,
alle 345 Datensätze, Kategorien nativ, Early Stopping wie bisher. Die Bereiche bleiben, wo möglich, die von
GPBoost: Damit unterscheiden sich die Bibliotheken nur in ihren Parametern, nicht in den Grenzen.

## XGBoost (`tree_method=hist`, `grow_policy=lossguide`, `max_depth=0`: Strategie num_leaves wie GPBoost)

| Parameter (XGBoost) | entspricht GPBoost | Bereich | Skala | Quelle |
|---|---|---|---|---|
| `eta` | learning_rate | [0.001, 1] | log | engine.py |
| `max_leaves` | num_leaves | [2, 1024] | log, int | engine.py |
| `min_child_weight` | min_data_in_leaf | Regression [1, 1000], Klassifikation [0.25, 250] | log | engine.py (Skala); Bereich = bisherige Umrechnung × 0.25 |
| `reg_lambda` | lambda_l2 | [1e-8, 100] | log | engine.py |
| `reg_alpha` | lambda_l1 | fest 0 | – | Vorgabe ohne L1 |
| `max_bin` | max_bin | [255, min(n, 10000)] | log, int | engine.py |
| `subsample` | bagging_fraction | [0.5, 1] | linear | engine.py |
| `colsample_bytree` | feature_fraction | [0.5, 1] | linear | engine.py |

Hinweise zu XGBoost:
- **min_child_weight** ist eine Hessian-Summe, keine Zeilenzahl. Für die logistische Verlustfunktion ist die
  Hessian je Zeile ≤ 0.25, daher der Faktor 0.25. Entschieden: so lassen; Fabio bestätigt noch.
- `gamma` (min_split_gain) wird nicht getunt, weil GPBoost es in unserem Raum auch nicht tunt.

## CatBoost (`grow_policy=SymmetricTree`, Standard)

| Parameter (CatBoost) | entspricht GPBoost | Bereich | Skala | Quelle |
|---|---|---|---|---|
| `learning_rate` | learning_rate | [0.001, 1] | log | engine.py |
| `depth` | **statt num_leaves** | [1, 10] | linear, int | engine.py (max_depth linear); CatBoost-Doku: symmetrische Bäume, num_leaves = 2^depth |
| `l2_leaf_reg` | lambda_l2 | [1e-8, 100] | log | engine.py (wie lambda_l2) |
| `border_count` | max_bin | [255, min(n, 1024)] | log, int | engine.py (wie max_bin); Obergrenze 1024 wie bisher (Laufzeit) |
| `subsample` | bagging_fraction | [0.5, 1] | linear | engine.py; braucht `bootstrap_type` Bernoulli oder MVS |
| `rsm` | feature_fraction | [0.5, 1] | linear | engine.py; CatBoost-Doku: nur CPU |
| – | min_data_in_leaf | nicht getunt | – | CatBoost-Doku: wirkt nur bei Depthwise/Lossguide |

Hinweise zu CatBoost:
- **depth statt num_leaves (entschieden):** Mit `depth` [1, 10] reicht die Baumgröße bis 2^10 = 1024 Blätter, also
  bis zur Obergrenze von num_leaves. In den Ergebnisdateien steht die Tiefe in der Spalte `max_depth`,
  `num_leaves` und `min_data_in_leaf` sind leer.
- **bootstrap_type Bernoulli (entschieden):** fest, sobald `subsample` gesetzt ist (alle Methoden außer Default).
- **Early Stopping bei Klassifikation auf Accuracy (24.11.2026):** GPBoost und XGBoost stoppen auf der Fehlerrate
  (`binary_error` bzw. `error`). CatBoost ignorierte `metric` und stoppte auf Logloss, das mit kleinen Lernraten
  fast immer weiter sinkt; so liefen viele Fits bis 1000 Iterationen. Gemessen (15 Versuche, 1 Fold):
  Wilt 53 → 3,3 s, eye_movements 266 → 26 s, nomao 1933 → 491 s. Mit `--scales gpboost` setzt `methods.py` daher
  `eval_metric="Accuracy"` (= 1 − Fehlerrate); Regression bleibt RMSE wie bei den anderen Bibliotheken.
- **Beste Iteration ab 1 (25.11.2026, XGBoost und CatBoost):** Beide Bibliotheken zaehlen die beste Iteration ab 0,
  GPBoost (und `num_boost_round` der Test-Refits) ab 1. Mit `--scales gpboost` gibt `methods.py` jetzt die Zahl der
  Baeume zurueck (+1); vorher bekamen die Refits einen Baum zu wenig, und bei bester Iteration 0 brach CatBoost mit
  "Iterations count should be positive" ab (gesehen bei HEBO auf seismic-bumps).
- **Was der alte Code tat:** Er ignorierte bei CatBoost `num_leaves` und `lambda_l1`, reichte `min_data_in_leaf`
  unverändert durch und überschrieb eine getunte Tiefe mit -1. Mit `--scales gpboost` ist das behoben; die
  Test-Refits nutzen die getunte Tiefe (geprüft).
- **Bibliothekstypische Zusatzparameter** wie `random_strength`, `bagging_temperature` und `one_hot_max_size`
  (z. B. in CatBoost-Benchmarks) werden nicht aufgenommen, weil GPBoost/XGBoost keine Entsprechung haben.

## Gitter (deterministisch und Zufallsgitter)

Die Gitter verwenden dieselben Wertelisten wie GPBoost, übersetzt, ohne `lambda_l1`. Das deterministische Gitter
hat wie beim Paper 135 Punkte (`--paper_grid`).
- **XGBoost:** learning_rate {0,01; 0,1; 1} × min_data_in_leaf {10, 100, 1000} × num_leaves {2, 4, 8, 32, 1024} ×
  lambda_l2 {0, 1, 10} = 135, wie bei GPBoost.
- **CatBoost:** `depth ∈ {1, 2, 3, 5, 10}` ersetzt `num_leaves ∈ {2, 4, 8, 32, 1024}` (gleiche Blattzahlen 2^depth),
  `min_data_in_leaf` fällt weg. Aufgefüllt wird mit `subsample ∈ {0,5; 0,75; 1}`: learning_rate (3) × lambda_l2 (3)
  × depth (5) × subsample (3) = 135. Begründung: subsample ist schon Teil des Suchraums und wirkt mit Bernoulli
  sicher. Das 270er-Gitter im Repo hat für das größere Budget ebenfalls bagging_fraction ergänzt, und die Werte
  stammen aus der Liste des Zufallsgitters.
- **Zufallsgitter CatBoost:** wie GPBoost, aber `depth` statt `num_leaves` und ohne `min_data_in_leaf`.

## Default

Default ist jeweils der Bibliotheks-Default (wie `run_default.py` heute je Bibliothek).
