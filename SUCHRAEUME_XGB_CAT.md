# Vorschlag: Suchräume und Skalen für XGBoost und CatBoost (zur Freigabe)

Stand 23.11.2026, Entwurf. Erst nach Freigabe werden Code (`methods.py`, `run_ablation.py --library`) und Jobs gebaut.

## Quellen, in dieser Reihenfolge

1. **Paper-Code a19b4ff:** enthält weder XGBoost noch CatBoost (`git grep -i "xgb\|catboost" a19b4ff`: keine Treffer).
   Von dort kommen nur die GPBoost-Bereiche, die in `methods.py` stehen.
2. **GPBoost `engine.py` (GitHub master, `_TUNING_PARAMS_*`, `_parse_tuning_search_space`):** legt die Skalen auch
   für die XGBoost-Aliase fest. Grundlage ist ein Abruf am 23.11.2026. Die Konstanten kamen dabei nur als
   Zusammenfassung zurück, nicht wörtlich, und sollten vor der Freigabe am Original geprüft werden.
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

Offen zu XGBoost:
- **min_child_weight** ist eine Hessian-Summe, keine Zeilenzahl. Für die logistische Verlustfunktion ist die
  Hessian je Zeile ≤ 0.25, daher der Faktor 0.25 (steht so schon im Code, dort mit „vorläufig, Fabio
  bestätigen“). Die Alternative wäre ein bibliothekstypischer Bereich wie log [1e-3? , 100] aus Benchmarks.
  Bitte entscheiden.
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

Offen zu CatBoost:
- **depth statt num_leaves:** Mit `depth` [1, 10] reicht die Baumgröße bis 2^10 = 1024 Blätter, also bis zur
  Obergrenze von num_leaves. Die Alternative `grow_policy=Lossguide` mit `max_leaves` [2, 1024] wäre näher an
  GPBoost, ist bei CatBoost aber unüblich und deutlich langsamer.
- **bootstrap_type:** Vorschlag `Bernoulli` fest, damit `subsample` sicher wirkt. Der Standard ist je nach Version
  `MVS` oder `Bayesian`, und bei `Bayesian` wird `subsample` nicht benutzt.
- **Was der heutige Code tut:** Er ignoriert bei CatBoost `num_leaves` und `lambda_l1` und reicht
  `min_data_in_leaf` unverändert durch. Mit der Strategie num_leaves tunt CatBoost dadurch die Baumgröße gar nicht.
  Das wird mit diesem Vorschlag behoben.
- **Bibliothekstypische Zusatzparameter** wie `random_strength`, `bagging_temperature` und `one_hot_max_size`
  (z. B. in CatBoost-Benchmarks) werden nicht aufgenommen, weil GPBoost/XGBoost keine Entsprechung haben.
  Bitte bestätigen.

## Gitter (deterministisch und Zufallsgitter)

Die Gitter verwenden dieselben Wertelisten wie GPBoost, übersetzt. Bei CatBoost ersetzt
`depth ∈ {1, 2, 3, 5, 10}` die Werte `num_leaves ∈ {2, 4, 8, 32, 1024}` (gleiche Blattzahlen), und
`min_data_in_leaf` fällt weg. Dadurch wird das CatBoost-Gitter kleiner (deterministisch 135/3 = 45 Punkte).
Bitte entscheiden: so lassen oder auf 135 auffüllen.

## Default

Default ist jeweils der Bibliotheks-Default (wie `run_default.py` heute je Bibliothek).
