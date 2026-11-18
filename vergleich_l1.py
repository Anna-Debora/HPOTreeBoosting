"""Vergleich ohne / mit lambda_l1 bei gleicher Versuchszahl t = 135 (Frage von Fabio), zwei Varianten:

  --variante alt (Standard, wie bisher; Ausgabe vergleich_l1.*)
    ohne l1   Results_paper   (135 Versuche, lambda_l1 = 0 fest, Skalen wie im Paper-Code: alles linear)
    mit l1    Results_l1_270  (270 Versuche, lambda_l1 linear in (0, 100)), Zwischenstand bei t = 135
  --variante gpb (Skalen nach GPBoost/engine.py; Ausgabe vergleich_l1_gpb.*)
    ohne l1   Results_gpb     (135 Versuche, --l1 off --scales gpboost) fuer SMAC, TPE, GP-BO; Zufallsgitter aus
                              Results_paper (gpb hat keins; das Gitter haengt nicht von --scales ab)
    mit l1    Results_gpb_l1  (270 Versuche, --l1 on --scales gpboost: lambda_l1 log in [1e-8, 100]), Stand t = 135
  --variante beide: beide nacheinander.
Methoden: Zufallsgitter, SMAC, TPE, GP-BO (Strategie num_leaves, GPBoost, 59 Aufgaben der Suiten 334-337).

Wert je (Aufgabe, Seed, Methode): Test-Score des bisher besten Versuchs nach t = 135 Versuchen (Spalten
current_best_*, Zeile iter = 134), Mittel ueber die 5 Folds; roh, nicht normiert (= s[k,m,l,T] in make_table5.py).
Es zaehlen nur Paare (Aufgabe, Seed), die in beiden Laeufen alle 5 Folds mit mindestens 135 Zeilen haben.
Metriken: Regression R2 und RMSE, Klassifikation Accuracy und Log Loss.

Verbesserung (positiv = mit l1 besser): R2/Accuracy: mit - ohne; RMSE/Log Loss: ohne - mit.
  je Aufgabe: Mittel ueber die Seeds der paarweisen Differenzen (gleiche Aufgabe, gleicher Seed);
  relativ:    dieses Mittel / |Mittel der Werte ohne l1 ueber dieselben Seeds| (in %).
Je Methode und Metrik ueber die Aufgaben: Mittel, Bootstrap-95-%-Intervall (Perzentil, 10000 Ziehungen der
Aufgaben, Seed 0), Zahl der Aufgaben besser / schlechter / gleich (|Differenz| <= 1e-12), Wilcoxon-Vorzeichen-
Rang-Test (zweiseitig, Nullen verworfen wie scipy zero_method='wilcox').

--pruefe: prueft mit kleinen Spielfunktionen, ob die ersten n Versuche eines Laufs mit 2n Versuchen denen eines
Laufs mit n Versuchen gleichen (Zufallsgitter, TPE, GP-BO, SMAC-Startdesign), ohne Modelle zu trainieren.

Aufruf:  python vergleich_l1.py [--variante alt|gpb|beide] [--ohne ORDNER] [--mit ORDNER] [--t 135] [--out NAME]
         (--ohne/--mit/--out ersetzen die Ordner bzw. den Namen der Variante; mit --variante beide nicht erlaubt)
         python vergleich_l1.py --pruefe
Ausgabe: <out>.csv (alle Zahlen), <out>.md (Tabellen und Einschraenkungen), <out>_aufgaben.csv (je Aufgabe);
         <out> = vergleich_l1 (alt) bzw. vergleich_l1_gpb (gpb)
"""
import argparse
import glob
import os
import re

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from task_types import is_classification

METHODS = {"random_search": "Random Grid", "SMAC": "SMAC", "tpe": "TPE", "gp_bo": "GP-BO"}
# Metrik -> (Spalte, hoeher besser)
METRICS = {"regression": {"R2": ("current_best_test_score", True), "RMSE": ("current_best_test_rmse", False)},
           "classification": {"Accuracy": ("current_best_test_score", True),
                              "Log Loss": ("current_best_test_log_loss", False)}}
FILE = re.compile(r"seed_(\d+)/gpboost/(\d+)_(\d+)__num_leaves\.csv$")
FOLDS = 5
VALUE_COLUMNS = ["current_best_test_score", "current_best_test_rmse", "current_best_test_log_loss"]
# Variante -> Ordner ohne l1 (je Methode), Ordner mit l1, Versuche der Laeufe, Ausgabename
VARIANTS = {
    "alt": dict(ohne={m: "Results_paper" for m in METHODS}, mit="Results_l1_270", out="vergleich_l1"),
    "gpb": dict(ohne={"random_search": "Results_paper", "SMAC": "Results_gpb", "tpe": "Results_gpb",
                      "gp_bo": "Results_gpb"}, mit="Results_gpb_l1", out="vergleich_l1_gpb"),
}


def folder_for(root, method):
    """root: ein Ordner fuer alle Methoden oder {Methode: Ordner}."""
    return root[method] if isinstance(root, dict) else root


def describe(root):
    """Ordnerangabe fuer die Ausgabe: 'Results_gpb' oder 'Results_gpb (Random Grid: Results_paper)'."""
    if not isinstance(root, dict):
        return f"`{root}`"
    main = max(set(root.values()), key=list(root.values()).count)
    other = [f"{METHODS[m]}: `{f}`" for m, f in root.items() if f != main]
    return f"`{main}`" + (f" ({', '.join(other)})" if other else "")


def values_at(root, method, t):
    """DataFrame (suite, task, seed, Spalten current_best_*) bei Versuch t, Mittel ueber 5 Folds."""
    rows = []
    root = folder_for(root, method)
    for path in sorted(glob.glob(os.path.join(root, method, "seed_*", "gpboost", "*__num_leaves.csv"))):
        match = FILE.search(path.replace(os.sep, "/"))
        if not match:
            continue
        seed, suite, task = map(int, match.groups())
        frame = pd.read_csv(path)
        at = frame[frame["iter"] == t - 1]
        if at["fold"].nunique() != FOLDS or len(at) != FOLDS:
            continue
        rows.append(dict(suite=suite, task=task, seed=seed, **at[VALUE_COLUMNS].mean().to_dict()))
    return pd.DataFrame(rows, columns=["suite", "task", "seed", *VALUE_COLUMNS])


def bootstrap_ci(x, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return np.percentile(means, [2.5, 97.5])


def compare(ohne, mit, t):
    per_task, summary = [], []
    for method, label in METHODS.items():
        a, b = values_at(ohne, method, t), values_at(mit, method, t)
        pairs = a.merge(b, on=["suite", "task", "seed"], suffixes=("_ohne", "_mit"))
        pairs["typ"] = ["classification" if is_classification(s) else "regression" for s in pairs["suite"]]
        for typ, metrics in METRICS.items():
            sub = pairs[pairs["typ"] == typ]
            for metric, (col, higher) in metrics.items():
                if sub.empty:      # z. B. Lauf mit l1 noch nicht fertig
                    continue
                diff = (sub[f"{col}_mit"] - sub[f"{col}_ohne"]) * (1 if higher else -1)
                g = sub.assign(diff=diff).groupby(["suite", "task"])
                task = pd.DataFrame({"verbesserung": g["diff"].mean(), "ohne": g[f"{col}_ohne"].mean(),
                                     "mit": g[f"{col}_mit"].mean(), "seeds": g.size()}).reset_index()
                task["relativ_%"] = 100 * task["verbesserung"] / task["ohne"].abs()
                task.insert(0, "metrik", metric)
                task.insert(0, "typ", typ)
                task.insert(0, "methode", label)
                per_task.append(task)
                d = task["verbesserung"].to_numpy()
                lo, hi = bootstrap_ci(d)
                rlo, rhi = bootstrap_ci(task["relativ_%"].to_numpy())
                nonzero = d[np.abs(d) > 1e-12]
                p = wilcoxon(nonzero).pvalue if len(nonzero) > 1 else np.nan
                summary.append(dict(
                    typ=typ, metrik=metric, methode=label, aufgaben=len(d), seeds_je_aufgabe=task["seeds"].min(),
                    mittel_ohne=task["ohne"].mean(), mittel_mit=task["mit"].mean(),
                    verbesserung=d.mean(), ci_unten=lo, ci_oben=hi,
                    relativ_pct=task["relativ_%"].mean(), rel_ci_unten=rlo, rel_ci_oben=rhi,
                    besser=int((d > 1e-12).sum()), schlechter=int((d < -1e-12).sum()),
                    gleich=int((np.abs(d) <= 1e-12).sum()), wilcoxon_p=p))
    return pd.DataFrame(summary), (pd.concat(per_task, ignore_index=True) if per_task else pd.DataFrame())


def fmt(x, digits=4):
    return "" if pd.isna(x) else f"{x:+.{digits}f}"


LIMITS_ALT = [
    "1. lambda_l1 war im Lauf mit l1 bei SMAC, TPE und GP-BO **linear in (0, 100)** gezogen: Werte unter 1 "
    "kommen nur in ~1 % der Ziehungen vor, also fast immer starke L1-Strafe. Das ist eine bekannt "
    "ungeeignete Skala (GPBoost: log mit unterer Grenze 1e-8); der Vergleich sagt etwas ueber diese Skala, "
    "nicht ueber l1 an sich. Random Grid zieht lambda_l1 aus {0, 1, 10, 100} (je 1/4, also auch 0), dafuer "
    "ist sein Gitter 4-mal so gross und die 135 Punkte decken die uebrigen Parameter duenner ab.",
    "2. **SMAC: bestaetigt.** Das Sobol-Startdesign hat min(10 x Parameter, 0.25 x n_trials) Punkte "
    "(smac 2.4.1, AbstractInitialDesign; Konstanten zaehlen bei der Parameterzahl mit, 8 Parameter): "
    "135-Lauf 33 Punkte, 270-Lauf 67 Punkte. Die ersten 135 Versuche eines 270-Laufs sind also 67 Sobol-"
    "Punkte + 68 modellbasierte statt 33 + 102 und nicht gleich einem 135-Lauf.",
    "3. **Random Grid, TPE, GP-BO: Ziehung gleich, Test-Score nicht ganz.** Die Ziehung haengt nicht von "
    "der Versuchszahl ab (Zufallsgitter: RandomState.choice ohne Zuruecklegen = permutation()[:n]; TPE: "
    "n_startup_trials = 10 fest; GP-BO: n_initial_points = 10 fest), die ersten 135 Konfigurationen sind "
    "also dieselben wie in einem 135-Lauf (bei gleichem Suchraum; geprueft mit `--pruefe`). Aber die "
    "Test-Refits aller Zeilen benutzen best_iter des besten Validierungsversuchs **aller** Versuche des "
    "Laufs (methods._compute_test_scores mit self.best_iter); im 270-Lauf kann das ein Versuch nach t = 135 "
    "sein. Das gilt auch fuer SMAC.",
    "4. Mit l1 aendert sich ausserdem der Suchraum fuer alle Methoden (eine Dimension mehr, andere "
    "Zufallsfolge), der Vergleich ist also Lauf gegen Lauf, nicht Konfiguration gegen Konfiguration."]

LIMITS_GPB = [
    "1. lambda_l1 ist im Lauf mit l1 bei SMAC, TPE und GP-BO **log-skaliert in [1e-8, 100]** wie in GPBoost "
    "(engine.py; untere Grenze 0 -> 1e-8), die uebrigen Parameter ebenfalls auf den GPBoost-Skalen (log fuer "
    "learning_rate, lambda_l2, num_leaves, min_data_in_leaf, max_bin) wie im Lauf ohne l1. In der Startphase "
    "(Sobol bzw. Zufall) liegen etwa 80 % der Werte unter 1 (8 von 10 Zehnerpotenzen) statt ~1 % bei der "
    "linearen Skala des alten Laufs; Werte nahe 1e-8 sind praktisch keine L1-Strafe. Der Vergleich sagt also "
    "etwas ueber lambda_l1 mit geeigneter Skala, aber nur fuer diesen Bereich und diese Skala.",
    "2. **Random Grid** haengt nicht von der Skala ab: lambda_l1 aus {0, 1, 10, 100} (je 1/4), Gitter 4-mal so "
    "gross. Ohne l1 kommt es aus Results_paper (der Lauf gpb hat kein Zufallsgitter), mit l1 aus Results_gpb_l1; "
    "dessen Konfigurationen sind dieselben wie in Results_l1_270 (geprueft: gleiche Ziehung, Scores bis auf "
    "Rundung ~1e-16 gleich). Die Zeile Random Grid ist daher dieselbe wie in der alten Variante.",
    "3. **SMAC: weiterhin 67 statt 33 Startpunkte.** Sobol-Startdesign min(10 x Parameter, 0.25 x n_trials) "
    "(smac 2.4.1; 8 Parameter in beiden Laeufen, ohne l1 als Konstante 0): 135-Lauf 33 Punkte, 270-Lauf 67. "
    "Die ersten 135 Versuche des Laufs mit l1 sind 67 Sobol-Punkte + 68 modellbasierte statt 33 + 102.",
    "4. **TPE, GP-BO: Ziehung praefixgleich, Test-Score nicht ganz.** Feste 10 Startpunkte (TPE "
    "n_startup_trials, GP-BO n_initial_points), die Ziehung haengt nicht von der Versuchszahl ab; aber die "
    "Test-Refits aller Zeilen benutzen best_iter des besten Validierungsversuchs **aller** 270 Versuche "
    "(methods._compute_test_scores mit self.best_iter), das kann ein Versuch nach t = 135 sein. Gilt auch "
    "fuer SMAC und Random Grid.",
    "5. Mit l1 hat der Suchraum eine Dimension mehr (andere Zufallsfolge, Modelle in 8 statt 7 Dimensionen): "
    "Lauf gegen Lauf, nicht Konfiguration gegen Konfiguration."]


def markdown(summary, ohne, mit, t, variante="alt"):
    title = "(vorlaeufig)" if variante == "alt" else "(Skalen nach GPBoost)"
    lines = [f"# Ohne vs. mit lambda_l1 bei t = {t} {title}", "",
             f"ohne l1: {describe(ohne)} (135 Versuche), mit l1: {describe(mit)} (270 Versuche, Zwischenstand "
             f"t = {t}). "
             "Wert: Test-Score des bisher besten Versuchs (current_best_*) bei t, Mittel ueber 5 Folds, roh. "
             "Verbesserung > 0 heisst: mit l1 besser. Paarweise je (Aufgabe, Seed), dann Mittel ueber die Seeds je "
             "Aufgabe; Intervall = Bootstrap 95 % ueber Aufgaben; p = Wilcoxon ueber Aufgaben.", ""]
    if summary.empty:
        lines += ["**Keine Paare** (Aufgabe, Seed) mit allen 5 Folds in beiden Laeufen gefunden.", ""]
    names = {"regression": "Regression", "classification": "Klassifikation"}
    for typ in ("regression", "classification"):
        part = summary[summary["typ"] == typ] if not summary.empty else summary
        if part.empty:
            continue
        lines += [f"## {names[typ]} ({part['aufgaben'].max()} Aufgaben)", ""]
        for metric in part["metrik"].unique():
            m = part[part["metrik"] == metric]
            lines += [f"### {metric}", "",
                      "| Methode | Mittel ohne | Mittel mit | Verbesserung | 95-%-KI | relativ | 95-%-KI relativ "
                      "| besser / schlechter / gleich | Wilcoxon p |",
                      "|---|---:|---:|---:|---|---:|---|---:|---:|"]
            for _, r in m.iterrows():
                lines.append(
                    f"| {r.methode} | {r.mittel_ohne:.4f} | {r.mittel_mit:.4f} | {fmt(r.verbesserung)} | "
                    f"[{fmt(r.ci_unten)}, {fmt(r.ci_oben)}] | {fmt(r.relativ_pct, 2)} % | "
                    f"[{fmt(r.rel_ci_unten, 2)}, {fmt(r.rel_ci_oben, 2)}] % | "
                    f"{r.besser} / {r.schlechter} / {r.gleich} | {r.wilcoxon_p:.3g} |")
            lines.append("")
    lines += ["## Einschraenkungen", "", *(LIMITS_ALT if variante == "alt" else LIMITS_GPB), ""]
    return "\n".join(lines)


# ------------------------------------------------------------------ --pruefe: Praefix-Eigenschaft
def pruefe(n=20):
    import optuna
    from skopt import gp_minimize
    from skopt.space import Real
    from ConfigSpace import ConfigurationSpace
    from smac import Scenario
    from smac.facade.hyperparameter_optimization_facade import HyperparameterOptimizationFacade as HPOFacade
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    out = {}

    a = np.random.RandomState(7).choice(a=10000, size=n, replace=False)
    b = np.random.RandomState(7).choice(a=10000, size=2 * n, replace=False)
    out["Random Grid"] = bool(np.array_equal(a, b[:n]))

    def tpe(k):
        study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=7))
        study.optimize(lambda tr: (tr.suggest_float("x", 0, 1) - .3) ** 2 + tr.suggest_float("y", 0, 1), n_trials=k)
        return [tr.params for tr in study.trials]
    out["TPE"] = tpe(n) == tpe(2 * n)[:n]

    def gp(k):
        return gp_minimize(lambda x: (x[0] - .3) ** 2 + x[1], [Real(0, 1), Real(0, 1)], n_calls=k,
                           random_state=7).x_iters
    out["GP-BO"] = np.allclose(gp(n), gp(2 * n)[:n])

    space = ConfigurationSpace(space={"a": (0.0, 1.0), "b": (0.0, 1.0), "c": (1, 1000), "l1": 0.0, "d": (0.5, 1.0),
                                      "e": (0.5, 1.0), "f": (255, 10000), "g": (2, 1024)})
    for trials in (135, 270):
        design = HPOFacade.get_initial_design(Scenario(space, n_trials=trials, seed=7))
        out[f"SMAC Startdesign n_trials={trials}"] = design._n_configs
    return out


def run_variant(variante, ohne, mit, t, out):
    summary, per_task = compare(ohne, mit, t)
    summary.to_csv(f"{out}.csv", index=False)
    per_task.to_csv(f"{out}_aufgaben.csv", index=False)
    text = markdown(summary, ohne, mit, t, variante)
    with open(f"{out}.md", "w") as handle:
        handle.write(text)
    print(text)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--variante", choices=(*VARIANTS, "beide"), default="alt",
                        help="alt: Results_paper gegen Results_l1_270 (Standard); gpb: Results_gpb gegen "
                             "Results_gpb_l1; beide")
    parser.add_argument("--ohne", help="Ordner ohne l1 (fuer alle Methoden; Standard je Variante)")
    parser.add_argument("--mit", help="Ordner mit l1 (Standard je Variante)")
    parser.add_argument("--t", type=int, default=135)
    parser.add_argument("--out", help="Ausgabename ohne Endung (Standard vergleich_l1 bzw. vergleich_l1_gpb)")
    parser.add_argument("--pruefe", action="store_true")
    args = parser.parse_args(argv)
    if args.pruefe:
        for key, value in pruefe().items():
            print(f"{key}: {value}")
        return
    if args.variante == "beide" and (args.ohne or args.mit or args.out):
        parser.error("--ohne/--mit/--out nur mit einer Variante")
    for variante in (VARIANTS if args.variante == "beide" else [args.variante]):
        v = VARIANTS[variante]
        run_variant(variante, args.ohne or v["ohne"], args.mit or v["mit"], args.t, args.out or v["out"])


if __name__ == "__main__":
    main()
