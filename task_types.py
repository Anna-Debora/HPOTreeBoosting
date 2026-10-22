"""Zentrale Stelle: Ist eine Suite/Aufgabe Klassifikation oder Regression?

Alte OpenML-Suiten: 334 und 337 = binaere Klassifikation, 335 und 336 = Regression.
Neue Daten (TabArena/TabZilla-Union) bekommen eigene Pseudo-Suiten:
    NEW_REGRESSION_SUITE = 1001, NEW_CLASSIFICATION_SUITE = 1002.
Alle Stellen im Code fragen nur noch is_classification(suite_id).
"""

OLD_CLASSIFICATION_SUITES = (334, 337)
OLD_REGRESSION_SUITES = (335, 336)
NEW_REGRESSION_SUITE = 1001
NEW_CLASSIFICATION_SUITE = 1002

CLASSIFICATION_SUITES = frozenset(OLD_CLASSIFICATION_SUITES + (NEW_CLASSIFICATION_SUITE,))
REGRESSION_SUITES = frozenset(OLD_REGRESSION_SUITES + (NEW_REGRESSION_SUITE,))
ALL_SUITES = tuple(sorted(CLASSIFICATION_SUITES | REGRESSION_SUITES))


def is_classification(suite_id):
    suite_id = int(suite_id)
    if suite_id not in CLASSIFICATION_SUITES and suite_id not in REGRESSION_SUITES:
        raise ValueError(f"Unbekannte suite_id {suite_id}; bekannt sind {ALL_SUITES}")
    return suite_id in CLASSIFICATION_SUITES
