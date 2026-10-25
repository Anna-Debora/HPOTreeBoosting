"""Zeigt die Zeilenzahlen der alten 59 Aufgaben (Suiten 334-337) aus den OpenML-Metadaten, ohne Download der Daten.
Aufruf:  python check_old_sizes.py
"""
import openml
import numpy as np

rows = []
for suite_id in (334, 335, 336, 337):
    for task_id in openml.study.get_suite(suite_id).tasks:
        task = openml.tasks.get_task(task_id, download_splits=False, download_data=False,
                                     download_qualities=False, download_features_meta_data=False)
        ds = openml.datasets.get_dataset(task.dataset_id, download_data=False,
                                         download_qualities=True, download_features_meta_data=False)
        rows.append((int(float(ds.qualities["NumberOfInstances"])), ds.name, suite_id))
rows.sort()
sizes = np.array([r[0] for r in rows])
print(f"{len(rows)} Aufgaben; kleinste 8:")
for r in rows[:8]:
    print("  ", r)
print("Median:", int(np.median(sizes)), " 10%-Quantil:", int(np.quantile(sizes, 0.1)), " Maximum:", sizes.max())
