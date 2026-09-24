"""prepare_one_dataset.py

Kleiner Hilfs-Downloader: holt EINEN einzelnen OpenML-Task herunter und legt ihn
im Format ab, das run_experiments.py erwartet:

    data/{suite_id}_{task_id}/{suite_id}_{task_id}_X.csv
    data/{suite_id}_{task_id}/{suite_id}_{task_id}_y.csv
    data/{suite_id}_{task_id}/{suite_id}_{task_id}_categorical_indicator.npy

Standardmaessig wird Suite 335, Task 361102 geladen -- genau der Task, den auch
im Originalprojekt zum Testen benutzt wurde.

Aufruf (Standard):        python prepare_one_dataset.py
Aufruf (anderer Task):    python prepare_one_dataset.py --suite_id 335 --task_id 361102
"""

import argparse
import os

import numpy as np
import openml


def main(args):
    suite_id = int(args.suite_id)
    task_id = int(args.task_id)

    # Zielordner anlegen, z.B. data/335_361102/
    path = f"data/{suite_id}_{task_id}"
    os.makedirs(path, exist_ok=True)

    print(f"Lade OpenML-Task {task_id} herunter (das kann einen Moment dauern) ...")

    # --- Genau der Ablauf aus dem auskommentierten Block in run_experiments.py ---
    task = openml.tasks.get_task(task_id)          # Task von OpenML holen
    dataset = task.get_dataset()                   # den zugehoerigen Datensatz
    X, y, categorical_indicator, attribute_names = dataset.get_data(
        dataset_format="dataframe",
        target=dataset.default_target_attribute,
    )
    # ---------------------------------------------------------------------------

    print(f"Heruntergeladen: {X.shape[0]} Zeilen, {X.shape[1]} Spalten.")

    # Als CSV / npy speichern, genau in den Dateinamen, die run_experiments.py liest
    X.to_csv(os.path.join(path, f"{suite_id}_{task_id}_X.csv"), index=False)
    y.to_csv(os.path.join(path, f"{suite_id}_{task_id}_y.csv"), index=False)
    np.save(
        os.path.join(path, f"{suite_id}_{task_id}_categorical_indicator.npy"),
        np.array(categorical_indicator),
    )

    print(f"Fertig. Dateien liegen in: {path}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite_id", default="335")
    parser.add_argument("--task_id", default="361102")
    args = parser.parse_args()
    main(args)
