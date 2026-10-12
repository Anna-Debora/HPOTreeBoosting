"""Replays the row selection of create_scores_dict (score branch) on one real result file.

Usage: python analyze_mini_layout.py <csv>
Hyperband/SMAC/NUM_ITER files are not needed to show how the main file's rows are assigned.
"""
import sys

import pandas as pd

df = pd.read_csv(sys.argv[1])
NUM_FOLDS = df["fold"].nunique()
N = df.groupby(["method", "try_num_leaves", "joint_tuning_depth_leaves", "try_num_iter"], sort=False)["fold"].size()
print(N.to_string(), "\n")

# create_scores_dict: LEN_DATA = rows with try_num_leaves; slices assume md | nl | joint blocks
LEN = int(df["try_num_leaves"].sum())
md = df[(~df["try_num_leaves"]) & (~df["joint_tuning_depth_leaves"])]
nl = df[df["try_num_leaves"]]
jt = df[df["joint_tuning_depth_leaves"]]
it = df[df["try_num_iter"]]
print(f"LEN_DATA (try_num_leaves rows)            = {LEN}")
print(f"score series max_depth  (nl=F & joint=F)  = {len(md)} rows, of which try_num_iter=True: {int(md['try_num_iter'].sum())}")
print(f"score series num_leaves (nl=T)            = {len(nl)}")
print(f"score series joint      (joint=T)         = {len(jt)}")
print(f"score series num_iter   (try_num_iter=T)  = {len(it)}")

# method/fold labels for the max_depth block come from data.loc[0:LEN-1]
labels = df.iloc[:LEN]
print(f"\nlabels used for max_depth: first {LEN} rows -> {dict(labels['method'].value_counts())}")
print(f"labels used for joint: rows {2*LEN}..{2*LEN + 270*3*NUM_FOLDS - 1} (hardcoded NUM_ITERS=270, 3 methods)")
jl = df.iloc[2 * LEN: 2 * LEN + 270 * 3 * NUM_FOLDS]
print(f"  -> actual rows there: {len(jl)}, flags: {jl[['try_num_leaves','joint_tuning_depth_leaves','try_num_iter']].drop_duplicates().to_dict('records')}")

# Compare the main file's own max_depth series positions with its labels
print("\nmax_depth series vs. labels (position-wise, main file only):")
print(f"  series length {len(md)}, label length {LEN}; rows of series beyond label length belong to: "
      f"{dict(md.iloc[LEN:][['method','try_num_iter']].value_counts())}")
