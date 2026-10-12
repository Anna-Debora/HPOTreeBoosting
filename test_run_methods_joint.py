"""run_methods() for all four flag combinations of run_experiments.py (few trials, one fold)."""
import make_mini_experiment as mm  # importing applies the trial-count monkeypatches

KEYS = ["method", "try_num_leaves", "joint_tuning_depth_leaves", "try_num_iter"]
EXPECTED = {
    "max_depth": {"grid_search", "random_search", "tpe", "gp_bo"},
    "num_leaves": {"grid_search", "random_search", "tpe", "gp_bo"},
    "joint": {"random_search", "tpe", "gp_bo"},
    "num_iter": {"random_search", "tpe", "gp_bo"},
}
COUNTS = {"grid_search": 270, "random_search": mm.N_RANDOM, "tpe": mm.N_TPE, "gp_bo": mm.N_GPBO}

for name, cfg in zip(EXPECTED, mm.CONFIGS):
    df = mm.run_config(cfg)
    blocks = df.groupby(KEYS, sort=False).size()
    print(f"\n== {name}: {cfg}")
    print(blocks.to_string())
    assert set(df["method"]) == EXPECTED[name], set(df["method"])
    for (method, *_), n in blocks.items():
        assert n == COUNTS[method], (method, n)
    assert df["fold"].nunique() == 1
print("\nOK: all four combinations ran; joint/num_iter contain only random_search/tpe/gp_bo")
