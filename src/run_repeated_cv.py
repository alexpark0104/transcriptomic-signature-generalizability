from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from analysis import repeated_nested_cv
from data_loader import load_discovery_data



expr, labels, _ = load_discovery_data(
    "data/rld.BMS038.20171011.csv",
    "data/bms038_clinical_data.csv",
)

results = repeated_nested_cv(
    expr.values,
    labels.values,
    n_repeats=2,
    n_genes=10,
    outer_splits=5,
    inner_splits=5,
    random_state=42,
)

Path("results").mkdir(exist_ok=True)
results.to_csv("results/repeated_nested_cv_test.csv", index=False)

print("\nSaved results/repeated_nested_cv_test.csv")
print(results)
print("\nSummary:")
print(results[["mean_outer_auc", "pooled_outer_auc"]].describe())
