"""Repeat the full nested-CV analysis with many different fold splits.

Run from the repo root:
    python src/run_repeated_cv.py          # 100 repeats per method (default)
    python src/run_repeated_cv.py 20       # a different number

This shows how much the nested AUC moves just because of how patients happen
to be assigned to folds. It does NOT capture all uncertainty: the patients and
the gene selection are the same in every repeat.
"""
import contextlib
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from analysis import METHODS, repeated_nested_cv
from data_loader import load_discovery_data

N_REPEATS = int(sys.argv[1]) if len(sys.argv) > 1 else 100
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

expr, labels, _ = load_discovery_data(
    "data/rld.BMS038.20171011.csv", "data/bms038_clinical_data.csv"
)
X = expr.values.astype(float)
y = labels.values.astype(int)

frames = []
for method in METHODS:
    print(f"Running {N_REPEATS} repeats of nested CV [{method}] ...")
    with contextlib.redirect_stdout(io.StringIO()):  # silence per-repeat prints
        frames.append(
            repeated_nested_cv(X, y, n_repeats=N_REPEATS, n_genes=10,
                               outer_splits=5, inner_splits=5,
                               random_state=42, method=method)
        )

results = pd.concat(frames, ignore_index=True)
results.to_csv(RESULTS_DIR / "repeated_nested_cv.csv", index=False)

rows = []
for method in METHODS:
    sub = results[results["method"] == method]
    for stat in ("mean_outer_auc", "pooled_outer_auc"):
        v = sub[stat].values
        rows.append({
            "method": method,
            "statistic": stat,
            "n_repeats": len(v),
            "mean": v.mean(),
            "sd": v.std(ddof=1),
            "min": v.min(),
            "p2.5": np.percentile(v, 2.5),
            "median": np.median(v),
            "p97.5": np.percentile(v, 97.5),
            "max": v.max(),
        })

summary = pd.DataFrame(rows)
summary.to_csv(RESULTS_DIR / "repeated_nested_cv_summary.csv", index=False)
pd.set_option("display.width", 200)
print("\n" + summary.round(4).to_string(index=False))
print("\nSaved results/repeated_nested_cv.csv and results/repeated_nested_cv_summary.csv")