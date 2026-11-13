"""Full permutation test of the apparent and nested-CV AUCs.

Run from the repo root:
    python src/run_permutation_test.py            # 1000 permutations (default)
    python src/run_permutation_test.py 200        # a different number

For each permutation the outcome labels are shuffled (class counts unchanged)
and the ENTIRE procedure is repeated: gene screening, model fitting, apparent
AUC and nested CV. The null distribution therefore shows what each statistic
looks like when there is no real association between expression and response.

The same shuffled labels are used for every screening method (paired), and
every statistic uses the same fixed fold seed (42) as the observed analysis.
The run is resumable: finished (method, permutation) rows are kept.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from analysis import METHODS, apparent_auc, nested_cv
from data_loader import load_discovery_data

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

N_PERMUTATIONS = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
RANDOM_SEED = 2026
N_GENES = 10
FOLD_SEED = 42
OUTPUT_PATH = RESULTS_DIR / "permutation_test_full.csv"
SUMMARY_PATH = RESULTS_DIR / "permutation_test_full_summary.csv"

expr, labels, _ = load_discovery_data(
    "data/rld.BMS038.20171011.csv",
    "data/bms038_clinical_data.csv",
)
X = expr.values.astype(float)
y = labels.values.astype(int)
print(f"Samples: {len(y)}, genes: {X.shape[1]}, "
      f"responders: {int(y.sum())}, non-responders: {int((y == 0).sum())}")


def statistics(X, y, method):
    apparent, _ = apparent_auc(X, y, n_genes=N_GENES, method=method)
    mean_auc, pooled_auc, _ = nested_cv(
        X, y, n_genes=N_GENES, outer_splits=5, inner_splits=5,
        random_state=FOLD_SEED, method=method,
    )
    return apparent, mean_auc, pooled_auc


# ---- observed values ----
observed = {}
for method in METHODS:
    observed[method] = statistics(X, y, method)
    a, m, p = observed[method]
    print(f"Observed [{method}]: apparent = {a:.4f}, nested mean = {m:.4f}, nested pooled = {p:.4f}")

# ---- permutations (resumable) ----
if OUTPUT_PATH.exists():
    rows = pd.read_csv(OUTPUT_PATH).to_dict("records")
    done = {(r["method"], int(r["permutation"])) for r in rows}
    print(f"Found {len(done)} completed (method, permutation) rows. Resuming.")
else:
    rows, done = [], set()

for i in range(1, N_PERMUTATIONS + 1):
    # Each permutation has its own reproducible random stream, so resuming
    # or changing N never changes any individual permutation.
    shuffled = np.random.default_rng([RANDOM_SEED, i]).permutation(y)
    for method in METHODS:
        if (method, i) in done:
            continue
        a, m, p = statistics(X, shuffled, method)
        rows.append({"method": method, "permutation": i,
                     "apparent_auc": a, "nested_mean_auc": m, "nested_pooled_auc": p})
    if i % 25 == 0 or i == N_PERMUTATIONS:
        pd.DataFrame(rows).to_csv(OUTPUT_PATH, index=False)
        print(f"  {i}/{N_PERMUTATIONS} permutations done")

pd.DataFrame(rows).to_csv(OUTPUT_PATH, index=False)

# ---- summary ----
null = pd.DataFrame(rows)
null = null[null["permutation"] <= N_PERMUTATIONS]
summary_rows = []
for method in METHODS:
    sub = null[null["method"] == method]
    for stat, obs in zip(("apparent_auc", "nested_mean_auc", "nested_pooled_auc"), observed[method]):
        values = sub[stat].values
        summary_rows.append({
            "method": method,
            "statistic": stat,
            "observed": obs,
            "null_mean": values.mean(),
            "null_2.5pct": np.percentile(values, 2.5),
            "null_97.5pct": np.percentile(values, 97.5),
            # One-sided: how often does shuffled-label data do at least as well?
            "p_upper": (1 + int((values >= obs).sum())) / (1 + len(values)),
            "n_permutations": len(values),
        })

summary = pd.DataFrame(summary_rows)
summary.to_csv(SUMMARY_PATH, index=False)
pd.set_option("display.width", 200)
print("\n" + summary.round(4).to_string(index=False))
print(f"\nSaved {OUTPUT_PATH} and {SUMMARY_PATH}")