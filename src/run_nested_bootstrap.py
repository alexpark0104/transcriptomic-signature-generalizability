from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))

from analysis import nested_cv
from data_loader import load_discovery_data


RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

expr, labels, _ = load_discovery_data(
    "data/rld.BMS038.20171011.csv",
    "data/bms038_clinical_data.csv",
)

X = expr.values
y = labels.values.astype(int)

print("Running one nested-CV analysis...")
mean_auc, pooled_auc, out_of_fold_predictions = nested_cv(
    X,
    y,
    n_genes=10,
    outer_splits=5,
    inner_splits=5,
    random_state=42,
)

observed_auc = roc_auc_score(
    y,
    out_of_fold_predictions,
)

print(f"Mean outer-fold AUC: {mean_auc:.3f}")
print(f"Pooled out-of-fold AUC: {observed_auc:.3f}")

rng = np.random.default_rng(42)
bootstrap_aucs = []

for _ in range(10000):
    indices = rng.integers(
        low=0,
        high=len(y),
        size=len(y),
    )

    sampled_y = y[indices]

    # AUC is undefined if a resample contains only one class.
    if len(np.unique(sampled_y)) < 2:
        continue

    sampled_predictions = out_of_fold_predictions[indices]

    bootstrap_aucs.append(
        roc_auc_score(sampled_y, sampled_predictions)
    )

bootstrap_aucs = np.asarray(bootstrap_aucs)
ci_lower, ci_upper = np.percentile(
    bootstrap_aucs,
    [2.5, 97.5],
)

summary = pd.DataFrame([{
    "mean_outer_auc": mean_auc,
    "pooled_outer_auc": observed_auc,
    "bootstrap_ci_lower": ci_lower,
    "bootstrap_ci_upper": ci_upper,
    "n_patients": len(y),
    "n_bootstrap_replicates": len(bootstrap_aucs),
}])

summary.to_csv(
    RESULTS_DIR / "nested_cv_bootstrap_summary.csv",
    index=False,
)

prediction_table = pd.DataFrame({
    "label": y,
    "out_of_fold_prediction": out_of_fold_predictions,
})

prediction_table.to_csv(
    RESULTS_DIR / "nested_cv_out_of_fold_predictions.csv",
    index=False,
)

print(f"Bootstrap 95% interval: {ci_lower:.3f} to {ci_upper:.3f}")
print(f"Valid bootstrap replicates: {len(bootstrap_aucs)}")
print("Saved nested-CV bootstrap results.")
