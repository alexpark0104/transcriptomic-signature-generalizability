"""Discovery-cohort analysis: apparent vs nested-CV AUC for both screening methods.

Run from the repo root:  python src/run_pipeline.py

The external-cohort analyses live in external_validation.py.
Primary method: inner_cv (the documented original). The other method is
reported alongside it, never instead of it.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from analysis import METHODS, apparent_auc, nested_cv
from data_loader import load_discovery_data

PRIMARY_METHOD = "inner_cv"
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

print("Loading discovery data...")
expr, labels, _ = load_discovery_data(
    "data/rld.BMS038.20171011.csv", "data/bms038_clinical_data.csv"
)
X, y = expr.values.astype(float), labels.values.astype(int)
print(f"Samples: {X.shape[0]}, Genes: {X.shape[1]}, "
      f"responders: {int(y.sum())}, non-responders: {int((y == 0).sum())}")

long_rows, gene_rows = [], []
for method in METHODS:
    print(f"\n[{method}]")
    apparent, top = apparent_auc(X, y, n_genes=10, method=method)
    mean_auc, pooled_auc, _ = nested_cv(X, y, n_genes=10, outer_splits=5,
                                        inner_splits=5, random_state=42, method=method)
    print(f"  apparent in-sample AUC: {apparent:.4f}")
    print(f"  nested CV mean AUC:     {mean_auc:.4f}")
    print(f"  nested CV pooled AUC:   {pooled_auc:.4f}")

    for estimate, value in (("apparent_in_sample", apparent),
                            ("nested_cv_mean", mean_auc),
                            ("nested_cv_pooled", pooled_auc)):
        long_rows.append({"method": method, "estimate": estimate,
                          "auc": value, "n_samples": len(y)})
    for rank, gene in enumerate(expr.columns[top], start=1):
        gene_rows.append({"method": method, "rank": rank, "discovery_gene_id": int(gene)})

long = pd.DataFrame(long_rows)
long.to_csv(RESULTS_DIR / "auc_summary_by_method.csv", index=False)
genes = pd.DataFrame(gene_rows)
genes.to_csv(RESULTS_DIR / "selected_discovery_genes_by_method.csv", index=False)

# Primary-method files keep the original column layout (plus a method column).
long[long["method"] == PRIMARY_METHOD][["estimate", "auc", "n_samples", "method"]] \
    .to_csv(RESULTS_DIR / "auc_summary.csv", index=False)
genes[genes["method"] == PRIMARY_METHOD][["rank", "discovery_gene_id"]] \
    .to_csv(RESULTS_DIR / "selected_discovery_genes.csv", index=False)

print("\nSaved results/auc_summary.csv, auc_summary_by_method.csv, "
      "selected_discovery_genes.csv, selected_discovery_genes_by_method.csv")