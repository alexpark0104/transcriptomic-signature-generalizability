from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))

from analysis import univariate_screening
from data_loader import load_discovery_data, load_validation_data


RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

DISCOVERY_EXPR = "data/rld.BMS038.20171011.csv"
DISCOVERY_CLIN = "data/bms038_clinical_data.csv"
EXTERNAL_SERIES = "data/GSE78220_series_matrix.txt.gz"
EXTERNAL_EXPR = "data/GSE78220_PatientFPKM.xlsx"

ID_TO_SYMBOL = {
    401613: "SERTM2",
    4534: "MTM1",
    221823: "PRPS1L1",
    10472: "ZBTB18",
    5376: "PMP22",
    3930: "LBR",
    9340: "GLP2R",
    132014: "IL17RE",
    22828: "SCAF8",
    5754: "PTK7",
}


def bootstrap_ci(labels, scores, n_boot=10000, seed=42):
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(labels), size=len(labels))
        sampled = labels[idx]
        if len(np.unique(sampled)) < 2:
            continue
        aucs.append(roc_auc_score(sampled, scores[idx]))
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    return lo, hi, len(aucs)


def main():
    expr_d, labels_d, _ = load_discovery_data(DISCOVERY_EXPR, DISCOVERY_CLIN)
    expr_v, labels_v = load_validation_data(EXTERNAL_SERIES, EXTERNAL_EXPR)
    expr_v = expr_v.loc[:, ~expr_v.columns.duplicated()]

    selected_indices, _ = univariate_screening(
        expr_d.values, labels_d.values, n_top=10, cv_splits=5, random_state=42,
    )
    selected_ids = [int(g) for g in expr_d.columns[selected_indices]]
    selected_symbols = [ID_TO_SYMBOL.get(g) for g in selected_ids]

    shared_positions = [
        i for i, s in enumerate(selected_symbols) if s in expr_v.columns
    ]
    shared_ids = [selected_ids[i] for i in shared_positions]
    shared_symbols = [selected_symbols[i] for i in shared_positions]
    missing_ids = [
        selected_ids[i] for i in range(len(selected_ids)) if i not in shared_positions
    ]

    # Fit the model only in the discovery cohort.
    X_d = expr_d.iloc[:, selected_indices].astype(float).values
    scaler = StandardScaler()
    model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
    model.fit(scaler.fit_transform(X_d), labels_d.values)
    coefs = model.coef_[0][shared_positions]

    # External data are raw FPKM-like values: log2(x + 1).
    X_v_raw = np.log2(expr_v[shared_symbols].astype(float).values + 1.0)
    y_v = labels_v.values

    # PRIMARY: each gene standardized WITHIN the external cohort.
    mu_v = X_v_raw.mean(axis=0)
    sd_v = X_v_raw.std(axis=0, ddof=1)
    keep = sd_v > 0
    dropped_flat = [shared_ids[i] for i in range(len(shared_ids)) if not keep[i]]
    z_v = (X_v_raw[:, keep] - mu_v[keep]) / sd_v[keep]
    primary_scores = z_v @ coefs[keep]
    primary_auc = roc_auc_score(y_v, primary_scores)
    ci_lo, ci_hi, n_valid = bootstrap_ci(y_v, primary_scores)

    # SECONDARY (reported for transparency only): discovery-cohort scaling,
    # which was the original method.
    disc_scaled = (X_v_raw - scaler.mean_[shared_positions]) / scaler.scale_[shared_positions]
    secondary_auc = roc_auc_score(y_v, disc_scaled @ coefs)

    audit = pd.DataFrame({
        "selected_rank": range(1, len(selected_ids) + 1),
        "discovery_gene_id": selected_ids,
        "gene_symbol": selected_symbols,
        "available_external": [i in shared_positions for i in range(len(selected_ids))],
        "zero_variance_external": [
            (selected_ids[i] in dropped_flat) for i in range(len(selected_ids))
        ],
    })
    audit.to_csv(RESULTS_DIR / "external_gene_audit.csv", index=False)

    summary = pd.DataFrame([{
        "analysis": "discovery_derived_9_gene_sensitivity",
        "auc": primary_auc,
        "bootstrap_ci_lower": ci_lo,
        "bootstrap_ci_upper": ci_hi,
        "n_external": len(labels_v),
        "n_responders": int(labels_v.sum()),
        "n_nonresponders": int((labels_v == 0).sum()),
        "n_selected_genes": len(selected_ids),
        "n_shared_genes": len(shared_ids),
        "n_genes_scored": int(keep.sum()),
        "missing_gene_ids": ",".join(map(str, missing_ids)),
        "zero_variance_gene_ids": ",".join(map(str, dropped_flat)),
        "scaling": "within_external_cohort",
        "auc_with_discovery_scaling": secondary_auc,
        "n_valid_bootstrap": n_valid,
    }])
    summary.to_csv(RESULTS_DIR / "external_sensitivity_summary.csv", index=False)

    print(f"Selected genes: {len(selected_ids)}")
    print(f"Shared external genes: {len(shared_ids)}")
    print(f"Missing gene IDs: {missing_ids}")
    print(f"Zero-variance external genes dropped: {dropped_flat}")
    print(f"Genes scored: {int(keep.sum())}")
    print(f"External AUC (within-cohort scaling, PRIMARY): {primary_auc:.3f}")
    print(f"Bootstrap 95% CI: {ci_lo:.3f} to {ci_hi:.3f}")
    print(f"External AUC (discovery scaling, secondary): {secondary_auc:.3f}")
    print("Saved external sensitivity audit files.")


if __name__ == "__main__":
    main()