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


def main():
    expr_d, labels_d, _ = load_discovery_data(
        DISCOVERY_EXPR,
        DISCOVERY_CLIN,
    )

    expr_v, labels_v = load_validation_data(
        EXTERNAL_SERIES,
        EXTERNAL_EXPR,
    )

    selected_indices, _ = univariate_screening(
        expr_d.values,
        labels_d.values,
        n_top=10,
        cv_splits=5,
        random_state=42,
    )

    selected_ids = [
        int(gene_id)
        for gene_id in expr_d.columns[selected_indices]
    ]

    selected_symbols = [
        ID_TO_SYMBOL.get(gene_id)
        for gene_id in selected_ids
    ]

    shared_positions = [
        index
        for index, symbol in enumerate(selected_symbols)
        if symbol in expr_v.columns
    ]

    shared_ids = [
        selected_ids[index]
        for index in shared_positions
    ]

    shared_symbols = [
        selected_symbols[index]
        for index in shared_positions
    ]

    missing_ids = [
        selected_ids[index]
        for index in range(len(selected_ids))
        if index not in shared_positions
    ]

    # Fit the model only in the discovery cohort.
    X_d = expr_d.iloc[:, selected_indices].astype(float).values

    scaler = StandardScaler()
    X_d_scaled = scaler.fit_transform(X_d)

    model = LogisticRegression(
        C=1.0,
        solver="lbfgs",
        max_iter=1000,
    )
    model.fit(scaler.transform(X_d), labels_d.values)

    # The external workbook contains raw FPKM-like values.
    # log2(x + 1) is used as an exploratory harmonization step.
    X_v_raw = np.log2(
        expr_v[shared_symbols].astype(float).values + 1.0
    )

    X_v_scaled = (
        X_v_raw - scaler.mean_[shared_positions]
    ) / scaler.scale_[shared_positions]

    external_scores = (
        X_v_scaled @ model.coef_[0][shared_positions]
    )

    observed_auc = roc_auc_score(
        labels_v.values,
        external_scores,
    )

    rng = np.random.default_rng(42)
    bootstrap_aucs = []

    for _ in range(10000):
        indices = rng.integers(
            0,
            len(labels_v),
            size=len(labels_v),
        )

        sampled_labels = labels_v.values[indices]

        if len(np.unique(sampled_labels)) < 2:
            continue

        sampled_scores = external_scores[indices]
        bootstrap_aucs.append(
            roc_auc_score(sampled_labels, sampled_scores)
        )

    ci_lower, ci_upper = np.percentile(
        bootstrap_aucs,
        [2.5, 97.5],
    )

    audit = pd.DataFrame({
        "selected_rank": range(1, len(selected_ids) + 1),
        "discovery_gene_id": selected_ids,
        "gene_symbol": selected_symbols,
        "available_external": [
            index in shared_positions
            for index in range(len(selected_ids))
        ],
    })

    audit.to_csv(
        RESULTS_DIR / "external_gene_audit.csv",
        index=False,
    )

    summary = pd.DataFrame([{
        "analysis": "discovery_derived_9_gene_sensitivity",
        "auc": observed_auc,
        "bootstrap_ci_lower": ci_lower,
        "bootstrap_ci_upper": ci_upper,
        "n_external": len(labels_v),
        "n_responders": int(labels_v.sum()),
        "n_nonresponders": int((labels_v == 0).sum()),
        "n_selected_genes": len(selected_ids),
        "n_shared_genes": len(shared_ids),
        "missing_gene_ids": ",".join(map(str, missing_ids)),
    }])

    summary.to_csv(
        RESULTS_DIR / "external_sensitivity_summary.csv",
        index=False,
    )

    print(f"Selected genes: {len(selected_ids)}")
    print(f"Shared external genes: {len(shared_ids)}")
    print(f"Missing gene IDs: {missing_ids}")
    print(f"External sensitivity AUC: {observed_auc:.3f}")
    print(
        "Bootstrap 95% CI: "
        f"{ci_lower:.3f} to {ci_upper:.3f}"
    )
    print("Saved external sensitivity audit files.")


if __name__ == "__main__":
    main()
