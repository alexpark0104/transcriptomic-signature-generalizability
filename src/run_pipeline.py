import os, pickle, pandas as pd, numpy as np
from data_loader import load_discovery_data, load_validation_data
from analysis import univariate_screening, nested_cv, apparent_auc
from validation import validate_on_gse78220
from config import SIGNATURE_GENES, COEFFICIENTS, ENTREZ_TO_SYMBOL

os.makedirs('results', exist_ok=True)
print("Loading discovery data...")
expr, labels, _ = load_discovery_data('data/rld.BMS038.20171011.csv', 'data/bms038_clinical_data.csv')
X, y = expr.values, labels.values
print(f"Samples: {X.shape[0]}, Genes: {X.shape[1]}")

print("Running apparent in-sample analysis...")
apparent, apparent_genes = apparent_auc(X, y, n_genes=10)
print(f"Apparent in-sample AUC: {apparent:.3f}")

selected_gene_ids = expr.columns[apparent_genes]

selected_genes_table = pd.DataFrame({
    "rank": np.arange(1, len(selected_gene_ids) + 1),
    "discovery_gene_id": selected_gene_ids,
})

selected_genes_table.to_csv(
    "results/selected_discovery_genes.csv",
    index=False,
)

print("Saved results/selected_discovery_genes.csv")

print("Running nested 5-fold CV...")
mean_auc, overall_auc, _ = nested_cv(X, y, n_genes=10)
print(f"Mean AUC: {mean_auc:.3f}, Overall AUC: {overall_auc:.3f}")

print("Running external validation...")
try:
    expr_val, labels_val = load_validation_data('data/GSE78220_series_matrix.txt.gz', 'data/GSE78220_PatientFPKM.xlsx')
    auc_val, _ = validate_on_gse78220(expr_val, labels_val, list(ENTREZ_TO_SYMBOL.values()), COEFFICIENTS)
    print(f"Validation AUC: {auc_val:.3f}")
    os.makedirs('results', exist_ok=True)

    # Primary analysis: all three estimates use the same
    # discovery feature-selection pipeline.
    primary_results = pd.DataFrame({
        'estimate': [
            'apparent_in_sample',
            'nested_cv_mean',
            'nested_cv_pooled',
        ],
        'auc': [
            apparent,
            mean_auc,
            overall_auc,
        ],
        'n_samples': [
            len(y),
            len(y),
            len(y),
        ]
    })

    primary_results.to_csv(
        'results/auc_summary.csv',
        index=False,
    )
    print("Saved primary results/auc_summary.csv")

    # Exploratory analysis: this is the repository's separate
    # prespecified fixed-signature validation.
    fixed_signature_result = pd.DataFrame([{
        'analysis': 'original_fixed_signature_external',
        'auc': auc_val,
        'n_external': len(labels_val),
        'n_responders': int(labels_val.sum()),
        'n_nonresponders': int((labels_val == 0).sum()),
    }])

    fixed_signature_result.to_csv(
        'results/fixed_signature_external.csv',
        index=False,
    )
    print("Saved results/fixed_signature_external.csv")

except Exception as e:
    print(f"Validation failed: {e}")
