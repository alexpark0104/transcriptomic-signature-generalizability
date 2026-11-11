import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from config import ENTREZ_TO_SYMBOL, COEFFICIENTS


def match_signature_genes(expr, gene_symbols, coefficients):
    """Match signature genes to expression columns by EXACT symbol
    (case-insensitive). No partial or substring matching.

    Returns the matched column names and their coefficients in the same order,
    plus the list of signature symbols that were not found.
    """
    lookup = {}
    for col in expr.columns:
        lookup.setdefault(str(col).upper(), col)

    present, weights, missing = [], [], []
    for sym, coef in zip(gene_symbols, coefficients):
        key = str(sym).upper()
        if key in lookup:
            present.append(lookup[key])
            weights.append(coef)
        else:
            missing.append(sym)
    return present, np.asarray(weights, dtype=float), missing


def compute_signature_score(expr, gene_symbols, coefficients):
    """Score every sample with the fixed signature.

    Steps: exact gene matching, log2(x + 1) if the data look like raw FPKM,
    z-scoring EACH GENE WITHIN THIS COHORT, then a weighted sum using the
    fixed discovery-derived coefficients.

    Returns (score, present_symbols), as before.
    """
    expr = expr.loc[:, ~expr.columns.duplicated()]
    present, weights, missing = match_signature_genes(expr, gene_symbols, coefficients)

    if len(present) < 5:
        raise ValueError(f"Too few signature genes available: {len(present)}")

    values = expr[present].values.astype(float)

    # Decide on the log transform using the whole matrix, not just these genes.
    if np.nanmax(expr.values.astype(float)) > 100:
        values = np.log2(values + 1)

    mu = values.mean(axis=0)
    sd = values.std(axis=0, ddof=1)
    keep = sd > 0
    if keep.sum() < 5:
        raise ValueError("Too few signature genes with nonzero variance")

    z = (values[:, keep] - mu[keep]) / sd[keep]
    kept_symbols = [p for p, k in zip(present, keep) if k]
    score = z @ weights[keep]

    if missing:
        print(f"Signature genes not found in this cohort: {missing}")
    print(f"Signature genes used: {len(kept_symbols)} of {len(gene_symbols)}")
    return score, kept_symbols


def validate_on_gse78220(expr, labels, gene_symbols, coefficients, save_plot=True):
    score, present = compute_signature_score(expr, gene_symbols, coefficients)
    y_true = labels.values

    # Orientation is fixed in advance: a higher score predicts response.
    # Do not flip the sign after looking at the result.
    auc = roc_auc_score(y_true, score)

    if save_plot:
        fpr, tpr, _ = roc_curve(y_true, score)
        plt.figure(figsize=(6, 6))
        plt.plot(fpr, tpr, lw=2.5, label=f'AUC = {auc:.3f}')
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Validation on GSE78220 (Hugo et al.)')
        plt.legend()
        os.makedirs('results/figures', exist_ok=True)
        plt.savefig('results/figures/validation_roc.png', dpi=300)
        plt.close()
    return auc, present