"""Do the inherited coefficients belong to the inherited Entrez IDs?

Run from the repo root:  python src/check_inherited_signature.py

If the coefficients in config.py were fitted on the genes with these Entrez IDs
in the discovery data, then scoring the discovery patients with those IDs and
coefficients should give a high AUC, far above what the same coefficients give
when applied to random sets of genes, or when the coefficients are shuffled
among the same ten genes. This is an in-sample diagnostic about provenance,
not a performance estimate.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))

from config import COEFFICIENTS, SIGNATURE_GENES
from data_loader import load_discovery_data

N_DRAWS = 2000
SEED = 2026

expr, labels, _ = load_discovery_data(
    "data/rld.BMS038.20171011.csv", "data/bms038_clinical_data.csv"
)
y = labels.values.astype(int)
X = expr.values.astype(float)
Z = (X - X.mean(axis=0)) / np.where(X.std(axis=0) == 0, 1.0, X.std(axis=0))

position = {int(c): i for i, c in enumerate(expr.columns)}
missing = [g for g in SIGNATURE_GENES if int(g) not in position]
if missing:
    sys.exit(f"Inherited IDs not found among the discovery columns: {missing}")

idx = [position[int(g)] for g in SIGNATURE_GENES]
coefs = np.asarray(COEFFICIENTS, dtype=float)
observed = roc_auc_score(y, Z[:, idx] @ coefs)
print(f"Samples: {len(y)} (responders {int(y.sum())}, non-responders {int((y == 0).sum())})")
print(f"\nInherited IDs + inherited coefficients, in-sample AUC on discovery data: {observed:.3f}")

rng = np.random.default_rng(SEED)

random_sets = np.array([
    roc_auc_score(y, Z[:, rng.choice(Z.shape[1], 10, replace=False)] @ coefs)
    for _ in range(N_DRAWS)
])
shuffled_coefs = np.array([
    roc_auc_score(y, Z[:, idx] @ rng.permutation(coefs)) for _ in range(N_DRAWS)
])

for name, values in (("same coefficients on RANDOM 10-gene sets", random_sets),
                     ("coefficients SHUFFLED among the inherited IDs", shuffled_coefs)):
    p = (1 + int((values >= observed).sum())) / (1 + len(values))
    print(f"\n{name} ({N_DRAWS} draws)")
    print(f"  mean {values.mean():.3f}, 95th percentile {np.percentile(values, 95):.3f}, "
          f"max {values.max():.3f}   -> share >= observed: p = {p:.4f}")

print("\nPer gene: coefficient sign versus the gene's own direction in the discovery data")
rows = []
for gene_id, coef, i in zip(SIGNATURE_GENES, coefs, idx):
    diff = X[y == 1, i].mean() - X[y == 0, i].mean()
    rows.append({
        "entrez_id": gene_id,
        "coefficient": coef,
        "responder_minus_nonresponder_mean": round(diff, 3),
        "sign_agrees": bool(np.sign(coef) == np.sign(diff)),
        "single_gene_auc": round(roc_auc_score(y, X[:, i]), 3),
    })
table = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(table.to_string(index=False))
print(f"\nSigns agree for {int(table['sign_agrees'].sum())} of {len(table)} genes.")