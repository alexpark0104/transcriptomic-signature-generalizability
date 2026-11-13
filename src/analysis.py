"""Apparent, nested-CV and repeated nested-CV AUC for small-sample gene signatures.

Two gene-screening methods are supported. Every function takes a `method`
argument, and apparent and nested analyses must always use the same one:

  "inner_cv"  : each gene is scored by its mean cross-validated AUC inside the
                training data (the original, documented method). The slope sign
                is learned on the inner training part and the AUC is measured
                on the inner validation part.
  "train_auc" : each gene is scored by its directional rank AUC,
                max(AUC, 1 - AUC), on the training data (a simpler filter).

Both are computed for all genes at once. For a single gene, a logistic
regression's slope has the sign of the class-mean difference, so its
validation AUC is the rank AUC, flipped when that sign is negative.

Ties are broken deterministically (see select_top_genes), so the selected
genes do not depend on the order of the columns.
"""
import warnings

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

METHODS = ("inner_cv", "train_auc")
DEFAULT_METHOD = "inner_cv"


def _rank_auc(X, y):
    """Forward rank (Mann-Whitney) AUC of every gene; class 1 is positive."""
    pos = y == 1
    n1 = int(pos.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        raise ValueError("Feature screening requires both outcome classes.")
    ranks = rankdata(X, axis=0, method="average")
    return (ranks[pos].sum(axis=0) - n1 * (n1 + 1) / 2) / (n1 * n0)


def _inner_cv_scores(X, y, n_splits, random_state):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_aucs = []
    for tr, va in cv.split(X, y):
        sign = np.sign(X[tr][y[tr] == 1].mean(axis=0) - X[tr][y[tr] == 0].mean(axis=0))
        auc = _rank_auc(X[va], y[va])
        fold_aucs.append(np.where(sign > 0, auc, np.where(sign < 0, 1.0 - auc, 0.5)))
    return np.mean(fold_aucs, axis=0)


def gene_scores(X, y, method=DEFAULT_METHOD, cv_splits=5, random_state=42):
    """Return (primary_score, tie_break_score) for every gene."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y).astype(int)
    if method == "inner_cv":
        primary = _inner_cv_scores(X, y, cv_splits, random_state)
    elif method == "train_auc":
        a = _rank_auc(X, y)
        primary = np.maximum(a, 1.0 - a)
    else:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")

    # Secondary key for ties: absolute standardized mean difference.
    sd = X.std(axis=0, ddof=1)
    sd[sd == 0] = np.inf
    secondary = np.abs(X[y == 1].mean(axis=0) - X[y == 0].mean(axis=0)) / sd
    return primary, secondary


def select_top_genes(X, y, n_top=10, method=DEFAULT_METHOD, cv_splits=5, random_state=42):
    primary, secondary = gene_scores(X, y, method, cv_splits, random_state)
    primary = np.round(primary, 12)  # float noise must not decide ties
    order = np.lexsort((np.arange(len(primary)), -secondary, -primary))
    top = order[:n_top]
    return top, primary[top].tolist()


def univariate_screening(X, y, n_top=10, cv_splits=5, random_state=42, method=DEFAULT_METHOD):
    """Backwards-compatible name. Returns (top_indices, top_scores)."""
    return select_top_genes(X, y, n_top, method, cv_splits, random_state)


def _fit_predict(X_train, y_train, X_test, top):
    scaler = StandardScaler()
    a = scaler.fit_transform(X_train[:, top])
    b = scaler.transform(X_test[:, top])
    model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000).fit(a, y_train)
    return model.predict_proba(b)[:, 1]


def nested_cv(X, y, n_genes=10, outer_splits=5, inner_splits=5, random_state=42,
              method=DEFAULT_METHOD):
    """Gene screening, scaling and model fitting happen only inside each outer
    training fold. Returns (mean fold AUC, pooled out-of-fold AUC, predictions)."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y).astype(int)
    outer_cv = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=random_state)
    outer_probs = np.zeros(len(y), dtype=float)
    fold_aucs = []
    for train_idx, test_idx in outer_cv.split(X, y):
        top, _ = select_top_genes(X[train_idx], y[train_idx], n_genes, method,
                                  inner_splits, random_state)
        probs = _fit_predict(X[train_idx], y[train_idx], X[test_idx], top)
        outer_probs[test_idx] = probs
        fold_aucs.append(roc_auc_score(y[test_idx], probs))
    return np.mean(fold_aucs), roc_auc_score(y, outer_probs), outer_probs


def apparent_auc(X, y, n_genes=10, method=DEFAULT_METHOD):
    """In-sample performance: selection and fitting both use all the data.
    Intentionally optimistic; it uses the same screening method as nested_cv."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y).astype(int)
    top, _ = select_top_genes(X, y, n_genes, method, 5, 42)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X[:, top])
    model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000).fit(Xs, y)
    return roc_auc_score(y, model.predict_proba(Xs)[:, 1]), top


def repeated_nested_cv(X, y, n_repeats=20, n_genes=10, outer_splits=5, inner_splits=5,
                       random_state=42, method=DEFAULT_METHOD):
    """Repeat the full nested procedure with different fold splits (the seed
    changes the outer splits and, for method="inner_cv", the inner splits)."""
    rows = []
    for repeat in range(n_repeats):
        seed = random_state + repeat
        mean_auc, pooled_auc, _ = nested_cv(X, y, n_genes, outer_splits, inner_splits,
                                            seed, method)
        rows.append({"repeat": repeat + 1, "random_state": seed, "method": method,
                     "mean_outer_auc": mean_auc, "pooled_outer_auc": pooled_auc})
        print(f"[{method}] repeat {repeat + 1}/{n_repeats}: "
              f"mean AUC = {mean_auc:.3f}, pooled AUC = {pooled_auc:.3f}")
    return pd.DataFrame(rows)