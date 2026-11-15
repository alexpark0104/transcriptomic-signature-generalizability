"""Pipeline variants for the sensitivity analysis (pure functions, no file I/O).

Every pipeline is a function  fit_predict(X_train, y_train, X_test, seed) -> P(responder)
that uses ONLY the training rows for every data-dependent step: any gene
pre-filter, gene screening, scaling, and any tuning of regularization strength.
The shared evaluation helpers below then wrap any pipeline in the same outer
cross-validation and "apparent" (fit and evaluate on all data) analyses.

The baseline pipeline here reuses analysis.select_top_genes and
analysis._fit_predict, so it reproduces analysis.nested_cv / analysis.apparent_auc
exactly (checked in tests/test_analysis.py and at the start of run_sensitivity.py).
analysis.py itself is deliberately not modified.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

import analysis


def make_screen_pipeline(n_genes=10, method="inner_cv", inner_splits=5, prefilter=None):
    """Univariate screening of the top `n_genes`, then unpenalized-ish logistic
    regression (C = 1) on z-scored genes, as in the main analysis.

    prefilter = "top_half_mean": before screening, keep only genes whose mean
    expression in the TRAINING rows is at or above the median gene mean. This is
    unsupervised (labels are not used) and is recomputed inside every training fold.
    """
    def fit_predict(X_train, y_train, X_test, seed):
        cols = np.arange(X_train.shape[1])
        if prefilter == "top_half_mean":
            means = X_train.mean(axis=0)
            cols = np.where(means >= np.median(means))[0]
        elif prefilter is not None:
            raise ValueError(f"unknown prefilter {prefilter!r}")
        top, _ = analysis.select_top_genes(X_train[:, cols], y_train, n_genes,
                                           method, inner_splits, seed)
        return analysis._fit_predict(X_train, y_train, X_test, cols[top])
    return fit_predict


def make_ridge_pipeline(Cs=(1e-4, 1e-3, 1e-2, 1e-1, 1.0), inner_splits=3):
    """L2-penalized logistic regression on ALL genes (z-scored), with the
    penalty strength C chosen by an inner cross-validation on the training rows.

    Scaling is refit inside every inner split as well, so nothing about an inner
    validation sample reaches the model that scores it. Ties between C values go
    to the smaller C (stronger regularization).
    """
    Cs = tuple(Cs)

    def fit_predict(X_train, y_train, X_test, seed):
        inner = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=seed)
        best_C, best_auc = Cs[0], -1.0
        for C in Cs:
            oof = np.zeros(len(y_train))
            for tr, va in inner.split(X_train, y_train):
                scaler = StandardScaler().fit(X_train[tr])
                model = LogisticRegression(C=C, max_iter=1000).fit(
                    scaler.transform(X_train[tr]), y_train[tr])
                oof[va] = model.decision_function(scaler.transform(X_train[va]))
            auc = roc_auc_score(y_train, oof)
            if auc > best_auc + 1e-12:
                best_auc, best_C = auc, C
        scaler = StandardScaler().fit(X_train)
        model = LogisticRegression(C=best_C, max_iter=1000).fit(
            scaler.transform(X_train), y_train)
        return model.predict_proba(scaler.transform(X_test))[:, 1]
    return fit_predict


CONFIGS = {
    "screen_10_baseline": make_screen_pipeline(10),
    "screen_5": make_screen_pipeline(5),
    "screen_20": make_screen_pipeline(20),
    "screen_50": make_screen_pipeline(50),
    "screen_10_prefilter_mean": make_screen_pipeline(10, prefilter="top_half_mean"),
    "ridge_tuned": make_ridge_pipeline(),
}

DESCRIPTIONS = {
    "screen_10_baseline": "Main analysis: top 10 genes by inner-CV AUC, logistic regression (C = 1)",
    "screen_5": "Top 5 genes",
    "screen_20": "Top 20 genes",
    "screen_50": "Top 50 genes",
    "screen_10_prefilter_mean": "Top 10 genes after keeping the half of genes with higher mean expression (training fold only)",
    "ridge_tuned": "L2 logistic regression on all genes, penalty tuned by inner 3-fold CV (training fold only)",
}


def nested_auc(fit_predict, X, y, outer_splits=5, seed=42):
    """Outer CV around any pipeline. Returns (mean fold AUC, pooled AUC)."""
    outer = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=seed)
    probs = np.zeros(len(y), dtype=float)
    fold_aucs = []
    for train_idx, test_idx in outer.split(X, y):
        p = fit_predict(X[train_idx], y[train_idx], X[test_idx], seed)
        probs[test_idx] = p
        fold_aucs.append(roc_auc_score(y[test_idx], p))
    return float(np.mean(fold_aucs)), float(roc_auc_score(y, probs))


def apparent_auc(fit_predict, X, y, seed=42):
    """In-sample AUC: the pipeline is fitted on all rows and scores the same rows."""
    return float(roc_auc_score(y, fit_predict(X, y, X, seed)))