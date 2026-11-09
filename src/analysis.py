import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def univariate_screening(X, y, n_top=10, cv_splits=5, random_state=42):
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    gene_aucs = []
    for g in range(X.shape[1]):
        x_g = X[:, g].reshape(-1, 1)
        auc_cv = []
        for train_idx, val_idx in cv.split(x_g, y):
            lr = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000)
            lr.fit(x_g[train_idx], y[train_idx])
            prob = lr.predict_proba(x_g[val_idx])[:, 1]
            auc_cv.append(roc_auc_score(y[val_idx], prob))
        gene_aucs.append(np.mean(auc_cv))
    top_indices = np.argsort(gene_aucs)[-n_top:][::-1]
    top_aucs = [gene_aucs[i] for i in top_indices]
    return top_indices, top_aucs

def nested_cv(X, y, n_genes=10, outer_splits=5, inner_splits=5, random_state=42):
    outer_cv = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=random_state)
    outer_probs = np.zeros(len(y))
    fold_aucs = []
    for train_idx, test_idx in outer_cv.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        inner_cv = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=random_state)
        gene_aucs = {}
        for g in range(X.shape[1]):
            x_g = X_train[:, g].reshape(-1, 1)
            auc_cv = []
            for inner_train, inner_val in inner_cv.split(x_g, y_train):
                lr = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000)
                lr.fit(x_g[inner_train], y_train[inner_train])
                prob = lr.predict_proba(x_g[inner_val])[:, 1]
                auc_cv.append(roc_auc_score(y_train[inner_val], prob))
            gene_aucs[g] = np.mean(auc_cv)
        top_genes = sorted(gene_aucs.items(), key=lambda x: x[1], reverse=True)[:n_genes]
        top_indices = [g for g, _ in top_genes]
        X_train_top = X_train[:, top_indices]
        X_test_top = X_test[:, top_indices]
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_top)
        X_test_scaled = scaler.transform(X_test_top)
        lr = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000)
        lr.fit(X_train_scaled, y_train)
        prob = lr.predict_proba(X_test_scaled)[:, 1]
        outer_probs[test_idx] = prob
        fold_aucs.append(roc_auc_score(y_test, prob))
    return np.mean(fold_aucs), roc_auc_score(y, outer_probs), outer_probs
def apparent_auc(X, y, n_genes=10):
    """
    Estimate apparent, in-sample performance.

    Feature selection and model fitting both use the full dataset.
    This is intentionally optimistic and is included only to quantify
    the difference between apparent and leakage-resistant performance.
    """
    top_indices, _ = univariate_screening(
        X,
        y,
        n_top=n_genes,
        cv_splits=5,
        random_state=42,
    )

    scaler = StandardScaler()
    X_selected = X[:, top_indices]
    X_scaled = scaler.fit_transform(X_selected)

    model = LogisticRegression(
        C=1.0,
        solver='lbfgs',
        max_iter=1000,
    )
    model.fit(X_scaled, y)

    probabilities = model.predict_proba(X_scaled)[:, 1]
    auc = roc_auc_score(y, probabilities)

    return auc, top_indices
