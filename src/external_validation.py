"""External validation on GSE78220 (Hugo et al.): all analyses in one place.

Run from the repo root:  python src/external_validation.py

Analyses (all use the SAME scoring rule, fixed before looking at any result):
  PRIMARY    discovery-derived model, screening method "inner_cv"
  SECONDARY  discovery-derived model, screening method "train_auc"
  INHERITED  the inherited candidate signature in config.py (undocumented
             derivation; reported separately, not an unbiased validation)

Scoring rule: each gene is log2(FPKM + 1) transformed and z-scored WITHIN the
external cohort; the score is the weighted sum using the weights fitted in the
discovery cohort (logistic-regression coefficients on z-scored discovery data,
or the inherited coefficients). Higher score = predicted responder. The
direction is fixed in advance and is never flipped after seeing a result.

Gene symbols come only from data/entrez_to_symbol.csv (NCBI Gene). If a current
symbol is absent from the external workbook but a former symbol (alias) is
present, the alias is used. Any gene with no match stops the script.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))

from analysis import select_top_genes
from config import COEFFICIENTS, SIGNATURE_GENES
from data_loader import load_discovery_data, load_validation_data
from validation import compute_signature_score

RESULTS_DIR = Path("results")
FIGURES_DIR = RESULTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
N_BOOT = 10000
SEED = 42


def load_mapping(path="data/entrez_to_symbol.csv"):
    table = pd.read_csv(path)
    mapping = {}
    for _, row in table.iterrows():
        alias = row["alias_found_in_external"]
        if bool(row["symbol_in_external"]):
            column, used_alias = row["symbol"], False
        elif isinstance(alias, str) and alias:
            column, used_alias = alias, True
        else:
            column, used_alias = None, False
        mapping[int(row["entrez_id"])] = {
            "symbol": row["symbol"], "external_column": column, "used_alias": used_alias,
        }
    return mapping


def bootstrap_ci(y, scores, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), size=len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y[idx], scores[idx]))
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    return lo, hi, len(aucs)


def main():
    mapping = load_mapping()

    expr_d, labels_d, _ = load_discovery_data(
        "data/rld.BMS038.20171011.csv", "data/bms038_clinical_data.csv"
    )
    X_d, y_d = expr_d.values.astype(float), labels_d.values.astype(int)

    expr_v, labels_v = load_validation_data(
        "data/GSE78220_series_matrix.txt.gz", "data/GSE78220_PatientFPKM.xlsx"
    )
    y_v = labels_v.values.astype(int)

    # ---- define the analyses: (name, role, entrez ids, weights) ----
    analyses = []
    for method, role in (("inner_cv", "primary"), ("train_auc", "secondary")):
        top, _ = select_top_genes(X_d, y_d, n_top=10, method=method)
        ids = [int(expr_d.columns[i]) for i in top]
        scaler = StandardScaler()
        model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
        model.fit(scaler.fit_transform(X_d[:, top]), y_d)
        analyses.append((f"discovery_derived_{method}", role, ids, model.coef_[0]))
    analyses.append(("inherited_candidate", "inherited",
                     [int(g) for g in SIGNATURE_GENES], np.asarray(COEFFICIENTS, float)))

    summary_rows, gene_rows = [], []
    for name, role, ids, weights in analyses:
        unresolved = [g for g in ids if mapping.get(g, {}).get("external_column") is None]
        if unresolved:
            sys.exit(f"[{name}] no external column for Entrez IDs {unresolved}. "
                     "Check data/entrez_to_symbol.csv.")
        columns = [mapping[g]["external_column"] for g in ids]

        print(f"\n=== {name} ({role}) ===")
        score, present = compute_signature_score(expr_v, columns, weights)
        present_upper = {str(p).upper() for p in present}
        dropped = [c for c in columns if str(c).upper() not in present_upper]

        auc = roc_auc_score(y_v, score)
        lo, hi, n_valid = bootstrap_ci(y_v, score)
        p_one = mannwhitneyu(score[y_v == 1], score[y_v == 0], alternative="greater").pvalue
        p_two = mannwhitneyu(score[y_v == 1], score[y_v == 0], alternative="two-sided").pvalue
        print(f"  external AUC = {auc:.3f}  (95% bootstrap CI {lo:.3f} to {hi:.3f})")
        print(f"  rank test: one-sided p (score higher in responders) = {p_one:.3f}, two-sided p = {p_two:.3f}")

        summary_rows.append({
            "analysis": name, "role": role, "auc": auc,
            "bootstrap_ci_lower": lo, "bootstrap_ci_upper": hi,
            "p_one_sided": p_one, "p_two_sided": p_two,
            "n_external": len(y_v), "n_responders": int(y_v.sum()),
            "n_nonresponders": int((y_v == 0).sum()),
            "n_genes_defined": len(ids), "n_genes_scored": len(present),
            "genes_dropped_zero_variance": ";".join(map(str, dropped)),
            "aliases_used": ";".join(
                f"{mapping[g]['symbol']}->{mapping[g]['external_column']}"
                for g in ids if mapping[g]["used_alias"]),
            "n_valid_bootstrap": n_valid,
        })
        for rank, (g, w) in enumerate(zip(ids, weights), start=1):
            col = mapping[g]["external_column"]
            gene_rows.append({
                "analysis": name, "rank": rank, "entrez_id": g,
                "symbol": mapping[g]["symbol"], "external_column": col,
                "used_alias": mapping[g]["used_alias"], "weight": float(w),
                "scored": str(col).upper() in present_upper,
            })

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(RESULTS_DIR / "external_validation.csv", index=False)
    pd.DataFrame(gene_rows).to_csv(RESULTS_DIR / "external_gene_audit.csv", index=False)

    # ---- figure: external AUC with bootstrap interval ----
    fig, ax = plt.subplots(figsize=(8, 1.4 + 0.8 * len(summary)))
    labels_plot = {"discovery_derived_inner_cv": "Discovery-derived (inner-CV screen)\nPRIMARY",
                   "discovery_derived_train_auc": "Discovery-derived (training-fold screen)\nsecondary",
                   "inherited_candidate": "Inherited candidate\n(undocumented derivation)"}
    ypos = np.arange(len(summary))[::-1]
    ax.errorbar(summary["auc"], ypos,
                xerr=[summary["auc"] - summary["bootstrap_ci_lower"],
                      summary["bootstrap_ci_upper"] - summary["auc"]],
                fmt="o", color="black", capsize=4)
    ax.axvline(0.5, color="gray", linestyle=":", label="Chance = 0.5")
    ax.set_yticks(ypos)
    ax.set_yticklabels([labels_plot.get(a, a) for a in summary["analysis"]], fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_xlabel(f"External AUC, GSE78220 (n = {len(y_v)}); bars = 95% bootstrap CI")
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "external_validation.png", dpi=300, bbox_inches="tight")

    pd.set_option("display.width", 220)
    print("\n" + summary[["analysis", "role", "auc", "bootstrap_ci_lower", "bootstrap_ci_upper",
                          "p_one_sided", "n_genes_defined", "n_genes_scored"]].round(3).to_string(index=False))
    print("\nSaved results/external_validation.csv, results/external_gene_audit.csv, "
          "results/figures/external_validation.png")


if __name__ == "__main__":
    main()