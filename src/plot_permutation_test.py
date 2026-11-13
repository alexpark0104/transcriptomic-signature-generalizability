"""Figure for the whole-pipeline label-permutation test.

Reads results/permutation_test_full.csv and results/permutation_test_full_summary.csv
(written by run_permutation_test.py). Observed values and p-values come from the
summary file, so nothing is typed in by hand.

Run from the repo root:  python src/plot_permutation_test.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

RESULTS_DIR = Path("results")
FIGURES_DIR = RESULTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

null = pd.read_csv(RESULTS_DIR / "permutation_test_full.csv")
summary = pd.read_csv(RESULTS_DIR / "permutation_test_full_summary.csv")

methods = list(summary["method"].unique())
method_titles = {
    "inner_cv": "Screening by inner-CV AUC",
    "train_auc": "Screening by training-fold AUC",
}
panels = [
    ("apparent_auc", "Apparent (in-sample) AUC", "#c0392b"),
    ("nested_pooled_auc", "Nested-CV pooled AUC", "#2980b9"),
]

fig, axes = plt.subplots(len(methods), len(panels), figsize=(12, 4.6 * len(methods)))
axes = axes.reshape(len(methods), len(panels))

for r, method in enumerate(methods):
    for c, (stat, label, color) in enumerate(panels):
        ax = axes[r, c]
        values = null[null["method"] == method][stat].values
        row = summary[(summary["method"] == method) & (summary["statistic"] == stat)].iloc[0]

        ax.hist(values, bins=40, color=color, alpha=0.75, edgecolor="black", linewidth=0.4)
        ax.axvline(row["observed"], color="black", linestyle="--", linewidth=2,
                   label=f"Observed = {row['observed']:.3f}")
        if stat == "apparent_auc":
            # Chance is not the relevant reference for an in-sample statistic;
            # zoom in on where the shuffled-label values actually fall.
            lo = min(values.min(), row["observed"]) - 0.01
            ax.set_xlim(lo, 1.005)
        else:
            ax.axvline(0.5, color="gray", linestyle=":", linewidth=1.5, label="Chance = 0.500")
        ax.axvspan(row["null_2.5pct"], row["null_97.5pct"], color="gray", alpha=0.12,
                   label="Null 95% range")

        ax.set_title(f"{method_titles.get(method, method)}\n{label}", fontsize=11)
        ax.set_xlabel("AUC with shuffled outcome labels")
        ax.set_ylabel("Number of permutations")
        ax.text(0.02, 0.96,
                f"p (null ≥ observed) = {row['p_upper']:.3f}\nn = {int(row['n_permutations'])} permutations",
                transform=ax.transAxes, va="top", fontsize=9)
        ax.legend(frameon=False, fontsize=8, loc="upper center")
        ax.grid(axis="y", alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

fig.suptitle("Whole-pipeline label-permutation test", fontsize=14, fontweight="bold")
fig.text(0.5, 0.005,
         "Each permutation repeated gene screening, model fitting, and evaluation on shuffled outcomes.",
         ha="center", fontsize=9)
fig.tight_layout(rect=(0, 0.02, 1, 0.96))

output_path = FIGURES_DIR / "permutation_test_full.png"
fig.savefig(output_path, dpi=300, bbox_inches="tight")
print(f"Saved {output_path}")