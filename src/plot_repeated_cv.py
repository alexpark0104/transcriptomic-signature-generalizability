"""Figure: spread of the nested-CV pooled AUC across many fold assignments.

Run from the repo root:  python src/plot_repeated_cv.py

Reads results/repeated_nested_cv.csv (both screening methods, written by
run_repeated_cv.py) and results/permutation_test_full_summary.csv.

Each panel is one screening method. The histogram is the pooled nested AUC over
all repeats (each repeat = a new random assignment of patients to folds).
The grey band is the 95% range of the SAME single-split statistic when outcome
labels are shuffled (whole-pipeline permutation test), so the two are directly
comparable. The spread across repeats reflects fold assignment only; it is not
a confidence interval for the true AUC.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

RESULTS_DIR = Path("results")
FIGURES_DIR = RESULTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

runs = pd.read_csv(RESULTS_DIR / "repeated_nested_cv.csv")
perm = pd.read_csv(RESULTS_DIR / "permutation_test_full_summary.csv")

titles = {"inner_cv": "Screening by inner-CV AUC (primary)",
          "train_auc": "Screening by training-fold AUC"}
methods = [m for m in titles if m in set(runs["method"])] or list(runs["method"].unique())

fig, axes = plt.subplots(1, len(methods), figsize=(6 * len(methods), 4.8), squeeze=False)
for ax, method in zip(axes[0], methods):
    values = runs.loc[runs["method"] == method, "pooled_outer_auc"].values
    n = len(values)
    lo, hi = values.min(), values.max()

    ax.hist(values, bins=20, color="#2980b9", alpha=0.75, edgecolor="black", linewidth=0.4)

    row = perm[(perm["method"] == method) & (perm["statistic"] == "nested_pooled_auc")]
    if len(row):
        row = row.iloc[0]
        ax.axvspan(row["null_2.5pct"], row["null_97.5pct"], color="gray", alpha=0.15,
                   label="Shuffled-label null, 95% range (single split)")
    ax.axvline(0.5, color="gray", linestyle=":", linewidth=1.5, label="Chance = 0.5")
    ax.axvline(values.mean(), color="#c0392b", linestyle="--", linewidth=1.8,
               label=f"Mean = {values.mean():.3f} (SD {values.std(ddof=1):.3f})")

    ax.set_title(f"{titles.get(method, method)}\n{n} repeats, range {lo:.3f} to {hi:.3f}", fontsize=10)
    ax.set_xlabel("Pooled outer-fold AUC")
    ax.set_ylabel("Number of repeats")
    ax.set_xlim(0.15, 0.85)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

fig.suptitle("Nested-CV AUC depends heavily on how patients are assigned to folds",
             fontsize=12, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.94))

output_path = FIGURES_DIR / "repeated_nested_cv_stability.png"
fig.savefig(output_path, dpi=300, bbox_inches="tight")
print(f"Saved {output_path}")