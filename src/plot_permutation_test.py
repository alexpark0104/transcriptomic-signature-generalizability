from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


RESULTS_DIR = Path("results")
FIGURES_DIR = RESULTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

permutations = pd.read_csv(
    RESULTS_DIR / "permutation_test_pilot.csv"
)

observed_apparent = 0.9951219512195122
observed_nested = 0.4536585365853658

fig, axes = plt.subplots(
    1,
    2,
    figsize=(12, 5),
    sharey=True,
)

panels = [
    (
        axes[0],
        permutations["apparent_auc"],
        observed_apparent,
        "Apparent AUC",
        "#c0392b",
    ),
    (
        axes[1],
        permutations["nested_pooled_auc"],
        observed_nested,
        "Nested pooled AUC",
        "#2980b9",
    ),
]

for ax, values, observed, label, color in panels:
    ax.hist(
        values,
        bins=5,
        color=color,
        alpha=0.75,
        edgecolor="black",
    )

    ax.axvline(
        observed,
        color="black",
        linestyle="--",
        linewidth=2,
        label=f"Observed = {observed:.3f}",
    )

    ax.axvline(
        0.5,
        color="gray",
        linestyle=":",
        linewidth=1.5,
        label="Chance = 0.500",
    )

    ax.set_xlabel(label)
    ax.set_ylabel("Number of permutations")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

fig.suptitle(
    "Whole-pipeline label-permutation pilot",
    fontsize=14,
    fontweight="bold",
)

fig.text(
    0.5,
    0.01,
    "10 permutations; each permutation repeated feature selection and evaluation",
    ha="center",
    fontsize=10,
)

fig.tight_layout(rect=(0, 0.05, 1, 0.93))

output_path = FIGURES_DIR / "permutation_test_pilot.png"
fig.savefig(output_path, dpi=300, bbox_inches="tight")
print(f"Saved {output_path}")
