from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


RESULTS_DIR = Path("results")
FIGURES_DIR = RESULTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

results = pd.read_csv(
    RESULTS_DIR / "repeated_nested_cv.csv"
)

fig, ax = plt.subplots(figsize=(8, 5.5))

x = results["repeat"]

ax.axhline(
    0.5,
    color="gray",
    linestyle=":",
    linewidth=1.5,
    label="Chance (AUC = 0.500)",
)

ax.plot(
    x,
    results["pooled_outer_auc"],
    marker="o",
    linewidth=2,
    markersize=7,
    color="#2980b9",
    label="Pooled outer-fold AUC",
)

mean_auc = results["pooled_outer_auc"].mean()
min_auc = results["pooled_outer_auc"].min()
max_auc = results["pooled_outer_auc"].max()

ax.axhline(
    mean_auc,
    color="#c0392b",
    linestyle="--",
    linewidth=1.8,
    label=f"Mean = {mean_auc:.3f}",
)

ax.fill_between(
    [x.min(), x.max()],
    min_auc,
    max_auc,
    color="#2980b9",
    alpha=0.10,
    label=f"Range = {min_auc:.3f}–{max_auc:.3f}",
)

ax.set_xticks(x)
ax.set_xlabel("Repeat")
ax.set_ylabel("Pooled outer-fold AUC")
ax.set_ylim(0.25, 0.70)
ax.set_title(
    "Nested-CV performance varies across fold assignments"
)
ax.legend(frameon=False, loc="best")
ax.grid(axis="y", alpha=0.25)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.text(
    0.5,
    0.01,
    "Five repeats; each repeat used a new 5-fold outer split",
    ha="center",
    fontsize=10,
)

fig.tight_layout(rect=(0, 0.05, 1, 1))

output_path = FIGURES_DIR / "repeated_nested_cv_stability.png"
fig.savefig(output_path, dpi=300, bbox_inches="tight")
print(f"Saved {output_path}")
