from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


RESULTS_PATH = Path("results/auc_summary.csv")
FIGURES_DIR = Path("results/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

results = pd.read_csv(RESULTS_PATH)

primary_order = [
    "apparent_in_sample",
    "nested_cv_mean",
    "nested_cv_pooled",
]

display_names = {
    "apparent_in_sample": "Apparent\n(discovery)",
    "nested_cv_mean": "Nested CV\n(mean)",
    "nested_cv_pooled": "Nested CV\n(pooled)",
}

primary = results[
    results["estimate"].isin(primary_order)
].copy()

primary["order"] = primary["estimate"].map(
    {name: index for index, name in enumerate(primary_order)}
)
primary = primary.sort_values("order")
primary["label"] = primary["estimate"].map(display_names)

colors = [
    "#c0392b",
    "#2980b9",
    "#3498db",
]

fig, ax = plt.subplots(figsize=(8, 6))

bars = ax.bar(
    primary["label"],
    primary["auc"],
    color=colors,
    edgecolor="black",
    linewidth=0.8,
)

ax.axhline(
    0.5,
    color="black",
    linestyle="--",
    linewidth=1.2,
    label="Chance (AUC = 0.50)",
)

for bar, auc in zip(bars, primary["auc"]):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        auc + 0.025,
        f"{auc:.3f}",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
    )

ax.set_ylim(0, 1.08)
ax.set_ylabel("Area under the ROC curve (AUC)")
ax.set_title(
    "Apparent performance collapses under nested cross-validation"
)
ax.legend(frameon=False, loc="upper right")
ax.grid(axis="y", alpha=0.25)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.text(
    0.5,
    0.01,
    "Same discovery feature-selection pipeline; discovery cohort n = 51",
    ha="center",
    fontsize=10,
)

fig.tight_layout(rect=(0, 0.04, 1, 1))

output_path = FIGURES_DIR / "primary_auc_comparison.png"
fig.savefig(output_path, dpi=300, bbox_inches="tight")
print(f"Saved {output_path}")

plt.show()
