"""Summary figure: in-sample performance versus honest estimates, with uncertainty.

Run from the repo root:  python src/plot_results.py

Every number is read from the results files written by the analysis scripts
(nothing is typed in by hand). Primary method: inner_cv screening.

Rows (top to bottom):
  1. Apparent, discovery (in-sample)      grey band = shuffled-label null range
  2. Nested CV, one split (seed 42)       whisker = 95% bootstrap CI over patients
  3. Nested CV, mean of repeated splits   whisker = 2.5th-97.5th percentile across splits
  4. External cohort GSE78220 (n = 28)    whisker = 95% bootstrap CI over patients
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

RESULTS = Path("results")
FIGURES = RESULTS / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)
METHOD = "inner_cv"


def need(name):
    path = RESULTS / name
    if not path.exists():
        raise SystemExit(f"Missing {path}. Run the script that creates it first.")
    return pd.read_csv(path)


auc = need("auc_summary.csv").set_index("estimate")["auc"]
boot = need("nested_cv_bootstrap_summary.csv").iloc[0]
repeated = need("repeated_nested_cv_summary.csv")
null = need("permutation_test_full_summary.csv")
external = need("external_validation.csv")

rep = repeated[(repeated["method"] == METHOD) & (repeated["statistic"] == "pooled_outer_auc")].iloc[0]
ext = external[external["role"] == "primary"].iloc[0]


def null_range(statistic):
    row = null[(null["method"] == METHOD) & (null["statistic"] == statistic)].iloc[0]
    return row["null_2.5pct"], row["null_97.5pct"]


rows = [
    dict(label="Apparent AUC, discovery\n(in-sample, n = 51)", value=auc["apparent_in_sample"],
         lo=None, hi=None, band=null_range("apparent_auc"), color="#c0392b"),
    dict(label="Nested CV, one split\n(whisker: 95% bootstrap CI)", value=auc["nested_cv_pooled"],
         lo=boot["bootstrap_ci_lower"], hi=boot["bootstrap_ci_upper"],
         band=null_range("nested_pooled_auc"), color="#2980b9"),
    dict(label=f"Nested CV, mean of {int(rep['n_repeats'])} splits\n(whisker: 95% range across splits)",
         value=rep["mean"], lo=rep["p2.5"], hi=rep["p97.5"],
         band=null_range("nested_pooled_auc"), color="#2980b9"),
    dict(label=f"External cohort GSE78220 (n = {int(ext['n_external'])})\n(whisker: 95% bootstrap CI)",
         value=ext["auc"], lo=ext["bootstrap_ci_lower"], hi=ext["bootstrap_ci_upper"],
         band=None, color="#27ae60"),
]

fig, ax = plt.subplots(figsize=(9.5, 5))
ypos = list(range(len(rows)))[::-1]
for y, r in zip(ypos, rows):
    if r["band"] is not None:
        ax.fill_betweenx([y - 0.32, y + 0.32], r["band"][0], r["band"][1],
                         color="gray", alpha=0.18, linewidth=0)
    if r["lo"] is not None:
        ax.errorbar(r["value"], y, xerr=[[r["value"] - r["lo"]], [r["hi"] - r["value"]]],
                    fmt="none", ecolor="black", capsize=5, linewidth=1.5)
    ax.plot(r["value"], y, "o", color=r["color"], markersize=11, markeredgecolor="black")
    if r["hi"] is None:   # no whisker: put the label to the left of the marker
        ax.text(r["value"] - 0.025, y, f"{r['value']:.3f}", va="center", ha="right",
                fontsize=10, fontweight="bold")
    else:
        ax.text(r["hi"] + 0.015, y, f"{r['value']:.3f}", va="center", fontsize=10, fontweight="bold")

ax.axvline(0.5, color="black", linestyle="--", linewidth=1.1, label="Chance (AUC = 0.5)")
ax.fill_betweenx([], [], [], color="gray", alpha=0.18, label="Shuffled-label null, 95% range")
ax.set_yticks(ypos)
ax.set_yticklabels([r["label"] for r in rows], fontsize=9)
ax.set_xlim(0, 1.08)
ax.set_xlabel("Area under the ROC curve (AUC)")
ax.set_title("A 10-gene signature: near-perfect in-sample, indistinguishable from chance out of sample",
             fontsize=11, fontweight="bold")
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=8)
ax.grid(axis="x", alpha=0.25)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
fig.text(0.5, 0.005,
         "Primary method: gene screening by inner cross-validation. Grey bands show what shuffled "
         "outcome labels produce for the same procedure.", ha="center", fontsize=8)
fig.tight_layout(rect=(0, 0.05, 1, 1))

output = FIGURES / "primary_auc_comparison.png"
fig.savefig(output, dpi=300, bbox_inches="tight")
print(f"Saved {output}")