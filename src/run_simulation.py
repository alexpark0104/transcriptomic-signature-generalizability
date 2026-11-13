"""Simulation: how optimistic is in-sample AUC, and what true signal can this
design detect?

Run from the repo root:
    python src/run_simulation.py           # full simulation
    python src/run_simulation.py quick     # small, fast check of the code

Design (everything is simulated; no patient data are used):
  * 22,333 genes of standard-normal noise, as in the discovery data.
  * Responders are ~19.6% of patients (10 of 51), as in the discovery cohort.
  * Signal: k informative genes, each shifted by d standard deviations in
    responders. "Oracle AUC" is the AUC of a model that KNEW which genes were
    informative, with unlimited data:  oracle AUC = Phi(d * sqrt(k) / sqrt(2)),
    so d = sqrt(2) * Phi^-1(oracle AUC) / sqrt(k).
  * Signal concentration matters. k = 1 is one strong gene (per-gene AUC equals
    the oracle AUC); k = 20 spreads the same total signal over 20 weak genes.
  * The analysis is the repository's primary pipeline (screening "inner_cv").
  * Power = share of simulated studies whose nested pooled AUC exceeds the 95th
    percentile of the no-signal results at the same sample size.
"""
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).parent))

from analysis import apparent_auc, nested_cv

QUICK = len(sys.argv) > 1 and sys.argv[1] == "quick"

N_GENES = 2000 if QUICK else 22333
POS_FRACTION = 10 / 51
N_GRID = [41, 51] if QUICK else [41, 51, 80, 120, 200]
K_GRID = [3] if QUICK else [1, 3, 20]
SIGNAL_GRID = [0.7] if QUICK else [0.6, 0.7, 0.8, 0.9]
REPS_NULL = 8 if QUICK else 200
REPS_SIGNAL = 4 if QUICK else 50
BASE_SEED = 2026
RESULTS_DIR = Path("results")
FIGURES_DIR = RESULTS_DIR / "figures"


def one_run(task):
    n, k, oracle_auc, rep = task
    rng = np.random.default_rng([BASE_SEED, n, k, int(round(oracle_auc * 100)), rep])
    n_pos = int(round(n * POS_FRACTION))
    y = np.array([1] * n_pos + [0] * (n - n_pos))
    X = rng.normal(size=(n, N_GENES))
    shift = 0.0
    if k > 0:
        shift = np.sqrt(2) * norm.ppf(oracle_auc) / np.sqrt(k)
        X[y == 1, :k] += shift

    apparent, _ = apparent_auc(X, y, n_genes=10, method="inner_cv")
    mean_auc, pooled_auc, _ = nested_cv(X, y, n_genes=10, outer_splits=5, inner_splits=5,
                                        random_state=42, method="inner_cv")
    return {"n": n, "n_responders": n_pos, "k_informative": k, "oracle_auc": oracle_auc,
            "per_gene_auc": float(norm.cdf(shift / np.sqrt(2))) if k > 0 else 0.5,
            "rep": rep, "apparent_auc": apparent,
            "nested_mean_auc": mean_auc, "nested_pooled_auc": pooled_auc}


def main():
    tasks = [(n, 0, 0.5, r) for n in N_GRID for r in range(REPS_NULL)]
    tasks += [(n, k, a, r) for n in N_GRID for k in K_GRID for a in SIGNAL_GRID
              for r in range(REPS_SIGNAL)]
    workers = max(1, (os.cpu_count() or 2) - 1)
    print(f"{len(tasks)} simulated studies, {workers} worker processes "
          f"({'quick' if QUICK else 'full'} mode, {N_GENES} genes)")

    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, row in enumerate(pool.map(one_run, tasks, chunksize=4), start=1):
            rows.append(row)
            if i % 200 == 0 or i == len(tasks):
                print(f"  {i}/{len(tasks)} done")

    RESULTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    runs = pd.DataFrame(rows)
    runs.to_csv(RESULTS_DIR / "simulation_runs.csv", index=False)

    # ---- summary with power ----
    null = runs[runs["k_informative"] == 0]
    threshold = null.groupby("n")["nested_pooled_auc"].quantile(0.95)
    summary_rows = []
    for (n, k, oracle), g in runs.groupby(["n", "k_informative", "oracle_auc"]):
        is_signal = k > 0
        summary_rows.append({
            "n": n, "n_responders": int(g["n_responders"].iloc[0]),
            "k_informative": k, "oracle_auc": oracle if is_signal else np.nan,
            "per_gene_auc": g["per_gene_auc"].iloc[0] if is_signal else np.nan,
            "n_studies": len(g),
            "apparent_auc_mean": g["apparent_auc"].mean(),
            "nested_pooled_auc_mean": g["nested_pooled_auc"].mean(),
            "nested_pooled_auc_p2.5": g["nested_pooled_auc"].quantile(0.025),
            "nested_pooled_auc_p97.5": g["nested_pooled_auc"].quantile(0.975),
            "optimism_mean": (g["apparent_auc"] - g["nested_pooled_auc"]).mean(),
            "null_95th_percentile": threshold[n],
            "power": float((g["nested_pooled_auc"] > threshold[n]).mean()) if is_signal else np.nan,
        })
    summary = pd.DataFrame(summary_rows).sort_values(["k_informative", "n", "oracle_auc"])
    summary.to_csv(RESULTS_DIR / "simulation_summary.csv", index=False)
    pd.set_option("display.width", 250)
    print("\n" + summary.round(3).to_string(index=False))

    # ---- figure ----
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    ax = axes[0]
    s0 = summary[summary["k_informative"] == 0].sort_values("n")
    ax.plot(s0["n"], s0["apparent_auc_mean"], "o-", color="#c0392b", label="Apparent (in-sample)")
    ax.plot(s0["n"], s0["nested_pooled_auc_mean"], "o-", color="#2980b9", label="Nested CV (pooled)")
    ax.fill_between(s0["n"], s0["nested_pooled_auc_p2.5"], s0["nested_pooled_auc_p97.5"],
                    color="#2980b9", alpha=0.15, label="Nested CV 95% range")
    ax.axhline(0.5, color="gray", linestyle=":")
    ax.set_xlabel("Patients (about 20% responders)")
    ax.set_ylabel("AUC")
    ax.set_title("No true signal: in-sample vs honest AUC")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    colors = {1: "#8e44ad", 3: "#16a085", 20: "#e67e22"}
    styles = {51: "-", 120: "--", 200: ":"}
    plot_n = [n for n in (51, 120, 200) if n in N_GRID] or N_GRID[-1:]
    for n in plot_n:
        for k in K_GRID:
            g = summary[(summary["n"] == n) & (summary["k_informative"] == k)].sort_values("oracle_auc")
            ax.plot(g["oracle_auc"], g["power"], styles.get(n, "-"), marker="o", markersize=4,
                    color=colors.get(k, "black"),
                    label=f"n = {n} ({int(g['n_responders'].iloc[0])} resp.), {k} informative gene{'s' if k > 1 else ''}")
    ax.axhline(0.8, color="gray", linestyle=":")
    ax.axhline(0.05, color="gray", linestyle=":")
    ax.set_xlabel("True signal strength (oracle AUC)")
    ax.set_ylabel("Power")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("What the design can detect\n(dotted lines: 80% power and the 5% false-positive rate)")
    ax.legend(frameon=False, fontsize=7, loc="upper left", ncol=1)
    for a in axes:
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "simulation.png", dpi=300, bbox_inches="tight")
    print("\nSaved results/simulation_runs.csv, simulation_summary.csv and figures/simulation.png")


if __name__ == "__main__":
    main()