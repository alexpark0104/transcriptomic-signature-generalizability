"""Sensitivity of the main conclusion to pipeline choices.

Run from the repo root:
    python src/run_sensitivity.py              # 200 permutations, 20 repeats
    python src/run_sensitivity.py 1000 100     # N_PERMUTATIONS, N_REPEATS

For each pipeline in sensitivity_pipelines.CONFIGS (number of screened genes,
a mean-expression pre-filter, and a tuned ridge model on all genes) this reports
  * apparent (in-sample) AUC,
  * nested-CV AUC for the seed-42 split and across N_REPEATS different splits,
  * the whole-pipeline permutation null for both the apparent and the nested
    pooled AUC (labels shuffled, class counts fixed, EVERY step repeated,
    including any pre-filter and any tuning).
The same shuffled labels are used for every pipeline (paired), and the
shuffles are the same streams as run_permutation_test.py (seed 2026, stream i).

Before anything else the script checks that the baseline pipeline reproduces the
main analysis exactly, so the sensitivity numbers are directly comparable.
Permutations run in parallel; finished (config, permutation) rows are kept, so
the run can be stopped and resumed.
"""
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

import analysis
from data_loader import load_discovery_data
from sensitivity_pipelines import CONFIGS, DESCRIPTIONS, apparent_auc, nested_auc

N_PERMUTATIONS = int(sys.argv[1]) if len(sys.argv) > 1 else 200
N_REPEATS = int(sys.argv[2]) if len(sys.argv) > 2 else 20
RANDOM_SEED = 2026
FOLD_SEED = 42
RESULTS_DIR = Path("results")
PERMS_PATH = RESULTS_DIR / "sensitivity_permutations.csv"
SUMMARY_PATH = RESULTS_DIR / "sensitivity_summary.csv"

_DATA = {}


def _init_worker(X, y):
    _DATA["X"], _DATA["y"] = X, y


def _permutation_task(task):
    name, i = task
    shuffled = np.random.default_rng([RANDOM_SEED, i]).permutation(_DATA["y"])
    fit_predict = CONFIGS[name]
    mean_auc, pooled_auc = nested_auc(fit_predict, _DATA["X"], shuffled, 5, FOLD_SEED)
    return {"config": name, "permutation": i,
            "apparent_auc": apparent_auc(fit_predict, _DATA["X"], shuffled, FOLD_SEED),
            "nested_mean_auc": mean_auc, "nested_pooled_auc": pooled_auc}


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    expr, labels, _ = load_discovery_data("data/rld.BMS038.20171011.csv",
                                          "data/bms038_clinical_data.csv")
    X, y = expr.values.astype(float), labels.values.astype(int)
    print(f"Samples: {len(y)}, genes: {X.shape[1]}, responders: {int(y.sum())}")

    # ---- the baseline pipeline must reproduce the main analysis exactly ----
    base = CONFIGS["screen_10_baseline"]
    ref_mean, ref_pooled, _ = analysis.nested_cv(X, y, n_genes=10, outer_splits=5,
                                                 inner_splits=5, random_state=FOLD_SEED,
                                                 method="inner_cv")
    ref_apparent, _ = analysis.apparent_auc(X, y, n_genes=10, method="inner_cv")
    mean_b, pooled_b = nested_auc(base, X, y, 5, FOLD_SEED)
    app_b = apparent_auc(base, X, y, FOLD_SEED)
    for name, got, want in (("nested mean", mean_b, ref_mean), ("nested pooled", pooled_b, ref_pooled),
                            ("apparent", app_b, ref_apparent)):
        if abs(got - want) > 1e-9:
            sys.exit(f"Baseline pipeline does not reproduce analysis.py ({name}: {got} vs {want}).")
    print("Baseline pipeline reproduces analysis.py exactly.")

    # ---- observed values (seed-42 split) and repeated splits ----
    observed, repeated = {}, {}
    for name, fit_predict in CONFIGS.items():
        a = apparent_auc(fit_predict, X, y, FOLD_SEED)
        m, p = nested_auc(fit_predict, X, y, 5, FOLD_SEED)
        observed[name] = (a, m, p)
        pooled = [nested_auc(fit_predict, X, y, 5, FOLD_SEED + r)[1] for r in range(N_REPEATS)]
        repeated[name] = np.asarray(pooled)
        print(f"[{name}] apparent {a:.4f}, nested mean {m:.4f}, pooled {p:.4f}; "
              f"{N_REPEATS} repeats pooled mean {repeated[name].mean():.4f}")

    # ---- permutation null (resumable, parallel) ----
    rows = pd.read_csv(PERMS_PATH).to_dict("records") if PERMS_PATH.exists() else []
    done = {(r["config"], int(r["permutation"])) for r in rows}
    workers = max(1, (os.cpu_count() or 2) - 1)
    print(f"\nPermutation test: {N_PERMUTATIONS} permutations x {len(CONFIGS)} pipelines, "
          f"{workers} worker processes")
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker,
                             initargs=(X, y)) as pool:
        for name in CONFIGS:
            todo = [(name, i) for i in range(1, N_PERMUTATIONS + 1) if (name, i) not in done]
            if todo:
                rows.extend(pool.map(_permutation_task, todo, chunksize=2))
                pd.DataFrame(rows).to_csv(PERMS_PATH, index=False)
            print(f"  [{name}] done ({N_PERMUTATIONS} permutations)")

    # ---- summary ----
    null = pd.DataFrame(rows)
    null = null[null["permutation"] <= N_PERMUTATIONS]
    out = []
    for name in CONFIGS:
        sub = null[null["config"] == name]
        a_obs, m_obs, p_obs = observed[name]
        rep = repeated[name]
        row = {"config": name, "description": DESCRIPTIONS[name],
               "apparent_auc": a_obs, "nested_mean_auc_seed42": m_obs,
               "nested_pooled_auc_seed42": p_obs,
               "nested_pooled_auc_repeat_mean": rep.mean(), "nested_pooled_auc_repeat_sd": rep.std(ddof=1),
               "nested_pooled_auc_repeat_min": rep.min(), "nested_pooled_auc_repeat_max": rep.max(),
               "n_repeats": len(rep), "n_permutations": len(sub)}
        for stat, obs in (("apparent_auc", a_obs), ("nested_pooled_auc", p_obs)):
            v = sub[stat].values
            row[f"null_{stat}_mean"] = v.mean()
            row[f"null_{stat}_p2.5"] = np.percentile(v, 2.5)
            row[f"null_{stat}_p97.5"] = np.percentile(v, 97.5)
            row[f"p_upper_{stat}"] = (1 + int((v >= obs).sum())) / (1 + len(v))
        out.append(row)
    summary = pd.DataFrame(out)
    summary.to_csv(SUMMARY_PATH, index=False)

    pd.set_option("display.width", 250)
    cols = ["config", "apparent_auc", "null_apparent_auc_mean", "nested_pooled_auc_seed42",
            "nested_pooled_auc_repeat_mean", "nested_pooled_auc_repeat_sd",
            "null_nested_pooled_auc_p2.5", "null_nested_pooled_auc_p97.5",
            "p_upper_nested_pooled_auc"]
    print("\n" + summary[cols].round(3).to_string(index=False))
    print(f"\nSaved {PERMS_PATH} and {SUMMARY_PATH}")


if __name__ == "__main__":
    main()