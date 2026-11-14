"""Write results/model_audit.txt and results/run_manifest.txt from the results files.

Run from the repo root AFTER all analysis scripts:  python src/make_audit.py

Every number is read from a results file; nothing is typed by hand and nothing
is interpreted. The script also runs consistency checks across the files and
warns if a results file is older than the code that produces it.
"""
import datetime
import hashlib
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

RESULTS = Path("results")
SRC = Path(__file__).parent
TOL = 1e-9
checks = []


def check(name, ok, detail=""):
    checks.append((name, bool(ok), detail))


def read(name):
    path = RESULTS / name
    if not path.exists():
        check(f"file exists: {name}", False, "missing - run the script that creates it")
        return None
    return pd.read_csv(path)


def table(df, cols=None):
    df = df if cols is None else df[cols]
    return df.to_string(index=False, float_format=lambda v: f"{v:.4f}")


def git(*args):
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# ------------------------------------------------------------------ load
by_method = read("auc_summary_by_method.csv")
primary = read("auc_summary.csv")
boot = read("nested_cv_bootstrap_summary.csv")
rep_summary = read("repeated_nested_cv_summary.csv")
rep_runs = read("repeated_nested_cv.csv")
perm_summary = read("permutation_test_full_summary.csv")
perm_runs = read("permutation_test_full.csv")
external = read("external_validation.csv")
gene_audit = read("external_gene_audit.csv")
genes_by_method = read("selected_discovery_genes_by_method.csv")
sim_summary = read("simulation_summary.csv")
mapping_path = Path("data/entrez_to_symbol.csv")
mapping = pd.read_csv(mapping_path) if mapping_path.exists() else None

# discovery cohort facts, read live from the data
try:
    from data_loader import load_discovery_data
    expr, labels, _ = load_discovery_data("data/rld.BMS038.20171011.csv",
                                          "data/bms038_clinical_data.csv")
    ids = [str(i) for i in expr.index]
    n_resp, n_non = int(labels.sum()), int((labels == 0).sum())
    suffixes = sorted({i.split("_")[-1] for i in ids})
    discovery_text = (f"Samples: {len(ids)}\nUnique sample IDs: {len(set(ids))}\n"
                      f"Unique patient prefixes: {len({i.split('_')[0] for i in ids})}\n"
                      f"Timepoint suffixes: {suffixes}\nGenes: {expr.shape[1]}\n"
                      f"Responders: {n_resp}\nNon-responders: {n_non}")
    check("discovery: one sample per patient", len({i.split('_')[0] for i in ids}) == len(ids))
    check("discovery: all samples pre-treatment", suffixes == ["Pre"], str(suffixes))
except Exception as error:
    discovery_text = f"(could not load discovery data: {error})"
    n_resp = n_non = None
    check("discovery data loads", False, str(error))

# ------------------------------------------------------- consistency checks
if by_method is not None and perm_summary is not None:
    stat_map = {"apparent_in_sample": "apparent_auc", "nested_cv_mean": "nested_mean_auc",
                "nested_cv_pooled": "nested_pooled_auc"}
    for _, row in by_method.iterrows():
        match = perm_summary[(perm_summary["method"] == row["method"]) &
                             (perm_summary["statistic"] == stat_map[row["estimate"]])]
        if len(match):
            check(f"permutation 'observed' equals pipeline: {row['method']} {row['estimate']}",
                  abs(match["observed"].iloc[0] - row["auc"]) < TOL,
                  f"{match['observed'].iloc[0]:.6f} vs {row['auc']:.6f}")

if by_method is not None and primary is not None:
    for _, row in primary.iterrows():
        m = by_method[(by_method["method"] == row["method"]) & (by_method["estimate"] == row["estimate"])]
        check(f"auc_summary.csv matches by-method file: {row['estimate']}",
              len(m) == 1 and abs(m["auc"].iloc[0] - row["auc"]) < TOL)

if by_method is not None and boot is not None:
    pooled = by_method[(by_method["method"] == "inner_cv") & (by_method["estimate"] == "nested_cv_pooled")]["auc"].iloc[0]
    mean_ = by_method[(by_method["method"] == "inner_cv") & (by_method["estimate"] == "nested_cv_mean")]["auc"].iloc[0]
    check("bootstrap summary matches pipeline (pooled)", abs(boot["pooled_outer_auc"].iloc[0] - pooled) < TOL)
    check("bootstrap summary matches pipeline (mean)", abs(boot["mean_outer_auc"].iloc[0] - mean_) < TOL)

if by_method is not None and rep_runs is not None:
    for method in rep_runs["method"].unique():
        first = rep_runs[(rep_runs["method"] == method) & (rep_runs["random_state"] == 42)]
        base = by_method[(by_method["method"] == method) & (by_method["estimate"] == "nested_cv_pooled")]
        if len(first) == 1 and len(base) == 1:
            check(f"repeated CV, seed 42 equals single split: {method}",
                  abs(first["pooled_outer_auc"].iloc[0] - base["auc"].iloc[0]) < TOL)

if perm_summary is not None:
    check("permutation test has >= 1000 permutations per row", (perm_summary["n_permutations"] >= 1000).all(),
          str(sorted(perm_summary["n_permutations"].unique())))

if external is not None:
    check("external cohort: 28 patients (15 responders, 13 non-responders)",
          (external["n_external"] == 28).all() and (external["n_responders"] == 15).all()
          and (external["n_nonresponders"] == 13).all())
    check("external: every defined gene was scored", (external["n_genes_defined"] == external["n_genes_scored"]).all(),
          str(external[["analysis", "n_genes_defined", "n_genes_scored"]].values.tolist()))
if gene_audit is not None:
    check("external gene audit: no gene unscored", bool(gene_audit["scored"].all()))

if mapping is not None:
    check("gene mapping: every symbol present", bool((mapping["symbol"].fillna("") != "").all()))

# staleness: results older than the code that makes them
code = [SRC / n for n in ("analysis.py", "data_loader.py", "config.py", "validation.py") if (SRC / n).exists()]
if code:
    newest_code = max(p.stat().st_mtime for p in code)
    for name in ("auc_summary_by_method.csv", "permutation_test_full.csv", "repeated_nested_cv.csv",
                 "nested_cv_bootstrap_summary.csv", "external_validation.csv", "simulation_summary.csv"):
        path = RESULTS / name
        if path.exists():
            check(f"not older than core code: {name}", path.stat().st_mtime >= newest_code,
                  "results are older than analysis/data_loader/config/validation - rerun")

# ------------------------------------------------------------- model_audit
sections = ["MODEL AUDIT (generated by src/make_audit.py; every number is read from results files)",
            f"Generated (UTC): {datetime.datetime.now(datetime.timezone.utc):%Y-%m-%d %H:%M:%S}",
            "", "1. DISCOVERY COHORT", discovery_text]

if genes_by_method is not None:
    g = genes_by_method.copy()
    if mapping is not None:
        g = g.merge(mapping[["entrez_id", "symbol"]], left_on="discovery_gene_id",
                    right_on="entrez_id", how="left").drop(columns="entrez_id")
    sections += ["", "2. GENES SELECTED ON THE FULL DISCOVERY DATA", table(g)]

if by_method is not None:
    sections += ["", "3. DISCOVERY PERFORMANCE BY SCREENING METHOD (one split, seed 42)",
                 table(by_method.pivot(index="method", columns="estimate", values="auc").reset_index())]
if boot is not None:
    sections += ["", "4. BOOTSTRAP UNCERTAINTY FOR THE PRIMARY (inner_cv) NESTED ESTIMATE", table(boot)]
if rep_summary is not None:
    sections += ["", "5. REPEATED NESTED CV (distribution across fold splits)", table(rep_summary)]
if perm_summary is not None:
    sections += ["", "6. WHOLE-PIPELINE PERMUTATION TEST", table(perm_summary)]
if external is not None:
    cols = [c for c in ["analysis", "role", "auc", "bootstrap_ci_lower", "bootstrap_ci_upper", "p_one_sided",
                        "p_two_sided", "n_external", "n_responders", "n_nonresponders", "n_genes_defined",
                        "n_genes_scored", "genes_dropped_zero_variance", "aliases_used"] if c in external.columns]
    sections += ["", "7. EXTERNAL VALIDATION (GSE78220)", external[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}")]
if sim_summary is not None:
    null = sim_summary[sim_summary["k_informative"] == 0]
    cols = ["n", "n_responders", "n_studies", "apparent_auc_mean", "nested_pooled_auc_mean",
            "nested_pooled_auc_p2.5", "nested_pooled_auc_p97.5", "null_95th_percentile"]
    strong = sim_summary[(sim_summary["k_informative"] > 0) & (sim_summary["oracle_auc"] == 0.9)]
    sections += ["", "8. SIMULATION: NO TRUE SIGNAL", table(null, cols),
                 "", "   SIMULATION: POWER AT ORACLE AUC 0.9 (by number of informative genes)",
                 table(strong, ["n", "n_responders", "k_informative", "per_gene_auc", "n_studies", "power"])]

failed = [c for c in checks if not c[1]]
sections += ["", "9. CONSISTENCY CHECKS"]
sections += [f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail and not ok else "")
             for name, ok, detail in checks]
sections += ["", f"Checks passed: {len(checks) - len(failed)} of {len(checks)}"]
(RESULTS / "model_audit.txt").write_text("\n".join(sections) + "\n")

# ----------------------------------------------------------- run_manifest
commit = git("rev-parse", "HEAD")
# Only code and data count: the results folder is output, and changes there do not make the code "dirty".
dirty = git("status", "--porcelain", "--", ".", ":(exclude)results")
packages = []
for name in ("numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "openpyxl", "truststore"):
    try:
        packages.append(f"{name}=={metadata.version(name)}")
    except metadata.PackageNotFoundError:
        pass

data_files = [Path(p) for p in ("data/rld.BMS038.20171011.csv", "data/bms038_clinical_data.csv",
                                "data/GSE78220_series_matrix.txt.gz", "data/GSE78220_PatientFPKM.xlsx",
                                "data/entrez_to_symbol.csv", "data/ncbi_gene_esummary.json")]
manifest = [
    "RUN MANIFEST (generated by src/make_audit.py)",
    f"Generated (UTC): {datetime.datetime.now(datetime.timezone.utc):%Y-%m-%d %H:%M:%S}",
    "", "Commit:", commit or "(no commit found, or git not available)",
    "Working tree: " + ("clean" if dirty == "" else "HAS UNCOMMITTED CHANGES - commit before citing these results"
                        if dirty else "unknown"),
    "", "Python:", platform.python_version(), "", "Packages:", *packages,
    "", "Random seeds (as set in the scripts):",
    "  cross-validation folds: 42 (repeated CV: 42 to 141)",
    "  permutation test: base seed 2026, one stream per permutation",
    "  bootstrap resampling: 42", "  simulation: base seed 2026",
    "", "Data files (SHA-256):",
    *[f"  {p}: {sha256(p)}" if p.exists() else f"  {p}: (not present)" for p in data_files],
    "", "Scripts, in the order they are run:",
    "  python src/run_pipeline.py", "  python src/run_nested_bootstrap.py",
    "  python src/run_repeated_cv.py 100", "  python src/run_permutation_test.py 1000",
    "  python src/build_gene_mapping.py", "  python src/external_validation.py",
    "  python src/run_simulation.py", "  python src/plot_results.py",
    "  python src/plot_permutation_test.py", "  python src/make_audit.py",
]
(RESULTS / "run_manifest.txt").write_text("\n".join(manifest) + "\n")

print(f"Checks passed: {len(checks) - len(failed)} of {len(checks)}")
for name, ok, detail in failed:
    print(f"  FAIL: {name}  {detail}")
print("Saved results/model_audit.txt and results/run_manifest.txt")