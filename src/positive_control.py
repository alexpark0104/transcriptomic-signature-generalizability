"""Positive control: can this pipeline detect a known, published immune signal?

Run from the repo root:  python src/positive_control.py

Signature: the "preliminary IFN-gamma" 10-gene signature listed in the Results
text of Ayers et al., J Clin Invest 2017;127(8):2930-2940 (doi:10.1172/JCI91190):
IFNG, STAT1, CCR5, CXCL9, CXCL10, CXCL11, IDO1, PRF1, GZMA, HLA-DRA.
It was derived in a different melanoma dataset (KEYNOTE-001, pembrolizumab,
NanoString), so it is independent of both cohorts used here.

Scoring (fixed in advance): the mean of the genes' log-scale expression, as in
the paper ("signature scores were calculated by averaging of the included
genes"). Direction is fixed: a higher score predicts response. No weights, no
fitting, nothing learned from either cohort.

PRIMARY CRITERION (fixed before running): the control "works" if the Stouffer-
combined one-sided rank-test p-value across the two cohorts is below 0.05.
A failure is only moderately informative: with 10 vs 41 and 15 vs 13 patients,
a true AUC of 0.70 is detected about 86% of the time when the cohorts are
combined, and a true AUC of 0.60 only about 37% of the time.

Secondary (reported, not used for the criterion): z-scored-gene variant, and a
discovery analysis restricted to PRCR vs PD (stable disease removed) so that the
response definition mirrors the external cohort.
"""
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, norm
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))

from data_loader import load_discovery_data, load_validation_data

RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)
LOCAL_JSON = Path("data/ncbi_positive_control_esummary.json")
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
N_BOOT = 10000
SEED = 42

# Symbols come from the paper. The Entrez IDs below are only CANDIDATES: the script
# checks each against NCBI Gene and stops if any official symbol differs.
SIGNATURE = {"IFNG": 3458, "STAT1": 6772, "CCR5": 1234, "CXCL9": 4283, "CXCL10": 3627,
             "CXCL11": 6373, "IDO1": 3620, "PRF1": 5551, "GZMA": 3001, "HLA-DRA": 3122}


def esummary_url(ids):
    query = urllib.parse.urlencode({"db": "gene", "id": ",".join(map(str, ids)), "retmode": "json"})
    return f"{EUTILS}?{query}"


def fetch_summaries(ids):
    if LOCAL_JSON.exists():
        print(f"Using local file {LOCAL_JSON}")
        with open(LOCAL_JSON) as handle:
            return json.load(handle)["result"]
    try:
        import truststore
        truststore.inject_into_ssl()
    except ImportError:
        pass
    try:
        with urllib.request.urlopen(esummary_url(ids), timeout=60) as response:
            return json.load(response)["result"]
    except Exception as error:
        sys.exit(
            f"Could not reach NCBI from Python ({error}).\n\n"
            "Open this address in your web browser, save the page text as\n"
            f"    {LOCAL_JSON}\nand rerun this script (do not edit the file):\n\n{esummary_url(ids)}"
        )


def bootstrap_ci(y, score):
    rng = np.random.default_rng(SEED)
    aucs = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, len(y), size=len(y))
        if len(np.unique(y[idx])) == 2:
            aucs.append(roc_auc_score(y[idx], score[idx]))
    return np.percentile(aucs, [2.5, 97.5])


def evaluate(name, cohort, variant, score, y):
    y = np.asarray(y).astype(int)
    score = np.asarray(score, dtype=float)
    auc = roc_auc_score(y, score)
    lo, hi = bootstrap_ci(y, score)
    p = mannwhitneyu(score[y == 1], score[y == 0], alternative="greater").pvalue
    print(f"  {name}: AUC = {auc:.3f} (95% CI {lo:.3f} to {hi:.3f}), one-sided p = {p:.3f}  "
          f"[{int(y.sum())} responders vs {int((y == 0).sum())}]")
    return {"analysis": name, "cohort": cohort, "score_variant": variant,
            "n_responders": int(y.sum()), "n_nonresponders": int((y == 0).sum()),
            "auc": auc, "ci_lower": lo, "ci_upper": hi, "p_one_sided": p}


def stouffer(ps):
    z = norm.isf(np.clip(np.asarray(ps, dtype=float), 1e-12, 1 - 1e-12))
    combined = z.sum() / np.sqrt(len(z))
    return combined, float(norm.sf(combined))


def mean_score(values, standardize):
    values = np.asarray(values, dtype=float)
    if standardize:
        values = (values - values.mean(axis=0)) / values.std(axis=0, ddof=1)
    return values.mean(axis=1)


def main():
    # ---- verify the symbol <-> Entrez ID pairs against NCBI Gene ----
    summaries = fetch_summaries(list(SIGNATURE.values()))
    problems = []
    for symbol, entrez in SIGNATURE.items():
        official = summaries.get(str(entrez), {}).get("name")
        if official != symbol:
            problems.append(f"  {symbol}: candidate Entrez ID {entrez} is {official!r} at NCBI")
    if problems:
        sys.exit("Symbol/ID check FAILED - stopping.\n" + "\n".join(problems))
    print(f"Verified all {len(SIGNATURE)} symbol/Entrez ID pairs against NCBI Gene.")

    expr_d, labels_d, matched_d = load_discovery_data(
        "data/rld.BMS038.20171011.csv", "data/bms038_clinical_data.csv")
    expr_v, labels_v = load_validation_data(
        "data/GSE78220_series_matrix.txt.gz", "data/GSE78220_PatientFPKM.xlsx")
    expr_v = expr_v.loc[:, ~expr_v.columns.duplicated()]

    disc_columns = {int(c): c for c in expr_d.columns}
    ext_columns = {str(c).upper(): c for c in expr_v.columns}
    gene_rows, disc_cols, ext_cols = [], [], []
    for symbol, entrez in SIGNATURE.items():
        in_d, in_v = entrez in disc_columns, symbol.upper() in ext_columns
        gene_rows.append({"symbol": symbol, "entrez_id": entrez, "ncbi_confirmed": True,
                          "in_discovery": in_d, "in_external": in_v})
        if in_d:
            disc_cols.append(disc_columns[entrez])
        if in_v:
            ext_cols.append(ext_columns[symbol.upper()])
    genes = pd.DataFrame(gene_rows)
    genes.to_csv(RESULTS / "positive_control_genes.csv", index=False)
    missing = genes[~(genes["in_discovery"] & genes["in_external"])]
    if len(missing):
        print("\nGenes not available in both cohorts (scored where present):")
        print(missing.to_string(index=False))
    print(f"\nGenes scored: discovery {len(disc_cols)} of {len(SIGNATURE)}, external {len(ext_cols)} of {len(SIGNATURE)}")

    y_d = labels_d.values.astype(int)
    y_v = labels_v.values.astype(int)
    d_values = expr_d[disc_cols].values.astype(float)
    v_values = expr_v[ext_cols].values.astype(float)
    if np.nanmax(expr_v.values.astype(float)) > 100:  # raw FPKM -> log2(x + 1), as in the pipeline
        v_values = np.log2(v_values + 1)

    rows = []
    print("\nPRIMARY: mean of log-scale expression (the paper's scoring)")
    rows.append(evaluate("discovery (Riaz, n=51)", "discovery", "mean_log", mean_score(d_values, False), y_d))
    rows.append(evaluate("external (Hugo, n=28)", "external", "mean_log", mean_score(v_values, False), y_v))
    z_primary, p_primary = stouffer([r["p_one_sided"] for r in rows])
    print(f"  COMBINED (Stouffer, one-sided): Z = {z_primary:.2f}, p = {p_primary:.4f}  "
          f"-> {'criterion MET' if p_primary < 0.05 else 'criterion NOT met'} (threshold fixed in advance: 0.05)")
    rows.append({"analysis": "combined, primary", "cohort": "both", "score_variant": "mean_log",
                 "p_one_sided": p_primary, "stouffer_z": z_primary})

    print("\nSecondary: each gene z-scored within its cohort, then averaged")
    sec = [evaluate("discovery (Riaz, n=51)", "discovery", "mean_z", mean_score(d_values, True), y_d),
           evaluate("external (Hugo, n=28)", "external", "mean_z", mean_score(v_values, True), y_v)]
    rows += sec
    z_sec, p_sec = stouffer([r["p_one_sided"] for r in sec])
    print(f"  COMBINED (Stouffer, one-sided): Z = {z_sec:.2f}, p = {p_sec:.4f}")
    rows.append({"analysis": "combined, z-scored", "cohort": "both", "score_variant": "mean_z",
                 "p_one_sided": p_sec, "stouffer_z": z_sec})

    print("\nSecondary: discovery restricted to PRCR vs PD (stable disease removed)")
    keep = matched_d["myBOR"].isin(["PRCR", "PD"]).values
    sub = evaluate("discovery, PRCR vs PD", "discovery", "mean_log", mean_score(d_values, False)[keep], y_d[keep])
    rows.append(sub)
    z_sub, p_sub = stouffer([sub["p_one_sided"], rows[1]["p_one_sided"]])
    print(f"  COMBINED with external (Stouffer, one-sided): Z = {z_sub:.2f}, p = {p_sub:.4f}")
    rows.append({"analysis": "combined, PRCR vs PD", "cohort": "both", "score_variant": "mean_log",
                 "p_one_sided": p_sub, "stouffer_z": z_sub})

    pd.DataFrame(rows).to_csv(RESULTS / "positive_control.csv", index=False)
    print("\nSaved results/positive_control.csv and results/positive_control_genes.csv")


if __name__ == "__main__":
    main()