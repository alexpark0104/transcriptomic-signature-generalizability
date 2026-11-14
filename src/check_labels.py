"""Verify the discovery-cohort labels and sample accounting.

Run from the repo root:  python src/check_labels.py

Read-only diagnostic: it changes no pipeline code and no results files.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

sys.path.insert(0, str(Path(__file__).parent))

from data_loader import load_discovery_data

EXPR_PATH = "data/rld.BMS038.20171011.csv"
CLIN_PATH = "data/bms038_clinical_data.csv"

pd.set_option("display.width", 200)
clin = pd.read_csv(CLIN_PATH, index_col=0)
expr, labels, matched = load_discovery_data(EXPR_PATH, CLIN_PATH)

print("=" * 70)
print("1. What the clinical columns contain")
print("=" * 70)
for col in ["SampleType", "Cohort", "TRTGRP", "myBOR", "myBOR2", "myBOR3", "IBOR"]:
    if col in clin.columns:
        print(f"\n{col}:")
        print(clin[col].value_counts(dropna=False).to_string())
for col in ["PFS", "PFSWK", "OS", "OSWK"]:
    if col in clin.columns:
        print(f"\n{col}: dtype {clin[col].dtype}, first values {clin[col].head(6).tolist()}")

print("\n" + "=" * 70)
print("2. Labels: loader's rule versus an exact mapping")
print("=" * 70)
exact = matched["myBOR"].map({"PRCR": 1, "PD": 0, "SD": 0})
print("Values outside PRCR/PD/SD among analysed patients:", int(exact.isna().sum()))
print("Loader labels identical to the exact mapping:", bool((exact.values == labels.values).all()))
print(f"Analysed patients: {len(labels)} (responders {int(labels.sum())}, "
      f"non-responders {int((labels == 0).sum())})")

print("\n" + "=" * 70)
print("3. Sample accounting between the expression and clinical files")
print("=" * 70)


def format_sample(s):  # same rule as data_loader.load_discovery_data
    parts = s.split("_")
    return f"{parts[0]}_{parts[1].capitalize()}" if len(parts) == 2 else s


expr_samples = set(pd.read_csv(EXPR_PATH, index_col=0, nrows=1).columns.astype(str))
clin_names = clin["Sample"].astype(str).apply(format_sample)
in_both = set(clin_names) & expr_samples
print(f"Clinical rows: {len(clin)}   expression samples: {len(expr_samples)}   matched by name: {len(in_both)}")
print("Clinical rows with no expression sample:", sorted(set(clin_names) - expr_samples)[:15])
print("Expression samples with no clinical row:", sorted(expr_samples - set(clin_names))[:15])
if "SampleType" in clin.columns:
    print("\nSample types of the matched clinical rows:")
    print(clin.assign(name=clin_names)[clin_names.isin(in_both)]["SampleType"].value_counts().to_string())

print("\n" + "=" * 70)
print("4. Composition of the analysed patients")
print("=" * 70)
for col in ["Cohort", "TRTGRP", "SampleType"]:
    if col in matched.columns:
        print(f"\n{col} by response (1 = responder):")
        print(pd.crosstab(matched[col], labels.values).to_string())

print("\n" + "=" * 70)
print("5. Internal consistency: do responders have longer progression-free survival?")
print("=" * 70)
print("(Labels and PFS come from the same clinical row, so this checks the clinical")
print(" file's own consistency, not the expression-to-clinical join.)")
if "PFSWK" in matched.columns:
    t = pd.to_numeric(matched["PFSWK"], errors="coerce")
    groups = matched["myBOR"]
    print("\nMedian PFS (weeks) by best overall response:")
    print(t.groupby(groups).agg(["count", "median"]).to_string())
    r, p = t[groups == "PRCR"].dropna(), t[groups == "PD"].dropna()
    if len(r) and len(p):
        print(f"\nPFS in PRCR vs PD: Mann-Whitney p = {mannwhitneyu(r, p).pvalue:.2g}")