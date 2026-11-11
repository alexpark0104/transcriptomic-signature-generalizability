from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from analysis import apparent_auc, nested_cv
from data_loader import load_discovery_data


RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

N_PERMUTATIONS = 10
RANDOM_SEED = 2026
OUTPUT_PATH = RESULTS_DIR / "permutation_test_pilot.csv"

expr, labels, _ = load_discovery_data(
    "data/rld.BMS038.20171011.csv",
    "data/bms038_clinical_data.csv",
)

X = expr.values
y = labels.values.astype(int)

print("Calculating observed results...")
observed_apparent, _ = apparent_auc(
    X,
    y,
    n_genes=10,
)

_, observed_nested, _ = nested_cv(
    X,
    y,
    n_genes=10,
    outer_splits=5,
    inner_splits=5,
    random_state=42,
)

print(f"Observed apparent AUC: {observed_apparent:.3f}")
print(f"Observed nested pooled AUC: {observed_nested:.3f}")

# Load completed rows if the file already exists.
if OUTPUT_PATH.exists():
    completed = pd.read_csv(OUTPUT_PATH)
    completed_numbers = set(
        completed["permutation"].astype(int)
    )
    rows = completed.to_dict("records")
    print(
        f"Found {len(completed_numbers)} completed permutations. "
        "Resuming."
    )
else:
    completed_numbers = set()
    rows = []

# Use one deterministic RNG stream. We advance it for every permutation,
# including permutations already completed, so future labels are unchanged.
rng = np.random.default_rng(RANDOM_SEED)

for permutation_index in range(N_PERMUTATIONS):
    permutation_number = permutation_index + 1
    shuffled_y = rng.permutation(y)

    if permutation_number in completed_numbers:
        print(
            f"Skipping completed permutation "
            f"{permutation_number}/{N_PERMUTATIONS}"
        )
        continue

    apparent_auc_value, _ = apparent_auc(
        X,
        shuffled_y,
        n_genes=10,
    )

    _, nested_auc_value, _ = nested_cv(
        X,
        shuffled_y,
        n_genes=10,
        outer_splits=5,
        inner_splits=5,
        random_state=42 + permutation_index,
    )

    row = {
        "permutation": permutation_number,
        "random_seed": RANDOM_SEED + permutation_index,
        "apparent_auc": apparent_auc_value,
        "nested_pooled_auc": nested_auc_value,
    }

    rows.append(row)
    rows = sorted(rows, key=lambda item: item["permutation"])

    pd.DataFrame(rows).to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(
        f"Permutation {permutation_number}/{N_PERMUTATIONS}: "
        f"apparent = {apparent_auc_value:.3f}, "
        f"nested pooled = {nested_auc_value:.3f}"
    )

print(f"\nSaved {OUTPUT_PATH}")
