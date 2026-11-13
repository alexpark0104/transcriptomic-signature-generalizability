"""Build data/entrez_to_symbol.csv from NCBI Gene (requires internet access).

Run from the repo root:  python src/build_gene_mapping.py

Genes included:
  * the 10 genes selected on the full discovery data by each screening method
  * the 10 genes of the inherited signature in config.py
For each gene it records the official symbol, description and aliases, whether
that symbol (or an alias) is present in the external GSE78220 workbook, and
where the gene is used. It also reports any disagreement with config.py.
"""
import datetime
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from analysis import METHODS, select_top_genes
from config import ENTREZ_TO_SYMBOL, SIGNATURE_GENES
from data_loader import load_discovery_data, load_validation_data

OUTPUT_PATH = Path("data/entrez_to_symbol.csv")
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"


LOCAL_JSON = Path("data/ncbi_gene_esummary.json")


def esummary_url(entrez_ids):
    query = urllib.parse.urlencode({
        "db": "gene",
        "id": ",".join(str(i) for i in entrez_ids),
        "retmode": "json",
    })
    return f"{EUTILS}?{query}"


def fetch_gene_summaries(entrez_ids):
    """One NCBI Gene esummary request. Returns the 'result' dictionary.

    If data/ncbi_gene_esummary.json exists (saved from a web browser), it is
    used instead of contacting NCBI from Python.
    """
    if LOCAL_JSON.exists():
        print(f"Using local file {LOCAL_JSON}")
        with open(LOCAL_JSON) as handle:
            return json.load(handle)["result"]

    try:  # use the operating system's certificate store when available
        import truststore
        truststore.inject_into_ssl()
    except ImportError:
        pass

    with urllib.request.urlopen(esummary_url(entrez_ids), timeout=60) as response:
        return json.load(response)["result"]


def main():
    expr, labels, _ = load_discovery_data(
        "data/rld.BMS038.20171011.csv", "data/bms038_clinical_data.csv"
    )
    X, y = expr.values.astype(float), labels.values.astype(int)

    roles = {}
    for method in METHODS:
        top, _ = select_top_genes(X, y, n_top=10, method=method)
        for gene_id in expr.columns[top]:
            roles.setdefault(int(gene_id), set()).add(f"discovery_{method}")
    for gene_id in SIGNATURE_GENES:
        roles.setdefault(int(gene_id), set()).add("inherited_config")

    ids = sorted(roles)
    print(f"Looking up {len(ids)} genes at NCBI Gene ...")
    try:
        summaries = fetch_gene_summaries(ids)
    except Exception as error:
        sys.exit(
            f"Could not reach NCBI from Python ({error}).\n\n"
            "Fix 1: run  pip install truststore  and then rerun this script.\n"
            "Fix 2: open the address below in your web browser, save the page text as\n"
            f"       {LOCAL_JSON}\n"
            "       and rerun this script. (Do not edit the file.)\n\n"
            f"{esummary_url(ids)}"
        )

    try:
        external_expr, _ = load_validation_data(
            "data/GSE78220_series_matrix.txt.gz", "data/GSE78220_PatientFPKM.xlsx"
        )
        external_columns = {str(c).upper(): c for c in external_expr.columns}
    except Exception as error:
        print(f"(External workbook not checked: {error})")
        external_columns = {}

    rows = []
    for gene_id in ids:
        info = summaries.get(str(gene_id), {})
        symbol = info.get("name")
        if not symbol:
            print(f"  WARNING: no NCBI record returned for Entrez ID {gene_id}")
        aliases = [a.strip() for a in str(info.get("otheraliases", "")).split(",") if a.strip()]

        in_external = bool(symbol) and symbol.upper() in external_columns
        alias_hit = next((a for a in aliases if a.upper() in external_columns), "")
        rows.append({
            "entrez_id": gene_id,
            "symbol": symbol or "",
            "description": info.get("description", ""),
            "aliases": ";".join(aliases),
            "symbol_in_external": in_external,
            "alias_found_in_external": "" if in_external else alias_hit,
            "used_in": ";".join(sorted(roles[gene_id])),
            "source": "NCBI Gene esummary",
            "retrieved": datetime.date.today().isoformat(),
        })

    table = pd.DataFrame(rows)
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    table.to_csv(OUTPUT_PATH, index=False)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_colwidth", 38)
    print("\n" + table[["entrez_id", "symbol", "symbol_in_external",
                        "alias_found_in_external", "used_in"]].to_string(index=False))

    print("\nChecks against config.py (inherited signature):")
    config_by_id = {int(k): v for k, v in ENTREZ_TO_SYMBOL.items()}
    for gene_id in SIGNATURE_GENES:
        ncbi_symbol = table.loc[table["entrez_id"] == int(gene_id), "symbol"].iloc[0]
        config_symbol = config_by_id.get(int(gene_id))
        status = "OK" if config_symbol == ncbi_symbol else "MISMATCH"
        print(f"  {gene_id}: config = {config_symbol!r}, NCBI = {ncbi_symbol!r}  [{status}]")

    print(f"\nSaved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()