# Data

The raw cohort files are not redistributed in this repository, because their
sources do not grant explicit redistribution rights: the study authors' analysis
repository declares no license, and GEO states that it cannot guarantee
unrestricted permission to copy or redistribute submitter-provided files.
Download them from the sources below and place them in this folder under the
exact names shown. The SHA-256 checksums let you confirm that you have the same
files that produced the committed results.

| File (place in `data/`) | Source | SHA-256 |
|---|---|---|
| `rld.BMS038.20171011.csv` | Study authors' analysis repository, `data/` folder: https://github.com/riazn/bms038_analysis (expression data associated with GEO GSE91061) | `ac69e754f36226a24c529fa8df6ae116fb0fbf30e365ed96757f9ee92d0dcec1` |
| `bms038_clinical_data.csv` | Same repository, `data/` folder | `bd5cc22279a1fe3e55eb3d9ba664d951a06b6393a83831f98fd9f72669bfef56` (see note 1) |
| `GSE78220_series_matrix.txt.gz` | GEO accession GSE78220, series matrix | See note 2 |
| `GSE78220_PatientFPKM.xlsx` | GEO accession GSE78220, supplementary file | `ae3b044f23a0a4cd2859da36726a220856c35216a169c17f93dfc3bd20b04de6` |

1. The authors' copy of the clinical file has different line endings (same rows
   and values). The checksum above is of the file with carriage returns removed:
   `tr -d '\r' < bms038_clinical_data.csv | shasum -a 256`.
2. GEO serves this file compressed (checksum
   `b2d7829dac364e7773390255252adb5c42f07104136a1f93d2a767e9522c086f`). The
   committed results were produced from its decompressed text (checksum
   `392b48f79222427f1a16a919b59d3d26cfc69370d30850562a2fe55ef92ddfe6`, as
   recorded in `results/run_manifest.txt`). The code accepts either form, so the
   compressed file can be used as downloaded.

GEO pages: GSE91061 (https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE91061)
and GSE78220 (https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE78220).

`entrez_to_symbol.csv` is created by this project from NCBI Gene records
(`src/build_gene_mapping.py`) and is kept in the repository; the source and
retrieval date are recorded in the file.
