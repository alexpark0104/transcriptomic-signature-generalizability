# A Proof‑of‑Concept Transcriptomic Analysis of Anti‑PD‑1 Response in Melanoma

 

## Overview
This repository contains the complete analysis pipeline for the study:
> *"A 10‑gene transcriptomic signature for anti‑PD‑1 response in melanoma shows limited cross‑validation and external generalizability."*

The analysis demonstrates the risk of overfitting in small‑sample, high‑dimensional biomarker discovery.

## Key Findings
- **Discovery cohort (GSE91061, n=51):** LOOCV AUC = 0.86 (overly optimistic).
- **Internal validation (nested 5‑fold CV):** Mean AUC ≈ 0.40 (unbiased estimate).
- **External validation (GSE78220, n=28):** AUC = 0.38 (below random).
- **Conclusion:** The signature does not generalise; the results serve as a cautionary tale.

## Datasets Used
1. **Discovery:** GSE91061 (Riaz et al., 2017) – `rld.BMS038.20171011.csv` and `bms038_clinical_data.csv`
2. **Validation:** GSE78220 (Hugo et al., 2016) – `GSE78220_series_matrix.txt.gz` and `GSE78220_PatientFPKM.xlsx`

## How to Reproduce
```bash
pip install -r requirements.txt
python src/run_pipeline.py
