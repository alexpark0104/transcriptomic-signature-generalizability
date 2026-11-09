# When Transcriptomic Signatures Fail to Generalize

## Research question

How much can apparent predictive performance be overstated when a transcriptomic signature is selected and evaluated in a small, high-dimensional discovery cohort?

This project examines that question using a melanoma anti-PD-1 response dataset. The analysis compares apparent in-sample performance with leakage-resistant nested cross-validation and exploratory external transportability analyses.

## Main finding

A feature-selection pipeline produced an apparent discovery-cohort AUC of approximately **0.995** when feature selection and model fitting were performed using all 51 discovery samples.

When the same feature-selection procedure was evaluated using nested 5-fold cross-validation, performance fell to:

- Mean outer-fold AUC: **0.478**
- Pooled out-of-fold AUC: **0.454**

The apparent estimate therefore exceeded the pooled nested-CV estimate by approximately **0.541 AUC points**.

This result demonstrates how strongly small-sample transcriptomic performance can be overstated when feature selection and evaluation are performed on the same patients.

## Primary analysis

The primary analysis uses the following evaluation levels:

1. **Apparent discovery performance:** feature selection and model fitting use the full discovery cohort, and predictions are evaluated on that same cohort.
2. **Nested cross-validation:** gene selection and model fitting occur within the training portion of each outer fold; performance is evaluated only on held-out patients.
3. **Pooled nested-CV performance:** all outer-fold held-out predictions are pooled before calculating the AUC.

The primary results are saved in:

```text
results/auc_summary.csv
