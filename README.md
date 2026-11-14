# Optimism in small-sample transcriptomic signatures: a melanoma anti-PD-1 case study

## Research question

How much can apparent predictive performance be overstated when a transcriptomic
signature is selected and evaluated in a small, high-dimensional cohort, and
what can such a cohort actually tell us?

## Summary

In a discovery cohort of 51 pre-treatment melanoma samples (10 responders,
41 non-responders, 22,333 genes), a 10-gene logistic-regression signature
reaches an in-sample AUC of about 0.99. That number is uninformative:

- **In-sample AUC is ~0.99 even when outcome labels are shuffled.** The
  whole-pipeline permutation null has mean 0.994 (95% range 0.961 to 1.000);
  the real labels give 0.993 (permutation p = 0.75).
- **Properly nested cross-validation is centred on chance.** Over 100 repeated
  5-fold splits the pooled out-of-fold AUC is 0.484 (SD 0.081, range 0.28 to
  0.75). A single split can land anywhere in roughly 0.25 to 0.75, which is also
  the range produced by shuffled labels.
- **External evaluation in an independent cohort shows no signal either**
  (GSE78220, n = 28): AUC 0.51, 95% bootstrap CI 0.28 to 0.73.
- **This design has little power to detect a real signal.** In simulation at
  n = 51, one strong informative gene (oracle AUC 0.9) is detected only 36% of
  the time; the same total signal spread over 3 or 20 genes is detected 8% and
  6% of the time (the false-positive rate is 5%).
- **A published positive control was not detected** (see below).

The correct reading is therefore not "a real signal failed to generalize." It is:
*no evidence of any signal was found, the in-sample metric is meaningless at
this sample size, and these cohorts could not have detected a modest signal
anyway.*

## Data

| Cohort | Role | Source | n (responders / non-responders) |
|---|---|---|---|
| Riaz et al. (BMS038), pre-treatment | Discovery | `data/rld.BMS038.20171011.csv`, `data/bms038_clinical_data.csv` | 51 (10 / 41) |
| Hugo et al., GSE78220 | External | `data/GSE78220_*` | 28 (15 / 13) |

Response: complete/partial response = responder; stable/progressive disease =
non-responder (discovery labels from `myBOR`, external labels from the GEO
series matrix). `src/check_labels.py` verifies the mapping.

TODO before release: cite both papers and the GEO/EGA accessions, and confirm
the data licenses permit redistribution in `data/`.

## Methods

Two gene-screening methods are implemented and always reported together; the
primary one is `inner_cv`.

- `inner_cv` (primary): each gene is scored by its mean cross-validated AUC
  inside the training data.
- `train_auc` (secondary): each gene is scored by its directional rank AUC on
  the training data.

The top 10 genes are z-scored and fitted with L2 logistic regression (C = 1).

Evaluation levels (`src/analysis.py`):

1. **Apparent:** screening and fitting on all 51 patients, evaluated on the
   same patients.
2. **Nested CV:** screening, scaling and fitting happen only inside each outer
   training fold; evaluation is on held-out patients only. Both the mean of the
   five fold AUCs and the pooled out-of-fold AUC are reported.
3. **Repeated nested CV:** step 2 with 100 different fold assignments.
4. **Whole-pipeline permutation test:** outcome labels are shuffled (class
   counts fixed) and every step above is repeated, 1000 times.
5. **External evaluation:** the discovery-fitted weights are applied to the
   external cohort, with genes log2(FPKM + 1) transformed and z-scored within
   that cohort. Direction is fixed in advance.
6. **Simulation** (`src/run_simulation.py`): 22,333 independent noise genes,
   about 20% responders, with k informative genes at a chosen oracle AUC; power
   is the share of simulated studies whose nested AUC exceeds the 95th
   percentile of the no-signal results at the same n.
7. **Positive control** (`src/positive_control.py`): the published 10-gene
   IFN-gamma signature (Ayers et al., J Clin Invest 2017), scored as the mean of
   log expression with no fitting. The success criterion (combined one-sided
   Stouffer p < 0.05 across both cohorts) was fixed before running.

## Results

Primary method (`inner_cv`); full tables in `results/model_audit.txt`.

| Estimate | AUC |
|---|---|
| Apparent, in-sample | 0.993 |
| Apparent under shuffled labels (null mean, 95% range) | 0.994 (0.961 to 1.000) |
| Nested CV, one split (seed 42): mean of folds / pooled | 0.465 / 0.437 |
| Nested CV, 100 repeated splits: pooled mean (SD) | 0.484 (0.081) |
| Nested CV under shuffled labels (single split, 95% range) | 0.249 to 0.746 |
| External cohort GSE78220, discovery-derived (95% bootstrap CI) | 0.508 (0.278 to 0.733) |

The `train_auc` method gives the same picture (apparent 0.990, repeated-CV
pooled mean 0.495).

**Positive control: criterion not met.** The IFN-gamma score gave AUC 0.65
(one-sided p = 0.076) in discovery and 0.49 (p = 0.54) externally; combined
p = 0.17 against the pre-specified threshold of 0.05. This means these
cohorts could not reliably detect even a published immune signature, which is
consistent with the low power seen in simulation. It is reported as a failure,
not as validation.

**Power (simulation, n = 51, 10 responders).**

| Informative genes | Oracle AUC 0.9: power |
|---|---|
| 1 | 0.36 |
| 3 | 0.08 |
| 20 | 0.06 |

Power at larger n (for example n = 200: 1.00, 1.00, 0.10 for k = 1, 3, 20)
shows how quickly detectability depends on how concentrated the signal is.
Cells use 50 simulated studies each, so individual power values carry Monte
Carlo error of several percentage points.

## What can and cannot be concluded

- Can: in-sample AUC from a small-sample, high-dimensional screen is not
  evidence of predictive value. Here it is indistinguishable from what pure
  noise produces.
- Can: nested-CV estimates at 10 responders are extremely noisy
  (about ±0.25 for a single split), so a single nested-CV number should not be
  reported without a null reference or repeated splits.
- Cannot: conclude that no transcriptomic signature of anti-PD-1 response
  exists, or that the signature "failed to generalize." The design has low power
  and the positive control also failed.
- Cannot: generalize beyond this pipeline family (univariate screen plus
  logistic regression, 10 genes) and these two cohorts.

## Limitations

- 10 responders in discovery; estimates are very noisy.
- Simulated genes are independent; real expression is correlated.
- One pipeline family and one setting (10 genes, C = 1, 5-fold CV); no
  sensitivity analysis over gene count or a regularized comparator yet.
- Discovery cohort composition (prior-treatment and cohort strata) has not yet
  been analysed as a source of heterogeneity. See `src/check_labels.py`.
- The two cohorts use different platforms and units (normalised counts vs
  FPKM); within-cohort z-scoring mitigates but does not remove this.
- A few selected genes were matched to the external cohort by former symbol
  (aliases are logged in `results/external_validation.csv`).
- Bootstrap intervals on out-of-fold predictions ignore the selection step and
  are descriptive only.
- 22 of the 73 pre-treatment clinical records had no pre-treatment expression sample in the released expression file (16 had no expression data at all, 6 only an on-treatment sample). Their response rate (5 of 22, 23%) did not differ from that of the analysed patients (10 of 51, 20%; Fisher exact p = 0.76).

## Note on the inherited candidate signature

`src/config.py` contains a 10-gene signature with undocumented provenance that
predates this analysis. Its in-sample discovery AUC is 0.98
(`src/check_inherited_signature.py`) and its external AUC is 0.41 (95% CI 0.20
to 0.64). Because its derivation cannot be shown to be independent of the
external cohort, it is reported separately and is not used for any claim above.

## Reproducing the results

Python and package versions used are recorded in `results/run_manifest.txt`.

```
pip install -r requirements.txt
python src/run_pipeline.py
python src/run_nested_bootstrap.py
python src/run_repeated_cv.py 100
python src/run_permutation_test.py 1000
python src/build_gene_mapping.py
python src/external_validation.py
python src/positive_control.py
python src/run_simulation.py
python src/plot_results.py
python src/plot_permutation_test.py
python src/plot_repeated_cv.py
python src/make_audit.py
```

Commit code changes before running `make_audit.py`, which records the commit
hash and whether the working tree is clean. `make_audit.py` checks that the
results files agree with one another and are not older than the code; passing
those checks means the files are consistent, not that the analysis is correct.

## Repository layout

```
src/        analysis code (analysis.py is the core; one script per analysis)
data/       input data and gene-symbol mapping
results/    CSV/text outputs, figures/, model_audit.txt, run_manifest.txt
```

## License

MIT (code). Data are subject to the licenses of their original sources.