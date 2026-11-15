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
- **The conclusion does not depend on the pipeline.** It is unchanged across
  five alternative pipelines (5, 20 or 50 screened genes, a mean-expression
  pre-filter, and a tuned ridge model on all genes).
- **This design has little power to detect a real signal.** In simulation at
  n = 51, one strong informative gene (oracle AUC 0.9) is detected only 36% of
  the time; the same total signal spread over 3 or 20 genes is detected 8% and
  6% of the time (the false-positive rate is 5%).
- **A published positive control was not detected** (see Results).

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
series matrix). `src/check_labels.py` verifies the mapping and that responders
have longer progression-free survival (median 87 vs 7 weeks for responders vs
progressive disease; Mann-Whitney p = 4.9e-6). See References for the source
papers.

<!-- TODO (author): confirm the data licenses permit redistribution in data/. -->

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
8. **Sensitivity to pipeline choices** (`src/run_sensitivity.py`): see below.

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

Each cell uses 50 simulated studies, so individual power values carry a 95%
margin of error of roughly ±8 to ±14 percentage points; read them as indicating
that power is low at this sample size and falls further when the signal is
spread over many genes, not as precise estimates. The weak-signal (k = 20) power
values are not monotone in n, which we attribute to this simulation noise.

### Sensitivity to pipeline choices

The main analysis uses one configuration (top 10 genes, C = 1, 5-fold CV).
`src/run_sensitivity.py` repeats the nested-CV and whole-pipeline permutation
analyses with 5, 20 and 50 screened genes; a mean-expression pre-filter (keep
the half of genes with higher mean expression in the training fold) before
screening; and an L2-penalized logistic regression on all genes with the
penalty strength tuned by an inner 3-fold CV. Every data-dependent step,
including the pre-filter and the tuning, is redone inside each training fold and
in each permutation. The baseline pipeline in this script reproduces the main
analysis exactly (checked at run time and in `tests/test_analysis.py`).

| Pipeline | Apparent AUC | Apparent AUC, shuffled-label null mean | Nested pooled AUC, mean of 20 splits (SD) | Nested pooled AUC, shuffled-label 95% range (single split) | Permutation p (nested, seed-42 split) |
|---|---|---|---|---|---|
| Top 10 genes (main analysis) | 0.993 | 0.992 | 0.483 (0.076) | 0.248 to 0.742 | 0.66 |
| Top 5 genes | 0.971 | 0.971 | 0.481 (0.095) | 0.275 to 0.766 | 0.81 |
| Top 20 genes | 0.990 | 1.000 | 0.504 (0.057) | 0.232 to 0.749 | 0.56 |
| Top 50 genes | 1.000 | 1.000 | 0.505 (0.045) | 0.236 to 0.761 | 0.56 |
| Top 10, mean-expression pre-filter | 0.990 | 0.984 | 0.553 (0.063) | 0.266 to 0.771 | 0.17 |
| Ridge, tuned, all genes | 1.000 | 1.000 | 0.527 (0.046) | 0.305 to 0.725 | 0.50 |

In all six pipelines the in-sample AUC (0.97 to 1.00) is indistinguishable from
the in-sample AUC obtained with shuffled labels, and the seed-42 nested-CV AUC
lies inside the shuffled-label 95% range for the same statistic (permutation
p = 0.17 to 0.81). Means over repeated splits are 0.48 to 0.55. The pipeline with
the highest repeated-split mean (the mean-expression pre-filter, 0.553; seed-42
p = 0.17) is the best of six, and a p-value that small is not unusual for the
best of six pipelines when there is no signal, so it is not treated as evidence
of an effect. These results show that the main conclusion does not depend on the
number of screened genes, on a pre-filter, or on a regularized tuned model. They
do not show that no signal exists, since the power analysis indicates that a
modest signal would not be detected at this sample size. Six pipelines were
examined, so any single nominal p-value should be read with that in mind. Ridge
was chosen rather than elastic net because it is standard for p >> n and its
implementation is stable across scikit-learn versions; other regularized models
were not tried.

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
- Cannot: generalize beyond this pipeline family (univariate screening or a
  ridge model, on these two cohorts).

## Limitations

- 10 responders in discovery; estimates are very noisy.
- The 51 analysed patients comprise 25 ipilimumab-naive and 26
  ipilimumab-progressed patients (response rates 24% and 15%, Fisher exact
  p = 0.50). Cohort-specific analyses were not attempted because of the small
  number of responders per group (6 and 4).
- 22 of the 73 pre-treatment clinical records had no pre-treatment expression
  sample in the released expression file (16 had no expression data at all, 6
  only an on-treatment sample). Their response rate (5 of 22, 23%) did not differ
  from that of the analysed patients (10 of 51, 20%; Fisher exact p = 0.76).
- Expression samples are matched to clinical records by sample name; that join is
  not independently verified.
- Simulated genes are independent; real expression is correlated.
- Only a small set of pipeline variants was examined (see Sensitivity); other
  models and settings were not tried.
- The two cohorts use different platforms and units (normalised counts vs
  FPKM); within-cohort z-scoring mitigates but does not remove this.
- A few selected genes were matched to the external cohort by former symbol
  (aliases are logged in `results/external_validation.csv`).
- Bootstrap intervals on out-of-fold predictions ignore the selection step and
  are descriptive only.

## Note on the inherited candidate signature

`src/config.py` contains a 10-gene signature with undocumented provenance that
predates this analysis. Its in-sample discovery AUC is 0.98
(`src/check_inherited_signature.py`) and its external AUC is 0.41 (95% CI 0.20
to 0.64). Because its derivation cannot be shown to be independent of the
external cohort, it is reported separately and is not used for any claim above.

## History and deviations

- An earlier version of this repository reported an apparent AUC of 0.995 and
  nested-CV AUCs of 0.478 (mean) and 0.454 (pooled) from a single split. Those
  numbers came from an earlier version of the pipeline and are superseded by the
  results above. The same earlier version contained a nine-gene sensitivity
  script with hard-coded gene symbols; it was removed because it overwrote the
  current audit file and contradicted the Entrez-to-symbol mapping.
- Fixed in advance of seeing results: the external scoring rule and direction,
  and the positive-control success criterion (combined one-sided p < 0.05),
  both stated in the source code.
- The positive control did not meet its pre-specified criterion (see Results).
  The criterion was not changed afterwards.
- The simulation, the second screening method, the sensitivity analysis and the
  reframing from "failure to generalize" to "no detectable signal" were added
  after earlier results were seen. They are reported alongside the original
  analysis, not instead of it.

<!-- TODO (author): check the last bullet against your own memory of the order in which
     analyses were added and edit it if any item is wrong. -->

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
python src/run_sensitivity.py 200 20
python src/plot_results.py
python src/plot_permutation_test.py
python src/plot_repeated_cv.py
python src/make_audit.py
```

Commit code changes before running `make_audit.py`, which records the commit
hash and whether the working tree is clean. `make_audit.py` checks that the
results files agree with one another and are not older than the code; passing
those checks means the files are consistent, not that the analysis is correct.

## Tests

`tests/test_analysis.py` checks the core code on synthetic data: the fast rank
AUC against scikit-learn, nested CV on pure noise (near 0.5, while the in-sample
AUC is high), a leakage test (every sample is excluded from gene screening in the
fold where it is tested, and the test fails on a deliberately leaky pipeline),
detection of a planted signal, and agreement between the sensitivity pipelines
and the main analysis.

```
python -m pytest tests -v
```

## Repository layout

```
src/        analysis code (analysis.py is the core; one script per analysis)
tests/      unit tests on synthetic data
data/       input data and gene-symbol mapping
results/    CSV/text outputs, figures/, model_audit.txt, run_manifest.txt
```

## References

- Riaz N, et al. Tumor and microenvironment evolution during immunotherapy with
  nivolumab. Cell 2017. Discovery cohort.
- Hugo W, et al. Genomic and transcriptomic features of response to anti-PD-1
  therapy in metastatic melanoma. Cell 2016. External cohort, GEO accession
  GSE78220.
- Ayers M, et al. IFN-gamma-related mRNA profile predicts clinical response to
  PD-1 blockade. J Clin Invest 2017;127(8):2930-2940. doi:10.1172/JCI91190.

<!-- TODO (author): verify these three citations against the papers and add
     volume, pages, DOIs, and the accession or download source for the Riaz data. -->

## License

MIT (code). Data are subject to the licenses of their original sources.