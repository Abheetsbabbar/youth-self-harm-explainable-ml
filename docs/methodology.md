# Methodology

## 1. Question and framing

**Unit:** the 153 upper-tier local authorities in England, of which 150 have a
published target.
**Outcome:** hospital admissions as a result of self-harm, ages 10–24, 2023/24
(Fingertips indicator 90813).

The outcome is a **count of events in a population**, so every model uses a
**Poisson loss** on the crude rate `admissions / population_10_24`, with the
population as the sample weight. This is mathematically equivalent to a Poisson
count model with a log-population offset. Large councils carry more information
and get more weight, and no model is rewarded for fitting noisy small areas. The
published age-standardised rate (DSR) is used for display. Within the narrow 10–24
age band, crude and standardised rates are almost identical.

## 2. Predictors and leakage control

The 23 predictors cover poverty, care and safeguarding, education, transitions,
family stress, community, adult mental health, population mix, environment and
housing. The full list and rationale are in [`config.py`](../src/selfharm_model/config.py)
and the [data dictionary](data_dictionary.md).

- **Temporal alignment:** each predictor uses the **latest period on or before the
  target year**. Only IMD 2025 and AHAH 2026 (structural, slow-moving) exist only
  later. They are flagged in `feature_catalogue.csv` and enforced in a test.
- **Admission-practice leakage:** other hospital-admission indicators (mental-health
  admissions under 18, alcohol under 18, substance misuse 15–24) were **excluded**.
  They share admission thresholds and coding practice with the target, so they
  would predict well for the wrong reason.
- **Missingness:** predictors missing in more than 10% of councils are dropped
  (none were). The GLM and random forest use median imputation inside the
  cross-validation pipeline. LightGBM handles missing values natively.

## 3. Models

| Model | Specification | Grid (inner CV) |
|---|---|---|
| England average | Weighted mean rate | – |
| Poisson GLM (ridge) | `StandardScaler → PoissonRegressor` | α ∈ {0.01, 0.1, 1, 10} |
| Random forest | 500 trees, squared-error splits, population weights | min_samples_leaf ∈ {3, 8}, max_features ∈ {0.33, 0.66} |
| LightGBM | `objective="poisson"`, learning rate 0.03, subsample 0.8, colsample 0.7, λ = 1 | num_leaves ∈ {3, 6}, n_estimators ∈ {200, 500}, min_child_samples ∈ {8, 15} |

Trees are deliberately shallow and grids small, because n = 150.

## 4. Validation design

1. **Nested, repeated cross-validation.** The outer loop is 5-fold × 5 repeats (25
   test folds). The inner loop is a 3-fold grid search on weighted Poisson
   deviance. Hyper-parameters are chosen without ever seeing the outer test fold.
2. **Spatial cross-validation (leave one region out, 9 folds).** Neighbouring
   councils share unmeasured traits: hospital trusts, police forces, coding
   practice, culture. Random CV lets a model learn those traits from a council's
   neighbours, which inflates performance. Holding out whole regions tests
   **transferability**.
3. **Out-of-time.** Train on 2022/23 outcomes with 2022/23-aligned predictors, then
   predict 2023/24 outcomes from 2023/24-aligned predictors.

**Metrics:**
- Mean weighted Poisson deviance.
- **D² = 1 − deviance / null deviance**, where the null is the training-fold
  weighted mean. It's the Poisson analogue of R² and can be negative on held-out
  data.
- Weighted MAE per 100,000.
- Spearman ρ.

| CV scheme | Baseline | Ridge GLM | Random forest | LightGBM |
|---|---|---|---|---|
| Random 5×5 (D²) | 0% | 29% | **37%** | 34% |
| Leave-one-region-out (D²) | 0% | **9%** | −11% | −14% |

**Interpretation:** the pattern is consistent with flexible models exploiting regional
structure that random CV leaks between folds. The regularised GLM transfers best, but
weakly. Tree-model D² varies widely across regions (SD about 0.55), and on mean absolute
error they remain slightly better than the baseline (104–105 vs 107 per 100,000).

## 5. Explainability

- **SHAP (TreeExplainer)** on the final LightGBM (tuned by 3-fold CV on all data).
  Values are on the log-rate scale, so exp(SHAP) is a multiplicative effect on the
  predicted rate.
- **Correlation clustering → quasi-Poisson GLM.** Predictors are clustered by
  1 − |Spearman ρ| (average linkage, cut at 0.3), keeping the most complete
  feature per cluster (16 kept). The GLM has a log-population offset and
  dispersion estimated by Pearson χ² (φ ≈ 23). Coefficients are reported as
  incidence-rate ratios per SD with 95% CIs. In-sample deviance explained is 51%.
  Only the AHAH index is individually significant (IRR 0.74, 95% CI 0.65–0.85).
  SEMH pupils (1.09) and adult loneliness (1.11) are borderline. With 16 correlated
  predictors and 150 councils, the signal is largely **shared** across predictors
  rather than attributable to any one of them.
- **Deviance decomposition.** Quasi-Poisson GLMs with region only, context only,
  and both explain 53%, 51% and 63% of in-sample deviance. The context model uses the 16
  de-correlated predictors, not all 23.

**Reading the top SHAP features.** `ethnic_minority_births_pct` and `ahah_index`
are both strongly linked to urban areas (Spearman ρ = 0.68 with each other; London
median AHAH 69 vs South West 3). Recorded youth self-harm admissions are known to
be lower in urban and more ethnically diverse areas. This may reflect true
differences, different help-seeking, or different hospital pathways. The model
cannot tell these apart. `under18_conception_rate` shows a negative *conditional*
effect although its raw correlation with the target is positive (+0.32). That's a
classic sign of collinearity, and a reason not to read SHAP causally.

## 6. Observed vs expected

- **Expected** admissions = out-of-fold LightGBM rate (averaged over the 5 repeats)
  × population. Using out-of-fold predictions avoids the model "explaining away" an
  area's own excess.
- **O/E** with an exact Poisson 99.8% CI:
  [χ²₀.₀₀₁(2O)/2, χ²₀.₉₉₉(2O+2)/2] / E.
- **Over-dispersion:** z = (O − E)/√E. φ = mean of squared winsorised (10/90%)
  z-scores = 14.6. Areas are flagged when z/√φ is beyond ±3.09 (99.8%). This
  matches the funnel-plot approach in Project 1.

## 7. Residual audit
Log(O/E) is compared across the 9 regions and the 5 IMD quintiles using
Kruskal–Wallis. There's **significant regional bias** (p = 0.0003), as the spatial
CV predicted. There's **no deprivation bias** (p = 0.47): the model neither
systematically over- nor under-predicts in poorer areas.

## References
- Roberts D.R. et al. (2017). Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. *Ecography* 40:913–929.
- Lundberg S.M. & Lee S.-I. (2017). A unified approach to interpreting model predictions. *NeurIPS*.
- Varma S. & Simon R. (2006). Bias in error estimation when using cross-validation for model selection. *BMC Bioinformatics* 7:91.
- Spiegelhalter D. (2005). Funnel plots for comparing institutional performance. *Statistics in Medicine* 24:1185–1202.
- OHID Fingertips. *Indicator 90813 definition and methodology.*
