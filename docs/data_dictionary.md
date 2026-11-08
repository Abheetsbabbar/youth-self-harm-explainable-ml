# Data dictionary

All files are in `data/processed/`.

## `model_table_2023_24.csv` (and `model_table_2022_23.csv`)
**Grain:** one row per upper-tier local authority in England (153 rows).

| Column | Definition |
|---|---|
| `area_code`, `area_name` | ONS code and name of the county or unitary authority |
| `region` | Government Office Region (9). Used as the group in spatial cross-validation |
| `selfharm_rate` | **Target (published):** hospital admissions as a result of self-harm, ages 10–24, directly age-standardised rate per 100,000 (Fingertips 90813) |
| `selfharm_rate_lci`, `selfharm_rate_uci` | 95% confidence interval of that rate |
| `selfharm_admissions` | Number of admissions (the count behind the rate) |
| `population_10_24` | Resident population aged 10–24 (the rate's denominator) |
| `crude_rate` *(computed at load)* | `selfharm_admissions / population_10_24`. The modelled outcome, with a Poisson loss and population weights |
| 23 predictor columns | See below |

## `feature_catalogue.csv`
One row per predictor. It records the Fingertips **indicator id and full name**, the
sex and age filter, **the period actually used**, whether that period is later than
the target year (`after_target_period`), and the share of councils with missing
values.

| Feature | Indicator | Theme |
|---|---|---|
| `imd_score` | Index of Multiple Deprivation 2025 score | Poverty |
| `child_absolute_low_income_pct` | Children (<16) in absolute low-income families | Poverty |
| `free_school_meals_pct` | Pupils eligible for free school meals | Poverty |
| `children_in_care_rate` | Looked-after children per 10,000 under-18s | Care and safeguarding |
| `lac_wellbeing_concern_pct` | Looked-after children whose emotional wellbeing is a cause for concern | Care and safeguarding |
| `persistent_absence_secondary_pct`, `persistent_absence_primary_pct` | Persistent school absence | Education |
| `semh_pupils_pct` | Pupils with social, emotional and mental-health needs | Education / mental health |
| `sen_pupils_pct` | Pupils with special educational needs | Education |
| `attainment8_score` | Average Attainment 8 (GCSE) | Education |
| `school_readiness_pct` | Good level of development at the end of Reception | Early years |
| `neet_16_17_pct` | 16–17 year olds not in education, employment or training | Transitions |
| `youth_justice_first_entrants_rate` | First-time entrants to youth justice, ages 10–17 | Community |
| `under18_conception_rate` | Conceptions per 1,000 females under 18 | Community |
| `domestic_abuse_rate` | Domestic-abuse incidents and crimes per 1,000 | Family stress |
| `homeless_families_rate` | Households with dependent children owed a homelessness duty | Family stress |
| `violent_crime_rate` | Violence offences per 1,000 | Community |
| `adult_loneliness_pct` | Adults who feel lonely often or always | Community |
| `smi_prevalence_pct` | GP-recorded serious mental illness prevalence (QOF) | Adult mental health |
| `adult_depression_prevalence_pct` | GP-recorded adult depression prevalence (QOF) | Adult mental health |
| `ethnic_minority_births_pct` | Deliveries to women from ethnic minority groups (proxy for population ethnic mix) | Population |
| `ahah_index` | Access to Healthy Assets & Hazards index | Environment |
| `housing_affordability_ratio` | House price to earnings ratio | Housing |

## Model outputs
| File | Content |
|---|---|
| `cv_results.csv` | One row per (CV scheme, outer fold, model): tuned hyper-parameters, held-out region (for spatial CV), Poisson deviance, D², weighted MAE, Spearman ρ |
| `glm_irr.csv` | Quasi-Poisson GLM incidence-rate ratios per standard deviation, with 95% CIs, for the de-correlated feature subset |
| `observed_vs_expected.csv` | One row per council: observed admissions, **out-of-fold** expected admissions, O/E ratio with exact 99.8% Poisson CI, over-dispersion-adjusted z-score and flag, region, IMD quintile |
| `utla_boundaries.geojson` | Simplified council boundaries (WGS84) |
| `../../reports/results.json` | Every number quoted in the README |
