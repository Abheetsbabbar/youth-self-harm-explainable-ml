# Model card, limitations and ethics

## Model card

| | |
|---|---|
| **Purpose** | Describe *area-level* variation in youth self-harm hospital admissions. Separate the part explained by an area's social context from the part that is not, so that public-health teams can ask *why* some areas differ from expectation |
| **Unit** | Upper-tier local authority (n = 150 with published data) |
| **Outcome** | Hospital admissions as a result of self-harm, ages 10–24, 2023/24 (OHID indicator 90813) |
| **Inputs** | 23 published contextual indicators, time-aligned to the target year |
| **Models** | Poisson GLM (ridge), random forest, **LightGBM with a Poisson objective**, plus a quasi-Poisson GLM for interpretation |
| **Evaluation** | Nested 5×5 repeated cross-validation; leave-one-region-out spatial cross-validation; out-of-time check (train 2022/23 → predict 2023/24) |
| **Intended users** | Public-health analysts, commissioners and researchers exploring local variation |
| **Not intended for** | Predicting, screening or making decisions about **any individual child**. Ranking councils as "good" or "bad". Allocating funding without wider evidence |

## Limitations

**Ecological fallacy.** Every relationship here is between *areas*. "Councils with
more children in care have higher admission rates" does **not** mean children in
care are at higher individual risk. That may be true, but these data cannot show it.

**Admissions are not the same as self-harm.** Only a minority of self-harm reaches
hospital, and admission depends on local A&E and paediatric practice, liaison
psychiatry and coding. An area with a high "observed ÷ expected" ratio may have
more self-harm, more complete recording, or a lower admission threshold. Other
admission outcomes were excluded as predictors for exactly this reason: they would
encode admission practice rather than need.

**Association is not causation.** SHAP values and GLM rate ratios describe what the
fitted models use. They do not show what would change admissions. Many predictors
are proxies for each other (poverty, care, absence). SHAP shares credit between
correlated features in ways that can look arbitrary, which is why a de-correlated
GLM is also reported.

**Small n.** 150 areas and 23 predictors leaves wide uncertainty. Hyper-parameter
grids were kept small and trees shallow. Fold-to-fold variation (the error bars in
figure 02) should be read as part of the result.

**Time alignment.** Predictors use the latest period **on or before** 2023/24.
Two slow-moving structural indices (IMD 2025, AHAH 2026) exist only for later years
and are flagged in `feature_catalogue.csv`.

**Ethnicity proxy.** The population's ethnic mix enters only through
"deliveries to women from ethnic minority groups". This is a crude proxy, used
because hospital-recorded self-harm rates differ by ethnic group. It must not be
read as a risk factor for any group of children.

## Ethical considerations
- **No individual data.** Every input is a published, disclosure-controlled
  aggregate. The project cannot identify, profile or score any child.
- **Avoiding harmful framing.** Areas are described as "higher or lower than
  expected given their context". They are not called "failing", and funnel limits
  are adjusted for over-dispersion so that differences due to chance aren't flagged.
- **Fairness audit.** Residuals are checked by region and deprivation quintile, so
  the model doesn't systematically under-predict need in more deprived or
  particular parts of the country (figure 07).
- **Sensitive topic.** Language follows NHS and OHID conventions. If you are
  affected by self-harm, support is available from
  [Samaritans](https://www.samaritans.org/) (116 123),
  [Shout](https://giveusashout.org/) (text 85258) and, for young people,
  [Childline](https://www.childline.org.uk/) (0800 1111) and
  [Papyrus HOPELINE247](https://www.papyrus-uk.org/) (0800 068 4141).
