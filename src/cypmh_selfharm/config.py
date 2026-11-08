"""Configuration: data sources, indicator catalogue, modelling settings."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "reports" / "figures"

FINGERTIPS_API = "https://fingertips.phe.org.uk/api"
UTLA_AREA_TYPE = 502          # Upper-tier local authorities (post April 2023), 153 in England
REGION_AREA_TYPE = 6          # Government Office Regions (9)
BOUNDARIES_URL = (
    "https://services1.arcgis.com/ESMARspQHYMw9BZ9/arcgis/rest/services/"
    "Counties_and_Unitary_Authorities_December_2024_Boundaries_UK_BUC/FeatureServer/0/query"
)

RAW_FILES = {
    "indicators": RAW / "fingertips_utla.csv",
    "regions": RAW / "utla_to_region.json",
    "region_names": RAW / "regions.json",
    "boundaries": RAW / "utla_boundaries.geojson",
}

# ----------------------------------------------------------------------------- target
TARGET_ID = 90813   # Hospital admissions as a result of self-harm (10-24 years), DSR per 100,000
TARGET_PERIOD = "2023/24"
PRIOR_PERIOD = "2022/23"  # used for the out-of-time stability check

# ----------------------------------------------------------------------------- predictors
# (indicator id, short name, Fingertips sex filter, age filter or None for the default)
# Chosen to describe the social context children grow up in: poverty, care, education,
# family and community stressors, adult mental health and population mix. Other HOSPITAL
# ADMISSION outcomes are deliberately excluded - they share local admission thresholds and
# coding practice with the target and would leak "how readily this area admits" into the model.
FEATURES = [
    (94240, "imd_score", "Persons", None),
    (93701, "child_absolute_low_income_pct", "Persons", None),
    (90922, "free_school_meals_pct", "Persons", None),
    (90803, "children_in_care_rate", "Persons", None),
    (92564, "persistent_absence_secondary_pct", "Persons", None),
    (92563, "persistent_absence_primary_pct", "Persons", None),
    (93203, "neet_16_17_pct", "Persons", None),
    (10401, "youth_justice_first_entrants_rate", "Persons", None),
    (91871, "semh_pupils_pct", "Persons", "School age"),
    (90898, "sen_pupils_pct", "Persons", None),
    (20401, "under18_conception_rate", "Female", None),
    (92863, "domestic_abuse_rate", "Persons", None),
    (93739, "homeless_families_rate", "Not applicable", None),
    (11202, "violent_crime_rate", "Persons", None),
    (94175, "adult_loneliness_pct", "Persons", None),
    (90581, "smi_prevalence_pct", "Persons", None),
    (848, "adult_depression_prevalence_pct", "Persons", None),
    (92973, "ethnic_minority_births_pct", "Female", None),
    (93378, "attainment8_score", "Persons", None),
    (90631, "school_readiness_pct", "Persons", None),
    (93074, "ahah_index", "Persons", None),
    (92315, "lac_wellbeing_concern_pct", "Persons", None),
    (93111, "housing_affordability_ratio", "Persons", None),
]
OTHER_ADMISSION_IDS = [90812, 92904, 90808]  # downloaded for context only - never used as predictors
MAX_MISSING_SHARE = 0.10  # drop a predictor if more than 10% of areas lack it

# ----------------------------------------------------------------------------- modelling
RANDOM_SEED = 42
CV_FOLDS = 5
CV_REPEATS = 5
INNER_FOLDS = 3
CORRELATION_CLUSTER_THRESHOLD = 0.3  # 1 - |Spearman rho| distance for the GLM's reduced feature set
