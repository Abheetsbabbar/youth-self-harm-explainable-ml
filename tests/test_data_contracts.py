"""Data contracts on the committed modelling table and outputs."""

import json
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pandera.pandas as pa
import pytest
from pandera import Check, Column

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"

TABLE = pa.DataFrameSchema(
    {
        "area_code": Column(str, unique=True, checks=Check.str_matches(r"^E(06|08|09|10)\d{6}$")),
        "region": Column(str, nullable=False),
        "selfharm_rate": Column(float, Check.in_range(0, 2000), nullable=True),
        "selfharm_admissions": Column(float, Check.ge(0), nullable=True),
        "population_10_24": Column(float, Check.gt(0), nullable=True),
        "imd_score": Column(float, Check.in_range(0, 100)),
        "free_school_meals_pct": Column(float, Check.in_range(0, 100)),
    },
    checks=Check(lambda d: ((d.selfharm_rate_lci <= d.selfharm_rate) & (d.selfharm_rate <= d.selfharm_rate_uci))
                 [d.selfharm_rate.notna()].all(), name="ci_brackets_value"),
    coerce=True,
)


@pytest.fixture(scope="module")
def table():
    return pd.read_csv(PROCESSED / "model_table_2023_24.csv")


def test_table_schema(table):
    TABLE.validate(table, lazy=True)


def test_all_upper_tier_councils_present(table):
    assert len(table) == 153
    assert table.region.nunique() == 9


def test_target_suppression_only_in_tiny_areas(table):
    missing = table[table.selfharm_rate.isna()].area_name.tolist()
    assert set(missing) <= {"Rutland", "Isles of Scilly", "City of London"}


def test_feature_catalogue_documents_time_alignment():
    cat = pd.read_csv(PROCESSED / "feature_catalogue.csv")
    assert {"feature", "indicator_id", "period_used", "after_target_period", "missing_share"} <= set(cat.columns)
    # only structural, slow-moving indices may come from after the target year
    assert set(cat.loc[cat.after_target_period, "feature"]) <= {"imd_score", "ahah_index"}


def test_no_hospital_admission_predictors():
    cat = pd.read_csv(PROCESSED / "feature_catalogue.csv")
    assert not cat.indicator_name.str.contains("admission", case=False).any()


def test_boundaries_cover_table(table):
    g = gpd.read_file(PROCESSED / "utla_boundaries.geojson")
    assert set(table.area_code) <= set(g.area_code)


def test_results_beat_baseline():
    r = json.loads((ROOT / "reports" / "results.json").read_text())
    cv = r["cv"]
    assert cv["Random 5-fold x5 | LightGBM (Poisson)"]["d2"] > cv["Random 5-fold x5 | England average (baseline)"]["d2"]
