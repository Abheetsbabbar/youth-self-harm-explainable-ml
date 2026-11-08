"""Build the modelling table: one row per upper-tier local authority.

Run:  python -m cypmh_selfharm.build
Outputs (committed):
    data/processed/model_table_<period>.csv   target + temporally aligned predictors
    data/processed/feature_catalogue.csv      which period each predictor came from, missingness
    data/processed/utla_boundaries.geojson    simplified council boundaries for maps
"""

from __future__ import annotations

import json
import logging

import geopandas as gpd
import pandas as pd

from . import config

log = logging.getLogger(__name__)


def load_fingertips() -> pd.DataFrame:
    d = pd.read_csv(config.RAW_FILES["indicators"], low_memory=False)
    d = d[(d["Area Type"] != "England") & d["Category Type"].isna()]
    return d.rename(columns={
        "Indicator ID": "indicator_id", "Indicator Name": "indicator_name", "Area Code": "area_code",
        "Area Name": "area_name", "Sex": "sex", "Age": "age", "Time period": "period",
        "Time period Sortable": "period_sort", "Value": "value", "Count": "count", "Denominator": "denominator",
        "Lower CI 95.0 limit": "lci95", "Upper CI 95.0 limit": "uci95",
    })


def _select(d: pd.DataFrame, indicator_id: int, sex: str, age: str | None) -> pd.DataFrame:
    x = d[(d.indicator_id == indicator_id) & (d.sex == sex)]
    if age is not None:
        x = x[x.age == age]
    elif x.age.nunique() > 1:  # default: the age group with the best coverage
        x = x[x.age == x.groupby("age").value.count().idxmax()]
    return x


def aligned_period(x: pd.DataFrame, target_sort: int) -> tuple[str, int]:
    """Latest period that does not start after the target period (avoid using the future);
    if the indicator only exists later (e.g. IMD 2025), take its earliest period."""
    periods = x.dropna(subset=["value"]).groupby("period_sort").period.first().sort_index()
    on_or_before = periods[periods.index <= target_sort]
    sort = on_or_before.index.max() if len(on_or_before) else periods.index.min()
    return periods.loc[sort], int(sort)


def build_table(d: pd.DataFrame, target_period: str, regions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    tgt = _select(d, config.TARGET_ID, "Persons", None)
    tgt = tgt[tgt.period == target_period]
    target_sort = int(tgt.period_sort.iloc[0])
    table = tgt[["area_code", "area_name", "value", "count", "denominator", "lci95", "uci95"]].rename(
        columns={"value": "selfharm_rate", "count": "selfharm_admissions", "denominator": "population_10_24",
                 "lci95": "selfharm_rate_lci", "uci95": "selfharm_rate_uci"})

    catalogue = []
    for ind_id, name, sex, age in config.FEATURES:
        x = _select(d, ind_id, sex, age)
        if x.empty:
            log.warning("no data for %s", name)
            continue
        period, sort = aligned_period(x, target_sort)
        vals = x[x.period_sort == sort][["area_code", "value"]].rename(columns={"value": name})
        table = table.merge(vals, on="area_code", how="left")
        catalogue.append({"feature": name, "indicator_id": ind_id, "indicator_name": x.indicator_name.iloc[0],
                          "sex": sex, "age": x.age.iloc[0], "period_used": period,
                          "after_target_period": sort > target_sort,
                          "missing_share": float(table[name].isna().mean())})
    catalogue = pd.DataFrame(catalogue)
    table = table.merge(regions, on="area_code", how="left")
    return table, catalogue


def load_regions() -> pd.DataFrame:
    mapping = json.loads(config.RAW_FILES["regions"].read_text())
    names = {r["Code"]: r["Name"].replace(" region (statistical)", "")
             for r in json.loads(config.RAW_FILES["region_names"].read_text())}
    rows = [{"area_code": c, "region": names[p]} for p, children in mapping.items() for c in children]
    return pd.DataFrame(rows)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config.PROCESSED.mkdir(parents=True, exist_ok=True)
    d = load_fingertips()
    regions = load_regions()
    for period in (config.TARGET_PERIOD, config.PRIOR_PERIOD):
        table, catalogue = build_table(d, period, regions)
        tag = period.replace("/", "_")
        table.to_csv(config.PROCESSED / f"model_table_{tag}.csv", index=False)
        if period == config.TARGET_PERIOD:
            catalogue.to_csv(config.PROCESSED / "feature_catalogue.csv", index=False)
        log.info("%s: %d areas, %d with target, %d predictors", period, len(table),
                 table.selfharm_rate.notna().sum(), len(catalogue))

    g = gpd.read_file(config.RAW_FILES["boundaries"]).rename(columns={"CTYUA24CD": "area_code",
                                                                     "CTYUA24NM": "boundary_name"})
    g["geometry"] = g.to_crs(27700).make_valid().simplify(200).to_crs(4326)
    g.to_file(config.PROCESSED / "utla_boundaries.geojson", driver="GeoJSON")


if __name__ == "__main__":
    main()
