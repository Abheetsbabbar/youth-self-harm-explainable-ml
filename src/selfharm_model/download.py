"""Download OHID Fingertips indicators, the region lookup and council boundaries (idempotent).

Run:  python -m selfharm_model.download
"""

from __future__ import annotations

import json
import logging

import requests

from . import config

log = logging.getLogger(__name__)
TIMEOUT = 600


def _get(url: str, params: dict | None = None) -> requests.Response:
    r = requests.get(url, params=params, timeout=TIMEOUT)
    r.raise_for_status()
    return r


def download_indicators() -> None:
    dest = config.RAW_FILES["indicators"]
    if dest.exists():
        log.info("cached   %s", dest.name)
        return
    ids = [config.TARGET_ID, *[f[0] for f in config.FEATURES], *config.OTHER_ADMISSION_IDS]
    log.info("download %d Fingertips indicators", len(ids))
    r = _get(f"{config.FINGERTIPS_API}/all_data/csv/by_indicator_id",
             {"indicator_ids": ",".join(map(str, ids)), "child_area_type_id": config.UTLA_AREA_TYPE,
              "parent_area_type_id": 15})
    dest.write_bytes(r.content)


def download_regions() -> None:
    if not config.RAW_FILES["regions"].exists():
        r = _get(f"{config.FINGERTIPS_API}/parent_to_child_areas",
                 {"child_area_type_id": config.UTLA_AREA_TYPE, "parent_area_type_id": config.REGION_AREA_TYPE})
        config.RAW_FILES["regions"].write_text(r.text)
    if not config.RAW_FILES["region_names"].exists():
        r = _get(f"{config.FINGERTIPS_API}/areas/by_area_type", {"area_type_id": config.REGION_AREA_TYPE})
        config.RAW_FILES["region_names"].write_text(r.text)


def download_boundaries() -> None:
    dest = config.RAW_FILES["boundaries"]
    if dest.exists():
        log.info("cached   %s", dest.name)
        return
    r = _get(config.BOUNDARIES_URL, {"where": "CTYUA24CD LIKE 'E%'", "outFields": "CTYUA24CD,CTYUA24NM",
                                     "outSR": 4326, "f": "geojson"})
    gj = r.json()
    log.info("boundaries: %d features", len(gj["features"]))
    dest.write_text(json.dumps(gj))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config.RAW.mkdir(parents=True, exist_ok=True)
    download_indicators()
    download_regions()
    download_boundaries()


if __name__ == "__main__":
    main()
