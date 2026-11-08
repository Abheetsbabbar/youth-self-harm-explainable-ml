# Data sources and licensing

All inputs are official, openly licensed and **aggregate** (one value per council).
No individual-level data is used, and none could be: the target is a count of
hospital admissions summarised by OHID with disclosure control.

| Dataset | Publisher | Used for | Licence |
|---|---|---|---|
| [Fingertips API](https://fingertips.phe.org.uk/) `all_data/csv/by_indicator_id`. Indicator **90813**: *Hospital admissions as a result of self-harm (10–24 years)* (directly standardised rate, count and population) | Office for Health Improvement and Disparities (OHID), from NHS England Hospital Episode Statistics | Target | OGL v3.0 |
| Fingertips: 23 contextual indicators (see [`data/processed/feature_catalogue.csv`](../data/processed/feature_catalogue.csv)) from the *Child and Maternal Health*, *Perinatal, Children and Young People's Mental Health*, *Suicide Prevention* and related profiles | OHID, using DfE, DWP, MoJ, ONS, Home Office, NHS England and MHCLG source data | Predictors | OGL v3.0 |
| Fingertips `parent_to_child_areas` (upper-tier LA → region) | OHID | Spatial cross-validation groups | OGL v3.0 |
| [Counties and Unitary Authorities (December 2024) Boundaries UK BUC](https://geoportal.statistics.gov.uk/) | ONS | Maps | OGL v3.0; contains OS data © Crown copyright 2026 |

**Geography:** the 153 upper-tier local authorities in England (Fingertips area type
502, post April 2023). Rutland, the Isles of Scilly and the City of London have
suppressed targets because of small numbers, which leaves 150 modelled areas.

**Hospital Episode Statistics:** the target derives from HES Admitted Patient Care.
Copyright © 2026, NHS England. Re-used with the permission of NHS England. All
rights reserved.

**Attribution:** Contains public sector information licensed under the Open
Government Licence v3.0. Source: OHID Fingertips; Office for National Statistics.

**Independence:** This is an independent portfolio project, not produced or
endorsed by OHID, NHS England or any local authority.
