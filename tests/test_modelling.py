"""Unit tests: metrics, observed/expected maths, correlation clustering and CV mechanics."""

import numpy as np
import pandas as pd
import pytest

from selfharm_model import modelling as md
from selfharm_model.build import aligned_period


def test_d2_is_zero_for_mean_prediction_and_one_for_perfect():
    y = np.array([0.001, 0.002, 0.003, 0.004])
    w = np.array([1e5, 2e5, 1e5, 3e5])
    mean = np.average(y, weights=w)
    assert md.weighted_metrics(y, np.full(4, mean), w, mean)["d2"] == pytest.approx(0, abs=1e-12)
    assert md.weighted_metrics(y, y, w, mean)["d2"] == pytest.approx(1)


def test_observed_expected_flags_extreme_areas_only():
    rng = np.random.default_rng(0)
    expected = rng.uniform(50, 500, 200)
    observed = rng.poisson(expected).astype(float)
    observed[0] = expected[0] * 3  # one clear outlier
    out = md.observed_expected(observed, expected)
    assert out.loc[0, "oe_flag"] == "Higher than expected"
    assert (out.oe_flag != "As expected").sum() <= 5
    assert (out.oe_lci_998 <= out.oe_ratio).all() and (out.oe_ratio <= out.oe_uci_998).all()


def test_correlation_clusters_drop_duplicates():
    rng = np.random.default_rng(1)
    a = rng.normal(size=200)
    X = pd.DataFrame({"a": a, "a_copy": a * 2 + 0.01 * rng.normal(size=200), "b": rng.normal(size=200)})
    keep = md.correlation_clusters(X, threshold=0.3)
    assert len(keep) == 2 and "b" in keep


def test_aligned_period_never_uses_future_when_past_exists():
    x = pd.DataFrame({"period": ["2021/22", "2022/23", "2024/25"], "period_sort": [20210000, 20220000, 20240000],
                      "value": [1.0, 2.0, 3.0]})
    assert aligned_period(x, 20230000) == ("2022/23", 20220000)
    # only later data -> earliest available
    y = pd.DataFrame({"period": ["2025"], "period_sort": [20250000], "value": [1.0]})
    assert aligned_period(y, 20230000) == ("2025", 20250000)


def test_lightgbm_learns_planted_signal():
    rng = np.random.default_rng(2)
    n = 300
    X = pd.DataFrame({"signal": rng.normal(size=n), "noise": rng.normal(size=n)})
    w = rng.uniform(5e4, 3e5, n)
    rate = 0.003 * np.exp(0.4 * X.signal)
    y = rng.poisson(rate * w) / w
    spec = next(s for s in md.MODELS if s.name.startswith("LightGBM"))
    m = md.fit(spec, {"num_leaves": 4, "n_estimators": 300, "min_child_samples": 10}, X, y, w)
    imp = dict(zip(X.columns, m.booster_.feature_importance(importance_type="gain"), strict=True))
    assert imp["signal"] > imp["noise"]
