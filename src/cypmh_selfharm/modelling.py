"""Models, nested cross-validation and spatial cross-validation.

Target: crude rate of self-harm admissions per person aged 10-24 (count / population),
modelled with a Poisson loss and the population as sample weight. That is equivalent to a
Poisson count model with a log-population offset, so every model is evaluated on the same
likelihood the data were generated under.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from scipy import stats
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_poisson_deviance
from sklearn.model_selection import GroupKFold, KFold, RepeatedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config

# --------------------------------------------------------------------------- model zoo


@dataclass
class ModelSpec:
    name: str
    build: callable  # params -> estimator
    grid: dict = field(default_factory=dict)


def _glm(alpha=1.0):
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         PoissonRegressor(alpha=alpha, max_iter=3000))


def _rf(min_samples_leaf=5, max_features=0.5):
    return make_pipeline(SimpleImputer(strategy="median"),
                         RandomForestRegressor(n_estimators=500, min_samples_leaf=min_samples_leaf,
                                               max_features=max_features, random_state=config.RANDOM_SEED,
                                               n_jobs=-1))


def _lgbm(num_leaves=4, n_estimators=300, min_child_samples=10, learning_rate=0.03):
    # Shallow, regularised trees: only ~150 areas. LightGBM handles missing values natively.
    return LGBMRegressor(objective="poisson", num_leaves=num_leaves, n_estimators=n_estimators,
                         min_child_samples=min_child_samples, learning_rate=learning_rate, subsample=0.8,
                         subsample_freq=1, colsample_bytree=0.7, reg_lambda=1.0, random_state=config.RANDOM_SEED,
                         verbose=-1, n_jobs=1)


MODELS = [
    ModelSpec("England average (baseline)", lambda: DummyRegressor(strategy="mean")),
    ModelSpec("Poisson GLM (ridge)", _glm, {"alpha": [0.01, 0.1, 1.0, 10.0]}),
    ModelSpec("Random forest", _rf, {"min_samples_leaf": [3, 8], "max_features": [0.33, 0.66]}),
    ModelSpec("LightGBM (Poisson)", _lgbm, {"num_leaves": [3, 6], "n_estimators": [200, 500],
                                            "min_child_samples": [8, 15]}),
]


# --------------------------------------------------------------------------- metrics


def weighted_metrics(y, pred, w, y_train_mean) -> dict:
    """Poisson deviance, deviance explained (D2), weighted MAE and Spearman rho."""
    dev = mean_poisson_deviance(y, pred, sample_weight=w)
    dev_null = mean_poisson_deviance(y, np.full_like(y, y_train_mean), sample_weight=w)
    return {"poisson_deviance": dev, "d2": 1 - dev / dev_null,
            "mae_per_100k": float(np.average(np.abs(y - pred), weights=w) * 1e5),
            "spearman": float(stats.spearmanr(y, pred)[0])}


# --------------------------------------------------------------------------- fitting


def fit(spec: ModelSpec, params: dict, X, y, w):
    est = spec.build(**params)
    if isinstance(est, LGBMRegressor | DummyRegressor):
        return est.fit(X, y, sample_weight=w)
    step = est.steps[-1][0]
    return est.fit(X, y, **{f"{step}__sample_weight": w})


def tune(spec: ModelSpec, X, y, w, seed: int) -> dict:
    """Inner cross-validation grid search on weighted Poisson deviance."""
    if not spec.grid:
        return {}
    keys = list(spec.grid)
    best, best_dev = None, np.inf
    inner = KFold(config.INNER_FOLDS, shuffle=True, random_state=seed)
    for combo in itertools.product(*spec.grid.values()):
        params = dict(zip(keys, combo, strict=True))
        devs = []
        for tr, va in inner.split(X):
            m = fit(spec, params, X.iloc[tr], y[tr], w[tr])
            devs.append(mean_poisson_deviance(y[va], np.clip(m.predict(X.iloc[va]), 1e-9, None), sample_weight=w[va]))
        if np.mean(devs) < best_dev:
            best, best_dev = params, np.mean(devs)
    return best


def cross_validate(X: pd.DataFrame, y: np.ndarray, w: np.ndarray, splitter, groups=None):
    """Nested CV: the outer split estimates performance, the inner split tunes hyper-parameters.

    Returns per-fold metrics and out-of-fold predictions (averaged over repeats).
    """
    rows = []
    oof = {s.name: np.zeros(len(y)) for s in MODELS}
    oof_n = np.zeros(len(y))
    for fold, (tr, te) in enumerate(splitter.split(X, y, groups)):
        oof_n[te] += 1
        for spec in MODELS:
            params = tune(spec, X.iloc[tr], y[tr], w[tr], seed=fold)
            m = fit(spec, params, X.iloc[tr], y[tr], w[tr])
            pred = np.clip(m.predict(X.iloc[te]), 1e-9, None)
            oof[spec.name][te] += pred
            rows.append({"fold": fold, "model": spec.name, "params": str(params),
                         "held_out_group": None if groups is None else groups[te][0],
                         **weighted_metrics(y[te], pred, w[te], np.average(y[tr], weights=w[tr]))})
    oof = {k: v / np.maximum(oof_n, 1) for k, v in oof.items()}
    return pd.DataFrame(rows), oof


def random_cv(X, y, w):
    return cross_validate(X, y, w, RepeatedKFold(n_splits=config.CV_FOLDS, n_repeats=config.CV_REPEATS,
                                                 random_state=config.RANDOM_SEED))


def spatial_cv(X, y, w, regions: np.ndarray):
    """Leave-one-region-out: tests whether the model transfers to a region it has never seen,
    guarding against optimism from neighbouring councils sharing unmeasured traits."""
    return cross_validate(X, y, w, GroupKFold(n_splits=len(np.unique(regions))), groups=regions)


# --------------------------------------------------------------------------- observed vs expected


def observed_expected(observed: np.ndarray, expected: np.ndarray, z: float = 3.09) -> pd.DataFrame:
    """O/E ratios with exact Poisson CIs and an over-dispersion-adjusted outlier flag."""
    o, e = np.asarray(observed, float), np.asarray(expected, float)
    lo = stats.chi2.ppf(0.001, 2 * o) / 2 / e
    hi = stats.chi2.ppf(0.999, 2 * (o + 1)) / 2 / e
    zs = (o - e) / np.sqrt(e)
    q10, q90 = np.quantile(zs, [0.1, 0.9])
    phi = max(1.0, float(np.mean(np.clip(zs, q10, q90) ** 2)))
    z_adj = zs / np.sqrt(phi)
    flag = np.where(z_adj > z, "Higher than expected", np.where(z_adj < -z, "Lower than expected", "As expected"))
    return pd.DataFrame({"oe_ratio": o / e, "oe_lci_998": lo, "oe_uci_998": hi, "z_adjusted": z_adj,
                         "oe_flag": flag, "phi": phi})


def correlation_clusters(X: pd.DataFrame, threshold: float) -> list[str]:
    """Hierarchical clustering on 1-|Spearman rho|; keep the most complete feature per cluster.
    Gives the GLM a non-collinear, interpretable subset."""
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    corr = X.corr(method="spearman").abs().fillna(0).to_numpy()
    dist = 1 - corr
    np.fill_diagonal(dist, 0)
    labels = fcluster(linkage(squareform(dist, checks=False), "average"), t=threshold, criterion="distance")
    keep = []
    for c in np.unique(labels):
        members = X.columns[labels == c]
        keep.append(X[members].notna().sum().idxmax() if len(members) > 1 else members[0])
    return sorted(keep)
