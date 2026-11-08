"""End-to-end analysis: CV comparison, explainability, observed-vs-expected, audits, figures.

Run:  python -m selfharm_model.analysis
"""

from __future__ import annotations

import json
import logging

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import statsmodels.api as sm
from matplotlib.colors import BoundaryNorm, ListedColormap
from scipy import stats

from . import config, style
from . import modelling as md

log = logging.getLogger(__name__)
SOURCE = "Source: OHID Fingertips (indicator 90813 and 23 contextual indicators), 2023/24; ONS boundaries."


def load(period_tag: str):
    df = pd.read_csv(config.PROCESSED / f"model_table_{period_tag}.csv")
    cat = pd.read_csv(config.PROCESSED / "feature_catalogue.csv")
    features = cat.loc[cat.missing_share <= config.MAX_MISSING_SHARE, "feature"].tolist()
    df = df.dropna(subset=["selfharm_admissions", "population_10_24"]).reset_index(drop=True)
    df["crude_rate"] = df.selfharm_admissions / df.population_10_24
    return df, features, cat


def label(f: str, cat: pd.DataFrame) -> str:
    name = cat.set_index("feature").loc[f, "indicator_name"]
    return (name[:48] + "…") if len(name) > 49 else name


# =========================================================================== steps


def evaluate(df, features, results):
    X, y, w = df[features], df.crude_rate.to_numpy(), df.population_10_24.to_numpy()
    log.info("random CV (5x5 nested)…")
    cv_r, oof_r = md.random_cv(X, y, w)
    log.info("spatial CV (leave-one-region-out)…")
    cv_s, oof_s = md.spatial_cv(X, y, w, df.region.to_numpy())
    cv_r["scheme"], cv_s["scheme"] = "Random 5-fold x5", "Leave-one-region-out"
    cv = pd.concat([cv_r, cv_s])
    cv.to_csv(config.PROCESSED / "cv_results.csv", index=False)
    summ = cv.groupby(["scheme", "model"])[["d2", "mae_per_100k", "spearman", "poisson_deviance"]].agg(["mean", "std"])
    results["cv"] = {f"{s} | {m}": {k: round(float(summ.loc[(s, m), (k, "mean")]), 4) for k in
                                    ["d2", "mae_per_100k", "spearman", "poisson_deviance"]}
                     | {"d2_sd": round(float(summ.loc[(s, m), ("d2", "std")]), 4)}
                     for s, m in summ.index}
    fig_model_comparison(cv)
    return oof_r, cv


def explain(df, features, cat, cv, results):
    X, y, w = df[features], df.crude_rate.to_numpy(), df.population_10_24.to_numpy()
    lgbm_spec = next(s for s in md.MODELS if s.name.startswith("LightGBM"))
    params = md.tune(lgbm_spec, X, y, w, seed=config.RANDOM_SEED)
    results["final_lightgbm_params"] = params
    model = md.fit(lgbm_spec, params, X, y, w)
    explainer = shap.TreeExplainer(model)
    sv = explainer(X)  # log-rate scale (Poisson link)
    imp = pd.Series(np.abs(sv.values).mean(0), index=features).sort_values(ascending=False)
    results["shap_mean_abs_top10"] = {f: round(float(v), 4) for f, v in imp.head(10).items()}
    # direction: Spearman between feature value and its SHAP value
    results["shap_direction_top10"] = {
        f: ("higher -> more admissions" if stats.spearmanr(X[f], sv.values[:, features.index(f)],
                                                             nan_policy="omit")[0] > 0
            else "higher -> fewer admissions") for f in imp.head(10).index}
    fig_shap(sv, X, imp, cat)

    # Interpretable GLM on a de-correlated subset (quasi-Poisson with population offset)
    subset = md.correlation_clusters(X, config.CORRELATION_CLUSTER_THRESHOLD)
    Xs = X[subset].fillna(X[subset].median())
    Z = sm.add_constant((Xs - Xs.mean()) / Xs.std(ddof=0))
    glm = sm.GLM(df.selfharm_admissions, Z, family=sm.families.Poisson(),
                 offset=np.log(df.population_10_24)).fit(scale="X2")
    ci = glm.conf_int()
    irr = pd.DataFrame({"irr_per_sd": np.exp(glm.params), "ci_low": np.exp(ci[0]), "ci_high": np.exp(ci[1]),
                        "p_value": glm.pvalues}).drop(index="const").sort_values("irr_per_sd")
    irr.to_csv(config.PROCESSED / "glm_irr.csv")
    results["glm"] = {"features": subset, "dispersion_scale": float(glm.scale),
                      "irr_per_sd": irr.round(4).to_dict(orient="index"),
                      "pseudo_r2_deviance": float(1 - glm.deviance / glm.null_deviance)}
    fig_glm(irr, cat)

    # How much is "where in England" vs "local social context"? (in-sample deviance explained)
    region_d = pd.get_dummies(df.region, drop_first=True, dtype=float)
    def pseudo_r2(design):
        m = sm.GLM(df.selfharm_admissions, sm.add_constant(design), family=sm.families.Poisson(),
                   offset=np.log(df.population_10_24)).fit(scale="X2")
        return float(1 - m.deviance / m.null_deviance)
    results["deviance_decomposition_in_sample"] = {
        "region_only": pseudo_r2(region_d),
        "context_only": results["glm"]["pseudo_r2_deviance"],
        "context_plus_region": pseudo_r2(pd.concat([Z.drop(columns="const"), region_d], axis=1)),
    }
    return model, params


def observed_vs_expected(df, oof, results):
    best = "LightGBM (Poisson)"
    expected = oof[best] * df.population_10_24
    oe = md.observed_expected(df.selfharm_admissions.to_numpy(), expected.to_numpy())
    out = pd.concat([df[["area_code", "area_name", "region", "selfharm_rate", "selfharm_admissions",
                         "population_10_24", "imd_score"]], oe], axis=1)
    out["expected_admissions"] = expected
    out["imd_quintile"] = pd.qcut(out.imd_score, 5, labels=["1 least deprived", "2", "3", "4", "5 most deprived"])
    out.to_csv(config.PROCESSED / "observed_vs_expected.csv", index=False)
    results["observed_vs_expected"] = {
        "phi": float(oe.phi.iloc[0]),
        "flag_counts": out.oe_flag.value_counts().to_dict(),
        "higher_than_expected": out[out.oe_flag == "Higher than expected"].sort_values("oe_ratio", ascending=False)[
            ["area_name", "region", "oe_ratio", "selfharm_rate"]].round(2).to_dict("records"),
        "lower_than_expected": out[out.oe_flag == "Lower than expected"].sort_values("oe_ratio")[
            ["area_name", "region", "oe_ratio", "selfharm_rate"]].round(2).to_dict("records"),
    }
    return out


def audit(out, results):
    """Does the model systematically over/under-predict for any region or deprivation group?"""
    out = out.copy()
    out["log_oe"] = np.log(out.oe_ratio)
    by_region = out.groupby("region").log_oe.agg(["mean", "count"])
    by_imd = out.groupby("imd_quintile", observed=True).log_oe.agg(["mean", "count"])
    kw_region = stats.kruskal(*[g.log_oe for _, g in out.groupby("region")])
    kw_imd = stats.kruskal(*[g.log_oe for _, g in out.groupby("imd_quintile", observed=True)])
    results["residual_audit"] = {
        "mean_oe_by_region": {k: round(float(np.exp(v)), 3) for k, v in by_region["mean"].items()},
        "mean_oe_by_imd_quintile": {str(k): round(float(np.exp(v)), 3) for k, v in by_imd["mean"].items()},
        "kruskal_region_p": float(kw_region.pvalue), "kruskal_imd_p": float(kw_imd.pvalue),
    }
    fig_audit(out)


def out_of_time(features, params, results):
    """Train on 2022/23 outcomes; predict 2023/24 outcomes from 2023/24 features."""
    prior, _, _ = load(config.PRIOR_PERIOD.replace("/", "_"))
    cur, _, _ = load(config.TARGET_PERIOD.replace("/", "_"))
    spec = next(s for s in md.MODELS if s.name.startswith("LightGBM"))
    m = md.fit(spec, params, prior[features], prior.crude_rate.to_numpy(), prior.population_10_24.to_numpy())
    pred = np.clip(m.predict(cur[features]), 1e-9, None)
    met = md.weighted_metrics(cur.crude_rate.to_numpy(), pred, cur.population_10_24.to_numpy(),
                              np.average(prior.crude_rate, weights=prior.population_10_24))
    both = prior[["area_code", "crude_rate"]].merge(cur[["area_code", "crude_rate"]], on="area_code",
                                                    suffixes=("_prior", "_cur"))
    results["out_of_time"] = {k: round(float(v), 4) for k, v in met.items()} | {
        "year_on_year_rate_spearman": round(float(stats.spearmanr(both.crude_rate_prior, both.crude_rate_cur)[0]), 4)}


# =========================================================================== figures


def fig_model_comparison(cv):
    s = cv.groupby(["scheme", "model"]).d2.agg(["mean", "std"]).reset_index()
    order = ["England average (baseline)", "Poisson GLM (ridge)", "Random forest", "LightGBM (Poisson)"]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    y = np.arange(len(order))
    for i, (scheme, c) in enumerate([("Random 5-fold x5", style.BLUE), ("Leave-one-region-out", style.ORANGE)]):
        d = s[s.scheme == scheme].set_index("model").loc[order]
        ax.barh(y + (i - 0.5) * 0.36, d["mean"] * 100, height=0.34, color=c, label=scheme,
                xerr=d["std"] * 100, error_kw={"ecolor": style.INK_2, "elinewidth": 0.8, "capsize": 2})
        for yi, v, sd in zip(y + (i - 0.5) * 0.36, d["mean"] * 100, d["std"] * 100, strict=True):
            x_txt, ha = (v + sd + 2, "left") if v >= 0 else (v - sd - 2, "right")
            ax.text(x_txt, yi, f"{v:.0f}%", va="center", ha=ha, fontsize=8, color=style.INK_2)
    ax.set_yticks(y, order)
    ax.axvline(0, color=style.AXIS, lw=1)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Poisson deviance explained on held-out councils (D², %)")
    ax.set_title("Model comparison under random and spatial cross-validation")
    ax.legend(fontsize=8, loc="lower right")
    style.source_note(fig, SOURCE + " Error bars: SD across outer folds; hyper-parameters tuned in inner CV.")
    fig.savefig(config.FIGURES / "02_model_comparison.png")
    plt.close(fig)


def fig_shap(sv, X, imp, cat):
    top = imp.head(12)
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.barh([label(f, cat) for f in top.index[::-1]], top.values[::-1], color=style.BLUE, height=0.6)
    ax.set_xlabel("Mean |SHAP value| (impact on log admission rate)")
    ax.set_title("What drives the model's predictions?")
    ax.grid(axis="y", visible=False)
    style.source_note(fig, SOURCE + " SHAP TreeExplainer on the final LightGBM model.")
    fig.savefig(config.FIGURES / "03_shap_importance.png")
    plt.close(fig)

    plt.figure()
    sv_named = shap.Explanation(sv.values, sv.base_values, data=X.to_numpy(),
                                feature_names=[label(f, cat) for f in X.columns])
    shap.plots.beeswarm(sv_named, max_display=12, show=False, plot_size=(10, 6))
    fig = plt.gcf()
    fig.axes[0].set_title("Direction of effects: each dot is a council", loc="left", fontweight="bold")
    fig.savefig(config.FIGURES / "04_shap_beeswarm.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    for ax, f in zip(axes, imp.index[:3], strict=True):
        i = list(X.columns).index(f)
        ax.scatter(X[f], np.exp(sv.values[:, i]), s=18, color=style.BLUE, edgecolor=style.SURFACE, lw=0.6)
        ax.axhline(1, color=style.AXIS, lw=1)
        ax.set_xlabel(label(f, cat), fontsize=8.5)
        ax.set_title(f.replace("_", " "), fontsize=10)
    axes[0].set_ylabel("Multiplicative effect on rate (exp SHAP)")
    fig.suptitle("How the top three factors shift predicted admissions", x=0.01, ha="left", fontweight="bold")
    style.source_note(fig, SOURCE)
    fig.savefig(config.FIGURES / "05_shap_dependence.png")
    plt.close(fig)


def fig_glm(irr, cat):
    fig, ax = plt.subplots(figsize=(8.5, 0.42 * len(irr) + 1.4))
    y = np.arange(len(irr))
    sig = irr.p_value < 0.05
    colors = np.where(~sig, style.NEUTRAL, np.where(irr.irr_per_sd > 1, style.ORANGE, style.BLUE))
    ax.hlines(y, irr.ci_low, irr.ci_high, color=colors, lw=2)
    ax.scatter(irr.irr_per_sd, y, color=colors, s=36, zorder=3, edgecolor=style.SURFACE)
    ax.axvline(1, color=style.INK_2, lw=1)
    ax.set_yticks(y, [label(f, cat) for f in irr.index])
    ax.set_xscale("log")
    ax.set_xticks([0.7, 0.8, 0.9, 1, 1.1, 1.25, 1.5], ["0.7", "0.8", "0.9", "1", "1.1", "1.25", "1.5"])
    ax.set_xlabel("Incidence-rate ratio per 1 SD increase (95% CI, quasi-Poisson)")
    ax.set_title("Interpretable model: de-correlated predictors")
    ax.grid(axis="y", visible=False)
    style.source_note(fig, SOURCE + " Grey = 95% CI includes 1. Population offset; dispersion estimated (X²).")
    fig.savefig(config.FIGURES / "06_glm_rate_ratios.png")
    plt.close(fig)


def fig_maps_and_funnel(out):
    g = gpd.read_file(config.PROCESSED / "utla_boundaries.geojson").merge(out, on="area_code", how="left").to_crs(27700)
    fig, axes = plt.subplots(1, 3, figsize=(15, 6.2), gridspec_kw={"width_ratios": [1, 1, 1.15]})
    # observed rate
    ax = axes[0]
    vals = g.selfharm_rate.dropna()
    bins = np.quantile(vals, np.linspace(0, 1, 7))
    bins[0] -= 1e-6
    bins[-1] += 1e-6
    cmap = ListedColormap(style.SEQUENTIAL[1:])
    g.plot(column="selfharm_rate", ax=ax, cmap=cmap, norm=BoundaryNorm(bins, cmap.N), edgecolor=style.SURFACE,
           lw=0.25, missing_kwds={"color": style.GRID})
    ax.set_axis_off()
    ax.set_title("Observed admissions per 100,000 aged 10-24", fontsize=10.5)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in style.SEQUENTIAL[1:]]
    ax.legend(handles, [f"{bins[i]:.0f}–{bins[i + 1]:.0f}" for i in range(6)], fontsize=7, loc="upper left")
    # O/E diverging
    ax = axes[1]
    oe_bins = [0, 0.67, 0.8, 0.91, 1.1, 1.25, 1.5, 10]
    cmap = ListedColormap(style.DIVERGING)
    g.plot(column="oe_ratio", ax=ax, cmap=cmap, norm=BoundaryNorm(oe_bins, cmap.N), edgecolor=style.SURFACE,
           lw=0.25, missing_kwds={"color": style.GRID})
    ax.set_axis_off()
    ax.set_title("Observed ÷ expected (given social context)", fontsize=10.5)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in style.DIVERGING]
    labels = ["< 0.67", "0.67–0.8", "0.8–0.91", "0.91–1.1 (as expected)", "1.1–1.25", "1.25–1.5", "> 1.5"]
    ax.legend(handles, labels, fontsize=7, loc="upper left")
    # funnel
    ax = axes[2]
    e = np.linspace(out.expected_admissions.min() * 0.8, out.expected_admissions.max() * 1.05, 300)
    phi = out.phi.iloc[0]
    for z, ls in [(1.96, "--"), (3.09, "-")]:
        ax.plot(e, 1 + z * np.sqrt(phi / e), ls, color=style.MUTED, lw=1)
        ax.plot(e, np.clip(1 - z * np.sqrt(phi / e), 0, None), ls, color=style.MUTED, lw=1)
    colors = np.where(out.oe_flag == "Higher than expected", style.ORANGE,
                      np.where(out.oe_flag == "Lower than expected", style.BLUE, style.NEUTRAL))
    ax.scatter(out.expected_admissions, out.oe_ratio, c=colors, s=26, edgecolor=style.SURFACE, lw=0.8, zorder=3)
    flagged = out[out.oe_flag != "As expected"].sort_values("oe_ratio")
    for j, (_, r) in enumerate(flagged.iterrows()):
        ax.annotate(r.area_name, (r.expected_admissions, r.oe_ratio), fontsize=7, color=style.INK_2,
                    xytext=(5, -9 if j % 2 else 4), textcoords="offset points")
    ax.axhline(1, color=style.INK_2, lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("Expected admissions (log scale)")
    ax.set_ylabel("Observed ÷ expected")
    ax.set_title("Funnel plot (over-dispersion adjusted)", fontsize=10.5)
    fig.suptitle("Youth self-harm admissions, 2023/24: where are they higher than the local context predicts?",
                 x=0.01, ha="left", fontweight="bold")
    style.source_note(fig, SOURCE + " Expected = out-of-fold LightGBM prediction x population.")
    fig.savefig(config.FIGURES / "01_maps_observed_vs_expected.png")
    plt.close(fig)


def fig_audit(out):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    fig.subplots_adjust(wspace=0.45)
    for ax, col, title in [(axes[0], "region", "By region"), (axes[1], "imd_quintile", "By deprivation quintile")]:
        groups = out.groupby(col, observed=True).oe_ratio
        order = groups.median().sort_values().index if col == "region" else groups.median().index
        for i, gname in enumerate(order):
            vals = out.loc[out[col] == gname, "oe_ratio"]
            jitter = np.random.default_rng(i).uniform(-0.18, 0.18, len(vals))
            ax.scatter(vals, i + jitter, s=12, color=style.BLUE, alpha=0.6, edgecolor="none")
            ax.plot([np.exp(np.log(vals).mean())] * 2, [i - 0.3, i + 0.3], color=style.INK, lw=2)
        ax.set_yticks(range(len(order)), [str(o) for o in order])
        ax.axvline(1, color=style.CRITICAL, lw=1, ls="--")
        ax.set_xscale("log")
        ax.set_xticks([0.33, 0.5, 0.75, 1, 1.5, 2, 3], ["0.33", "0.5", "0.75", "1", "1.5", "2", "3"])
        ax.minorticks_off()
        ax.set_xlabel("Observed ÷ expected (bar = geometric mean)")
        ax.set_title(title, fontsize=11)
        ax.grid(axis="y", visible=False)
    fig.suptitle("Residual audit: is the model systematically biased for any group?", x=0.01, ha="left",
                 fontweight="bold")
    style.source_note(fig, SOURCE)
    fig.savefig(config.FIGURES / "07_residual_audit.png")
    plt.close(fig)


# =========================================================================== main


def run() -> dict:
    style.apply()
    config.FIGURES.mkdir(parents=True, exist_ok=True)
    df, features, cat = load(config.TARGET_PERIOD.replace("/", "_"))
    results: dict = {"n_areas": len(df), "n_features": len(features), "features": features,
                     "england_crude_rate_per_100k":
                         float(df.selfharm_admissions.sum() / df.population_10_24.sum() * 1e5),
                     "rate_range_per_100k": [float(df.selfharm_rate.min()), float(df.selfharm_rate.max())]}
    oof, cv = evaluate(df, features, results)
    model, params = explain(df, features, cat, cv, results)
    out = observed_vs_expected(df, oof, results)
    audit(out, results)
    out_of_time(features, params, results)
    fig_maps_and_funnel(out)
    (config.ROOT / "reports" / "results.json").write_text(json.dumps(_json_safe(results), indent=2))
    log.info("done")
    return results


def _json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float) and np.isnan(obj):
        return None
    return obj


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run()
