"""SHAP explanations from LightGBM's built-in exact TreeSHAP (`pred_contrib=True`).

Values are in the raw model's log-odds: a feature's SHAP value is how far it pushed this
customer's score up (riskier) or down from the average customer. Calibration is a monotone
map applied afterwards, so it changes the size of a probability, never which way a driver
pushes.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from churn.eval.metrics import BLUE, INK, MUTED, ORANGE, figure, style

if TYPE_CHECKING:
    from churn.config import Settings
    from churn.models.model import ChurnModel


def explain(model: ChurnModel, X: pd.DataFrame) -> pd.DataFrame:
    """Per-customer SHAP values, one column per feature, plus `base` (the average score)."""
    contrib = model.booster.predict(X[model.features], pred_contrib=True)
    return pd.DataFrame(contrib, columns=[*model.features, "base"], index=X.index)


def driver_sentence(values: pd.Series, shap: np.ndarray, churned: np.ndarray) -> dict:
    """Turn one feature's SHAP pattern into a rule with real churn rates on both sides.

    The rule is the split of the feature that best matches where SHAP is positive
    (pushes risk up); the churn rates are observed outcomes, not model output.
    """
    risky = shap > 0
    if isinstance(values.dtype, pd.CategoricalDtype):
        mean_shap = pd.Series(shap).groupby(values.to_numpy(), observed=True).mean()
        risky_levels = sorted(mean_shap[mean_shap > 0].index.tolist())
        safe_levels = sorted(mean_shap[mean_shap <= 0].index.tolist())
        group = values.isin(risky_levels).to_numpy()
        # name whichever list is shorter, so the sentence stays readable
        if len(risky_levels) <= len(safe_levels):
            rule = f"{values.name} in {risky_levels}"
        else:
            rule = f"{values.name} not in {safe_levels}"
    else:
        v = values.to_numpy(dtype=float)
        known = ~np.isnan(v)
        best = (-1.0, 0.0, True)
        for t in np.unique(np.nanquantile(v, np.linspace(0.01, 0.99, 99))):
            for above in (True, False):
                g = (v > t) if above else (v <= t)
                score = (g[known] == risky[known]).mean()
                if score > best[0]:
                    best = (score, t, above)
        _, t, above = best
        group = known & ((v > t) if above else (v <= t))
        rule = f"{values.name} {'>' if above else '<='} {t:g}"
    r_in, r_out = churned[group].mean(), churned[~group].mean()
    sentence = (
        f"Customers with {rule} ({group.mean():.0%} of customers) churn at {r_in:.1%} "
        f"vs {r_out:.1%} for everyone else: {r_in / r_out:.1f}x the risk."
    )
    if r_in <= r_out:  # SHAP is conditional on the other features; the raw split disagrees
        sentence += (
            " The model only treats this as risky in combination with other features;"
            " on its own the group is not riskier."
        )
    return {
        "feature": values.name,
        "rule": rule,
        "share_of_customers": float(group.mean()),
        "churn_rate_in_group": float(r_in),
        "churn_rate_rest": float(r_out),
        "risk_ratio": float(r_in / r_out) if r_out else float("inf"),
        "sentence": sentence,
    }


def report(settings: Settings, top: int = 15) -> list[dict]:
    """Global importance + top-3 dependence charts and driver sentences on the test cohort."""
    from churn.config import PROJECT_ROOT
    from churn.features.build import TARGET, load_test_cohort

    test, model = load_test_cohort(settings)
    shap = explain(model, test)
    # self-check: SHAP values plus the base must reproduce the model's raw log-odds
    raw = model.predict_raw(test)
    assert np.allclose(1 / (1 + np.exp(-shap.sum(axis=1))), raw, atol=1e-6)

    importance = shap.drop(columns="base").abs().mean().sort_values(ascending=False)
    churned = test[TARGET].to_numpy()
    drivers = [driver_sentence(test[f], shap[f].to_numpy(), churned) for f in importance.index[:5]]
    for d, f in zip(drivers, importance.index, strict=False):
        d["mean_abs_shap"] = float(importance[f])

    figures = PROJECT_ROOT / "reports" / "figures"
    _importance_plot(importance.head(top), figures / "shap_importance.png")
    _dependence_plot(test, shap, list(importance.index[:3]), figures / "shap_dependence.png")
    (settings.model_dir / "drivers.json").write_text(json.dumps(drivers, indent=2))
    return drivers


def _importance_plot(importance: pd.Series, path) -> None:
    fig = figure(figsize=(7, 0.32 * len(importance) + 1))
    ax = fig.subplots()
    style(ax)
    ax.barh(range(len(importance)), importance.to_numpy(), color=BLUE, height=0.6)
    ax.set_yticks(range(len(importance)), importance.index)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Mean |SHAP value| (log-odds), test cohort")
    ax.set_title("What drives the churn score", loc="left", fontsize=11)
    fig.savefig(path, bbox_inches="tight", dpi=110)


def _dependence_plot(X: pd.DataFrame, shap: pd.DataFrame, features: list[str], path) -> None:
    sample = X.sample(min(len(X), 5000), random_state=0).index
    fig = figure(figsize=(4.2 * len(features), 3.6))
    for ax, f in zip(fig.subplots(1, len(features)), features, strict=True):
        style(ax)
        x, s = X.loc[sample, f], shap.loc[sample, f]
        if isinstance(x.dtype, pd.CategoricalDtype):
            mean = s.groupby(x.to_numpy(), observed=True).mean().sort_values()
            ax.bar(range(len(mean)), mean.to_numpy(), color=np.where(mean > 0, ORANGE, BLUE))
            ax.set_xticks(range(len(mean)), mean.index, rotation=90, fontsize=7)
            ax.set_ylabel("Mean SHAP value")
        else:
            ax.scatter(x, s, s=6, alpha=0.35, color=np.where(s > 0, ORANGE, BLUE), lw=0)
            ax.set_ylabel("SHAP value (log-odds)")
        ax.axhline(0, color=MUTED, lw=1)
        ax.set_xlabel(f)
        ax.set_title(f, loc="left", fontsize=11)
    fig.suptitle("Orange pushes risk up, blue pushes it down (5,000 test customers)",
                 x=0.01, ha="left", y=1.02, color=INK, fontsize=10)  # fmt: skip
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", dpi=110)
