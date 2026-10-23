"""Who to contact: expected net savings of a retention campaign at every targeting depth.

Contacting a customer costs `contact_cost`. If they would have churned, the offer keeps them
with probability `success_rate`, worth `customer_value`. So contacting customer i is worth
    p_i * success_rate * customer_value - contact_cost
and it pays exactly when p_i exceeds the break-even probability
    contact_cost / (success_rate * customer_value).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import brentq
from scipy.special import expit, logit

from churn.eval.metrics import BLUE, INK, MUTED, ORANGE, SURFACE, figure, style

if TYPE_CHECKING:
    from churn.config import Settings


def break_even(contact_cost: float, success_rate: float, customer_value: float) -> float:
    return contact_cost / (success_rate * customer_value)


def net_savings_curve(
    rank_score: np.ndarray,
    outcome: np.ndarray,
    contact_cost: float,
    success_rate: float,
    customer_value: float,
) -> np.ndarray:
    """Net savings from contacting the top k customers by `rank_score`, for k = 0..n.

    `outcome` is a churn probability (expected savings) or a 0/1 label (realised savings).
    """
    order = np.argsort(-np.asarray(rank_score), kind="stable")
    gain = np.asarray(outcome, dtype=float)[order] * success_rate * customer_value - contact_cost
    return np.concatenate([[0.0], np.cumsum(gain)])


def shift_base_rate(p: np.ndarray, target_rate: float) -> np.ndarray:
    """Move every probability by the same log-odds shift so their mean is `target_rate`.

    Simulates a month whose churn rate differs from what the model expects, keeping the
    ranking unchanged.
    """
    z = logit(np.clip(p, 1e-6, 1 - 1e-6))
    shift = brentq(lambda d: expit(z + d).mean() - target_rate, -20, 20)
    return expit(z + shift)


def report(settings: Settings) -> dict:
    """Savings curves on the test cohort -> reports/figures/net_savings.png + economics.json."""
    from churn.config import PROJECT_ROOT
    from churn.features.build import TARGET, load_test_cohort

    test, model = load_test_cohort(settings)
    y = test[TARGET].to_numpy()
    rank, p = model.predict_raw(test), model.predict(test)  # rank on raw: calibration ties
    econ = dict(
        contact_cost=settings.contact_cost,
        success_rate=settings.success_rate,
        customer_value=settings.customer_value,
    )
    realised = net_savings_curve(rank, y, **econ)
    expected = net_savings_curve(rank, p, **econ)
    p_low, p_high = shift_base_rate(p, p.mean() / 2), shift_base_rate(p, p.mean() * 2)
    low = net_savings_curve(rank, p_low, **econ)
    high = net_savings_curve(rank, p_high, **econ)

    n, per = len(y), 1000 / len(y)  # report per 1,000 customers in the cohort
    threshold = break_even(**econ)
    k_rule = int((p >= threshold).sum())  # p rises with rank (isotonic), so this is a top-k
    k_low = int((p_low >= threshold).sum())
    k_best = int(np.argmax(realised))
    summary = {
        "assumptions": econ,
        "break_even_probability": threshold,
        "rule": "contact every customer whose churn probability >= break-even",
        "rule_contact_share": k_rule / n,
        "rule_net_savings_per_1000": realised[k_rule] * per,
        "best_possible_contact_share": k_best / n,
        "best_possible_net_savings_per_1000": realised[k_best] * per,
        "contact_top_10pct_net_savings_per_1000": realised[round(n * 0.1)] * per,
        "contact_everyone_net_savings_per_1000": realised[-1] * per,
        # if churn were half as common, the rule contacts fewer people; expected value then
        "rule_contact_share_if_churn_half": k_low / n,
        "rule_expected_savings_per_1000_if_churn_half": low[k_low] * per,
        "test_customers": n,
        "test_churn_rate": float(y.mean()),
    }
    (settings.model_dir / "economics.json").write_text(json.dumps(summary, indent=2))
    _plot(realised * per, expected * per, low * per, high * per, k_rule / n, summary,
          PROJECT_ROOT / "reports" / "figures" / "net_savings.png")  # fmt: skip
    return summary


def _plot(realised, expected, low, high, rule_share, summary, path) -> None:
    x = np.arange(len(realised)) / (len(realised) - 1) * 100
    fig = figure(figsize=(8.5, 4.4))
    ax = fig.subplots()
    style(ax)
    ax.fill_between(x, low, high, color=ORANGE, alpha=0.15, lw=0,
                    label="expected if the churn rate were ×0.5 to ×2")  # fmt: skip
    ax.plot(x, expected, color=ORANGE, lw=2, label="expected (model probabilities)")
    ax.plot(x, realised, color=BLUE, lw=2, label="realised (actual test outcomes)")
    ax.axhline(0, color=MUTED, lw=1)
    k = round(rule_share * (len(realised) - 1))
    ax.plot(x[k], realised[k], "o", ms=8, color=BLUE, mec=SURFACE, mew=2)
    ax.annotate(
        f"contact the top {rule_share:.1%}\n"
        f"(churn probability ≥ {summary['break_even_probability']:.0%})\n"
        f"net {realised[k]:,.0f} NTD per 1,000 customers",
        (x[k], realised[k]), xytext=(40, -115), textcoords="offset points", color=INK, fontsize=9,
        arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8),
    )  # fmt: skip
    shown = x <= 40
    ax.set_xlim(0, 40)
    ax.set_ylim(min(low[shown].min(), realised[shown].min()) * 1.1, high[shown].max() * 1.1)
    ax.set_xlabel("Customers contacted, highest risk first (%)")
    ax.set_ylabel("Net savings (NTD per 1,000 customers)")
    ax.set_title("Net savings of a retention campaign, test cohort (2017-02)", loc="left",
                 fontsize=11)  # fmt: skip
    ax.legend(frameon=False, loc="lower left", fontsize=9)
    fig.savefig(path, bbox_inches="tight", dpi=110)
