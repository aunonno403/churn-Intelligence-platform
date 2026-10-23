"""Module B: who does a contact actually change? Uplift on the Hillstrom e-mail RCT.

Hillstrom (MineThatData, 2008) randomly sent 64k customers a men's e-mail, a women's
e-mail, or nothing. We use women's e-mail vs nothing, outcome = site visit within two weeks.
Randomisation is what makes uplift estimable: treated and control customers are alike in
everything but the e-mail. This cannot be done on the observational KKBox data.

T-learner: one outcome model per arm; uplift = P(visit | e-mail) - P(visit | none).
X-learner (Kunzel et al. 2019): impute each customer's effect with the other arm's model,
regress those effects, and blend the two effect models by the treatment propensity.

The split here is random, not temporal: the campaign is a single snapshot with no time axis.
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
from typing import TYPE_CHECKING

import lightgbm as lgb
import numpy as np
import pandas as pd

from churn.eval.metrics import AQUA, BLUE, INK, MUTED, ORANGE, figure, style

if TYPE_CHECKING:
    from churn.config import Settings

URL = (
    "http://www.minethatdata.com/"
    "Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv"
)
# The server is plain HTTP, so pin the exact file every published number came from.
SHA256 = "0e5893329d8b93cefecc571777672028290ab69865718020c78c7284f291aece"
TREATMENT, CONTROL = "Womens E-Mail", "No E-Mail"
FEATURES = ["recency", "history", "mens", "womens", "zip_code", "newbie", "channel"]
# Smallest effect worth naming: below it a customer counts as unaffected.
MIN_EFFECT = 0.01

_LGB = dict(
    n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=200,
    subsample=0.8, subsample_freq=1, verbose=-1,
)  # fmt: skip


def load_hillstrom(settings: Settings) -> pd.DataFrame:
    path = settings.data_dir / "hillstrom" / "hillstrom.csv"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL, path)
    if (digest := hashlib.sha256(path.read_bytes()).hexdigest()) != SHA256:
        raise ValueError(
            f"{path} has SHA-256 {digest}, expected {SHA256}. "
            "The file is corrupt or changed; delete it to download it again."
        )
    df = pd.read_csv(path)
    df = df[df.segment.isin([TREATMENT, CONTROL])].reset_index(drop=True)
    for c in ("zip_code", "channel"):
        df[c] = df[c].astype("category")
    df["treated"] = (df.segment == TREATMENT).astype(int)
    return df


class TLearner:
    def __init__(self, seed: int = 0) -> None:
        self.m0 = lgb.LGBMClassifier(random_state=seed, **_LGB)
        self.m1 = lgb.LGBMClassifier(random_state=seed, **_LGB)

    def fit(self, X: pd.DataFrame, t: np.ndarray, y: np.ndarray) -> TLearner:
        self.m0.fit(X[t == 0], y[t == 0])
        self.m1.fit(X[t == 1], y[t == 1])
        return self

    def outcomes(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """P(visit | no e-mail), P(visit | e-mail)."""
        return self.m0.predict_proba(X)[:, 1], self.m1.predict_proba(X)[:, 1]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        p0, p1 = self.outcomes(X)
        return p1 - p0


class XLearner:
    def __init__(self, seed: int = 0) -> None:
        self.seed = seed

    def fit(self, X: pd.DataFrame, t: np.ndarray, y: np.ndarray) -> XLearner:
        base = TLearner(self.seed).fit(X, t, y)
        treated, control = t == 1, t == 0
        # imputed individual effects: observed outcome vs the other arm's prediction
        d1 = y[treated] - base.m0.predict_proba(X[treated])[:, 1]
        d0 = base.m1.predict_proba(X[control])[:, 1] - y[control]
        self.tau1 = lgb.LGBMRegressor(random_state=self.seed, **_LGB).fit(X[treated], d1)
        self.tau0 = lgb.LGBMRegressor(random_state=self.seed, **_LGB).fit(X[control], d0)
        self.propensity = float(t.mean())  # constant: treatment was randomised
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        e = self.propensity
        return e * self.tau0.predict(X) + (1 - e) * self.tau1.predict(X)


def segments(p0: np.ndarray, uplift: np.ndarray) -> np.ndarray:
    """The four classic groups. Unaffected customers split by whether they act anyway."""
    return np.select(
        [uplift > MIN_EFFECT, uplift < -MIN_EFFECT, p0 >= np.median(p0)],
        ["persuadable", "sleeping dog", "sure thing"],
        "lost cause",
    )


SEGMENTS = ["persuadable", "sleeping dog", "sure thing", "lost cause"]
STRATEGIES = {  # name -> colour
    "uplift (X-learner)": BLUE,
    "uplift (T-learner)": AQUA,
    "highest risk": ORANGE,  # least likely to visit unprompted: what a churn model ranks
}


def _effect(y: np.ndarray, t: np.ndarray) -> tuple[float | None, float | None]:
    """Observed RCT uplift (treated minus control visit rate) and its 95% half-width.
    None when either arm is empty: there is nothing to compare (and NaN is not JSON)."""
    a, b = y[t == 1], y[t == 0]
    if not len(a) or not len(b):
        return None, None
    se = np.sqrt(a.var() / len(a) + b.var() / len(b))
    return float(a.mean() - b.mean()), float(1.96 * se)


def _pp(v: float | None) -> float | None:
    return None if v is None else v * 100


def _fit_and_score(df: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, dict, np.ndarray]:
    from sklearn.model_selection import train_test_split

    train, test = train_test_split(
        df, test_size=0.3, random_state=seed, stratify=df.treated * 2 + df.visit
    )
    fit = dict(X=train[FEATURES], t=train.treated.to_numpy(), y=train.visit.to_numpy())
    t_model = TLearner(seed).fit(**fit)
    p0, _ = t_model.outcomes(test[FEATURES])
    scores = {
        "uplift (X-learner)": XLearner(seed).fit(**fit).predict(test[FEATURES]),
        "uplift (T-learner)": t_model.predict(test[FEATURES]),
        "highest risk": 1 - p0,
    }
    return test, scores, p0


def report(settings: Settings, budget: float = 0.3, splits: int = 5) -> dict:
    """Fit on 70% of the campaign and judge on the other 30% with the observed RCT outcomes,
    repeated over `splits` random splits so the comparison is not one lucky draw."""
    from churn.config import PROJECT_ROOT
    from churn.eval.metrics import qini_coefficient, qini_curve

    df = load_hillstrom(settings)
    at_budget = {name: [] for name in STRATEGIES}
    qini = {name: [] for name in STRATEGIES}
    for i in range(splits):
        test, scores, p0 = _fit_and_score(df, settings.seed + i)
        y, t = test.visit.to_numpy(), test.treated.to_numpy()
        for name, s in scores.items():
            # Qini counts gains in treated-group units; / treated share = gain if all top-k
            # were e-mailed (exact in expectation, since treatment was randomised)
            gain = qini_curve(y, t, s) / t.mean() * 1000 / len(y)
            at_budget[name].append(float(gain[round(len(y) * budget)]))
            qini[name].append(qini_coefficient(qini_curve(y, t, s)))
            if i == 0:
                scores[name] = (s, gain)
        if i == 0:
            first = (y, t, scores, p0)

    y, t, scores, p0 = first
    by_fifth = {}
    for name in ("uplift (X-learner)", "highest risk"):
        order = np.argsort(-scores[name][0], kind="stable")
        by_fifth[name] = [_effect(y[part], t[part]) for part in np.array_split(order, 5)]
    seg = segments(p0, scores["uplift (X-learner)"][0])
    seg_effects = {s: (float((seg == s).mean()), *_effect(y[seg == s], t[seg == s]))
                   for s in SEGMENTS}  # fmt: skip

    summary = {
        "treatment": f"{TREATMENT} vs {CONTROL}",
        "outcome": "visit",
        "test_customers_per_split": len(y),
        "splits": splits,
        "average_uplift_pp": _effect(y, t)[0] * 100,
        "budget_share": budget,
        "extra_visits_per_1000_at_budget": {
            n: {"mean": float(np.mean(v)), "min": min(v), "max": max(v)}
            for n, v in at_budget.items()
        },
        # every curve ends at the average effect; random targeting gets that share of it
        "random_targeting_at_budget_split0": float(scores["highest risk"][1][-1] * budget),
        "qini_coefficient_mean": {n: float(np.mean(v)) for n, v in qini.items()},
        "observed_uplift_pp_by_fifth_split0": {
            n: [{"uplift": u * 100, "ci95": c * 100} for u, c in v] for n, v in by_fifth.items()
        },
        "segments_split0": {
            s: {"share": sh, "observed_uplift_pp": _pp(u), "ci95_pp": _pp(c)}
            for s, (sh, u, c) in seg_effects.items()
        },
        "segment_rule": f"persuadable / sleeping dog: predicted uplift above +/-{MIN_EFFECT:.0%}; "
        "otherwise sure thing / lost cause by P(visit without e-mail) above / below median",
    }
    settings.model_dir.mkdir(parents=True, exist_ok=True)
    (settings.model_dir / "uplift.json").write_text(json.dumps(summary, indent=2))
    figures = PROJECT_ROOT / "reports" / "figures"
    _gain_plot({n: g for n, (_, g) in scores.items()}, summary, figures / "uplift_vs_risk.png")
    _fifths_plot(by_fifth, figures / "uplift_by_fifth.png")
    return summary


def _gain_plot(gains: dict, summary: dict, path) -> None:
    budget = summary["budget_share"]
    fig = figure(figsize=(8.5, 4.6))
    ax = fig.subplots()
    style(ax)
    for name, g in gains.items():
        s = summary["extra_visits_per_1000_at_budget"][name]
        ax.plot(np.linspace(0, 100, len(g)), g, lw=2, color=STRATEGIES[name],
                label=f"target by {name}: {s['mean']:.0f} at {budget:.0%} "
                      f"(range {s['min']:.0f}-{s['max']:.0f})")  # fmt: skip
    end = next(iter(gains.values()))[-1]
    ax.plot([0, 100], [0, end], ls="--", lw=1, color=MUTED,
            label=f"random targeting: {end * budget:.0f} at {budget:.0%}")  # fmt: skip
    ax.axvline(budget * 100, color=MUTED, lw=0.8, ls=":")
    ax.set_xlabel("Customers e-mailed, in the order each strategy ranks them (%)")
    ax.set_ylabel("Extra site visits per 1,000 customers")
    ax.set_title(
        "Targeting by uplift vs by risk (Hillstrom; curves from one split, "
        f"legend = mean of {summary['splits']} splits)",
        loc="left",
        fontsize=10,
    )
    ax.set_xlim(0, 100)
    ax.legend(frameon=False, loc="lower right", fontsize=8.5)
    fig.savefig(path, bbox_inches="tight", dpi=110)


def _fifths_plot(by_fifth: dict, path) -> None:
    fig = figure(figsize=(10, 3.6))
    axes = fig.subplots(1, 2, sharey=True)
    for ax, (name, effects) in zip(axes, by_fifth.items(), strict=True):
        style(ax)
        u = [e[0] * 100 for e in effects]
        ci = [e[1] * 100 for e in effects]
        ax.bar(range(5), u, yerr=ci, color=STRATEGIES[name], width=0.6,
               error_kw=dict(ecolor=INK, lw=1, capsize=3))  # fmt: skip
        ax.axhline(0, color=MUTED, lw=1)
        ax.set_xticks(range(5), ["top 20%", "2nd", "3rd", "4th", "bottom 20%"])
        ax.grid(axis="x", visible=False)
        ax.set_title(f"Ranked by {name}", loc="left", fontsize=11)
    axes[0].set_ylabel("Observed uplift in visit rate (pp)")
    fig.suptitle(
        "Does the ranking find the people the e-mail moves? Actual test outcomes, 95% intervals",
        x=0.01,
        ha="left",
        y=1.03,
        fontsize=10,
        color=INK,
    )
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", dpi=110)
