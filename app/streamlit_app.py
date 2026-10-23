"""Churn Intelligence: overview, risk explorer, campaign simulator, model card.

Runs only on app/bundle/ (built by `churn bundle`), never on the DuckDB database, so it
fits on a free host. Run: streamlit run app/streamlit_app.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Import the project's code straight from src/ so a host needs only app/requirements.txt,
# not the training stack (MLflow, Optuna, DuckDB, Kaggle).
sys.path.insert(0, str(ROOT / "src"))

import altair as alt  # noqa: E402
import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from churn.eval.economics import break_even, net_savings_curve  # noqa: E402
from churn.eval.metrics import BLUE, MUTED, ORANGE  # noqa: E402
from churn.explain.shap_utils import explain  # noqa: E402

BUNDLE = ROOT / "app" / "bundle"
FEATURES_SHOWN = {
    "last_auto_renew": "Auto-renew on",
    "last_plan_days": "Plan length (days)",
    "last_amount_paid": "Last paid (NTD)",
    "days_since_last_tx": "Days since last payment",
    "minutes_30d": "Minutes listened, last 30 days",
    "active_days_30d": "Active days, last 30 days",
}

st.set_page_config(page_title="Churn Intelligence", layout="wide")


# `version` is the bundle's file time: part of the cache key, so a rebuilt bundle reloads.
@st.cache_resource
def load_model(version: float):
    return joblib.load(BUNDLE / "model.joblib")


@st.cache_data
def load_bundle(version: float) -> tuple[pd.DataFrame, dict]:
    customers = pd.read_parquet(BUNDLE / "customers.parquet")
    summary = json.loads((BUNDLE / "summary.json").read_text())
    return customers.sort_values("score", ascending=False).reset_index(drop=True), summary


def _clamp(v, lo, hi):
    """Slider start values must lie inside the slider, whatever the bundle's settings say."""
    return min(max(v, lo), hi)


def _show(v) -> str:
    if pd.isna(v):
        return "missing"
    return f"{v:,.4g}" if isinstance(v, float | np.floating) else str(v)


if not (BUNDLE / "customers.parquet").exists():
    st.error("No data bundle found. Build it with `churn bundle` (see the README).")
    st.stop()

version = (BUNDLE / "customers.parquet").stat().st_mtime  # `churn bundle` rewrites every file
customers, summary = load_bundle(version)
model = load_model(version)
cfg = summary["settings"]
test_metrics = summary["metrics"]["metrics"]

st.title("Churn Intelligence")
st.caption(
    f"KKBox music subscriptions · test month {cfg['test_month']} · "
    f"{cfg['bundle_customers']:,} sampled customers of {cfg['test_customers']:,} in the "
    f"{cfg['sample_share_of_kkbox_users']:.0%} sample"
)
overview, explorer, simulator, card = st.tabs(
    ["Overview", "Risk explorer", "Campaign simulator", "Model card"]
)

# --- Overview ------------------------------------------------------------------------
with overview:
    lgbm = test_metrics["lightgbm"]["test"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Churn rate, test month", f"{lgbm['churn_rate']:.1%}")
    c2.metric("PR-AUC", f"{lgbm['pr_auc']:.3f}", help="Random ranking would score the churn rate.")
    c3.metric("Churners in top 10% of scores", f"{lgbm['lift_10']:.1f}x the average")
    econ = summary["economics"]
    c4.metric(
        "Net savings, break-even targeting",
        f"{econ['rule_net_savings_per_1000']:,.0f} NTD",
        help=f"Per 1,000 customers, contacting everyone above the break-even probability; "
        f"all {cfg['test_customers']:,} test customers.",
    )

    left, right = st.columns(2)
    with left:
        st.subheader("Churn rate by expiry month")
        trend = pd.DataFrame(summary["churn_by_cohort"])
        st.altair_chart(
            alt.Chart(trend)
            .mark_bar(color=BLUE, cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
            .encode(
                x=alt.X("cohort:N", title="Membership expires in", axis=alt.Axis(labelAngle=0)),
                y=alt.Y("churn_rate:Q", title="Churn rate", axis=alt.Axis(format="%")),
                tooltip=[
                    alt.Tooltip("cohort:N", title="Month"),
                    alt.Tooltip("split:N", title="Used for"),
                    alt.Tooltip("customers:Q", title="Customers", format=","),
                    alt.Tooltip("churn_rate:Q", title="Churn rate", format=".1%"),
                ],
            ),
            width="stretch",
        )
        st.caption("Gap months (Nov, Jan) are left out so no label leaks across splits.")
    with right:
        st.subheader("Cohort retention")
        retention = pd.DataFrame(summary["retention"])
        st.altair_chart(
            alt.Chart(retention)
            .mark_rect()
            .encode(
                x=alt.X("months_since:O", title="Months since first payment"),
                y=alt.Y("cohort:O", title="First paid in"),
                color=alt.Color(
                    "retained:Q",
                    title="Still subscribed",
                    scale=alt.Scale(domain=[0, 1], range=["#cde2fb", BLUE, "#0d366b"]),
                    legend=alt.Legend(format="%"),
                ),
                tooltip=[
                    alt.Tooltip("cohort:O", title="Cohort"),
                    alt.Tooltip("months_since:O", title="Months since"),
                    alt.Tooltip("retained:Q", title="Still subscribed", format=".0%"),
                    alt.Tooltip("cohort_size:Q", title="Cohort size", format=","),
                ],
            ),
            width="stretch",
        )
    st.subheader("What drives churn")
    for d in summary["drivers"]:
        st.markdown(f"- {d['sentence']}")

# --- Risk explorer -------------------------------------------------------------------
with explorer:
    f1, f2, f3 = st.columns(3)
    renew = f1.radio("Auto-renew", ["All", "On", "Off"], horizontal=True)
    floor = f2.slider("Minimum churn probability", 0.0, 1.0, 0.10, 0.01, format="%.2f")
    shown = f3.slider("Customers to list", 10, 500, 100, 10)
    view = customers[customers.churn_probability >= floor]
    if renew != "All":
        view = view[view.last_auto_renew == (renew == "On")]
    st.caption(f"{len(view):,} customers match, highest risk first.")

    table = view.head(shown)[["customer", "churn_probability", *FEATURES_SHOWN, "is_churn"]]
    st.dataframe(
        table.assign(
            churn_probability=table.churn_probability * 100,
            last_auto_renew=table.last_auto_renew.astype(bool),
            is_churn=table.is_churn.astype(bool),
        ),
        column_config={
            "customer": "Customer",
            "churn_probability": st.column_config.ProgressColumn(
                "Churn probability", format="%.0f%%", min_value=0, max_value=100
            ),
            **{k: st.column_config.Column(v) for k, v in FEATURES_SHOWN.items()},
            "minutes_30d": st.column_config.NumberColumn(
                FEATURES_SHOWN["minutes_30d"], format="%.0f"
            ),
            "is_churn": st.column_config.CheckboxColumn(
                "Churned (actual)", help="What really happened in the test month."
            ),
        },
        hide_index=True,
        width="stretch",
        height=320,
    )

    if len(view):
        who = st.selectbox("Explain one customer", view.head(shown).customer)
        row = view[view.customer == who]
        p = float(row.churn_probability.iloc[0])
        st.markdown(
            f"**{who}**: churn probability **{p:.0%}**, risk rank "
            f"**#{row.index[0] + 1:,}** of {len(customers):,}."  # customers is sorted by score
        )
        contrib = explain(model, row).drop(columns="base").iloc[0]
        top = contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(10)
        values = row.iloc[0]
        why = pd.DataFrame(
            {
                "feature": [f"{f} = {_show(values[f])}" for f in top.index],
                "push": top.to_numpy(),
                "direction": np.where(top > 0, "raises risk", "lowers risk"),
            }
        )
        st.altair_chart(
            alt.Chart(why)
            .mark_bar(cornerRadius=3)
            .encode(
                x=alt.X("push:Q", title="Push on the risk score (log-odds, SHAP)"),
                y=alt.Y("feature:N", sort=None, title=None, axis=alt.Axis(labelLimit=400)),
                color=alt.Color(
                    "direction:N",
                    scale=alt.Scale(domain=["raises risk", "lowers risk"], range=[ORANGE, BLUE]),
                    legend=alt.Legend(title=None, orient="top"),
                ),
                tooltip=[
                    alt.Tooltip("feature:N", title="Feature"),
                    alt.Tooltip("push:Q", title="Push", format="+.3f"),
                ],
            ),
            width="stretch",
        )
    else:
        st.info("No customers match these filters.")

# --- Campaign simulator --------------------------------------------------------------
with simulator:
    st.markdown(
        "Contacting a customer costs money. If they would have churned, the offer keeps "
        "them with the success rate below, worth the customer value. Contact everyone "
        "whose churn probability is above the **break-even** point, up to your budget."
    )
    st.caption(
        f"Computed live on the {len(customers):,}-customer sample, so the default plan's "
        f"savings differ slightly from the Overview figure for all {cfg['test_customers']:,} "
        "test customers."
    )
    s1, s2, s3, s4 = st.columns(4)
    cost = s1.slider("Cost per contact (NTD)", 5, 300, _clamp(int(cfg["contact_cost"]), 5, 300), 5)
    success = s2.slider("Offer success rate", 0.05, 0.80,
                        _clamp(float(cfg["success_rate"]), 0.05, 0.80), 0.05,
                        format="%.2f")  # fmt: skip
    value = s3.slider("Value of a kept customer (NTD)", 300, 6000,
                      _clamp(int(cfg["customer_value"]), 300, 6000), 50)  # fmt: skip
    cap = s4.slider("Budget: max share contacted", 0.01, 1.0, 1.0, 0.01, format="%.2f")

    econ_args = dict(contact_cost=cost, success_rate=success, customer_value=value)
    threshold = break_even(**econ_args)
    n = len(customers)
    prob, actual = customers.churn_probability.to_numpy(), customers.is_churn.to_numpy()
    expected = net_savings_curve(customers.score, prob, **econ_args)
    realised = net_savings_curve(customers.score, actual, **econ_args)
    # every contact above break-even adds expected value, so the best budget-limited plan is
    # the top min(k_rule, cap) customers (they are already sorted by score)
    k = min(int((prob >= threshold).sum()), int(cap * n))
    per = 1000 / n

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Break-even churn probability", f"{threshold:.1%}")
    m2.metric("Customers to contact", f"{k:,} ({k / n:.1%})")
    m3.metric("Expected net savings", f"{expected[k] * per:,.0f} NTD",
              help="Per 1,000 customers, using the model's probabilities.")  # fmt: skip
    m4.metric(
        "What actually happened",
        f"{realised[k] * per:,.0f} NTD",
        delta=f"{(realised[k] - expected[k]) * per:+,.0f} vs expected",
        help="Per 1,000 customers, using the real test-month outcomes.",
    )

    # show where decisions happen: up to 40% contacted, or twice the plan if that is larger
    last = int(n * max(0.4, min(1.0, 2 * k / n)))
    idx = np.arange(0, last + 1, max(1, last // 400))
    curve = pd.DataFrame(
        {"share": idx / n, "expected": expected[idx] * per, "realised": realised[idx] * per}
    ).melt("share", var_name="outcome", value_name="savings")
    lines = (
        alt.Chart(curve)
        .mark_line(strokeWidth=2)
        .encode(
            x=alt.X(
                "share:Q",
                title="Customers contacted, highest risk first",
                axis=alt.Axis(format="%"),
            ),  # fmt: skip
            y=alt.Y("savings:Q", title="Net savings (NTD per 1,000 customers)"),
            color=alt.Color(
                "outcome:N",
                scale=alt.Scale(domain=["expected", "realised"], range=[ORANGE, BLUE]),
                legend=alt.Legend(title=None, orient="top"),
            ),
            tooltip=[
                alt.Tooltip("share:Q", title="Contacted", format=".1%"),
                alt.Tooltip("outcome:N", title="Curve"),
                alt.Tooltip("savings:Q", title="Net savings", format=",.0f"),
            ],
        )
    )
    marker = alt.Chart(pd.DataFrame({"share": [k / n]})).mark_rule(color=MUTED, strokeDash=[4, 4])
    st.altair_chart((lines + marker.encode(x="share:Q")).properties(height=320),
                    width="stretch")  # fmt: skip

    target = customers.head(k)[["customer", "churn_probability", *FEATURES_SHOWN]]
    st.markdown(f"**Who to contact** ({k:,} customers)")
    st.dataframe(target.head(200), hide_index=True, width="stretch", height=240)
    st.download_button(
        "Download the full list (CSV)",
        target.to_csv(index=False).encode(),
        file_name="customers_to_contact.csv",
        mime="text/csv",
        disabled=k == 0,
    )

# --- Model card ----------------------------------------------------------------------
with card:
    st.subheader("Model card")
    st.markdown(
        f"""
**Predicts** whether a KKBox subscriber churns: no valid renewal within
{cfg["churn_window_days"]} days after their membership expires (the WSDM Cup definition,
matching KKBox's published labelling script). **Model:** LightGBM, Optuna-tuned, with
isotonic calibration fitted on out-of-fold training scores.

**Temporal validation:** train on expiry months {", ".join(cfg["train_months"])};
validate on {cfg["valid_month"]}; test on {cfg["test_month"]}. A one-month gap between
splits means the model never trains on labels that did not exist yet at prediction time.
Features use only data from before each month's cutoff, and a test proves it.
"""
    )
    rows = [
        {"model": name, "split": split, **{k: m[k] for k in
         ("pr_auc", "roc_auc", "brier", "lift_10", "churn_rate", "mean_predicted")}}
        for name, by in test_metrics.items() for split, m in by.items()
    ]  # fmt: skip
    st.dataframe(
        pd.DataFrame(rows).round(4), hide_index=True, width="stretch",
        column_config={"lift_10": "lift @10%", "mean_predicted": "mean predicted"},
    )  # fmt: skip
    a, b = st.columns(2)
    a.image(str(BUNDLE / "reliability_test.png"), caption="Calibration on the test month")
    lv = summary["label_validation"]
    rates = [c["churn_rate"] for c in summary["churn_by_cohort"]]
    b.markdown(
        f"""
**Label audit.** Our labels vs KKBox's official Feb-2017 file: {lv["agreement"]:.1%}
agreement, {lv["churner_precision"]:.1%} precision, {lv["churner_recall"]:.0%} recall of
official churners. The disagreements contradict KKBox's own published script, so the
official file is treated as a diagnostic, not ground truth.

**Limitations**
- Probabilities move with the monthly churn rate ({min(rates):.1%} to {max(rates):.1%}
  across months); a single calibration cannot follow that. Budget forecasts should use the
  sensitivity range.
- The model ranks risk; it does not say a contact will work. Campaign economics assume
  the offer works equally on every churner, which the uplift study below contradicts.
- Trained on a {cfg["sample_share_of_kkbox_users"]:.0%} user sample; in a learning-curve
  check, each doubling of data added roughly +0.01 PR-AUC.
- Member attributes (city, age, gender) are an undated snapshot.
"""
    )
    st.subheader("Who should we contact? Uplift vs risk (Hillstrom e-mail experiment)")
    up = summary["uplift"]
    gains = up["extra_visits_per_1000_at_budget"]
    st.markdown(
        f"""
A separate, randomised e-mail campaign (64k customers) answers the causal question the
churn model cannot. E-mailing {up["budget_share"]:.0%} of customers, ranked by predicted
**uplift**, brings **{gains["uplift (X-learner)"]["mean"]:.0f}** extra visits per 1,000
customers; ranked by **risk**, {gains["highest risk"]["mean"]:.0f}; at random,
{up["random_targeting_at_budget_split0"]:.0f} (mean of {up["splits"]} random splits).
Risk and responsiveness are unrelated, so the riskiest customers are not the ones a
contact moves.
"""
    )
    c, d = st.columns(2)
    c.image(str(BUNDLE / "uplift_vs_risk.png"))
    d.image(str(BUNDLE / "uplift_by_fifth.png"))
