"""Ranking and probability-quality metrics for churn scores."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure  # no pyplot: never changes a notebook's backend
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def lift_at(y: np.ndarray, p: np.ndarray, frac: float) -> float:
    """Churn rate among the top `frac` highest-scored users, divided by the overall rate."""
    if np.ptp(p) == 0:
        return 1.0  # a constant score ranks nobody; any top-k would be arbitrary row order
    k = max(1, math.ceil(len(y) * frac))
    top = np.argsort(-p, kind="stable")[:k]
    return float(y[top].mean() / y.mean())


def evaluate(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    y, p = np.asarray(y), np.asarray(p)
    return {
        "pr_auc": float(average_precision_score(y, p)),
        "roc_auc": float(roc_auc_score(y, p)) if len(set(p)) > 1 else 0.5,
        "brier": float(brier_score_loss(y, p)),
        "lift_10": lift_at(y, p, 0.10),
        "lift_20": lift_at(y, p, 0.20),
        "churn_rate": float(y.mean()),
        "mean_predicted": float(p.mean()),  # vs churn_rate: calibration-in-the-large
    }


def reliability_plot(y: np.ndarray, curves: dict[str, np.ndarray], title: str, path: Path) -> None:
    """Observed churn rate vs mean predicted probability, in equal-count bins."""
    colors = ["#2a78d6", "#eb6834"]  # reference palette, slots 1-2
    fig = Figure(figsize=(5.2, 5.2), facecolor="#fcfcfb")
    ax = fig.subplots()
    ax.set_facecolor("#fcfcfb")
    top = max(float(np.quantile(p, 0.995)) for p in curves.values()) * 1.1
    ax.plot([0, top], [0, top], color="#898781", ls="--", lw=1, label="perfect calibration")
    for (name, p), color in zip(curves.items(), colors, strict=False):
        observed, predicted = calibration_curve(y, p, n_bins=15, strategy="quantile")
        ax.plot(predicted, observed, marker="o", ms=5, lw=2, color=color, label=name)
    ax.set(xlim=(0, top), ylim=(0, top), xlabel="Mean predicted churn probability",
           ylabel="Observed churn rate")  # fmt: skip
    ax.set_title(title, loc="left", fontsize=11)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(color="#e1e0d9", lw=0.6)
    ax.legend(frameon=False, loc="upper left")
    fig.savefig(path, bbox_inches="tight", dpi=110)
