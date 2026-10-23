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


# Chart palette, shared by every figure: validated reference palette, series slots 1-4 in
# order, plus the light chart surface and its ink. Change colours here only.
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#52514e", "#898781", "#e1e0d9"


def figure(**kwargs) -> Figure:
    return Figure(facecolor=SURFACE, **kwargs)


def style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(color=GRID, lw=0.6)


def reliability_plot(y: np.ndarray, curves: dict[str, np.ndarray], title: str, path: Path) -> None:
    """Observed churn rate vs mean predicted probability, in equal-count bins."""
    fig = figure(figsize=(5.2, 5.2))
    ax = fig.subplots()
    style(ax)
    top = max(float(np.quantile(p, 0.995)) for p in curves.values()) * 1.1
    ax.plot([0, top], [0, top], color=MUTED, ls="--", lw=1, label="perfect calibration")
    for (name, p), color in zip(curves.items(), [BLUE, ORANGE], strict=False):
        observed, predicted = calibration_curve(y, p, n_bins=15, strategy="quantile")
        ax.plot(predicted, observed, marker="o", ms=5, lw=2, color=color, label=name)
    ax.set(xlim=(0, top), ylim=(0, top), xlabel="Mean predicted churn probability",
           ylabel="Observed churn rate")  # fmt: skip
    ax.set_title(title, loc="left", fontsize=11)
    ax.legend(frameon=False, loc="upper left")
    fig.savefig(path, bbox_inches="tight", dpi=110)


def qini_curve(y: np.ndarray, treated: np.ndarray, score: np.ndarray) -> np.ndarray:
    """Incremental responders from treating the top k customers by `score`, for k = 0..n.

    Q(k) = R_t(k) - R_c(k) * N_t(k) / N_c(k): responders among the treated in the top k,
    minus control responders scaled up to the same group size (valid only in an RCT).
    """
    order = np.argsort(-np.asarray(score), kind="stable")
    y, t = np.asarray(y, float)[order], np.asarray(treated, float)[order]
    r_t, r_c = np.cumsum(y * t), np.cumsum(y * (1 - t))
    n_t, n_c = np.cumsum(t), np.cumsum(1 - t)
    scaled_c = np.divide(r_c * n_t, n_c, out=np.zeros_like(r_c), where=n_c > 0)
    return np.concatenate([[0.0], r_t - scaled_c])


def qini_coefficient(curve: np.ndarray) -> float:
    """Area between a Qini curve and random targeting, per customer (x runs 0..1)."""
    n = len(curve) - 1
    x = np.linspace(0, 1, n + 1)
    return float(np.trapezoid((curve - x * curve[-1]) / n, x))
