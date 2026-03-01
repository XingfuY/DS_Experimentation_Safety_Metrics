"""Visualization utilities for experimentation and safety metrics."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from typing import Optional


# Style defaults
COLORS = {
    "control": "#4A90D9",
    "treatment": "#E74C3C",
    "neutral": "#95A5A6",
    "accent": "#2ECC71",
    "warning": "#F39C12",
    "danger": "#E91E63",
}

def set_style():
    """Set consistent matplotlib style for all plots."""
    plt.rcParams.update({
        "figure.figsize": (10, 6),
        "figure.dpi": 100,
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.fontsize": 10,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.3,
    })


def plot_ab_test_results(
    control: np.ndarray, treatment: np.ndarray,
    metric_name: str = "Metric", test_result: Optional[dict] = None,
    ax: Optional[plt.Axes] = None
) -> plt.Figure:
    """Plot A/B test results with distributions and effect size.

    Args:
        control: Control group metric values.
        treatment: Treatment group metric values.
        metric_name: Name of the metric.
        test_result: Optional dict from statistical test with p_value, ci_lower, ci_upper.
        ax: Optional matplotlib axes.

    Returns:
        matplotlib Figure.
    """
    set_style()
    if ax is None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    else:
        fig = ax.figure
        axes = [ax, ax]

    # Distribution comparison
    axes[0].hist(control, bins=50, alpha=0.6, color=COLORS["control"],
                 label=f"Control (n={len(control):,})", density=True)
    axes[0].hist(treatment, bins=50, alpha=0.6, color=COLORS["treatment"],
                 label=f"Treatment (n={len(treatment):,})", density=True)
    axes[0].axvline(np.mean(control), color=COLORS["control"], linestyle="--", linewidth=2)
    axes[0].axvline(np.mean(treatment), color=COLORS["treatment"], linestyle="--", linewidth=2)
    axes[0].set_xlabel(metric_name)
    axes[0].set_ylabel("Density")
    axes[0].set_title(f"{metric_name} Distribution by Group")
    axes[0].legend()

    # Effect size with CI
    if test_result:
        effect = test_result.get("effect_size", np.mean(treatment) - np.mean(control))
        ci_lower = test_result.get("ci_lower", effect - 0.1)
        ci_upper = test_result.get("ci_upper", effect + 0.1)
        p_value = test_result.get("p_value", 1.0)

        color = COLORS["accent"] if p_value < 0.05 else COLORS["neutral"]
        axes[1].barh(0, effect, color=color, alpha=0.7, height=0.3)
        axes[1].errorbar(effect, 0, xerr=[[effect - ci_lower], [ci_upper - effect]],
                        fmt="o", color="black", capsize=5, markersize=8)
        axes[1].axvline(0, color="gray", linestyle="-", linewidth=1)
        axes[1].set_xlabel(f"Effect Size ({metric_name})")
        axes[1].set_title(f"Treatment Effect (p={p_value:.4f})")
        axes[1].set_yticks([])
        sig_text = "Significant" if p_value < 0.05 else "Not Significant"
        axes[1].text(effect, 0.2, f"{sig_text}\n95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]",
                    ha="center", fontsize=10)

    plt.tight_layout()
    return fig


def plot_sequential_test(
    results: list[dict], alpha: float = 0.05,
    ax: Optional[plt.Axes] = None
) -> plt.Figure:
    """Plot sequential test monitoring with alpha spending boundaries.

    Args:
        results: List of dicts from sequential_test function.
        alpha: Overall significance level.
        ax: Optional matplotlib axes.

    Returns:
        matplotlib Figure.
    """
    set_style()
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    else:
        fig = ax.figure

    days = [r["day"] for r in results]
    z_stats = [abs(r["z_stat"]) for r in results]
    boundaries = [abs(stats.norm.ppf(r["alpha_threshold"] / 2)) for r in results]

    ax.plot(days, z_stats, "o-", color=COLORS["treatment"], label="|Z-statistic|", linewidth=2)
    ax.plot(days, boundaries, "--", color=COLORS["danger"], label="Rejection boundary", linewidth=2)

    # Mark rejection points
    for r in results:
        if r["reject"]:
            ax.axvline(r["day"], color=COLORS["warning"], alpha=0.3, linewidth=3)
            ax.annotate("Reject H₀", xy=(r["day"], abs(r["z_stat"])),
                       xytext=(10, 10), textcoords="offset points",
                       fontsize=9, color=COLORS["danger"])

    ax.set_xlabel("Day")
    ax.set_ylabel("|Z-statistic|")
    ax.set_title("Sequential Test Monitoring")
    ax.legend()
    plt.tight_layout()
    return fig


def plot_did(
    df: pd.DataFrame, outcome_col: str, treat_col: str, time_col: str,
    did_result: Optional[dict] = None, ax: Optional[plt.Axes] = None
) -> plt.Figure:
    """Plot Difference-in-Differences with parallel trends visualization.

    Args:
        df: Panel data.
        outcome_col: Outcome variable.
        treat_col: Treatment indicator.
        time_col: Time period.
        did_result: Optional DID estimation result.
        ax: Optional matplotlib axes.

    Returns:
        matplotlib Figure.
    """
    set_style()
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    else:
        fig = ax.figure

    for group, color, label in [(0, COLORS["control"], "Control"), (1, COLORS["treatment"], "Treatment")]:
        group_data = df[df[treat_col] == group].groupby(time_col)[outcome_col].agg(["mean", "sem"])
        ax.plot(group_data.index, group_data["mean"], "o-", color=color, label=label, linewidth=2)
        ax.fill_between(group_data.index,
                       group_data["mean"] - 1.96 * group_data["sem"],
                       group_data["mean"] + 1.96 * group_data["sem"],
                       alpha=0.15, color=color)

    if did_result:
        ax.set_title(f"DID Estimate: {did_result['did_estimate']:.4f} "
                    f"(p={did_result['p_value']:.4f})")
    else:
        ax.set_title("Difference-in-Differences")

    ax.set_xlabel("Time Period")
    ax.set_ylabel(outcome_col)
    ax.legend()
    plt.tight_layout()
    return fig


def plot_rdd(
    df: pd.DataFrame, outcome_col: str, running_col: str,
    cutoff: float = 0, rdd_result: Optional[dict] = None,
    ax: Optional[plt.Axes] = None
) -> plt.Figure:
    """Plot Regression Discontinuity Design.

    Args:
        df: DataFrame.
        outcome_col: Outcome variable.
        running_col: Running variable.
        cutoff: Cutoff value.
        rdd_result: Optional RDD estimation result.
        ax: Optional matplotlib axes.

    Returns:
        matplotlib Figure.
    """
    set_style()
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    else:
        fig = ax.figure

    left = df[df[running_col] < cutoff]
    right = df[df[running_col] >= cutoff]

    ax.scatter(left[running_col], left[outcome_col], alpha=0.3, s=10,
              color=COLORS["control"], label="Below cutoff")
    ax.scatter(right[running_col], right[outcome_col], alpha=0.3, s=10,
              color=COLORS["treatment"], label="Above cutoff")

    # Local polynomial fits
    for subset, color in [(left, COLORS["control"]), (right, COLORS["treatment"])]:
        if len(subset) > 5:
            z = np.polyfit(subset[running_col], subset[outcome_col], 2)
            p = np.poly1d(z)
            x_smooth = np.linspace(subset[running_col].min(), subset[running_col].max(), 100)
            ax.plot(x_smooth, p(x_smooth), color=color, linewidth=2.5)

    ax.axvline(cutoff, color="gray", linestyle="--", linewidth=1.5, label=f"Cutoff = {cutoff}")

    if rdd_result:
        ax.set_title(f"RDD Estimate: {rdd_result['rdd_estimate']:.4f} "
                    f"(p={rdd_result['p_value']:.4f})")
    else:
        ax.set_title("Regression Discontinuity Design")

    ax.set_xlabel(running_col)
    ax.set_ylabel(outcome_col)
    ax.legend()
    plt.tight_layout()
    return fig


def plot_propensity_balance(
    balance: dict, ax: Optional[plt.Axes] = None
) -> plt.Figure:
    """Plot covariate balance (standardized mean differences) after matching.

    Args:
        balance: Dict from propensity_score_matching result.
        ax: Optional matplotlib axes.

    Returns:
        matplotlib Figure.
    """
    set_style()
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, max(4, len(balance) * 0.5)))
    else:
        fig = ax.figure

    covariates = list(balance.keys())
    smds = [balance[c]["smd"] for c in covariates]
    colors = [COLORS["accent"] if s < 0.1 else COLORS["danger"] for s in smds]

    y_pos = range(len(covariates))
    ax.barh(y_pos, smds, color=colors, alpha=0.7)
    ax.axvline(0.1, color=COLORS["danger"], linestyle="--", linewidth=1.5, label="Threshold (0.1)")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(covariates)
    ax.set_xlabel("Standardized Mean Difference (|SMD|)")
    ax.set_title("Covariate Balance After Matching")
    ax.legend()
    plt.tight_layout()
    return fig


def plot_anomaly_detection(
    timeseries: np.ndarray, anomaly_result: dict,
    dates: Optional[np.ndarray] = None, metric_name: str = "Metric",
    ax: Optional[plt.Axes] = None
) -> plt.Figure:
    """Plot time series with detected anomalies highlighted.

    Args:
        timeseries: Array of metric values.
        anomaly_result: Dict from anomaly detection function.
        dates: Optional date labels.
        metric_name: Name of the metric.
        ax: Optional matplotlib axes.

    Returns:
        matplotlib Figure.
    """
    set_style()
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 5))
    else:
        fig = ax.figure

    x = dates if dates is not None else np.arange(len(timeseries))

    ax.plot(x, timeseries, color=COLORS["control"], linewidth=1.5, alpha=0.8)

    if "upper_bound" in anomaly_result:
        ax.fill_between(x, anomaly_result["lower_bound"], anomaly_result["upper_bound"],
                       alpha=0.15, color=COLORS["control"], label="Normal range")

    anomaly_idx = anomaly_result.get("anomaly_indices", [])
    if len(anomaly_idx) > 0:
        ax.scatter(np.array(x)[anomaly_idx], timeseries[anomaly_idx],
                  color=COLORS["danger"], s=50, zorder=5, label=f"Anomalies ({len(anomaly_idx)})")

    ax.set_xlabel("Date" if dates is not None else "Time")
    ax.set_ylabel(metric_name)
    ax.set_title(f"Anomaly Detection: {metric_name}")
    ax.legend()
    plt.tight_layout()
    return fig


def plot_metric_tradeoff(
    precision: np.ndarray, recall: np.ndarray,
    thresholds: np.ndarray, current_threshold: float = 0.5,
    ax: Optional[plt.Axes] = None
) -> plt.Figure:
    """Plot precision-recall tradeoff with threshold selector.

    Args:
        precision: Precision values at each threshold.
        recall: Recall values at each threshold.
        thresholds: Threshold values.
        current_threshold: Current operating threshold.
        ax: Optional matplotlib axes.

    Returns:
        matplotlib Figure.
    """
    set_style()
    if ax is None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    else:
        fig = ax.figure
        axes = [ax, ax]

    # PR curve
    axes[0].plot(recall, precision, color=COLORS["treatment"], linewidth=2)
    idx = np.argmin(np.abs(thresholds - current_threshold))
    axes[0].plot(recall[idx], precision[idx], "o", color=COLORS["danger"],
                markersize=12, label=f"Current (t={current_threshold})")
    axes[0].set_xlabel("Recall")
    axes[0].set_ylabel("Precision")
    axes[0].set_title("Precision-Recall Curve")
    axes[0].legend()

    # Threshold sweep
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    axes[1].plot(thresholds, precision, label="Precision", color=COLORS["control"], linewidth=2)
    axes[1].plot(thresholds, recall, label="Recall", color=COLORS["treatment"], linewidth=2)
    axes[1].plot(thresholds, f1, label="F1", color=COLORS["accent"], linewidth=2, linestyle="--")
    axes[1].axvline(current_threshold, color="gray", linestyle=":", label=f"Current ({current_threshold})")
    axes[1].set_xlabel("Threshold")
    axes[1].set_ylabel("Score")
    axes[1].set_title("Metrics vs Threshold")
    axes[1].legend()

    plt.tight_layout()
    return fig
