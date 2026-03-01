"""Safety metrics library for content platform trust & safety."""

import numpy as np
import pandas as pd
from scipy import stats


def violation_rate(violations: np.ndarray, total: np.ndarray) -> dict:
    """Content violation rate with confidence interval.

    Args:
        violations: Number of violations per period.
        total: Total content items per period.

    Returns:
        Dict with rate, CI, and trend information.
    """
    rate = violations.sum() / total.sum()
    se = np.sqrt(rate * (1 - rate) / total.sum())
    return {
        "rate": rate,
        "se": se,
        "ci_lower": rate - 1.96 * se,
        "ci_upper": rate + 1.96 * se,
        "total_violations": int(violations.sum()),
        "total_content": int(total.sum()),
    }


def precision_recall_at_threshold(
    scores: np.ndarray, labels: np.ndarray, threshold: float = 0.5
) -> dict:
    """Compute precision, recall, F1, and confusion matrix at a threshold.

    Args:
        scores: Classifier confidence scores (0-1).
        labels: True binary labels.
        threshold: Classification threshold.

    Returns:
        Dict with precision, recall, F1, FPR, confusion matrix.
    """
    preds = (scores >= threshold).astype(int)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "threshold": threshold,
    }


def precision_recall_curve(
    scores: np.ndarray, labels: np.ndarray, n_thresholds: int = 100
) -> pd.DataFrame:
    """Compute precision-recall curve across thresholds.

    Args:
        scores: Classifier confidence scores.
        labels: True binary labels.
        n_thresholds: Number of threshold points.

    Returns:
        DataFrame with threshold, precision, recall, f1, fpr columns.
    """
    thresholds = np.linspace(0.01, 0.99, n_thresholds)
    results = []
    for t in thresholds:
        metrics = precision_recall_at_threshold(scores, labels, t)
        results.append(metrics)
    return pd.DataFrame(results)


def time_to_action(action_times: np.ndarray, targets: dict = None) -> dict:
    """Compute time-to-action statistics for moderation response.

    Args:
        action_times: Array of response times (in hours).
        targets: Optional dict with SLA targets, e.g. {"p50": 1.0, "p95": 24.0}.

    Returns:
        Dict with percentile stats and SLA compliance rates.
    """
    if targets is None:
        targets = {"p50": 1.0, "p95": 24.0}

    valid = action_times[action_times > 0]
    result = {
        "mean": float(np.mean(valid)),
        "median": float(np.median(valid)),
        "p50": float(np.percentile(valid, 50)),
        "p75": float(np.percentile(valid, 75)),
        "p90": float(np.percentile(valid, 90)),
        "p95": float(np.percentile(valid, 95)),
        "p99": float(np.percentile(valid, 99)),
        "std": float(np.std(valid)),
        "n": len(valid),
    }

    for label, target in targets.items():
        percentile = float(label.replace("p", ""))
        actual = np.percentile(valid, percentile)
        result[f"sla_{label}_target"] = target
        result[f"sla_{label}_actual"] = actual
        result[f"sla_{label}_met"] = actual <= target

    return result


def recidivism_rate(
    user_violations: pd.DataFrame, user_col: str = "user_id",
    time_col: str = "timestamp", window_days: int = 30
) -> dict:
    """Compute recidivism rate: fraction of users who re-offend within window.

    Args:
        user_violations: DataFrame of violation events.
        user_col: User identifier column.
        time_col: Timestamp column.
        window_days: Window for measuring repeat violations.

    Returns:
        Dict with recidivism rate, repeat offender stats.
    """
    df = user_violations.sort_values([user_col, time_col])
    df[time_col] = pd.to_datetime(df[time_col])

    user_counts = df.groupby(user_col).size()
    total_offenders = len(user_counts)
    repeat_offenders = (user_counts > 1).sum()

    # Time-windowed recidivism
    recid_count = 0
    for uid, group in df.groupby(user_col):
        if len(group) < 2:
            continue
        times = group[time_col].values
        for i in range(len(times) - 1):
            diff = (times[i + 1] - times[i]) / np.timedelta64(1, "D")
            if diff <= window_days:
                recid_count += 1
                break

    return {
        "recidivism_rate": recid_count / total_offenders if total_offenders > 0 else 0,
        "repeat_offenders": int(repeat_offenders),
        "total_offenders": int(total_offenders),
        "recidivists_in_window": recid_count,
        "window_days": window_days,
        "avg_violations_per_offender": float(user_counts.mean()),
        "max_violations": int(user_counts.max()),
    }


def composite_safety_score(
    metrics: dict, weights: dict = None
) -> dict:
    """Compute weighted composite safety score from multiple metrics.

    Normalizes each metric to [0,1] and computes weighted average.
    Higher score = safer platform.

    Args:
        metrics: Dict of metric_name → value. Some are "higher is better"
                 (precision, recall) and some are "lower is better" (violation_rate, fpr).
        weights: Dict of metric_name → weight. Defaults to equal weights.

    Returns:
        Dict with composite score, component scores, and breakdown.
    """
    higher_is_better = {"precision", "recall", "f1", "trust_score", "satisfaction"}
    lower_is_better = {"violation_rate", "fpr", "time_to_action", "recidivism_rate", "churn_rate"}

    if weights is None:
        weights = {k: 1.0 / len(metrics) for k in metrics}

    # Normalize weights
    total_w = sum(weights.values())
    weights = {k: v / total_w for k, v in weights.items()}

    components = {}
    for name, value in metrics.items():
        if name in higher_is_better:
            normalized = min(max(value, 0), 1)
        elif name in lower_is_better:
            normalized = 1 - min(max(value, 0), 1)
        else:
            normalized = min(max(value, 0), 1)

        components[name] = {
            "raw": value,
            "normalized": normalized,
            "weight": weights.get(name, 0),
            "weighted": normalized * weights.get(name, 0),
        }

    composite = sum(c["weighted"] for c in components.values())

    return {
        "composite_score": composite,
        "components": components,
        "interpretation": (
            "excellent" if composite >= 0.9 else
            "good" if composite >= 0.7 else
            "needs_improvement" if composite >= 0.5 else
            "critical"
        ),
    }


def anomaly_detection_zscore(
    timeseries: np.ndarray, window: int = 30, threshold: float = 3.0
) -> dict:
    """Z-score based anomaly detection on time series.

    Args:
        timeseries: Array of metric values over time.
        window: Rolling window size for computing mean/std.
        threshold: Z-score threshold for anomaly flag.

    Returns:
        Dict with anomaly flags, z-scores, and detected anomalies.
    """
    series = pd.Series(timeseries)
    rolling_mean = series.rolling(window=window, min_periods=max(1, window // 2)).mean()
    rolling_std = series.rolling(window=window, min_periods=max(1, window // 2)).std()
    rolling_std = rolling_std.replace(0, np.nan).ffill().fillna(1)

    z_scores = ((series - rolling_mean) / rolling_std).fillna(0).values
    is_anomaly = np.abs(z_scores) > threshold

    anomaly_indices = np.where(is_anomaly)[0]
    return {
        "z_scores": z_scores,
        "is_anomaly": is_anomaly,
        "anomaly_indices": anomaly_indices.tolist(),
        "n_anomalies": int(is_anomaly.sum()),
        "anomaly_rate": float(is_anomaly.mean()),
        "threshold": threshold,
        "window": window,
    }


def ewma_anomaly_detection(
    timeseries: np.ndarray, span: int = 20, n_std: float = 3.0
) -> dict:
    """Exponentially Weighted Moving Average anomaly detection.

    More responsive to recent changes than simple rolling z-score.

    Args:
        timeseries: Array of metric values.
        span: EWMA span parameter.
        n_std: Number of standard deviations for threshold.

    Returns:
        Dict with EWMA values, bounds, and anomaly flags.
    """
    series = pd.Series(timeseries)
    ewma = series.ewm(span=span).mean()
    ewma_std = series.ewm(span=span).std()
    ewma_std = ewma_std.replace(0, np.nan).ffill().fillna(1)

    upper = ewma + n_std * ewma_std
    lower = ewma - n_std * ewma_std
    is_anomaly = ((series > upper) | (series < lower)).values

    return {
        "ewma": ewma.values,
        "upper_bound": upper.values,
        "lower_bound": lower.values,
        "is_anomaly": is_anomaly,
        "anomaly_indices": np.where(is_anomaly)[0].tolist(),
        "n_anomalies": int(is_anomaly.sum()),
    }


def change_point_detection(
    timeseries: np.ndarray, min_segment: int = 10, penalty: float = 3.0
) -> dict:
    """Simple change point detection using cumulative sum (CUSUM).

    Args:
        timeseries: Array of metric values.
        min_segment: Minimum segment length between change points.
        penalty: Penalty for adding a change point (higher = fewer points).

    Returns:
        Dict with detected change points and segment statistics.
    """
    n = len(timeseries)
    mean_overall = np.mean(timeseries)
    cumsum = np.cumsum(timeseries - mean_overall)

    # Find candidate change points using CUSUM max deviation
    change_points = []
    segments = [(0, n)]

    while segments:
        start, end = segments.pop(0)
        if end - start < 2 * min_segment:
            continue

        segment = timeseries[start:end]
        seg_mean = np.mean(segment)
        seg_cumsum = np.cumsum(segment - seg_mean)
        max_dev_idx = np.argmax(np.abs(seg_cumsum))

        # Test significance
        left = segment[:max_dev_idx + 1]
        right = segment[max_dev_idx + 1:]
        if len(left) < min_segment or len(right) < min_segment:
            continue

        t_stat, p_val = stats.ttest_ind(left, right)
        if abs(t_stat) > penalty:
            cp = start + max_dev_idx
            change_points.append(cp)
            segments.append((start, cp))
            segments.append((cp, end))

    change_points = sorted(set(change_points))

    # Compute segment statistics
    boundaries = [0] + change_points + [n]
    segment_stats = []
    for i in range(len(boundaries) - 1):
        seg = timeseries[boundaries[i]:boundaries[i + 1]]
        segment_stats.append({
            "start": boundaries[i],
            "end": boundaries[i + 1],
            "mean": float(np.mean(seg)),
            "std": float(np.std(seg)),
            "n": len(seg),
        })

    return {
        "change_points": change_points,
        "n_change_points": len(change_points),
        "segment_stats": segment_stats,
        "cumsum": cumsum,
    }
