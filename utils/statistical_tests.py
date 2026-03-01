"""Statistical testing utilities for A/B testing and experimentation."""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Optional


def two_sample_z_test(
    x1: np.ndarray, x2: np.ndarray, alternative: str = "two-sided"
) -> dict:
    """Two-sample z-test for comparing means of large samples.

    Args:
        x1: Control group observations.
        x2: Treatment group observations.
        alternative: 'two-sided', 'less', or 'greater'.

    Returns:
        Dict with z_stat, p_value, effect_size, ci_lower, ci_upper.
    """
    n1, n2 = len(x1), len(x2)
    mean1, mean2 = np.mean(x1), np.mean(x2)
    var1, var2 = np.var(x1, ddof=1), np.var(x2, ddof=1)
    se = np.sqrt(var1 / n1 + var2 / n2)
    z = (mean2 - mean1) / se

    if alternative == "two-sided":
        p = 2 * (1 - stats.norm.cdf(abs(z)))
    elif alternative == "greater":
        p = 1 - stats.norm.cdf(z)
    else:
        p = stats.norm.cdf(z)

    ci_lower = (mean2 - mean1) - 1.96 * se
    ci_upper = (mean2 - mean1) + 1.96 * se
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    cohens_d = (mean2 - mean1) / pooled_std if pooled_std > 0 else 0.0

    return {
        "z_stat": z,
        "p_value": p,
        "effect_size": mean2 - mean1,
        "cohens_d": cohens_d,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "control_mean": mean1,
        "treatment_mean": mean2,
        "control_n": n1,
        "treatment_n": n2,
    }


def two_sample_t_test(
    x1: np.ndarray, x2: np.ndarray, equal_var: bool = False,
    alternative: str = "two-sided"
) -> dict:
    """Two-sample t-test (Welch's by default).

    Args:
        x1: Control group observations.
        x2: Treatment group observations.
        equal_var: If True, use pooled variance (Student's t-test).
        alternative: 'two-sided', 'less', or 'greater'.

    Returns:
        Dict with t_stat, p_value, effect_size, degrees of freedom, CI.
    """
    result = stats.ttest_ind(x1, x2, equal_var=equal_var, alternative=alternative)
    n1, n2 = len(x1), len(x2)
    mean1, mean2 = np.mean(x1), np.mean(x2)
    var1, var2 = np.var(x1, ddof=1), np.var(x2, ddof=1)
    se = np.sqrt(var1 / n1 + var2 / n2)

    # Welch-Satterthwaite degrees of freedom
    if not equal_var:
        num = (var1 / n1 + var2 / n2) ** 2
        denom = (var1 / n1) ** 2 / (n1 - 1) + (var2 / n2) ** 2 / (n2 - 1)
        df = num / denom if denom > 0 else n1 + n2 - 2
    else:
        df = n1 + n2 - 2

    t_crit = stats.t.ppf(0.975, df)
    ci_lower = (mean2 - mean1) - t_crit * se
    ci_upper = (mean2 - mean1) + t_crit * se

    return {
        "t_stat": result.statistic,
        "p_value": result.pvalue,
        "effect_size": mean2 - mean1,
        "df": df,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "control_mean": mean1,
        "treatment_mean": mean2,
    }


def chi_squared_test(contingency_table: np.ndarray) -> dict:
    """Chi-squared test of independence.

    Args:
        contingency_table: 2D array of observed frequencies.

    Returns:
        Dict with chi2, p_value, dof, expected frequencies, cramers_v.
    """
    chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
    n = contingency_table.sum()
    k = min(contingency_table.shape) - 1
    cramers_v = np.sqrt(chi2 / (n * k)) if k > 0 and n > 0 else 0.0

    return {
        "chi2": chi2,
        "p_value": p,
        "dof": dof,
        "expected": expected,
        "cramers_v": cramers_v,
    }


def proportion_z_test(
    successes1: int, n1: int, successes2: int, n2: int,
    alternative: str = "two-sided"
) -> dict:
    """Two-proportion z-test.

    Args:
        successes1: Number of successes in control.
        n1: Total observations in control.
        successes2: Number of successes in treatment.
        n2: Total observations in treatment.
        alternative: 'two-sided', 'less', or 'greater'.

    Returns:
        Dict with z_stat, p_value, proportions, CI for difference.
    """
    p1 = successes1 / n1
    p2 = successes2 / n2
    p_pool = (successes1 + successes2) / (n1 + n2)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    z = (p2 - p1) / se if se > 0 else 0.0

    if alternative == "two-sided":
        p_val = 2 * (1 - stats.norm.cdf(abs(z)))
    elif alternative == "greater":
        p_val = 1 - stats.norm.cdf(z)
    else:
        p_val = stats.norm.cdf(z)

    se_diff = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    ci_lower = (p2 - p1) - 1.96 * se_diff
    ci_upper = (p2 - p1) + 1.96 * se_diff

    return {
        "z_stat": z,
        "p_value": p_val,
        "p_control": p1,
        "p_treatment": p2,
        "difference": p2 - p1,
        "relative_lift": (p2 - p1) / p1 if p1 > 0 else float("inf"),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
    }


def sample_size_proportion(
    baseline_rate: float, mde: float, alpha: float = 0.05,
    power: float = 0.8, two_sided: bool = True
) -> int:
    """Calculate required sample size per group for a proportion test.

    Args:
        baseline_rate: Expected proportion in control group.
        mde: Minimum detectable effect (absolute difference).
        alpha: Significance level.
        power: Statistical power (1 - beta).
        two_sided: Whether the test is two-sided.

    Returns:
        Required sample size per group.
    """
    p1 = baseline_rate
    p2 = baseline_rate + mde
    if two_sided:
        z_alpha = stats.norm.ppf(1 - alpha / 2)
    else:
        z_alpha = stats.norm.ppf(1 - alpha)
    z_beta = stats.norm.ppf(power)

    n = ((z_alpha * np.sqrt(2 * p1 * (1 - p1)) +
          z_beta * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) / mde) ** 2
    return int(np.ceil(n))


def sample_size_continuous(
    std: float, mde: float, alpha: float = 0.05,
    power: float = 0.8, two_sided: bool = True
) -> int:
    """Calculate required sample size per group for a continuous metric.

    Args:
        std: Expected standard deviation of the metric.
        mde: Minimum detectable effect (absolute difference in means).
        alpha: Significance level.
        power: Statistical power.
        two_sided: Whether the test is two-sided.

    Returns:
        Required sample size per group.
    """
    if two_sided:
        z_alpha = stats.norm.ppf(1 - alpha / 2)
    else:
        z_alpha = stats.norm.ppf(1 - alpha)
    z_beta = stats.norm.ppf(power)
    n = 2 * ((z_alpha + z_beta) * std / mde) ** 2
    return int(np.ceil(n))


def cuped_adjustment(
    metric_post: np.ndarray, metric_pre: np.ndarray,
    group: np.ndarray
) -> tuple[np.ndarray, float]:
    """CUPED (Controlled-experiment Using Pre-Experiment Data) variance reduction.

    Uses pre-experiment covariate to reduce variance of the treatment effect
    estimator. Returns adjusted metric values and variance reduction ratio.

    Args:
        metric_post: Post-experiment metric values.
        metric_pre: Pre-experiment covariate (same metric in pre-period).
        group: Binary array (0=control, 1=treatment).

    Returns:
        Tuple of (adjusted_metric, variance_reduction_ratio).
    """
    theta = np.cov(metric_post, metric_pre)[0, 1] / np.var(metric_pre)
    adjusted = metric_post - theta * (metric_pre - np.mean(metric_pre))
    var_original = np.var(metric_post)
    var_adjusted = np.var(adjusted)
    reduction = 1 - var_adjusted / var_original if var_original > 0 else 0.0
    return adjusted, reduction


def sequential_test(
    data: pd.DataFrame, metric_col: str, group_col: str,
    alpha: float = 0.05, spending_func: str = "obrien_fleming"
) -> list[dict]:
    """Sequential testing with alpha spending function.

    Performs interim analyses at each day in the data, adjusting significance
    thresholds to control overall Type I error.

    Args:
        data: DataFrame with daily accumulated data.
        metric_col: Column name for the metric.
        group_col: Column name for group assignment.
        alpha: Overall significance level.
        spending_func: 'obrien_fleming' or 'pocock'.

    Returns:
        List of dicts with interim analysis results.
    """
    days = sorted(data["day"].unique())
    n_looks = len(days)
    results = []

    for i, day in enumerate(days):
        info_frac = (i + 1) / n_looks
        if spending_func == "obrien_fleming":
            alpha_spent = 2 * (1 - stats.norm.cdf(
                stats.norm.ppf(1 - alpha / 2) / np.sqrt(info_frac)
            ))
        else:  # pocock
            alpha_spent = alpha * np.log(1 + (np.e - 1) * info_frac)

        subset = data[data["day"] <= day]
        control = subset[subset[group_col] == "control"][metric_col].values
        treatment = subset[subset[group_col] == "treatment"][metric_col].values

        if len(control) < 10 or len(treatment) < 10:
            continue

        test = two_sample_z_test(control, treatment)
        results.append({
            "day": day,
            "info_fraction": info_frac,
            "alpha_threshold": alpha_spent,
            "z_stat": test["z_stat"],
            "p_value": test["p_value"],
            "reject": test["p_value"] < alpha_spent,
            "effect_size": test["effect_size"],
            "n_control": len(control),
            "n_treatment": len(treatment),
        })

    return results


def bootstrap_ci(
    x: np.ndarray, stat_func=np.mean, n_bootstrap: int = 10000,
    alpha: float = 0.05, seed: int = 42
) -> dict:
    """Bootstrap confidence interval.

    Args:
        x: Data array.
        stat_func: Statistic to compute (default: mean).
        n_bootstrap: Number of bootstrap samples.
        alpha: Significance level for CI.
        seed: Random seed.

    Returns:
        Dict with observed statistic, CI bounds, and standard error.
    """
    rng = np.random.default_rng(seed)
    observed = stat_func(x)
    boot_stats = np.array([
        stat_func(rng.choice(x, size=len(x), replace=True))
        for _ in range(n_bootstrap)
    ])
    ci_lower = np.percentile(boot_stats, 100 * alpha / 2)
    ci_upper = np.percentile(boot_stats, 100 * (1 - alpha / 2))

    return {
        "observed": observed,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "se": np.std(boot_stats),
        "bootstrap_distribution": boot_stats,
    }


def multiple_testing_correction(
    p_values: np.ndarray, method: str = "bonferroni", alpha: float = 0.05
) -> dict:
    """Apply multiple testing correction.

    Args:
        p_values: Array of p-values.
        method: 'bonferroni', 'holm', or 'bh' (Benjamini-Hochberg).
        alpha: Family-wise or false discovery rate level.

    Returns:
        Dict with adjusted p-values, rejection decisions, adjusted alpha.
    """
    m = len(p_values)
    sorted_idx = np.argsort(p_values)
    sorted_p = p_values[sorted_idx]

    if method == "bonferroni":
        adjusted = np.minimum(p_values * m, 1.0)
        rejected = adjusted < alpha
    elif method == "holm":
        adjusted = np.zeros(m)
        for i, idx in enumerate(sorted_idx):
            adjusted[idx] = sorted_p[i] * (m - i)
        adjusted = np.minimum(adjusted, 1.0)
        # Enforce monotonicity
        cum_max = np.zeros(m)
        for i, idx in enumerate(sorted_idx):
            if i == 0:
                cum_max[idx] = adjusted[idx]
            else:
                cum_max[idx] = max(cum_max[sorted_idx[i - 1]], adjusted[idx])
        adjusted = cum_max
        rejected = adjusted < alpha
    elif method == "bh":
        adjusted = np.zeros(m)
        for i, idx in enumerate(sorted_idx):
            adjusted[idx] = sorted_p[i] * m / (i + 1)
        adjusted = np.minimum(adjusted, 1.0)
        # Enforce monotonicity (reverse direction)
        for i in range(m - 2, -1, -1):
            idx = sorted_idx[i]
            next_idx = sorted_idx[i + 1]
            adjusted[idx] = min(adjusted[idx], adjusted[next_idx])
        rejected = adjusted < alpha
    else:
        raise ValueError(f"Unknown method: {method}")

    return {
        "original_p_values": p_values,
        "adjusted_p_values": adjusted,
        "rejected": rejected,
        "n_rejected": int(np.sum(rejected)),
        "method": method,
    }
