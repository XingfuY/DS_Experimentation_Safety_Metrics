"""Causal inference estimators implemented from scratch with statsmodels."""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from typing import Optional


def difference_in_differences(
    df: pd.DataFrame, outcome_col: str, treat_col: str, post_col: str,
    cluster_col: Optional[str] = None
) -> dict:
    """Difference-in-Differences estimator with optional clustered SEs.

    Args:
        df: DataFrame with outcome, treatment, and post-period indicators.
        outcome_col: Column name for the outcome variable.
        treat_col: Column name for treatment group indicator (0/1).
        post_col: Column name for post-period indicator (0/1).
        cluster_col: Column name for clustering standard errors.

    Returns:
        Dict with DID estimate, standard error, p-value, CI, and model summary.
    """
    df = df.copy()
    df["treat_x_post"] = df[treat_col] * df[post_col]
    X = sm.add_constant(df[[treat_col, post_col, "treat_x_post"]])
    y = df[outcome_col]

    if cluster_col is not None:
        model = sm.OLS(y, X).fit(
            cov_type="cluster", cov_kwds={"groups": df[cluster_col]}
        )
    else:
        model = sm.OLS(y, X).fit(cov_type="HC1")

    did_coef = model.params["treat_x_post"]
    did_se = model.bse["treat_x_post"]
    did_p = model.pvalues["treat_x_post"]
    ci = model.conf_int().loc["treat_x_post"]

    return {
        "did_estimate": did_coef,
        "se": did_se,
        "p_value": did_p,
        "ci_lower": ci[0],
        "ci_upper": ci[1],
        "pre_control_mean": df[(df[treat_col] == 0) & (df[post_col] == 0)][outcome_col].mean(),
        "pre_treat_mean": df[(df[treat_col] == 1) & (df[post_col] == 0)][outcome_col].mean(),
        "post_control_mean": df[(df[treat_col] == 0) & (df[post_col] == 1)][outcome_col].mean(),
        "post_treat_mean": df[(df[treat_col] == 1) & (df[post_col] == 1)][outcome_col].mean(),
        "model": model,
    }


def parallel_trends_test(
    df: pd.DataFrame, outcome_col: str, treat_col: str, time_col: str,
    n_pre_periods: int = 4
) -> dict:
    """Test parallel trends assumption for DID.

    Estimates treatment-time interactions in pre-period. Under parallel trends,
    these should be jointly insignificant.

    Args:
        df: Panel data with time periods.
        outcome_col: Outcome variable.
        treat_col: Treatment group indicator.
        time_col: Time period column (integer).
        n_pre_periods: Number of pre-treatment periods to test.

    Returns:
        Dict with F-test result and period-specific estimates.
    """
    df = df.copy().reset_index(drop=True)
    periods = sorted(df[time_col].unique())
    pre_periods = periods[:n_pre_periods]

    # Create time dummies interacted with treatment
    interaction_cols = []
    for t in pre_periods[1:]:  # omit first period as reference
        col = f"treat_x_t{t}"
        df[col] = (df[treat_col] * (df[time_col] == t)).astype(float)
        interaction_cols.append(col)

    time_dummies = pd.get_dummies(df[time_col], prefix="t", drop_first=True).astype(float)
    X = pd.concat([
        df[[treat_col]].rename(columns={treat_col: "treat"}),
        time_dummies,
        df[interaction_cols],
    ], axis=1)
    X = sm.add_constant(X)
    y = df[outcome_col]

    model = sm.OLS(y, X).fit(cov_type="HC1")

    # F-test for joint significance of interaction terms
    r_matrix = np.zeros((len(interaction_cols), len(model.params)))
    for i, col in enumerate(interaction_cols):
        j = list(model.params.index).index(col)
        r_matrix[i, j] = 1
    f_test = model.f_test(r_matrix)

    period_estimates = {
        col: {"coef": model.params[col], "se": model.bse[col], "p": model.pvalues[col]}
        for col in interaction_cols
    }

    return {
        "f_stat": float(f_test.fvalue),
        "f_pvalue": float(f_test.pvalue),
        "parallel_trends_holds": float(f_test.pvalue) > 0.05,
        "period_estimates": period_estimates,
        "model": model,
    }


def regression_discontinuity(
    df: pd.DataFrame, outcome_col: str, running_col: str, cutoff: float = 0,
    bandwidth: Optional[float] = None, kernel: str = "triangular",
    polynomial_order: int = 1
) -> dict:
    """Sharp Regression Discontinuity Design.

    Args:
        df: DataFrame with outcome and running variable.
        outcome_col: Outcome variable.
        running_col: Running variable (treatment assigned when >= cutoff).
        cutoff: Cutoff value for the running variable.
        bandwidth: Bandwidth for local regression. If None, uses IK bandwidth.
        kernel: 'triangular', 'uniform', or 'epanechnikov'.
        polynomial_order: Order of local polynomial (1=linear, 2=quadratic).

    Returns:
        Dict with RDD estimate, SE, p-value, CI, and diagnostics.
    """
    df = df.copy()
    df["centered"] = df[running_col] - cutoff
    df["treated"] = (df["centered"] >= 0).astype(float)

    # IK bandwidth if not specified
    if bandwidth is None:
        bandwidth = 1.06 * df["centered"].std() * len(df) ** (-1 / 5)

    # Filter to bandwidth
    mask = df["centered"].abs() <= bandwidth
    df_bw = df[mask].copy()

    # Kernel weights
    if kernel == "triangular":
        df_bw["weight"] = (1 - df_bw["centered"].abs() / bandwidth).clip(lower=0)
    elif kernel == "epanechnikov":
        u = (df_bw["centered"] / bandwidth).values
        df_bw["weight"] = np.maximum(0.75 * (1 - u ** 2), 0)
    else:
        df_bw["weight"] = 1.0

    # Build polynomial terms
    terms = {"treated": df_bw["treated"]}
    for p in range(1, polynomial_order + 1):
        terms[f"centered_{p}"] = df_bw["centered"] ** p
        terms[f"treated_centered_{p}"] = df_bw["treated"] * df_bw["centered"] ** p

    X = sm.add_constant(pd.DataFrame(terms))
    y = df_bw[outcome_col]

    model = sm.WLS(y, X, weights=df_bw["weight"]).fit(cov_type="HC1")

    return {
        "rdd_estimate": model.params["treated"],
        "se": model.bse["treated"],
        "p_value": model.pvalues["treated"],
        "ci_lower": model.conf_int().loc["treated"][0],
        "ci_upper": model.conf_int().loc["treated"][1],
        "bandwidth": bandwidth,
        "n_left": int((df_bw["treated"] == 0).sum()),
        "n_right": int((df_bw["treated"] == 1).sum()),
        "model": model,
    }


def mccrary_density_test(
    running_var: np.ndarray, cutoff: float = 0, n_bins: int = 50
) -> dict:
    """McCrary density test for manipulation of the running variable.

    Tests whether there is a discontinuity in the density of the running
    variable at the cutoff, which would suggest manipulation.

    Args:
        running_var: Running variable values.
        cutoff: Cutoff value.
        n_bins: Number of histogram bins.

    Returns:
        Dict with test statistic, p-value, and bin counts.
    """
    centered = running_var - cutoff
    bins = np.linspace(centered.min(), centered.max(), n_bins + 1)
    counts, _ = np.histogram(centered, bins=bins)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    left_mask = bin_centers < 0
    right_mask = bin_centers >= 0

    left_counts = counts[left_mask]
    right_counts = counts[right_mask]

    # Compare density just left and right of cutoff
    n_near = max(3, n_bins // 10)
    left_near = left_counts[-n_near:] if len(left_counts) >= n_near else left_counts
    right_near = right_counts[:n_near] if len(right_counts) >= n_near else right_counts

    left_density = left_near.mean()
    right_density = right_near.mean()
    se = np.sqrt(left_near.var() / len(left_near) + right_near.var() / len(right_near))
    t_stat = (right_density - left_density) / se if se > 0 else 0
    p_value = 2 * (1 - stats.norm.cdf(abs(t_stat)))

    return {
        "t_stat": t_stat,
        "p_value": p_value,
        "no_manipulation": p_value > 0.05,
        "left_density": left_density,
        "right_density": right_density,
        "bin_centers": bin_centers,
        "counts": counts,
    }


def instrumental_variables_2sls(
    df: pd.DataFrame, outcome_col: str, endogenous_col: str,
    instrument_col: str, covariates: Optional[list[str]] = None
) -> dict:
    """Two-Stage Least Squares (2SLS) instrumental variables estimator.

    Args:
        df: DataFrame.
        outcome_col: Outcome variable (Y).
        endogenous_col: Endogenous treatment variable (D).
        instrument_col: Instrument (Z).
        covariates: Optional list of exogenous control variables.

    Returns:
        Dict with IV estimate, first-stage F-stat, and diagnostics.
    """
    exog_cols = [instrument_col]
    if covariates:
        exog_cols += covariates

    # First stage: D = a + b*Z + controls + e
    X_first = sm.add_constant(df[exog_cols])
    first_stage = sm.OLS(df[endogenous_col], X_first).fit()
    first_stage_f = first_stage.fvalue
    d_hat = first_stage.fittedvalues

    # Second stage: Y = a + b*D_hat + controls + u
    second_stage_vars = ["d_hat"]
    df_2s = df.copy()
    df_2s["d_hat"] = d_hat
    if covariates:
        second_stage_vars += covariates
    X_second = sm.add_constant(df_2s[second_stage_vars])
    second_stage = sm.OLS(df_2s[outcome_col], X_second).fit()

    # Correct standard errors (use original D, not D_hat)
    resid = df[outcome_col] - second_stage.params["const"] - second_stage.params["d_hat"] * df[endogenous_col]
    if covariates:
        for c in covariates:
            resid -= second_stage.params[c] * df[c]
    sigma2 = (resid ** 2).sum() / (len(df) - len(second_stage.params))
    XtX_inv = np.linalg.inv(X_second.T @ X_second)
    corrected_se = np.sqrt(sigma2 * np.diag(XtX_inv))
    iv_se = corrected_se[list(X_second.columns).index("d_hat")]
    iv_estimate = second_stage.params["d_hat"]
    t_stat = iv_estimate / iv_se
    p_value = 2 * (1 - stats.norm.cdf(abs(t_stat)))

    return {
        "iv_estimate": iv_estimate,
        "se": iv_se,
        "t_stat": t_stat,
        "p_value": p_value,
        "ci_lower": iv_estimate - 1.96 * iv_se,
        "ci_upper": iv_estimate + 1.96 * iv_se,
        "first_stage_f": first_stage_f,
        "weak_instrument": first_stage_f < 10,
        "first_stage_coef": first_stage.params[instrument_col],
        "first_stage_model": first_stage,
        "second_stage_model": second_stage,
    }


def propensity_score_matching(
    df: pd.DataFrame, treatment_col: str, outcome_col: str,
    covariates: list[str], n_neighbors: int = 1, caliper: Optional[float] = None
) -> dict:
    """Propensity Score Matching with nearest-neighbor matching.

    Args:
        df: DataFrame.
        treatment_col: Binary treatment indicator.
        outcome_col: Outcome variable.
        covariates: List of covariate columns for propensity model.
        n_neighbors: Number of nearest neighbors to match.
        caliper: Maximum propensity score distance for a valid match.

    Returns:
        Dict with ATT estimate, balance diagnostics, matched sample.
    """
    # Estimate propensity scores
    X = df[covariates].values
    y = df[treatment_col].values
    ps_model = LogisticRegression(max_iter=1000, random_state=42)
    ps_model.fit(X, y)
    df = df.copy()
    df["propensity_score"] = ps_model.predict_proba(X)[:, 1]

    # Match treated to control
    treated = df[df[treatment_col] == 1]
    control = df[df[treatment_col] == 0]

    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
    nn.fit(control[["propensity_score"]].values)
    distances, indices = nn.kneighbors(treated[["propensity_score"]].values)

    # Apply caliper
    if caliper is not None:
        valid = distances[:, 0] <= caliper
        treated_matched = treated[valid]
        matched_control_idx = indices[valid]
    else:
        treated_matched = treated
        matched_control_idx = indices

    control_matched = control.iloc[matched_control_idx[:, 0]]

    # ATT
    att = treated_matched[outcome_col].mean() - control_matched[outcome_col].values.mean()
    se = np.sqrt(
        treated_matched[outcome_col].var() / len(treated_matched) +
        control_matched[outcome_col].values.var() / len(control_matched)
    )
    t_stat = att / se if se > 0 else 0
    p_value = 2 * (1 - stats.norm.cdf(abs(t_stat)))

    # Balance diagnostics (standardized mean differences)
    balance = {}
    for cov in covariates:
        treated_mean = treated_matched[cov].mean()
        control_mean = control_matched[cov].values.mean()
        pooled_std = np.sqrt(
            (treated_matched[cov].var() + control.loc[control_matched.index, cov].var()) / 2
        )
        smd = (treated_mean - control_mean) / pooled_std if pooled_std > 0 else 0
        balance[cov] = {
            "treated_mean": treated_mean,
            "control_mean": control_mean,
            "smd": abs(smd),
            "balanced": abs(smd) < 0.1,
        }

    return {
        "att": att,
        "se": se,
        "t_stat": t_stat,
        "p_value": p_value,
        "ci_lower": att - 1.96 * se,
        "ci_upper": att + 1.96 * se,
        "n_treated_matched": len(treated_matched),
        "n_treated_total": len(treated),
        "balance": balance,
        "all_balanced": all(b["balanced"] for b in balance.values()),
        "propensity_scores": df["propensity_score"].values,
    }


def inverse_probability_weighting(
    df: pd.DataFrame, treatment_col: str, outcome_col: str,
    covariates: list[str]
) -> dict:
    """Inverse Probability Weighting (IPW) estimator for ATE.

    Args:
        df: DataFrame.
        treatment_col: Binary treatment indicator.
        outcome_col: Outcome variable.
        covariates: Covariate columns for propensity model.

    Returns:
        Dict with ATE estimate, SE, and diagnostics.
    """
    X = df[covariates].values
    y = df[treatment_col].values
    ps_model = LogisticRegression(max_iter=1000, random_state=42)
    ps_model.fit(X, y)
    ps = ps_model.predict_proba(X)[:, 1]

    # Trim extreme propensity scores
    ps = np.clip(ps, 0.01, 0.99)

    treated = df[treatment_col].values
    outcome = df[outcome_col].values

    # Horvitz-Thompson estimator
    ate_treated = (treated * outcome / ps).mean()
    ate_control = ((1 - treated) * outcome / (1 - ps)).mean()
    ate = ate_treated - ate_control

    # Bootstrap SE
    rng = np.random.default_rng(42)
    n = len(df)
    boot_ates = []
    for _ in range(1000):
        idx = rng.choice(n, size=n, replace=True)
        t_b = treated[idx]
        o_b = outcome[idx]
        ps_b = ps[idx]
        ate_b = (t_b * o_b / ps_b).mean() - ((1 - t_b) * o_b / (1 - ps_b)).mean()
        boot_ates.append(ate_b)
    se = np.std(boot_ates)

    return {
        "ate": ate,
        "se": se,
        "p_value": 2 * (1 - stats.norm.cdf(abs(ate / se))) if se > 0 else 1.0,
        "ci_lower": ate - 1.96 * se,
        "ci_upper": ate + 1.96 * se,
        "propensity_scores": ps,
        "effective_sample_size": 1 / np.sum((1 / ps[treated == 1]) ** 2) * (np.sum(1 / ps[treated == 1])) ** 2,
    }
