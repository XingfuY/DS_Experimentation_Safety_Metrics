"""
Synthetic data generators for content safety policy A/B experiments.

Generates realistic experimental data for a TikTok-like short-video platform,
covering standard A/B tests, friction interventions, sequential monitoring,
and multivariate (multi-arm) designs. All generators produce pandas DataFrames
ready for statistical analysis with CUPED variance reduction, sequential
testing, and multiple-comparison corrections.
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_REGIONS = ["NA", "LATAM", "EMEA", "APAC", "SEA"]
_PLATFORMS = ["ios", "android", "web"]
_REGION_WEIGHTS = [0.30, 0.10, 0.20, 0.25, 0.15]
_PLATFORM_WEIGHTS = [0.45, 0.40, 0.15]


def _assign_demographics(rng: np.random.Generator, n: int) -> dict:
    """Return dict of demographic arrays shared across generators."""
    region = rng.choice(_REGIONS, size=n, p=_REGION_WEIGHTS)
    platform = rng.choice(_PLATFORMS, size=n, p=_PLATFORM_WEIGHTS)
    account_age_days = rng.lognormal(mean=5.5, sigma=1.0, size=n).astype(int)
    account_age_days = np.clip(account_age_days, 1, 3650)
    # Older accounts more likely to be creators
    creator_prob = np.clip(0.05 + 0.10 * (account_age_days / 3650), 0.02, 0.20)
    is_creator = rng.binomial(1, creator_prob).astype(bool)
    return {
        "region": region,
        "platform": platform,
        "account_age_days": account_age_days,
        "is_creator": is_creator,
    }


# ---------------------------------------------------------------------------
# 1. Standard A/B test — AI content label
# ---------------------------------------------------------------------------

def generate_ab_test_data(
    n_users: int = 10_000,
    effect_size: float = 0.05,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic A/B test data for an AI content label feature.

    The treatment group sees explicit "AI-generated" labels on synthetic
    content.  The hypothesis is that transparency increases user trust but
    slightly reduces engagement (users scroll past labelled content more
    often).

    Parameters
    ----------
    n_users : int
        Total number of users split 50/50 between control and treatment.
    effect_size : float
        Absolute lift in trust score for the treatment group (on a 0-1 scale).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        One row per user with pre/post metrics, covariates, and group
        assignment.  Suitable for CUPED analysis (pre-period covariates
        provided).
    """
    rng = np.random.default_rng(seed)
    n_treat = n_users // 2
    n_ctrl = n_users - n_treat

    groups = np.array(["control"] * n_ctrl + ["treatment"] * n_treat)
    rng.shuffle(groups)
    is_treatment = groups == "treatment"

    demos = _assign_demographics(rng, n_users)

    # --- Pre-experiment covariates (identical distribution for both arms) ---
    # Base engagement drawn from a Beta so it lives in (0, 1)
    user_quality = rng.normal(0, 1, n_users)  # latent user factor
    pre_engagement = np.clip(
        rng.beta(2, 5, n_users) + 0.02 * user_quality, 0.01, 0.99
    )
    pre_trust = np.clip(
        rng.beta(5, 3, n_users) + 0.03 * user_quality, 0.05, 0.99
    )

    # --- Post-experiment metrics ---
    # Natural drift (small)
    drift_engagement = rng.normal(0.0, 0.02, n_users)
    drift_trust = rng.normal(0.0, 0.02, n_users)

    # Heterogeneous treatment effect: newer users respond more
    age_modifier = np.clip(
        1.5 - demos["account_age_days"] / 1500, 0.5, 2.0
    )
    # Individual-level noise on the treatment effect
    individual_te_trust = rng.normal(effect_size, effect_size * 0.4, n_users)
    individual_te_engagement = rng.normal(-0.02, 0.01, n_users)

    post_trust = pre_trust + drift_trust
    post_engagement = pre_engagement + drift_engagement

    post_trust[is_treatment] += (
        individual_te_trust[is_treatment] * age_modifier[is_treatment]
    )
    post_engagement[is_treatment] += (
        individual_te_engagement[is_treatment] * age_modifier[is_treatment]
    )

    post_trust = np.clip(post_trust, 0.0, 1.0)
    post_engagement = np.clip(post_engagement, 0.0, 1.0)

    # --- Behavioural counters ---
    session_base = rng.poisson(12, n_users) + 1
    time_base = rng.gamma(3, 8, n_users)  # minutes
    content_created = rng.poisson(
        np.where(demos["is_creator"], 5, 0.3), n_users
    )
    content_shared = rng.poisson(2, n_users)

    # Violations: low-base-rate event (~2 %)
    violation_rate = np.clip(0.02 - 0.005 * user_quality, 0.001, 0.10)
    violations_reported = rng.poisson(violation_rate * session_base)
    # ~60 % of reported violations are confirmed
    violations_confirmed = rng.binomial(violations_reported, 0.60)

    # AI content interaction (treatment only)
    ai_content_interacted = np.zeros(n_users, dtype=int)
    ai_content_interacted[is_treatment] = rng.poisson(3, n_treat)

    df = pd.DataFrame(
        {
            "user_id": np.arange(1, n_users + 1),
            "group": groups,
            "pre_engagement_rate": np.round(pre_engagement, 4),
            "post_engagement_rate": np.round(post_engagement, 4),
            "pre_trust_score": np.round(pre_trust, 4),
            "post_trust_score": np.round(post_trust, 4),
            "sessions": session_base,
            "time_spent_minutes": np.round(time_base, 1),
            "content_created": content_created,
            "content_shared": content_shared,
            "violations_reported": violations_reported,
            "violations_confirmed": violations_confirmed,
            "ai_content_interacted": ai_content_interacted,
            "region": demos["region"],
            "platform": demos["platform"],
            "account_age_days": demos["account_age_days"],
            "is_creator": demos["is_creator"],
        }
    )
    return df


# ---------------------------------------------------------------------------
# 2. Friction warning test
# ---------------------------------------------------------------------------

def generate_friction_warning_test(
    n_users: int = 8_000,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate data for a friction-warning intervention experiment.

    Treatment users see an interstitial warning ("This may violate community
    guidelines — are you sure?") before their post is published when a
    lightweight classifier flags the content.  The warning reduces confirmed
    violations by ~30 % but increases short-term churn by ~5 %.

    Heterogeneous effects: users with accounts < 90 days old respond more
    strongly to the warning (larger violation reduction, but also higher
    churn sensitivity).

    Parameters
    ----------
    n_users : int
        Total users, split 50/50.
    seed : int
        Random seed.

    Returns
    -------
    pd.DataFrame
        One row per user with posting behaviour, warning outcomes, violation
        counts, satisfaction, and churn indicators.
    """
    rng = np.random.default_rng(seed)
    n_treat = n_users // 2
    n_ctrl = n_users - n_treat

    groups = np.array(["control"] * n_ctrl + ["treatment"] * n_treat)
    rng.shuffle(groups)
    is_treatment = groups == "treatment"

    demos = _assign_demographics(rng, n_users)
    is_new_user = demos["account_age_days"] < 90

    # --- Posting behaviour ---
    posts_attempted = rng.poisson(8, n_users) + 1

    # Warning trigger rate ~ 15 % of attempted posts for treatment
    warning_rate = np.where(is_new_user, 0.20, 0.12)
    posts_with_warning = np.zeros(n_users, dtype=int)
    posts_with_warning[is_treatment] = rng.binomial(
        posts_attempted[is_treatment],
        warning_rate[is_treatment],
    )

    # Probability of heeding the warning (not posting)
    heed_prob = np.where(is_new_user, 0.55, 0.35)
    warnings_heeded_count = np.zeros(n_users, dtype=int)
    warnings_heeded_count[is_treatment] = rng.binomial(
        posts_with_warning[is_treatment],
        heed_prob[is_treatment],
    )
    # Boolean: did user heed at least one warning?
    warning_heeded = warnings_heeded_count > 0

    # Posts completed = attempted minus heeded warnings
    posts_completed = posts_attempted - warnings_heeded_count

    # --- Violations ---
    base_violation_rate = np.where(is_new_user, 0.06, 0.03)
    violations_flagged_ctrl = rng.binomial(
        posts_completed, base_violation_rate
    )
    # Treatment reduces violations (the 30 % effect comes from heeding +
    # behavioural shift even among those who post anyway)
    treatment_violation_rate = base_violation_rate * np.where(
        is_new_user, 0.60, 0.75
    )
    violations_flagged_treat = rng.binomial(
        posts_completed, treatment_violation_rate
    )
    violations_flagged = np.where(
        is_treatment, violations_flagged_treat, violations_flagged_ctrl
    )
    violations_confirmed = rng.binomial(violations_flagged, 0.65)

    # --- Satisfaction (1-5 Likert) ---
    base_satisfaction = rng.normal(3.5, 0.8, n_users)
    # Warning introduces mild annoyance
    base_satisfaction[is_treatment] -= 0.15
    # New users who got warnings are slightly more annoyed
    base_satisfaction[is_treatment & is_new_user] -= 0.10
    satisfaction = np.clip(np.round(base_satisfaction), 1, 5).astype(int)

    # --- Churn ---
    base_churn_7d = 0.08
    base_churn_30d = 0.15
    churn_7d_prob = np.full(n_users, base_churn_7d)
    churn_30d_prob = np.full(n_users, base_churn_30d)

    # Treatment lifts churn ~5 pp overall; new users +8 pp
    churn_7d_prob[is_treatment] += np.where(
        is_new_user[is_treatment], 0.08, 0.04
    )
    churn_30d_prob[is_treatment] += np.where(
        is_new_user[is_treatment], 0.10, 0.05
    )
    churn_7d = rng.binomial(1, np.clip(churn_7d_prob, 0, 1)).astype(bool)
    churn_30d = rng.binomial(1, np.clip(churn_30d_prob, 0, 1)).astype(bool)
    # If churned at 7d, definitely churned at 30d
    churn_30d = churn_30d | churn_7d

    df = pd.DataFrame(
        {
            "user_id": np.arange(1, n_users + 1),
            "group": groups,
            "posts_attempted": posts_attempted,
            "posts_completed": posts_completed,
            "posts_with_warning": posts_with_warning,
            "warning_heeded": warning_heeded,
            "violations_flagged": violations_flagged,
            "violations_confirmed": violations_confirmed,
            "user_satisfaction_score": satisfaction,
            "churn_7d": churn_7d,
            "churn_30d": churn_30d,
            "region": demos["region"],
            "platform": demos["platform"],
            "account_age_days": demos["account_age_days"],
            "is_creator": demos["is_creator"],
        }
    )
    return df


# ---------------------------------------------------------------------------
# 3. Sequential test data (daily accumulation)
# ---------------------------------------------------------------------------

def generate_sequential_test_data(
    n_users_per_day: int = 500,
    n_days: int = 30,
    effect_size: float = 0.03,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate daily-accruing A/B test data for sequential analysis.

    Designed for group-sequential or always-valid inference methods.
    The true treatment effect ramps up over the first 7 days (novelty
    period) before stabilising, mimicking real platform dynamics where
    users habituate to a new feature.

    Parameters
    ----------
    n_users_per_day : int
        New users entering the experiment each day (split 50/50).
    n_days : int
        Duration of the experiment in days.
    effect_size : float
        Steady-state treatment effect on the primary metric (after novelty
        period subsides).
    seed : int
        Random seed.

    Returns
    -------
    pd.DataFrame
        One row per user per day of exposure.  Columns include day,
        user_id, group, metric_value, and cumulative_users.
    """
    rng = np.random.default_rng(seed)

    records = []
    user_counter = 0
    cumulative_users = 0

    for day in range(1, n_days + 1):
        n_today = n_users_per_day + rng.integers(-50, 51)  # slight daily jitter
        n_today = max(n_today, 100)
        cumulative_users += n_today

        groups = rng.choice(
            ["control", "treatment"], size=n_today, p=[0.5, 0.5]
        )
        is_treatment = groups == "treatment"

        # Novelty curve: effect is amplified in first 7 days, then settles
        if day <= 7:
            # Novelty inflates the apparent effect (users engage more with
            # something new), peaking around day 3-4
            novelty_multiplier = 1.0 + 1.2 * np.exp(-0.3 * (day - 3.5) ** 2)
        else:
            novelty_multiplier = 1.0

        day_effect = effect_size * novelty_multiplier

        # Base metric ~ N(0.50, 0.15)
        base = rng.normal(0.50, 0.15, n_today)

        # Add small day-of-week seasonality (weekends slightly higher)
        dow = day % 7
        if dow in (0, 6):
            base += 0.02

        metric = base.copy()
        # Treatment effect with individual-level noise
        te = rng.normal(day_effect, day_effect * 0.5, n_today)
        metric[is_treatment] += te[is_treatment]

        user_ids = np.arange(
            user_counter + 1, user_counter + n_today + 1
        )
        user_counter += n_today

        day_df = pd.DataFrame(
            {
                "day": day,
                "user_id": user_ids,
                "group": groups,
                "metric_value": np.round(metric, 4),
                "cumulative_users": cumulative_users,
            }
        )
        records.append(day_df)

    df = pd.concat(records, ignore_index=True)
    return df


# ---------------------------------------------------------------------------
# 4. Multivariate (multi-arm) test
# ---------------------------------------------------------------------------

def generate_multivariate_test(
    n_users: int = 15_000,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate multi-arm experiment data for content safety interventions.

    Four arms are compared:
    - **control**: No intervention.
    - **mild_warning**: Small, dismissible banner ("Reminder: check
      community guidelines").
    - **strong_warning**: Full-screen interstitial requiring acknowledgement.
    - **educational_prompt**: Short educational tooltip explaining *why*
      the content may be problematic.

    Each arm has a different trade-off between violation reduction and
    engagement / satisfaction impact.

    Parameters
    ----------
    n_users : int
        Total users, split roughly equally across four arms.
    seed : int
        Random seed.

    Returns
    -------
    pd.DataFrame
        One row per user with pre/post violation and engagement metrics
        plus satisfaction.  Pre-period columns enable CUPED adjustment.
    """
    rng = np.random.default_rng(seed)

    arms = ["control", "mild_warning", "strong_warning", "educational_prompt"]
    arm_assignment = rng.choice(arms, size=n_users)

    # Effect sizes (relative reduction in violations, absolute change in
    # engagement, absolute change in satisfaction on 1-5 scale)
    #                          violation_rr  engagement_delta  satisfaction_delta
    arm_effects = {
        "control":             (1.00,          0.00,             0.00),
        "mild_warning":        (0.88,         -0.005,           -0.05),
        "strong_warning":      (0.65,         -0.03,            -0.30),
        "educational_prompt":  (0.78,         -0.01,            +0.10),
    }

    # --- Pre-experiment (identical across arms by randomisation) ---
    user_quality = rng.normal(0, 1, n_users)
    pre_violations = rng.poisson(
        np.clip(2.0 - 0.3 * user_quality, 0.2, 6.0)
    )
    pre_engagement = np.clip(
        rng.beta(3, 4, n_users) + 0.03 * user_quality, 0.01, 0.99
    )

    # --- Post-experiment ---
    post_violations = np.zeros(n_users, dtype=int)
    post_engagement = np.zeros(n_users, dtype=float)
    satisfaction = np.zeros(n_users, dtype=float)

    for arm in arms:
        mask = arm_assignment == arm
        n_arm = mask.sum()
        viol_rr, eng_delta, sat_delta = arm_effects[arm]

        # Violations: Poisson with rate scaled by relative risk
        base_rate = np.clip(2.0 - 0.3 * user_quality[mask], 0.2, 6.0)
        post_rate = base_rate * viol_rr
        # Add individual-level noise to the relative risk
        individual_rr_noise = rng.normal(1.0, 0.1, n_arm)
        post_rate = post_rate * np.clip(individual_rr_noise, 0.5, 1.5)
        post_violations[mask] = rng.poisson(np.clip(post_rate, 0.05, 10.0))

        # Engagement
        individual_eng_noise = rng.normal(0, 0.02, n_arm)
        post_engagement[mask] = np.clip(
            pre_engagement[mask] + eng_delta + individual_eng_noise,
            0.0, 1.0,
        )

        # Satisfaction (1-5)
        base_sat = rng.normal(3.5 + sat_delta, 0.7, n_arm)
        satisfaction[mask] = np.clip(np.round(base_sat), 1, 5)

    df = pd.DataFrame(
        {
            "user_id": np.arange(1, n_users + 1),
            "arm": arm_assignment,
            "pre_violations": pre_violations,
            "violations_post": post_violations,
            "pre_engagement": np.round(pre_engagement, 4),
            "engagement_post": np.round(post_engagement, 4),
            "satisfaction": satisfaction.astype(int),
        }
    )
    return df


# ---------------------------------------------------------------------------
# CLI convenience
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Generating policy experiment datasets...\n")

    ab = generate_ab_test_data()
    print(f"AB test:          {ab.shape[0]:>6,} rows, {ab.shape[1]} cols")
    print(f"  Treatment trust lift:  "
          f"{ab.loc[ab.group == 'treatment', 'post_trust_score'].mean() - ab.loc[ab.group == 'control', 'post_trust_score'].mean():.4f}")
    print(f"  Treatment engagement:  "
          f"{ab.loc[ab.group == 'treatment', 'post_engagement_rate'].mean() - ab.loc[ab.group == 'control', 'post_engagement_rate'].mean():.4f}")

    fw = generate_friction_warning_test()
    print(f"\nFriction warning: {fw.shape[0]:>6,} rows, {fw.shape[1]} cols")
    ctrl_viol = fw.loc[fw.group == "control", "violations_confirmed"].mean()
    treat_viol = fw.loc[fw.group == "treatment", "violations_confirmed"].mean()
    print(f"  Violation reduction:   {1 - treat_viol / ctrl_viol:.1%}")
    print(f"  Churn 7d lift:         "
          f"{fw.loc[fw.group == 'treatment', 'churn_7d'].mean() - fw.loc[fw.group == 'control', 'churn_7d'].mean():.3f}")

    seq = generate_sequential_test_data()
    print(f"\nSequential test:  {seq.shape[0]:>6,} rows, {seq.shape[1]} cols")
    print(f"  Days: {seq.day.nunique()}, Users: {seq.user_id.nunique():,}")

    mv = generate_multivariate_test()
    print(f"\nMultivariate:     {mv.shape[0]:>6,} rows, {mv.shape[1]} cols")
    print("  Mean violations_post by arm:")
    print(mv.groupby("arm")["violations_post"].mean().to_string(float_format="    %.3f".format))

    print("\nDone.")
