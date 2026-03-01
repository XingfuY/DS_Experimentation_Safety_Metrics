"""Synthetic data generator for content moderation scenarios on a short-video platform.

Generates three datasets suitable for causal inference analysis:
  1. Content stream with classifier performance and moderation actions
  2. Before/after policy change for difference-in-differences and RDD
  3. Creator cohorts for propensity score matching

All data is synthetic. Seeds are fixed for reproducibility.
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONTENT_TYPES = ["video", "live", "comment"]
CONTENT_TYPE_WEIGHTS = [0.60, 0.10, 0.30]

CATEGORIES = ["entertainment", "education", "news", "commerce", "other"]
CATEGORY_WEIGHTS = [0.40, 0.15, 0.20, 0.15, 0.10]

VIOLATION_TYPES = [
    "hate_speech",
    "misinformation",
    "spam",
    "nudity",
    "violence",
    "copyright",
]
VIOLATION_TYPE_WEIGHTS = [0.25, 0.20, 0.20, 0.15, 0.10, 0.10]

REGIONS = ["US", "EU", "APAC", "LATAM"]
REGION_WEIGHTS = [0.35, 0.25, 0.25, 0.15]

MODERATION_ACTIONS = ["warning", "remove", "restrict", "ban"]
MODERATION_ACTION_WEIGHTS = [0.35, 0.40, 0.15, 0.10]


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _clamp(arr: np.ndarray, lo: float = 0.0, hi: float = 1.0) -> np.ndarray:
    """Clip array values to [lo, hi]."""
    return np.clip(arr, lo, hi)


def _generate_timestamps(
    rng: np.random.Generator, n: int, start: str = "2025-01-01", days: int = 90
) -> pd.DatetimeIndex:
    """Generate uniformly distributed timestamps over a date range."""
    start_ts = pd.Timestamp(start)
    offsets = pd.to_timedelta(rng.uniform(0, days * 24 * 3600, size=n), unit="s")
    return pd.DatetimeIndex(start_ts + offsets)


# ---------------------------------------------------------------------------
# 1. Content stream
# ---------------------------------------------------------------------------


def generate_content_stream(
    n_items: int = 50_000,
    violation_rate: float = 0.03,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a content-moderation stream with classifier predictions.

    Simulates content items flowing through a moderation pipeline on a
    short-video platform.  Each item has ground-truth violation labels,
    a calibrated classifier score, engagement metrics, and the resulting
    moderation action.

    Classifier performance targets:
        - Precision ~85 %
        - Recall    ~90 %

    Parameters
    ----------
    n_items : int
        Number of content items to generate.
    violation_rate : float
        Base probability that any given item is a true violation.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        One row per content item with columns described below.

    Columns
    -------
    content_id : str
        Unique content identifier (e.g. ``"C00001"``).
    user_id : str
        Creator identifier; ~20 % of n_items unique users.
    timestamp : datetime64[ns]
        When the content was posted (within a 90-day window).
    content_type : str
        One of ``video``, ``live``, ``comment``.
    category : str
        One of ``entertainment``, ``education``, ``news``, ``commerce``,
        ``other``.
    violation_type : str
        Ground-truth violation type or ``"none"``.
    is_violation : bool
        Whether the item is a true violation.
    classifier_score : float
        Model confidence in [0, 1], calibrated higher for violations.
    classifier_prediction : bool
        Binary prediction at threshold 0.5.
    true_positive, false_positive, false_negative, true_negative : bool
        Confusion-matrix flags.
    reports_count : int
        Number of user reports (higher for violations).
    views, likes, shares, comments : int
        Engagement metrics; violations receive fewer likes but more
        reports.
    moderation_action : str
        Action taken: ``none``, ``warning``, ``remove``, ``restrict``,
        or ``ban``.
    time_to_action_hours : float
        Hours between posting and moderation action (NaN when action is
        ``none``).
    """
    rng = np.random.default_rng(seed)
    n = n_items

    # --- identifiers -------------------------------------------------------
    content_ids = np.array([f"C{i:05d}" for i in range(1, n + 1)])
    n_users = max(1, int(n * 0.20))
    user_pool = np.array([f"U{i:05d}" for i in range(1, n_users + 1)])
    user_ids = rng.choice(user_pool, size=n)

    # --- timestamps --------------------------------------------------------
    timestamps = _generate_timestamps(rng, n)

    # --- content metadata --------------------------------------------------
    content_types = rng.choice(CONTENT_TYPES, size=n, p=CONTENT_TYPE_WEIGHTS)
    categories = rng.choice(CATEGORIES, size=n, p=CATEGORY_WEIGHTS)

    # --- ground-truth violations -------------------------------------------
    # Violation rate varies by category: news and commerce slightly higher
    category_violation_multiplier = {
        "entertainment": 0.8,
        "education": 0.5,
        "news": 1.6,
        "commerce": 1.4,
        "other": 1.0,
    }
    per_item_rate = np.array(
        [violation_rate * category_violation_multiplier[c] for c in categories]
    )
    is_violation = rng.random(n) < per_item_rate

    violation_types = np.where(
        is_violation,
        rng.choice(VIOLATION_TYPES, size=n, p=VIOLATION_TYPE_WEIGHTS),
        "none",
    )

    # --- classifier scores -------------------------------------------------
    # Non-violations: Beta(2, 8) -> mean ~0.20
    # Violations:     Beta(8, 2) -> mean ~0.80
    scores_clean = rng.beta(2, 8, size=n)
    scores_viol = rng.beta(8, 2, size=n)
    classifier_score = np.where(is_violation, scores_viol, scores_clean)

    # To achieve ~85% precision and ~90% recall at threshold 0.5 we need
    # a controlled amount of overlap.  The Beta distributions above give
    # roughly the right shape, but we add a small amount of noise to fine-
    # tune.  We also nudge a fraction of clean items above 0.5 (false
    # positives) and a fraction of violations below 0.5 (false negatives).

    # Target counts (approximate)
    n_viol = int(is_violation.sum())
    n_clean = n - n_viol

    # Desired recall ~90%  -> ~10% of violations below threshold
    # Desired precision ~85% -> FP / (TP + FP) ~15%
    # TP ~ 0.90 * n_viol
    # FP ~ TP * (0.15 / 0.85) = TP * 0.1765
    target_fn_rate = 0.10
    target_tp = int(0.90 * n_viol)
    target_fp = int(target_tp * 0.15 / 0.85)
    target_fp_rate = target_fp / max(n_clean, 1)

    # Force false negatives: push some violation scores below 0.5
    viol_idx = np.where(is_violation)[0]
    n_force_fn = int(target_fn_rate * len(viol_idx))
    if n_force_fn > 0 and len(viol_idx) > 0:
        fn_idx = rng.choice(viol_idx, size=n_force_fn, replace=False)
        classifier_score[fn_idx] = rng.beta(2, 5, size=n_force_fn)  # push below 0.5

    # Force false positives: push some clean scores above 0.5
    clean_idx = np.where(~is_violation)[0]
    n_force_fp = int(target_fp_rate * len(clean_idx))
    if n_force_fp > 0 and len(clean_idx) > 0:
        fp_idx = rng.choice(clean_idx, size=n_force_fp, replace=False)
        classifier_score[fp_idx] = rng.beta(5, 2, size=n_force_fp)  # push above 0.5

    classifier_score = _clamp(classifier_score)

    # --- classifier prediction at 0.5 threshold ---------------------------
    classifier_prediction = classifier_score >= 0.5

    # --- confusion matrix flags --------------------------------------------
    tp = is_violation & classifier_prediction
    fp = ~is_violation & classifier_prediction
    fn = is_violation & ~classifier_prediction
    tn = ~is_violation & ~classifier_prediction

    # --- engagement metrics ------------------------------------------------
    # Base views drawn from a log-normal (heavy-tailed like real platforms)
    log_views = rng.normal(loc=8.0, scale=1.5, size=n)  # median ~2981
    views = np.maximum(1, np.round(np.exp(log_views))).astype(int)

    # Violations get a view penalty (often caught early -> less distribution)
    views = np.where(is_violation, (views * rng.uniform(0.1, 0.5, n)).astype(int), views)
    views = np.maximum(1, views)

    # Likes: fraction of views, lower for violations
    like_rate_clean = rng.beta(2, 10, size=n)  # mean ~17%
    like_rate_viol = rng.beta(1, 20, size=n)   # mean ~5%
    like_rate = np.where(is_violation, like_rate_viol, like_rate_clean)
    likes = np.maximum(0, np.round(views * like_rate)).astype(int)

    # Shares: fraction of likes
    share_rate = rng.beta(1.5, 15, size=n)
    shares = np.maximum(0, np.round(likes * share_rate)).astype(int)

    # Comments: correlated with views
    comment_rate = rng.beta(1.5, 20, size=n)
    comments_count = np.maximum(0, np.round(views * comment_rate)).astype(int)

    # Reports: rare for clean content, more common for violations
    reports_clean = rng.poisson(0.05, size=n)
    reports_viol = rng.poisson(3.0, size=n)
    reports_count = np.where(is_violation, reports_viol, reports_clean)
    reports_count = np.maximum(0, reports_count)

    # --- moderation actions ------------------------------------------------
    # Items flagged by classifier or with high reports get moderation
    flagged = classifier_prediction | (reports_count >= 3)

    moderation_action = np.full(n, "none", dtype=object)
    flagged_idx = np.where(flagged)[0]
    if len(flagged_idx) > 0:
        actions = rng.choice(
            MODERATION_ACTIONS, size=len(flagged_idx), p=MODERATION_ACTION_WEIGHTS
        )
        moderation_action[flagged_idx] = actions

    # Time to action: log-normal hours for actioned items
    time_to_action = np.full(n, np.nan)
    actioned_idx = np.where(moderation_action != "none")[0]
    if len(actioned_idx) > 0:
        # Automated catches are fast (~0.1-2h), manual review takes longer
        auto_mask = classifier_prediction[actioned_idx]
        times = np.where(
            auto_mask,
            rng.lognormal(mean=-0.5, sigma=1.0, size=len(actioned_idx)),  # fast
            rng.lognormal(mean=2.5, sigma=0.8, size=len(actioned_idx)),   # slow
        )
        time_to_action[actioned_idx] = np.round(times, 2)

    # --- assemble ----------------------------------------------------------
    df = pd.DataFrame(
        {
            "content_id": content_ids,
            "user_id": user_ids,
            "timestamp": timestamps,
            "content_type": content_types,
            "category": categories,
            "violation_type": violation_types,
            "is_violation": is_violation,
            "classifier_score": np.round(classifier_score, 4),
            "classifier_prediction": classifier_prediction,
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
            "true_negative": tn,
            "reports_count": reports_count,
            "views": views,
            "likes": likes,
            "shares": shares,
            "comments": comments_count,
            "moderation_action": moderation_action,
            "time_to_action_hours": time_to_action,
        }
    )
    return df.sort_values("timestamp").reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. Policy change (DiD / RDD)
# ---------------------------------------------------------------------------


def generate_moderation_policy_change(
    n_users: int = 10_000,
    seed: int = 42,
) -> pd.DataFrame:
    """Simulate a stricter hate-speech policy change for DiD and RDD analysis.

    Each user is observed in a *before* period and an *after* period.  The
    policy change is modeled as a treatment that is stronger in the EU
    (GDPR compliance) and weaker elsewhere.  A running variable
    ``days_since_policy`` enables regression discontinuity designs.

    Parameters
    ----------
    n_users : int
        Number of unique users to simulate.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Two rows per user (before / after) with columns described below.

    Columns
    -------
    user_id : str
        Unique user identifier.
    period : str
        ``"before"`` or ``"after"`` the policy change.
    days_since_policy : int
        Running variable centered on the policy change date.  Negative
        values are pre-policy; positive values are post-policy.  Suitable
        for RDD analysis.
    region : str
        One of ``US``, ``EU``, ``APAC``, ``LATAM``.
    violations_count : int
        Number of violations in the observation window.
    appeals_count : int
        Number of moderation appeals filed.
    appeal_success_rate : float
        Fraction of appeals that were overturned (0-1).
    content_removed : int
        Number of content items removed by moderators.
    user_warnings : int
        Number of warnings issued to the user.
    account_restricted : bool
        Whether the account was placed on restricted status.
    """
    rng = np.random.default_rng(seed)

    user_ids = np.array([f"U{i:05d}" for i in range(1, n_users + 1)])
    regions = rng.choice(REGIONS, size=n_users, p=REGION_WEIGHTS)

    # Region-level treatment effects (reduction multiplier on violations)
    # EU sees the strongest effect due to GDPR enforcement
    region_treatment_effect = {
        "US": 0.75,    # 25% reduction
        "EU": 0.55,    # 45% reduction (strongest)
        "APAC": 0.70,  # 30% reduction
        "LATAM": 0.80, # 20% reduction
    }

    rows = []
    for period in ["before", "after"]:
        for i in range(n_users):
            uid = user_ids[i]
            region = regions[i]

            # Running variable: days relative to policy change
            if period == "before":
                days = -rng.integers(1, 91)  # -90 to -1
            else:
                days = rng.integers(1, 91)   # 1 to 90

            # Baseline violation rate varies by region
            base_violations_mu = {
                "US": 2.5, "EU": 2.0, "APAC": 3.0, "LATAM": 2.8
            }[region]

            # User-level heterogeneity
            user_violation_propensity = rng.gamma(2, 1)
            mu = base_violations_mu * user_violation_propensity

            if period == "after":
                # Apply treatment effect
                mu *= region_treatment_effect[region]
                # Gradual ramp: effect strengthens over time post-policy
                ramp = min(1.0, abs(days) / 60.0)
                mu = base_violations_mu * user_violation_propensity * (
                    1.0 - (1.0 - region_treatment_effect[region]) * ramp
                )

            violations = max(0, rng.poisson(mu))

            # Appeals: users with violations may appeal
            appeals = rng.binomial(violations, 0.3) if violations > 0 else 0

            # Appeal success rate: lower after policy (stricter rules)
            if appeals > 0:
                base_success = 0.40
                if period == "after":
                    base_success = 0.25 if region == "EU" else 0.32
                successes = rng.binomial(appeals, base_success)
                appeal_success_rate = round(successes / appeals, 3)
            else:
                appeal_success_rate = 0.0

            # Content removed: correlated with violations
            content_removed = rng.binomial(violations, 0.6) if violations > 0 else 0
            if period == "after":
                # Stricter enforcement -> higher removal rate
                removal_rate = 0.80 if region == "EU" else 0.70
                content_removed = (
                    rng.binomial(violations, removal_rate) if violations > 0 else 0
                )

            # Warnings
            warnings = rng.binomial(violations, 0.5) if violations > 0 else 0

            # Account restriction: more likely post-policy for repeat offenders
            restrict_prob = 0.02 + 0.03 * min(violations, 5)
            if period == "after":
                restrict_prob *= 1.5 if region != "EU" else 2.0
            account_restricted = bool(rng.random() < restrict_prob)

            rows.append(
                {
                    "user_id": uid,
                    "period": period,
                    "days_since_policy": int(days),
                    "region": region,
                    "violations_count": violations,
                    "appeals_count": appeals,
                    "appeal_success_rate": appeal_success_rate,
                    "content_removed": content_removed,
                    "user_warnings": warnings,
                    "account_restricted": account_restricted,
                }
            )

    df = pd.DataFrame(rows)
    return df.sort_values(["user_id", "period"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3. Creator cohorts (propensity score matching)
# ---------------------------------------------------------------------------


def generate_creator_cohorts(
    n_creators: int = 5_000,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate creator-level data for propensity score matching.

    Simulates an educational intervention program where creators with
    higher prior violation histories are more likely to be selected
    (confounding by indication).  The true average treatment effect on
    the treated (ATT) is approximately a 20 % reduction in post-
    intervention violation rates.

    Parameters
    ----------
    n_creators : int
        Number of creators to simulate.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        One row per creator with columns described below.

    Columns
    -------
    creator_id : str
        Unique creator identifier.
    follower_count : int
        Number of followers (log-normal distributed).
    account_age_days : int
        Days since account creation.
    content_frequency : float
        Average posts per week.
    prior_violations : int
        Number of violations in the pre-intervention period.
    category : str
        Primary content category.
    region : str
        Creator's region.
    received_intervention : bool
        Whether the creator received the education intervention.
    post_violation_rate : float
        Violations per 100 posts in the post-intervention period.
    post_engagement_rate : float
        Average engagement rate (likes + comments + shares) / views
        in the post-intervention period.
    """
    rng = np.random.default_rng(seed)
    n = n_creators

    creator_ids = np.array([f"CR{i:05d}" for i in range(1, n + 1)])

    # --- creator characteristics (pre-treatment covariates) ----------------

    # Follower count: log-normal, median ~5000
    follower_count = np.maximum(
        10, np.round(np.exp(rng.normal(8.5, 1.2, size=n)))
    ).astype(int)

    # Account age: 30 to 1500 days
    account_age_days = rng.integers(30, 1500, size=n)

    # Content frequency: posts per week, gamma-distributed
    content_frequency = np.round(rng.gamma(3, 2, size=n), 1)
    content_frequency = np.maximum(0.5, content_frequency)

    # Prior violations: Poisson, correlated with content frequency
    prior_violation_mu = 0.5 + 0.3 * content_frequency
    prior_violations = rng.poisson(prior_violation_mu)

    categories = rng.choice(CATEGORIES, size=n, p=CATEGORY_WEIGHTS)
    regions = rng.choice(REGIONS, size=n, p=REGION_WEIGHTS)

    # --- treatment assignment (confounded) ---------------------------------
    # Higher prior violations -> higher probability of receiving intervention
    # Also: newer accounts and higher-frequency posters are more likely
    logit = (
        -2.0
        + 0.6 * prior_violations
        + 0.3 * (content_frequency - 6.0) / 4.0
        - 0.3 * (account_age_days - 750) / 500.0
        + 0.2 * np.where(regions == "US", 1, 0)
    )
    propensity = 1.0 / (1.0 + np.exp(-logit))
    received_intervention = rng.random(n) < propensity

    # --- outcomes ----------------------------------------------------------
    # Base post-violation rate per 100 posts
    base_violation_rate = (
        1.0
        + 0.5 * prior_violations
        + 0.2 * content_frequency / 6.0
        - 0.1 * np.log1p(follower_count) / 10.0
        + rng.normal(0, 0.5, size=n)
    )
    base_violation_rate = np.maximum(0.0, base_violation_rate)

    # Treatment effect: ~20% reduction (heterogeneous)
    treatment_effect = rng.normal(0.20, 0.05, size=n)
    treatment_effect = np.clip(treatment_effect, 0.05, 0.40)

    post_violation_rate = np.where(
        received_intervention,
        base_violation_rate * (1.0 - treatment_effect),
        base_violation_rate,
    )
    post_violation_rate = np.round(np.maximum(0.0, post_violation_rate), 3)

    # Post engagement rate: slightly higher for intervention group
    # (educational content tends to improve quality)
    base_engagement = rng.beta(2, 20, size=n)  # mean ~9%
    engagement_boost = np.where(received_intervention, rng.normal(0.01, 0.005, n), 0)
    post_engagement_rate = np.round(
        _clamp(base_engagement + engagement_boost, 0.001, 0.5), 4
    )

    # --- assemble ----------------------------------------------------------
    df = pd.DataFrame(
        {
            "creator_id": creator_ids,
            "follower_count": follower_count,
            "account_age_days": account_age_days,
            "content_frequency": content_frequency,
            "prior_violations": prior_violations,
            "category": categories,
            "region": regions,
            "received_intervention": received_intervention,
            "post_violation_rate": post_violation_rate,
            "post_engagement_rate": post_engagement_rate,
        }
    )
    return df


# ---------------------------------------------------------------------------
# Quick sanity checks when run as a script
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("Content Moderation Synthetic Data Generator")
    print("=" * 70)

    # --- 1. Content stream -------------------------------------------------
    print("\n--- 1. Content Stream ---")
    df_stream = generate_content_stream()
    n_total = len(df_stream)
    n_viol = df_stream["is_violation"].sum()
    n_pred_pos = df_stream["classifier_prediction"].sum()
    tp = df_stream["true_positive"].sum()
    fp = df_stream["false_positive"].sum()
    fn = df_stream["false_negative"].sum()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    print(f"  Total items:        {n_total:,}")
    print(f"  True violations:    {n_viol:,} ({n_viol/n_total:.1%})")
    print(f"  Classifier +ve:     {n_pred_pos:,}")
    print(f"  Precision:          {precision:.3f}")
    print(f"  Recall:             {recall:.3f}")
    print(f"  Moderation actions: {(df_stream['moderation_action'] != 'none').sum():,}")
    print(f"\n  Sample rows:\n{df_stream.head(3).to_string(index=False)}")

    # --- 2. Policy change --------------------------------------------------
    print("\n--- 2. Policy Change ---")
    df_policy = generate_moderation_policy_change()
    before = df_policy[df_policy["period"] == "before"]
    after = df_policy[df_policy["period"] == "after"]
    print(f"  Total rows:         {len(df_policy):,}")
    print(f"  Users:              {df_policy['user_id'].nunique():,}")
    print(f"  Mean violations (before): {before['violations_count'].mean():.2f}")
    print(f"  Mean violations (after):  {after['violations_count'].mean():.2f}")
    for r in REGIONS:
        b = before[before["region"] == r]["violations_count"].mean()
        a = after[after["region"] == r]["violations_count"].mean()
        print(f"    {r}: {b:.2f} -> {a:.2f}  (change: {(a-b)/b:.1%})")

    # --- 3. Creator cohorts ------------------------------------------------
    print("\n--- 3. Creator Cohorts ---")
    df_creators = generate_creator_cohorts()
    treated = df_creators[df_creators["received_intervention"]]
    control = df_creators[~df_creators["received_intervention"]]
    print(f"  Total creators:     {len(df_creators):,}")
    print(f"  Treated:            {len(treated):,} ({len(treated)/len(df_creators):.1%})")
    print(f"  Control:            {len(control):,}")
    print(
        f"  Mean prior violations (treated):  {treated['prior_violations'].mean():.2f}"
    )
    print(
        f"  Mean prior violations (control):  {control['prior_violations'].mean():.2f}"
    )
    print(
        f"  Mean post violation rate (treated): "
        f"{treated['post_violation_rate'].mean():.3f}"
    )
    print(
        f"  Mean post violation rate (control): "
        f"{control['post_violation_rate'].mean():.3f}"
    )

    print("\nDone.")
