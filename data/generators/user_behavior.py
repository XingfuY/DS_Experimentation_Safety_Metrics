"""Synthetic user behavior data generators for a TikTok-like content platform.

Generates realistic user cohorts, longitudinal journey panels, and social
network graphs for trust & safety experimentation and analysis. All data
is fully synthetic with seeded randomness for reproducibility.
"""

import numpy as np
import pandas as pd
from typing import Tuple


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _weighted_choice(rng: np.random.Generator, options: list, weights: list, size: int) -> np.ndarray:
    """Draw *size* samples from *options* according to *weights*."""
    probs = np.array(weights, dtype=float)
    probs /= probs.sum()
    return rng.choice(options, size=size, p=probs)


# ---------------------------------------------------------------------------
# 1. User cohorts
# ---------------------------------------------------------------------------

def generate_user_cohorts(n_users: int = 10_000, seed: int = 42) -> pd.DataFrame:
    """Generate a cross-sectional DataFrame of synthetic platform users.

    Each row represents one user with demographic, engagement, and safety
    attributes. Realistic correlations are baked in:

    * Creators have higher follower counts and engagement.
    * Bot suspects exhibit abnormally high activity with low variability.
    * Risk scores are driven by violation history and report signals.
    * Regional differences: APAC users have higher daily minutes; US has
      a higher creator fraction.

    Parameters
    ----------
    n_users : int
        Number of users to generate (default 10 000).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        One row per user with columns described in the module docstring.
    """
    rng = np.random.default_rng(seed)

    user_ids = np.arange(1, n_users + 1)

    # -- Region (with slight distribution skew) ----------------------------
    regions = _weighted_choice(
        rng,
        ["US", "EU", "APAC", "LATAM", "MEA"],
        [0.30, 0.25, 0.25, 0.12, 0.08],
        n_users,
    )

    # -- Platform ----------------------------------------------------------
    platforms = _weighted_choice(
        rng, ["ios", "android", "web"], [0.45, 0.45, 0.10], n_users
    )

    # -- Signup date (last ~3 years) & account age -------------------------
    max_age_days = 1095  # ~3 years
    account_age_days = rng.integers(1, max_age_days + 1, size=n_users)
    reference_date = pd.Timestamp("2026-02-28")
    signup_dates = pd.to_datetime(
        [reference_date - pd.Timedelta(days=int(d)) for d in account_age_days]
    )

    # -- Age bucket --------------------------------------------------------
    age_buckets = _weighted_choice(
        rng,
        ["13-17", "18-24", "25-34", "35-44", "45+"],
        [0.08, 0.35, 0.30, 0.17, 0.10],
        n_users,
    )

    # -- Creator flag (US has more creators) -------------------------------
    creator_base_prob = np.where(regions == "US", 0.18, 0.12)
    is_creator = rng.random(n_users) < creator_base_prob

    # -- Follower / following counts ---------------------------------------
    # Creators get a multiplicative boost.
    base_followers = rng.lognormal(mean=4.0, sigma=1.8, size=n_users).astype(int)
    base_followers = np.clip(base_followers, 0, 5_000_000)
    follower_count = np.where(is_creator, base_followers * rng.integers(2, 10, size=n_users), base_followers)

    following_count = rng.lognormal(mean=4.5, sigma=1.2, size=n_users).astype(int)
    following_count = np.clip(following_count, 0, 10_000)

    # -- Daily active minutes (APAC higher) --------------------------------
    base_minutes = rng.lognormal(mean=3.0, sigma=0.8, size=n_users)
    regional_multiplier = np.where(regions == "APAC", 1.35, 1.0)
    daily_active_minutes = np.round(base_minutes * regional_multiplier, 1)
    daily_active_minutes = np.clip(daily_active_minutes, 0, 600)

    # -- Weekly sessions ---------------------------------------------------
    weekly_sessions = rng.poisson(lam=12, size=n_users)
    weekly_sessions = np.clip(weekly_sessions, 0, 100)

    # -- Content metrics (30 day windows) ----------------------------------
    content_created_30d = np.where(
        is_creator,
        rng.poisson(lam=15, size=n_users),
        rng.poisson(lam=0.5, size=n_users),
    )
    content_consumed_30d = rng.poisson(lam=300, size=n_users) + (daily_active_minutes * 2).astype(int)

    likes_given_30d = rng.poisson(lam=80, size=n_users)
    comments_given_30d = rng.poisson(lam=12, size=n_users)
    shares_30d = rng.poisson(lam=5, size=n_users)

    # -- Safety signals ----------------------------------------------------
    reports_filed_30d = rng.poisson(lam=0.3, size=n_users)
    violations_30d = rng.poisson(lam=0.15, size=n_users)
    # Users with violations tend to receive more reports.
    reports_received_30d = (
        rng.poisson(lam=0.2, size=n_users) + violations_30d * rng.integers(1, 4, size=n_users)
    )

    # -- Risk score (composite: 0-1) ---------------------------------------
    risk_raw = (
        0.40 * np.clip(violations_30d / 3.0, 0, 1)
        + 0.30 * np.clip(reports_received_30d / 5.0, 0, 1)
        + 0.15 * np.clip(reports_filed_30d / 3.0, 0, 1)  # serial reporters
        + 0.10 * (1.0 - np.clip(account_age_days / 365, 0, 1))  # new accounts riskier
        + 0.05 * rng.random(n_users)  # noise
    )
    risk_score = np.round(np.clip(risk_raw, 0, 1), 4)

    # -- Trust tier --------------------------------------------------------
    trust_tier = np.full(n_users, "standard", dtype=object)
    trust_tier[account_age_days < 30] = "new"
    trust_tier[(account_age_days >= 180) & (violations_30d == 0) & (risk_score < 0.15)] = "trusted"
    trust_tier[
        (account_age_days >= 365)
        & (violations_30d == 0)
        & (risk_score < 0.08)
        & is_creator
    ] = "verified"

    # -- Bot suspects (~2%) ------------------------------------------------
    n_bots = int(round(n_users * 0.02))
    bot_indices = rng.choice(n_users, size=n_bots, replace=False)
    is_bot_suspect = np.zeros(n_users, dtype=bool)
    is_bot_suspect[bot_indices] = True

    # Bots: high activity, unnaturally consistent, lots of likes/shares
    daily_active_minutes[bot_indices] = rng.uniform(180, 400, size=n_bots).round(1)
    weekly_sessions[bot_indices] = rng.integers(50, 100, size=n_bots)
    likes_given_30d[bot_indices] = rng.integers(500, 2000, size=n_bots)
    shares_30d[bot_indices] = rng.integers(100, 500, size=n_bots)
    comments_given_30d[bot_indices] = rng.integers(0, 3, size=n_bots)  # bots rarely comment
    follower_count[bot_indices] = rng.integers(0, 15, size=n_bots)  # low organic following
    following_count[bot_indices] = rng.integers(500, 5000, size=n_bots)
    risk_score[bot_indices] = np.round(rng.uniform(0.5, 0.95, size=n_bots), 4)

    # -- Assemble ----------------------------------------------------------
    df = pd.DataFrame(
        {
            "user_id": user_ids,
            "signup_date": signup_dates,
            "region": regions,
            "platform": platforms,
            "account_age_days": account_age_days,
            "is_creator": is_creator,
            "follower_count": follower_count,
            "following_count": following_count,
            "daily_active_minutes": daily_active_minutes,
            "weekly_sessions": weekly_sessions,
            "content_created_30d": content_created_30d,
            "content_consumed_30d": content_consumed_30d,
            "likes_given_30d": likes_given_30d,
            "comments_given_30d": comments_given_30d,
            "shares_30d": shares_30d,
            "reports_filed_30d": reports_filed_30d,
            "reports_received_30d": reports_received_30d,
            "violations_30d": violations_30d,
            "risk_score": risk_score,
            "trust_tier": trust_tier,
            "is_bot_suspect": is_bot_suspect,
            "age_bucket": age_buckets,
        }
    )

    return df


# ---------------------------------------------------------------------------
# 2. User journey (longitudinal daily panel)
# ---------------------------------------------------------------------------

def generate_user_journey(
    n_users: int = 5_000, n_days: int = 90, seed: int = 42
) -> pd.DataFrame:
    """Generate a daily panel of user behavior over time.

    Simulates churn (some users stop appearing), weekend effects (higher
    engagement on Sat/Sun), enforcement actions (warnings and restrictions),
    and behavioural responses to those actions.

    Parameters
    ----------
    n_users : int
        Number of unique users (default 5 000).
    n_days : int
        Number of calendar days to simulate (default 90).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Daily-level records with columns: user_id, date, daily_minutes,
        sessions, content_viewed, content_created, violations,
        reports_filed, was_warned, was_restricted, engagement_score,
        risk_score_daily.
    """
    rng = np.random.default_rng(seed)

    start_date = pd.Timestamp("2025-12-01")
    dates = pd.date_range(start_date, periods=n_days, freq="D")

    # Per-user baseline traits (drawn once) --------------------------------
    base_minutes = rng.lognormal(mean=3.0, sigma=0.7, size=n_users)
    base_sessions = rng.poisson(lam=4, size=n_users).clip(1, None)
    base_violation_rate = rng.exponential(scale=0.02, size=n_users)  # most users ~0
    churn_day = rng.integers(30, n_days + 30, size=n_users)  # some churn within window

    # Pre-allocate output lists for speed ----------------------------------
    records: list[dict] = []

    for uid in range(n_users):
        user_id = uid + 1
        cumulative_violations = 0
        warned = False
        restricted = False
        escalation_tendency = rng.random()  # 0 = will reform, 1 = will escalate

        for day_idx in range(n_days):
            # Churn: user stops appearing after their churn day
            if day_idx >= churn_day[uid]:
                break

            current_date = dates[day_idx]
            is_weekend = current_date.dayofweek >= 5

            # -- Seasonal / weekend multiplier ---
            weekend_mult = 1.3 if is_weekend else 1.0
            # Slight linear trend (engagement grows slightly over time)
            trend_mult = 1.0 + 0.001 * day_idx

            # -- Enforcement effects ---
            enforcement_mult = 1.0
            violation_rate_mult = 1.0
            if restricted:
                enforcement_mult = 0.3  # heavily suppressed engagement
                # Some users reform, others try to circumvent
                if escalation_tendency < 0.6:
                    violation_rate_mult = 0.1  # reform
                else:
                    violation_rate_mult = 1.5  # escalate / circumvent
            elif warned:
                enforcement_mult = 0.8
                if escalation_tendency < 0.5:
                    violation_rate_mult = 0.4  # scared straight
                else:
                    violation_rate_mult = 1.2  # mild escalation

            # -- Daily minutes ---
            noise = rng.normal(1.0, 0.2)
            daily_minutes = round(
                max(0, base_minutes[uid] * weekend_mult * trend_mult * enforcement_mult * noise), 1
            )

            # -- Sessions ---
            sessions = max(0, int(base_sessions[uid] * weekend_mult * enforcement_mult + rng.normal(0, 1)))

            # -- Content ---
            content_viewed = max(0, int(daily_minutes * rng.uniform(1.5, 3.0)))
            content_created = max(0, rng.poisson(lam=0.3 * enforcement_mult))

            # -- Violations ---
            violations = rng.poisson(lam=base_violation_rate[uid] * violation_rate_mult)
            cumulative_violations += violations

            # -- Reports filed ---
            reports_filed = rng.poisson(lam=0.05)

            # -- Enforcement actions (based on cumulative violations) ---
            was_warned = False
            was_restricted = False
            if cumulative_violations >= 2 and not warned:
                was_warned = True
                warned = True
            if cumulative_violations >= 5 and not restricted:
                was_restricted = True
                restricted = True

            # -- Engagement score (0-1 normalised proxy) ---
            engagement_score = round(
                min(1.0, (daily_minutes / 120) * 0.4 + (sessions / 15) * 0.3 + (content_viewed / 200) * 0.3),
                4,
            )

            # -- Daily risk score ---
            risk_score_daily = round(
                min(
                    1.0,
                    0.5 * min(violations / 2.0, 1.0)
                    + 0.3 * min(cumulative_violations / 8.0, 1.0)
                    + 0.2 * rng.random(),
                ),
                4,
            )

            records.append(
                {
                    "user_id": user_id,
                    "date": current_date,
                    "daily_minutes": daily_minutes,
                    "sessions": sessions,
                    "content_viewed": content_viewed,
                    "content_created": content_created,
                    "violations": violations,
                    "reports_filed": reports_filed,
                    "was_warned": was_warned,
                    "was_restricted": was_restricted,
                    "engagement_score": engagement_score,
                    "risk_score_daily": risk_score_daily,
                }
            )

    df = pd.DataFrame(records)
    return df


# ---------------------------------------------------------------------------
# 3. Network graph for coordinated inauthentic behavior (CIB) detection
# ---------------------------------------------------------------------------

def generate_network_graph(
    n_users: int = 2_000, avg_connections: int = 15, seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate a synthetic social graph with embedded inauthentic clusters.

    Approximately 5 % of users are flagged as inauthentic. These users form
    dense, tightly-connected clusters with high mutual interaction weights,
    while authentic users have sparser, more diverse connection patterns.

    Parameters
    ----------
    n_users : int
        Total number of users / nodes (default 2 000).
    avg_connections : int
        Average outgoing edges per authentic user (default 15).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        ``(nodes_df, edges_df)``

        *nodes_df* columns: user_id, is_inauthentic, account_age_days,
        follower_count, content_similarity_score.

        *edges_df* columns: source, target, interaction_type, weight,
        timestamp.
    """
    rng = np.random.default_rng(seed)

    user_ids = np.arange(1, n_users + 1)

    # -- Inauthentic users (~5 %, arranged in clusters) --------------------
    n_inauthentic = int(round(n_users * 0.05))
    inauthentic_ids = set(rng.choice(user_ids, size=n_inauthentic, replace=False).tolist())

    # Partition inauthentic users into clusters of size 5-15.
    inauthentic_list = sorted(inauthentic_ids)
    clusters: list[list[int]] = []
    idx = 0
    while idx < len(inauthentic_list):
        cluster_size = min(rng.integers(5, 16), len(inauthentic_list) - idx)
        clusters.append(inauthentic_list[idx : idx + cluster_size])
        idx += cluster_size

    is_inauthentic = np.array([uid in inauthentic_ids for uid in user_ids])

    # -- Node attributes ---------------------------------------------------
    account_age_days = np.where(
        is_inauthentic,
        rng.integers(5, 90, size=n_users),  # inauthentic = newer accounts
        rng.integers(10, 1095, size=n_users),
    )

    follower_count = np.where(
        is_inauthentic,
        rng.integers(10, 500, size=n_users),
        rng.lognormal(mean=4.0, sigma=1.5, size=n_users).astype(int),
    )

    # Content similarity: inauthentic users post near-identical content.
    content_similarity_score = np.where(
        is_inauthentic,
        np.round(rng.uniform(0.80, 0.99, size=n_users), 4),
        np.round(rng.uniform(0.05, 0.50, size=n_users), 4),
    )

    nodes_df = pd.DataFrame(
        {
            "user_id": user_ids,
            "is_inauthentic": is_inauthentic,
            "account_age_days": account_age_days,
            "follower_count": follower_count,
            "content_similarity_score": content_similarity_score,
        }
    )

    # -- Edge generation ---------------------------------------------------
    interaction_types = ["follow", "like", "share", "comment"]
    interaction_weights_auth = [0.35, 0.35, 0.15, 0.15]
    interaction_weights_inauth = [0.20, 0.40, 0.30, 0.10]  # more shares among bots

    reference_ts = pd.Timestamp("2026-01-01")
    edge_records: list[dict] = []

    # Helper: create an edge dict
    def _make_edge(src: int, tgt: int, itype: str, weight: float, ts: pd.Timestamp) -> dict:
        return {
            "source": src,
            "target": tgt,
            "interaction_type": itype,
            "weight": round(weight, 4),
            "timestamp": ts,
        }

    # (a) Inauthentic cluster edges (dense, high-weight) -------------------
    for cluster in clusters:
        n_c = len(cluster)
        for i in range(n_c):
            for j in range(n_c):
                if i == j:
                    continue
                # High probability of mutual connection within cluster
                if rng.random() < 0.85:
                    itype = _weighted_choice(rng, interaction_types, interaction_weights_inauth, 1)[0]
                    weight = rng.uniform(0.7, 1.0)
                    ts = reference_ts + pd.Timedelta(minutes=int(rng.integers(0, 60 * 24 * 90)))
                    edge_records.append(_make_edge(cluster[i], cluster[j], itype, weight, ts))

        # Some edges from inauthentic users to authentic users (camouflage)
        n_camouflage = rng.integers(2, 8) * n_c
        authentic_targets = rng.choice(
            [uid for uid in user_ids if uid not in inauthentic_ids],
            size=min(n_camouflage, n_users - n_inauthentic),
            replace=True,
        )
        for tgt in authentic_targets:
            src = rng.choice(cluster)
            itype = _weighted_choice(rng, interaction_types, interaction_weights_auth, 1)[0]
            weight = rng.uniform(0.1, 0.4)
            ts = reference_ts + pd.Timedelta(minutes=int(rng.integers(0, 60 * 24 * 90)))
            edge_records.append(_make_edge(src, tgt, itype, weight, ts))

    # (b) Authentic user edges (sparse, diverse) ---------------------------
    authentic_ids = [uid for uid in user_ids if uid not in inauthentic_ids]
    for uid in authentic_ids:
        n_edges = max(1, rng.poisson(lam=avg_connections))
        targets = rng.choice(
            [u for u in user_ids if u != uid], size=min(n_edges, n_users - 1), replace=False
        )
        for tgt in targets:
            itype = _weighted_choice(rng, interaction_types, interaction_weights_auth, 1)[0]
            weight = rng.uniform(0.05, 0.6)
            ts = reference_ts + pd.Timedelta(minutes=int(rng.integers(0, 60 * 24 * 90)))
            edge_records.append(_make_edge(uid, tgt, itype, weight, ts))

    edges_df = pd.DataFrame(edge_records)

    return nodes_df, edges_df


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Generating user cohorts...")
    cohorts = generate_user_cohorts()
    print(f"  shape: {cohorts.shape}")
    print(f"  columns: {list(cohorts.columns)}")
    print(f"  region distribution:\n{cohorts['region'].value_counts().to_string()}")
    print(f"  bot suspects: {cohorts['is_bot_suspect'].sum()}")
    print(f"  trust tier distribution:\n{cohorts['trust_tier'].value_counts().to_string()}")
    print()

    print("Generating user journeys...")
    journeys = generate_user_journey(n_users=500, n_days=30)
    print(f"  shape: {journeys.shape}")
    print(f"  unique users: {journeys['user_id'].nunique()}")
    print(f"  date range: {journeys['date'].min()} to {journeys['date'].max()}")
    print(f"  warned events: {journeys['was_warned'].sum()}")
    print(f"  restricted events: {journeys['was_restricted'].sum()}")
    print()

    print("Generating network graph...")
    nodes, edges = generate_network_graph(n_users=500, avg_connections=10)
    print(f"  nodes shape: {nodes.shape}")
    print(f"  edges shape: {edges.shape}")
    print(f"  inauthentic nodes: {nodes['is_inauthentic'].sum()}")
    print(f"  interaction types:\n{edges['interaction_type'].value_counts().to_string()}")
