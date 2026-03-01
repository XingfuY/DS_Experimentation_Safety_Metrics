"""Synthetic data generators for safety incident monitoring and anomaly detection.

Generates realistic time series data mimicking a TikTok-like platform's trust
and safety operational metrics. All generators produce reproducible output via
configurable random seeds.

Patterns modeled:
    - Weekly seasonality (higher weekend engagement)
    - Diurnal cycles (peak evening, trough early morning)
    - Long-term growth trends
    - Injected anomalies: spikes, gradual drifts, seasonal breaks
    - Policy change structural breaks with adaptation dynamics
    - Coordinated attack bursts in real-time streams
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _weekly_seasonality(day_indices: np.ndarray) -> np.ndarray:
    """Return a multiplicative weekly factor.

    Weekends (Sat=5, Sun=6) get ~20-30 % more activity; Tuesday is the
    trough.  The pattern uses a mix of two harmonics for realism.
    """
    dow = day_indices % 7  # 0=Mon ... 6=Sun
    return 1.0 + 0.12 * np.sin(2 * np.pi * dow / 7) + 0.08 * np.cos(4 * np.pi * dow / 7)


def _growth_trend(day_indices: np.ndarray, rate: float = 0.0008) -> np.ndarray:
    """Gentle exponential growth over the observation window."""
    return np.exp(rate * day_indices)


def _diurnal_pattern(hours: np.ndarray) -> np.ndarray:
    """Multiplicative hourly factor peaking around 20:00, trough near 04:00.

    Models a mix of harmonics calibrated to social-media usage curves.
    """
    return (
        1.0
        + 0.45 * np.sin(2 * np.pi * (hours - 6) / 24)
        + 0.15 * np.sin(4 * np.pi * (hours - 6) / 24)
    )


# ---------------------------------------------------------------------------
# 1. Daily incident time series
# ---------------------------------------------------------------------------

def generate_incident_timeseries(
    n_days: int = 365,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate daily safety-incident time series with injected anomalies.

    Parameters
    ----------
    n_days : int
        Number of days to generate (default 365, i.e. one year).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Columns:
            date, total_reports, hate_speech_reports, misinformation_reports,
            spam_reports, nudity_reports, violence_reports, copyright_reports,
            auto_removed, manual_reviewed, appeals_filed, appeals_overturned,
            false_positive_rate, avg_time_to_action_hours,
            active_users, new_users, content_volume,
            is_anomaly, anomaly_type

    Injected anomalies
    ------------------
    1. Day 60-62 : Sudden hate-speech spike (3x baseline) — type ``spike``
    2. Day 150-180 : Gradual spam drift (linear ramp to 2x) — type ``drift``
    3. Day 250-255 : Holiday seasonal break (all reports +40 %) — type ``seasonal_break``
    4. Day 320 : Single-day violence spike (4x) — type ``spike``
    """
    rng = np.random.default_rng(seed)

    day_idx = np.arange(n_days)
    dates = pd.date_range("2025-01-01", periods=n_days, freq="D")

    weekly = _weekly_seasonality(day_idx)
    trend = _growth_trend(day_idx)

    # --- base rates (reports per day at launch scale) ---
    base_total = 12_000
    category_shares = {
        "hate_speech": 0.18,
        "misinformation": 0.14,
        "spam": 0.28,
        "nudity": 0.15,
        "violence": 0.10,
        "copyright": 0.15,
    }

    # Smooth baseline with noise
    noise = 1.0 + rng.normal(0, 0.04, n_days)
    baseline = base_total * weekly * trend * noise

    # Per-category
    categories = {}
    for cat, share in category_shares.items():
        cat_noise = 1.0 + rng.normal(0, 0.06, n_days)
        categories[cat] = np.maximum(0, baseline * share * cat_noise).astype(int)

    # --- Inject anomalies ---
    anomaly_flags = np.zeros(n_days, dtype=bool)
    anomaly_types = np.full(n_days, "none", dtype=object)

    # Anomaly 1: hate-speech spike days 60-62
    for d in range(60, min(63, n_days)):
        categories["hate_speech"][d] = int(categories["hate_speech"][d] * rng.uniform(2.8, 3.5))
        anomaly_flags[d] = True
        anomaly_types[d] = "spike"

    # Anomaly 2: gradual spam drift days 150-180
    if n_days > 150:
        drift_end = min(180, n_days)
        drift_len = drift_end - 150
        drift_multiplier = np.linspace(1.0, 2.0, drift_len)
        for i, d in enumerate(range(150, drift_end)):
            categories["spam"][d] = int(categories["spam"][d] * drift_multiplier[i])
            anomaly_flags[d] = True
            anomaly_types[d] = "drift"

    # Anomaly 3: holiday seasonal break days 250-255
    if n_days > 250:
        break_end = min(256, n_days)
        for d in range(250, break_end):
            for cat in categories:
                categories[cat][d] = int(categories[cat][d] * rng.uniform(1.35, 1.50))
            anomaly_flags[d] = True
            anomaly_types[d] = "seasonal_break"

    # Anomaly 4: single-day violence spike day 320
    if n_days > 320:
        categories["violence"][320] = int(categories["violence"][320] * rng.uniform(3.5, 4.5))
        anomaly_flags[320] = True
        anomaly_types[320] = "spike"

    # Recompute total after anomaly injection
    total_reports = sum(categories.values())

    # --- Derived operational metrics ---
    auto_rate = 0.65 + 0.05 * np.sin(2 * np.pi * day_idx / 365)  # automation improves seasonally
    auto_removed = (total_reports * auto_rate * (1.0 + rng.normal(0, 0.02, n_days))).astype(int)
    manual_reviewed = (total_reports - auto_removed) + rng.integers(-50, 50, n_days)
    manual_reviewed = np.maximum(0, manual_reviewed)

    appeals_rate = 0.08 + rng.normal(0, 0.005, n_days)
    appeals_filed = (auto_removed * np.clip(appeals_rate, 0.03, 0.15)).astype(int)
    overturn_rate = 0.22 + rng.normal(0, 0.03, n_days)
    appeals_overturned = (appeals_filed * np.clip(overturn_rate, 0.10, 0.40)).astype(int)

    # False positive rate correlates with auto-removal rate
    fpr_base = 0.05 + 0.02 * (auto_rate - auto_rate.mean()) / auto_rate.std()
    false_positive_rate = np.clip(fpr_base + rng.normal(0, 0.005, n_days), 0.01, 0.15)

    # Time to action (hours) — lower on weekdays when more moderators are on
    tta_base = 2.5 - 0.4 * np.cos(2 * np.pi * day_idx / 7)
    avg_tta = np.clip(tta_base + rng.normal(0, 0.3, n_days), 0.5, 8.0)

    # --- Platform-level metrics ---
    active_users_base = 180_000_000
    active_users = (
        active_users_base * trend * weekly * (1 + rng.normal(0, 0.01, n_days))
    ).astype(int)
    new_users = (rng.poisson(500_000, n_days) * trend * weekly).astype(int)
    content_volume = (
        50_000_000 * trend * weekly * (1 + rng.normal(0, 0.02, n_days))
    ).astype(int)

    df = pd.DataFrame(
        {
            "date": dates,
            "total_reports": total_reports,
            "hate_speech_reports": categories["hate_speech"],
            "misinformation_reports": categories["misinformation"],
            "spam_reports": categories["spam"],
            "nudity_reports": categories["nudity"],
            "violence_reports": categories["violence"],
            "copyright_reports": categories["copyright"],
            "auto_removed": auto_removed,
            "manual_reviewed": manual_reviewed,
            "appeals_filed": appeals_filed,
            "appeals_overturned": appeals_overturned,
            "false_positive_rate": np.round(false_positive_rate, 4),
            "avg_time_to_action_hours": np.round(avg_tta, 2),
            "active_users": active_users,
            "new_users": new_users,
            "content_volume": content_volume,
            "is_anomaly": anomaly_flags,
            "anomaly_type": anomaly_types,
        }
    )
    return df


# ---------------------------------------------------------------------------
# 2. Hourly incident time series
# ---------------------------------------------------------------------------

def generate_hourly_incidents(
    n_days: int = 30,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate hourly safety-incident data with diurnal patterns and anomalies.

    Parameters
    ----------
    n_days : int
        Number of days (default 30). Produces ``n_days * 24`` rows.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Columns:
            timestamp, hour, day_of_week, reports, violations_detected,
            classifier_confidence_mean, queue_depth, moderator_actions,
            avg_response_time_minutes, is_anomaly, anomaly_type

    Injected anomalies
    ------------------
    1. Day 10, hours 14-18 : Coordinated spam burst — reports 5x, queue
       depth spikes, response time degrades. Type ``spam_burst``.
    2. Day 20-23 : Classifier degradation — mean confidence drops from
       ~0.92 to ~0.75, false detections rise. Type ``classifier_degradation``.
    3. Day 25, hour 2 : Flash spike in violations (bot wave). Type ``spike``.
    """
    rng = np.random.default_rng(seed)

    n_hours = n_days * 24
    timestamps = pd.date_range("2025-06-01", periods=n_hours, freq="h")
    hours = timestamps.hour.values
    dow = timestamps.dayofweek.values  # 0=Mon

    diurnal = _diurnal_pattern(hours.astype(float))
    weekend_boost = np.where((dow == 5) | (dow == 6), 1.20, 1.0)

    # Baseline reports per hour
    base_reports = 500
    reports = (
        base_reports * diurnal * weekend_boost * (1 + rng.normal(0, 0.08, n_hours))
    )
    reports = np.maximum(10, reports).astype(int)

    # Violations detected (subset of reports + proactive scans)
    violation_rate = 0.40 + rng.normal(0, 0.03, n_hours)
    violations = (reports * np.clip(violation_rate, 0.25, 0.60)).astype(int)

    # Classifier confidence
    confidence_base = 0.92 + rng.normal(0, 0.015, n_hours)
    confidence = np.clip(confidence_base, 0.70, 0.99)

    # Queue depth: accumulates during peak, drains during off-peak
    queue_base = np.clip(reports - 0.85 * reports.mean(), 0, None)
    queue_depth = (queue_base + rng.poisson(30, n_hours)).astype(int)

    # Moderator actions: capacity-limited
    mod_capacity_per_hour = 420  # moderator team throughput
    moderator_actions = np.minimum(
        reports,
        (mod_capacity_per_hour * (1 + rng.normal(0, 0.05, n_hours))).astype(int),
    )
    moderator_actions = np.maximum(0, moderator_actions)

    # Response time (minutes): inversely related to moderator slack
    utilization = np.clip(reports / (mod_capacity_per_hour + 1), 0.3, 2.5)
    response_time = 15.0 * utilization + rng.exponential(3.0, n_hours)
    response_time = np.clip(response_time, 2.0, 180.0)

    # --- Anomaly labels ---
    anomaly_flags = np.zeros(n_hours, dtype=bool)
    anomaly_types = np.full(n_hours, "none", dtype=object)

    # Anomaly 1: coordinated spam burst, day 10 hours 14-18
    burst_start = 10 * 24 + 14
    burst_end = 10 * 24 + 19
    if burst_end <= n_hours:
        for h in range(burst_start, burst_end):
            reports[h] = int(reports[h] * rng.uniform(4.5, 5.5))
            queue_depth[h] = int(queue_depth[h] * rng.uniform(6, 10))
            response_time[h] = response_time[h] * rng.uniform(2.5, 4.0)
            anomaly_flags[h] = True
            anomaly_types[h] = "spam_burst"

    # Anomaly 2: classifier degradation, days 20-23
    degrade_start = 20 * 24
    degrade_end = min(24 * 24, n_hours)
    if degrade_start < n_hours:
        for h in range(degrade_start, degrade_end):
            day_in_window = (h - degrade_start) / (degrade_end - degrade_start)
            drop = 0.17 * (1 - 0.3 * day_in_window)  # partial recovery toward end
            confidence[h] = np.clip(confidence[h] - drop, 0.55, 0.85)
            violations[h] = int(violations[h] * rng.uniform(0.6, 0.8))  # fewer correct detections
            anomaly_flags[h] = True
            anomaly_types[h] = "classifier_degradation"

    # Anomaly 3: flash bot-wave spike, day 25 hour 2
    flash_idx = 25 * 24 + 2
    if flash_idx < n_hours:
        reports[flash_idx] = int(reports[flash_idx] * rng.uniform(7, 10))
        violations[flash_idx] = int(violations[flash_idx] * rng.uniform(5, 7))
        anomaly_flags[flash_idx] = True
        anomaly_types[flash_idx] = "spike"

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "hour": hours,
            "day_of_week": dow,
            "reports": reports,
            "violations_detected": violations,
            "classifier_confidence_mean": np.round(confidence, 4),
            "queue_depth": queue_depth,
            "moderator_actions": moderator_actions,
            "avg_response_time_minutes": np.round(response_time, 2),
            "is_anomaly": anomaly_flags,
            "anomaly_type": anomaly_types,
        }
    )
    return df


# ---------------------------------------------------------------------------
# 3. Policy impact time series (interrupted time series design)
# ---------------------------------------------------------------------------

def generate_policy_impact_timeseries(
    n_days: int = 180,
    policy_day: int = 90,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate time series around a policy change for interrupted time series analysis.

    Models a platform policy tightening (e.g., stricter nudity guidelines):
    - Pre-policy: stable trend with slight upward drift in violations.
    - Immediate post-policy: sharp drop in violations, spike in false
      positives, dip in user satisfaction.
    - Long-run: gradual adaptation — false positives decline, satisfaction
      recovers, violations settle at new equilibrium ~35 % below pre-policy.

    Parameters
    ----------
    n_days : int
        Total observation window (default 180).
    policy_day : int
        Day index when the policy takes effect (default 90, midpoint).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Columns:
            date, day_index, post_policy, violations, reports,
            false_positives, user_satisfaction, creator_churn_rate
    """
    rng = np.random.default_rng(seed)

    day_idx = np.arange(n_days)
    dates = pd.date_range("2025-03-01", periods=n_days, freq="D")
    post_policy = day_idx >= policy_day

    # --- Violations ---
    # Pre-policy: mild upward trend + noise
    pre_trend = 5000 + 8 * day_idx  # ~8 more violations/day organic growth
    # Post-policy: immediate drop + gradual settling
    days_since_policy = np.maximum(day_idx - policy_day, 0).astype(float)
    # Exponential adaptation: sharp initial drop, gradual settling
    immediate_effect = -1800  # sudden drop
    adaptation = 500 * (1 - np.exp(-days_since_policy / 30))  # partial recovery
    post_effect = np.where(post_policy, immediate_effect + adaptation, 0)

    violations_base = pre_trend + post_effect
    weekly = _weekly_seasonality(day_idx)
    violations = np.maximum(
        0,
        (violations_base * weekly + rng.normal(0, 200, n_days)).astype(int),
    )

    # --- Reports (user-submitted) ---
    # Reports spike right after policy change (confusion), then settle
    report_base = 8000 + 5 * day_idx
    report_spike = np.where(
        post_policy,
        1200 * np.exp(-days_since_policy / 15),  # transient spike
        0,
    )
    reports = np.maximum(
        0,
        ((report_base + report_spike) * weekly + rng.normal(0, 250, n_days)).astype(int),
    )

    # --- False positives ---
    # Jump after policy, then decay as classifiers are re-tuned
    fp_pre = 250 + rng.normal(0, 30, n_days)
    fp_post_boost = np.where(
        post_policy,
        600 * np.exp(-days_since_policy / 25),  # initial surge
        0,
    )
    false_positives = np.maximum(0, (fp_pre + fp_post_boost).astype(int))

    # --- User satisfaction (0-100 scale) ---
    sat_pre = 72.0 + 0.02 * day_idx  # gentle upward trend
    sat_shock = np.where(
        post_policy,
        -8.0 * np.exp(-days_since_policy / 20),  # initial dip
        0,
    )
    sat_recovery = np.where(
        post_policy,
        3.0 * (1 - np.exp(-days_since_policy / 60)),  # long-run improvement
        0,
    )
    user_satisfaction = np.clip(
        sat_pre + sat_shock + sat_recovery + rng.normal(0, 1.2, n_days),
        0,
        100,
    )

    # --- Creator churn rate (daily, fraction) ---
    churn_base = 0.0025 + 0.000005 * day_idx  # baseline ~0.25 %/day
    churn_spike = np.where(
        post_policy,
        0.004 * np.exp(-days_since_policy / 12),  # creators leave briefly
        0,
    )
    creator_churn_rate = np.clip(
        churn_base + churn_spike + rng.normal(0, 0.0003, n_days),
        0.0005,
        0.02,
    )

    df = pd.DataFrame(
        {
            "date": dates,
            "day_index": day_idx,
            "post_policy": post_policy,
            "violations": violations,
            "reports": reports,
            "false_positives": false_positives,
            "user_satisfaction": np.round(user_satisfaction, 2),
            "creator_churn_rate": np.round(creator_churn_rate, 5),
        }
    )
    return df


# ---------------------------------------------------------------------------
# 4. Real-time event stream
# ---------------------------------------------------------------------------

def generate_realtime_stream(
    n_events: int = 10_000,
    attack_start: int = 5000,
    attack_duration: int = 500,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate event-level data simulating a real-time safety monitoring stream.

    Events arrive according to a non-homogeneous Poisson process with a
    diurnal rate.  During the attack window, event rate increases sharply
    and severity distribution shifts toward ``high``/``critical``.

    Parameters
    ----------
    n_events : int
        Total number of events to generate (default 10 000).
    attack_start : int
        Event index at which the coordinated attack begins (default 5000).
    attack_duration : int
        Number of events that constitute the attack burst (default 500).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Columns:
            event_id, timestamp, event_type, severity, category,
            processing_time_ms, is_attack_related
    """
    rng = np.random.default_rng(seed)

    attack_end = attack_start + attack_duration

    # --- Event types and their baseline probabilities ---
    event_types = ["report", "auto_detect", "appeal", "escalation"]
    normal_type_probs = [0.45, 0.35, 0.12, 0.08]
    attack_type_probs = [0.60, 0.25, 0.05, 0.10]  # more reports during attack

    # --- Severity levels ---
    severities = ["low", "medium", "high", "critical"]
    normal_sev_probs = [0.40, 0.35, 0.18, 0.07]
    attack_sev_probs = [0.10, 0.20, 0.40, 0.30]  # severity skews up

    # --- Content categories ---
    categories = [
        "hate_speech",
        "misinformation",
        "spam",
        "nudity",
        "violence",
        "copyright",
        "harassment",
        "self_harm",
    ]
    normal_cat_probs = [0.15, 0.12, 0.25, 0.14, 0.10, 0.12, 0.08, 0.04]
    # Attack is concentrated in hate_speech and harassment
    attack_cat_probs = [0.35, 0.05, 0.10, 0.03, 0.05, 0.02, 0.35, 0.05]

    # --- Generate timestamps ---
    # Simulate inter-arrival times: baseline ~5 s between events,
    # compressed to ~1 s during attack
    inter_arrival = np.empty(n_events)
    for i in range(n_events):
        if attack_start <= i < attack_end:
            inter_arrival[i] = rng.exponential(1.0)  # fast arrivals
        else:
            inter_arrival[i] = rng.exponential(5.0)  # normal pace

    cum_seconds = np.cumsum(inter_arrival)
    base_ts = pd.Timestamp("2025-07-15 00:00:00")
    timestamps = pd.to_datetime(base_ts) + pd.to_timedelta(cum_seconds, unit="s")

    # --- Assign attributes per event ---
    event_type_list = []
    severity_list = []
    category_list = []
    processing_time_list = []
    is_attack_list = []

    for i in range(n_events):
        is_attack = attack_start <= i < attack_end

        # Event type
        if is_attack:
            et = rng.choice(event_types, p=attack_type_probs)
        else:
            et = rng.choice(event_types, p=normal_type_probs)
        event_type_list.append(et)

        # Severity
        if is_attack:
            sev = rng.choice(severities, p=attack_sev_probs)
        else:
            sev = rng.choice(severities, p=normal_sev_probs)
        severity_list.append(sev)

        # Category
        if is_attack:
            cat = rng.choice(categories, p=attack_cat_probs)
        else:
            cat = rng.choice(categories, p=normal_cat_probs)
        category_list.append(cat)

        # Processing time: attack events queue up, increasing latency
        if is_attack:
            # Linearly increasing congestion within the attack window
            congestion = 1.0 + 2.0 * (i - attack_start) / attack_duration
            pt = rng.lognormal(np.log(300 * congestion), 0.5)
        else:
            # Normal processing: ~120 ms median
            pt = rng.lognormal(np.log(120), 0.4)
        processing_time_list.append(pt)

        is_attack_list.append(is_attack)

    df = pd.DataFrame(
        {
            "event_id": np.arange(n_events),
            "timestamp": timestamps,
            "event_type": event_type_list,
            "severity": severity_list,
            "category": category_list,
            "processing_time_ms": np.round(processing_time_list, 1),
            "is_attack_related": is_attack_list,
        }
    )
    return df


# ---------------------------------------------------------------------------
# CLI convenience
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Generating daily incident time series ...")
    daily = generate_incident_timeseries()
    print(f"  Shape: {daily.shape}")
    print(f"  Anomaly days: {daily['is_anomaly'].sum()}")
    print(f"  Date range: {daily['date'].min()} to {daily['date'].max()}")
    print()

    print("Generating hourly incidents ...")
    hourly = generate_hourly_incidents()
    print(f"  Shape: {hourly.shape}")
    print(f"  Anomaly hours: {hourly['is_anomaly'].sum()}")
    print()

    print("Generating policy impact time series ...")
    policy = generate_policy_impact_timeseries()
    print(f"  Shape: {policy.shape}")
    print(f"  Pre-policy mean violations: {policy.loc[~policy['post_policy'], 'violations'].mean():.0f}")
    print(f"  Post-policy mean violations: {policy.loc[policy['post_policy'], 'violations'].mean():.0f}")
    print()

    print("Generating real-time event stream ...")
    stream = generate_realtime_stream()
    print(f"  Shape: {stream.shape}")
    print(f"  Attack events: {stream['is_attack_related'].sum()}")
    print(f"  Severity distribution:")
    print(stream["severity"].value_counts().to_string(header=False))
