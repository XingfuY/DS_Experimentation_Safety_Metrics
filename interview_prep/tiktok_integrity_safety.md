# TikTok Integrity & Safety — Data Scientist Interview Preparation Guide

**Target Role**: Data Scientist, Integrity & Safety
**Company**: TikTok (ByteDance)
**Interview Format**: 3 rounds — SQL, Tech Lead (Stats/ML/Product), Hiring Manager (Resume Deep-Dive + Open-Ended)
**Key Lesson from Prior Attempt**: The HM wants technical depth and quantitative rigor. Every answer must demonstrate specific methods, numbers, and reasoning. Collaboration stories without technical substance will not pass.

---

## Section 1: Role Analysis

### 1.1 Job Description Deep-Dive

The Data Scientist role within TikTok's Integrity & Safety (I&S) team sits at the intersection of three critical functions: Trust & Safety policy enforcement, ML/algorithm development, and product analytics. Understanding the precise responsibilities is essential for framing every interview answer.

**Metrics Systems for Content Safety.** The core deliverable is designing, maintaining, and improving the metrics that quantify platform safety. This includes defining north-star metrics (e.g., violating view rate — the fraction of total content views that land on content that violates community guidelines), guardrail metrics (false positive rate on content removals, creator appeal rate, time-to-action on reported content), and diagnostic metrics (per-policy violation rates, per-geography enforcement consistency, classifier precision/recall by violation category). The DS owns the metric definitions, monitors them daily, and surfaces anomalies. This means you must be fluent in metric decomposition: if the overall violation rate increases 10%, can you decompose it into a volume effect (more content posted) versus a rate effect (higher fraction of violating content) versus a detection effect (classifier recall changed)? Metric trees — where a top-level KPI branches into additive or multiplicative components — are a daily tool.

**Anomaly Detection.** The platform processes billions of content items daily. The DS must detect sudden changes in violation patterns — a new spam campaign, a coordinated harassment operation, a sudden increase in CSAM attempts, or a policy change that inadvertently increases false positives. This requires statistical process control (Shewhart charts, CUSUM, EWMA), time-series decomposition (STL decomposition to separate trend, seasonality, residual), and real-time alerting with appropriate thresholds that balance sensitivity against alert fatigue. You should be prepared to discuss how you'd set alert thresholds: too sensitive (alpha too high) and the on-call DS is paged for noise; too conservative and real incidents go undetected for hours.

**A/B Testing for Security Policies.** Unlike typical product A/B tests where the unit of randomization is a user and the outcome is engagement, experimentation in safety has unique challenges. Network effects are pervasive: if you throttle one bad seller's content in the treatment group, their coordinated network of accounts in the control group may also change behavior. Compliance is imperfect: a policy may be "launched" but enforcement depends on classifier accuracy, which varies by content type and language. Outcome metrics are rare events (violations might be 0.01% of content), requiring enormous sample sizes or alternative methods like ratio metrics and variance reduction. The DS must be comfortable with interference-robust designs (cluster randomization by creator community, switchback designs for time-based policies), pre-experiment power calculations under low base rates, and post-hoc sensitivity analyses.

**Causal Inference.** Many safety interventions cannot be randomized. If the platform globally deploys a new hate speech classifier, there is no control group. The DS must use quasi-experimental methods: difference-in-differences (comparing enforcement rates before/after deployment, against a control metric that should be unaffected), regression discontinuity (exploiting a threshold — accounts with violation scores above X get restricted), instrumental variables (using exogenous variation like server-side rollout timing), and synthetic control (constructing a counterfactual from a weighted combination of unaffected regions). Every answer about measuring effectiveness should demonstrate fluency with at least two of these methods and their identifying assumptions.

**Data Mining.** Exploratory analysis to uncover new abuse patterns, identify gaps in existing classifiers, and characterize emerging threats. This includes clustering analysis on account behavior features, association rule mining on violation co-occurrence, and network analysis on account interaction graphs.

### 1.2 Team Context

Integrity & Safety is not a standalone team — it operates as a bridge:

- **Upstream**: Trust & Safety (T&S) policy team defines what constitutes a violation (community guidelines, legal requirements, advertiser safety standards). The DS translates policy definitions into measurable, operationalizable metrics.
- **Downstream**: ML/algorithm teams build classifiers, recommendation filters, and enforcement systems. The DS evaluates their performance in production, designs A/B tests for new models, and identifies failure modes.
- **Lateral**: Product teams own the user experience (creator tools, appeals flow, transparency reports). The DS quantifies the impact of safety actions on creator experience and platform health.

The DS is the quantitative backbone connecting these teams. The HM is looking for someone who can independently drive technical analysis, not someone who primarily facilitates between teams.

### 1.3 0-90 Day Success Metrics

**Week 1-2: Orientation and Metric Literacy**
- Map the complete metrics stack: identify every dashboard, understand the data lineage from raw event logs to aggregated KPIs, document the metric definitions and any known limitations.
- Identify the top 5 metrics the team reviews weekly. For each, understand: definition, data source, refresh cadence, historical trends, known seasonality, and current owner.
- Get access to the experimentation platform. Run a simple AA test to verify the randomization infrastructure and baseline variance.
- Review the last 3 months of incident reports and post-mortems to understand common failure modes.

**Month 1: Own One Metric Deep-Dive**
- Select one metric (e.g., false positive rate on content removal) and produce a comprehensive deep-dive: trend decomposition, segment-level analysis (by content type, geography, language, violation category), correlation with upstream classifier changes, and identification of any measurement gaps.
- Deliverable: a written report with specific findings and 2-3 actionable recommendations (e.g., "FPR for Spanish-language content is 2.3x higher than English due to classifier training data imbalance; recommend prioritizing Spanish-language labeling").

**Month 2-3: Design and Analyze First Experiment**
- Design an A/B test for a proposed policy or classifier change. Full design document including: hypothesis, primary and guardrail metrics, unit of randomization, sample size calculation, expected duration, pre-registration of analysis plan.
- Analyze an existing experiment using the pre-registered plan. Produce results with confidence intervals, subgroup analyses, and a clear recommendation.
- Propose one metric improvement: either a new metric that fills a measurement gap, or a refinement to an existing metric definition that reduces noise or bias.

### 1.4 Key Competencies (Calibrated to HM Feedback)

The rejection signal was unambiguous: the HM wanted **technical depth, not collaboration stories**. This means:

1. **Quantitative Rigor**: Every claim must be backed by a number, a method, or a formula. "I improved the model" is insufficient. "I reduced the false negative rate from 18% to 7% by engineering 12 temporal features and tuning the XGBoost max_depth from 6 to 4 with 5-fold cross-validation, which traded 2pp of false positive rate for 11pp of false negative rate reduction — an acceptable tradeoff given the cost asymmetry where a missed violation costs approximately 50x a false removal based on user impact analysis" is what the HM wants to hear.

2. **Open-Ended Problem Structuring**: When given a vague question like "evaluate effectiveness of throttling," the HM wants to see you independently decompose it into measurable components, propose a causal identification strategy, enumerate threats to validity, and suggest concrete next steps — all without being prompted.

3. **Statistical Sophistication**: Beyond textbook definitions, the HM wants to see that you understand when standard methods fail. When does a t-test give wrong answers? (Non-normal data with small n, correlated observations, multiple comparisons.) When does DID fail? (Parallel trends assumption violated.) When is XGBoost worse than logistic regression? (Very small data, need for interpretability, need for calibrated probabilities.)

4. **Domain Application**: Every technical concept should be immediately connected to a content safety example. Don't explain precision and recall abstractly — explain them as "precision is the fraction of removed content that actually violated policy (wrongful removal rate = 1 - precision)" and "recall is the fraction of violating content that we actually caught (miss rate = 1 - recall)."

---

## Section 2: SQL Mastery

All problems use the following schema (content safety domain):

```sql
-- content_items: every piece of content posted
CREATE TABLE content_items (
    content_id BIGINT PRIMARY KEY,
    creator_id BIGINT,
    content_type VARCHAR(20),   -- 'video', 'comment', 'live', 'profile'
    created_at TIMESTAMP,
    region VARCHAR(10),
    language VARCHAR(10),
    view_count BIGINT,
    is_removed BOOLEAN,
    removal_reason VARCHAR(50)
);

-- content_reports: user-submitted reports
CREATE TABLE content_reports (
    report_id BIGINT PRIMARY KEY,
    content_id BIGINT,
    reporter_id BIGINT,
    report_reason VARCHAR(50),  -- 'hate_speech', 'spam', 'violence', 'nudity', etc.
    reported_at TIMESTAMP
);

-- moderation_actions: actions taken by human moderators or automated systems
CREATE TABLE moderation_actions (
    action_id BIGINT PRIMARY KEY,
    content_id BIGINT,
    moderator_type VARCHAR(20), -- 'human', 'automated', 'appeal_review'
    action VARCHAR(20),         -- 'remove', 'restrict', 'warn', 'no_action', 'reinstate'
    violation_category VARCHAR(50),
    actioned_at TIMESTAMP,
    confidence_score FLOAT      -- for automated actions
);

-- user_violations: aggregated violation history per user
CREATE TABLE user_violations (
    user_id BIGINT,
    violation_date DATE,
    violation_category VARCHAR(50),
    strike_count INT,
    account_status VARCHAR(20)  -- 'active', 'warned', 'restricted', 'banned'
);
```

### 2.1 PostgreSQL

All 15 problems below use PostgreSQL syntax. For MySQL 8+ equivalents, see **2.2**. For a dialect comparison chart, see **2.3**.

#### Problem 1 (PG): Basic Ratio — Content Removal Rate by Type

**Problem**: Calculate the content removal rate (fraction of content items that were removed) for each content_type, ordered from highest to lowest.

```sql
SELECT
    content_type,
    COUNT(*) AS total_content,
    SUM(CASE WHEN is_removed THEN 1 ELSE 0 END) AS removed_count,
    ROUND(SUM(CASE WHEN is_removed THEN 1 ELSE 0 END)::DECIMAL / COUNT(*), 4) AS removal_rate
FROM content_items
GROUP BY content_type
ORDER BY removal_rate DESC;
```

**Explanation**: Straightforward aggregation with a conditional sum. The `::DECIMAL` cast prevents integer division. This is the kind of ratio the team monitors daily — if `live` suddenly has 2x the removal rate of `video`, that signals either an emerging abuse vector in live streams or a classifier calibration issue.

#### Problem 2 (PG): Date Filtering — Reports in the Last 30 Days by Reason

**Problem**: Count the number of reports filed in the last 30 days, broken down by report_reason. Only include reasons with more than 100 reports.

```sql
SELECT
    report_reason,
    COUNT(*) AS report_count
FROM content_reports
WHERE reported_at >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY report_reason
HAVING COUNT(*) > 100
ORDER BY report_count DESC;
```

**Explanation**: `HAVING` filters after aggregation, `WHERE` filters before. The `INTERVAL` syntax is PostgreSQL-standard. In a safety context, this query is a quick health check: if "spam" reports suddenly dominate, it may indicate a spam wave; if "hate_speech" reports increase in a specific region, it may follow a real-world event.

#### Problem 3 (PG): JOIN — False Positive Rate by Moderator Type

**Problem**: For each moderator_type, calculate the false positive rate — defined as the fraction of 'remove' actions that were later 'reinstate'd on appeal.

```sql
SELECT
    r.moderator_type,
    COUNT(DISTINCT r.content_id) AS total_removals,
    COUNT(DISTINCT a.content_id) AS reinstated_count,
    ROUND(COUNT(DISTINCT a.content_id)::DECIMAL / NULLIF(COUNT(DISTINCT r.content_id), 0), 4) AS fpr
FROM moderation_actions r
LEFT JOIN moderation_actions a
    ON r.content_id = a.content_id
    AND a.action = 'reinstate'
    AND a.moderator_type = 'appeal_review'
WHERE r.action = 'remove'
GROUP BY r.moderator_type
ORDER BY fpr DESC;
```

**Explanation**: Self-join on `moderation_actions` — removals joined to reinstatements on the same content_id. `NULLIF` prevents division by zero. `LEFT JOIN` ensures we count content that was removed but never appealed/reinstated. This metric directly measures enforcement quality: if `automated` has 8% FPR and `human` has 2%, the automated system needs calibration improvement.

#### Problem 4 (PG): GROUP BY with Multiple Dimensions — Violation Heatmap

**Problem**: For each (region, violation_category) pair, calculate the total number of violations in 2025. Return only pairs with more than 50 violations. Order by violation count descending.

```sql
SELECT
    region,
    violation_category,
    COUNT(*) AS violation_count
FROM content_items ci
JOIN moderation_actions ma ON ci.content_id = ma.content_id
WHERE ma.action = 'remove'
  AND ma.actioned_at >= '2025-01-01'
  AND ma.actioned_at < '2026-01-01'
GROUP BY region, violation_category
HAVING COUNT(*) > 50
ORDER BY violation_count DESC;
```

**Explanation**: Two-dimensional GROUP BY to build a heatmap. This is a core safety analytics query — identifying geographic hotspots for specific violation types helps prioritize classifier improvement (e.g., if "hate_speech" is disproportionately high in region X, check if the classifier supports that region's dominant languages).

#### Problem 5 (PG): Subquery — Repeat Offenders

**Problem**: Find users who have been removed content more than 5 times in the last 90 days but are still in 'active' status.

```sql
SELECT
    ci.creator_id,
    COUNT(DISTINCT ma.content_id) AS removal_count,
    uv.account_status
FROM content_items ci
JOIN moderation_actions ma
    ON ci.content_id = ma.content_id
    AND ma.action = 'remove'
    AND ma.actioned_at >= CURRENT_DATE - INTERVAL '90 days'
JOIN (
    SELECT user_id, account_status
    FROM user_violations
    WHERE violation_date = (SELECT MAX(violation_date) FROM user_violations uv2 WHERE uv2.user_id = user_violations.user_id)
) uv ON ci.creator_id = uv.user_id
WHERE uv.account_status = 'active'
GROUP BY ci.creator_id, uv.account_status
HAVING COUNT(DISTINCT ma.content_id) > 5
ORDER BY removal_count DESC;
```

**Explanation**: Correlated subquery to get each user's most recent status. The main query joins content items to moderation actions, filters for removals in the last 90 days, and aggregates by creator. The HAVING clause filters for repeat offenders. These are enforcement gaps — users who should have been escalated to 'warned' or 'restricted' but slipped through.

#### Problem 6 (PG): Window Function — Running Violation Count per User

**Problem**: For each user violation record, compute the cumulative violation count for that user up to that date.

```sql
SELECT
    user_id,
    violation_date,
    violation_category,
    strike_count,
    SUM(strike_count) OVER (
        PARTITION BY user_id
        ORDER BY violation_date
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS cumulative_strikes
FROM user_violations
ORDER BY user_id, violation_date;
```

**Explanation**: `SUM() OVER(PARTITION BY ... ORDER BY ... ROWS BETWEEN ...)` computes a running total. The frame specification `ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW` is the default for ordered window functions but it is good practice to state it explicitly. This tracks how users escalate over time — useful for calibrating the strike-to-ban policy.

#### Problem 7 (PG): 7-Day Rolling Average — Daily Removal Rate

**Problem**: Compute the 7-day rolling average of the daily content removal rate.

```sql
WITH daily_stats AS (
    SELECT
        DATE(created_at) AS dt,
        COUNT(*) AS total_content,
        SUM(CASE WHEN is_removed THEN 1 ELSE 0 END) AS removed_count,
        SUM(CASE WHEN is_removed THEN 1 ELSE 0 END)::DECIMAL / COUNT(*) AS daily_removal_rate
    FROM content_items
    GROUP BY DATE(created_at)
)
SELECT
    dt,
    daily_removal_rate,
    AVG(daily_removal_rate) OVER (
        ORDER BY dt
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ) AS rolling_7d_avg
FROM daily_stats
ORDER BY dt;
```

**Explanation**: First CTE aggregates to daily grain. Window function `AVG() OVER(ROWS BETWEEN 6 PRECEDING AND CURRENT ROW)` computes the 7-day trailing average (current day + 6 prior days = 7 days). This is the exact pattern from the Round 1 interview. In production, this query powers the daily safety dashboard — the 7-day rolling average smooths out day-of-week effects (weekends typically have different content volumes and moderation staffing).

#### Problem 8 (PG): CTE + Self-Join — Time Between Report and Action

**Problem**: Calculate the median time (in hours) between a content report and the first moderation action, broken down by violation_category.

```sql
WITH first_report AS (
    SELECT
        content_id,
        MIN(reported_at) AS first_reported_at
    FROM content_reports
    GROUP BY content_id
),
first_action AS (
    SELECT
        content_id,
        violation_category,
        MIN(actioned_at) AS first_actioned_at
    FROM moderation_actions
    WHERE action IN ('remove', 'restrict', 'warn', 'no_action')
    GROUP BY content_id, violation_category
)
SELECT
    fa.violation_category,
    PERCENTILE_CONT(0.5) WITHIN GROUP (
        ORDER BY EXTRACT(EPOCH FROM (fa.first_actioned_at - fr.first_reported_at)) / 3600.0
    ) AS median_hours_to_action,
    COUNT(*) AS sample_size
FROM first_report fr
JOIN first_action fa ON fr.content_id = fa.content_id
WHERE fa.first_actioned_at >= fr.first_reported_at
GROUP BY fa.violation_category
ORDER BY median_hours_to_action DESC;
```

**Explanation**: Two CTEs compute the earliest report and earliest action per content item. The join computes the time delta. `PERCENTILE_CONT(0.5)` gives the exact median (interpolated). This metric — time-to-action — is a critical SLA metric for safety teams. If median time-to-action for "CSAM" is anything above a few minutes, that's an emergency.

#### Problem 9 (PG): LAG/LEAD — Detecting Escalation Patterns

**Problem**: For each user, identify when they escalated from 'warned' to 'restricted' status. Show the user, the date they were warned, the date they were restricted, and the number of days between.

```sql
WITH status_changes AS (
    SELECT
        user_id,
        violation_date,
        account_status,
        LAG(account_status) OVER (PARTITION BY user_id ORDER BY violation_date) AS prev_status,
        LAG(violation_date) OVER (PARTITION BY user_id ORDER BY violation_date) AS prev_date
    FROM user_violations
)
SELECT
    user_id,
    prev_date AS warned_date,
    violation_date AS restricted_date,
    violation_date - prev_date AS days_between
FROM status_changes
WHERE prev_status = 'warned'
  AND account_status = 'restricted'
ORDER BY days_between ASC;
```

**Explanation**: `LAG()` accesses the previous row within each user's timeline. Filtering for transitions from 'warned' to 'restricted' identifies escalation events. The `days_between` distribution informs policy: if 80% of escalations happen within 3 days, perhaps the warning is ineffective and we should consider earlier restriction.

#### Problem 10 (PG): RANK + Filtering — Top Reported Creators per Region

**Problem**: For each region, find the top 3 most-reported creators in the last 30 days.

```sql
WITH creator_reports AS (
    SELECT
        ci.region,
        ci.creator_id,
        COUNT(DISTINCT cr.report_id) AS report_count,
        RANK() OVER (
            PARTITION BY ci.region
            ORDER BY COUNT(DISTINCT cr.report_id) DESC
        ) AS rnk
    FROM content_reports cr
    JOIN content_items ci ON cr.content_id = ci.content_id
    WHERE cr.reported_at >= CURRENT_DATE - INTERVAL '30 days'
    GROUP BY ci.region, ci.creator_id
)
SELECT region, creator_id, report_count, rnk
FROM creator_reports
WHERE rnk <= 3
ORDER BY region, rnk;
```

**Explanation**: `RANK()` within each region partition. Using `RANK()` (not `ROW_NUMBER()`) means ties at the boundary get the same rank — if two creators are tied at #3, both appear. This is a daily operations query: the top-reported creators are candidates for manual review or escalated enforcement.

#### Problem 11 (PG): Sessionization — User Report Sessions

**Problem**: Group a user's reports into sessions where consecutive reports within 5 minutes of each other belong to the same session. Count the number of sessions and average reports per session for each reporter.

```sql
WITH ordered_reports AS (
    SELECT
        reporter_id,
        reported_at,
        CASE
            WHEN reported_at - LAG(reported_at) OVER (PARTITION BY reporter_id ORDER BY reported_at) > INTERVAL '5 minutes'
            THEN 1
            ELSE 0
        END AS new_session_flag
    FROM content_reports
),
sessioned AS (
    SELECT
        reporter_id,
        reported_at,
        SUM(new_session_flag) OVER (
            PARTITION BY reporter_id
            ORDER BY reported_at
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS session_id
    FROM ordered_reports
)
SELECT
    reporter_id,
    COUNT(DISTINCT session_id) AS num_sessions,
    COUNT(*)::DECIMAL / NULLIF(COUNT(DISTINCT session_id), 0) AS avg_reports_per_session
FROM sessioned
GROUP BY reporter_id
HAVING COUNT(*) >= 10
ORDER BY avg_reports_per_session DESC;
```

**Explanation**: Classic sessionization pattern. `LAG()` computes time gap between consecutive reports. If gap > 5 minutes, mark as new session. Cumulative sum of the new_session_flag gives session IDs. Users with very high avg_reports_per_session (e.g., 50+ reports in a single session) may be "report bombers" — weaponizing the report system to harass creators. This feature is directly useful for abuse detection.

#### Problem 12 (PG): Funnel Analysis — Content Moderation Pipeline

**Problem**: Build a funnel showing: (1) total content posted, (2) content flagged by automated system, (3) content reviewed by human moderator, (4) content actually removed. Calculate conversion rates at each step.

```sql
WITH funnel AS (
    SELECT
        DATE_TRUNC('week', ci.created_at) AS week,
        COUNT(DISTINCT ci.content_id) AS total_posted,
        COUNT(DISTINCT CASE WHEN auto.content_id IS NOT NULL THEN ci.content_id END) AS auto_flagged,
        COUNT(DISTINCT CASE WHEN human.content_id IS NOT NULL THEN ci.content_id END) AS human_reviewed,
        COUNT(DISTINCT CASE WHEN removed.content_id IS NOT NULL THEN ci.content_id END) AS removed
    FROM content_items ci
    LEFT JOIN moderation_actions auto
        ON ci.content_id = auto.content_id AND auto.moderator_type = 'automated'
    LEFT JOIN moderation_actions human
        ON ci.content_id = human.content_id AND human.moderator_type = 'human'
    LEFT JOIN moderation_actions removed
        ON ci.content_id = removed.content_id AND removed.action = 'remove'
    GROUP BY DATE_TRUNC('week', ci.created_at)
)
SELECT
    week,
    total_posted,
    auto_flagged,
    ROUND(auto_flagged::DECIMAL / NULLIF(total_posted, 0), 4) AS flag_rate,
    human_reviewed,
    ROUND(human_reviewed::DECIMAL / NULLIF(auto_flagged, 0), 4) AS human_review_rate,
    removed,
    ROUND(removed::DECIMAL / NULLIF(human_reviewed, 0), 4) AS removal_rate_given_review
FROM funnel
ORDER BY week;
```

**Explanation**: LEFT JOINs preserve all content in the funnel. Each subsequent step is a subset. The conversion rates between steps reveal bottlenecks: if auto-flag rate is 5% but human_review_rate is only 20%, the human moderation team is overwhelmed (80% of flagged content is never human-reviewed). This directly informs staffing and automation investment decisions.

#### Problem 13 (PG): Cohort Retention — Creator Retention After First Violation

**Problem**: For creators who received their first violation in each month, what fraction continued posting content 30, 60, and 90 days later?

```sql
WITH first_violation AS (
    SELECT
        ci.creator_id,
        DATE_TRUNC('month', MIN(ma.actioned_at)) AS cohort_month,
        MIN(ma.actioned_at) AS first_violation_date
    FROM moderation_actions ma
    JOIN content_items ci ON ma.content_id = ci.content_id
    WHERE ma.action = 'remove'
    GROUP BY ci.creator_id
),
post_activity AS (
    SELECT DISTINCT
        fv.creator_id,
        fv.cohort_month,
        CASE WHEN EXISTS (
            SELECT 1 FROM content_items ci2
            WHERE ci2.creator_id = fv.creator_id
              AND ci2.created_at BETWEEN fv.first_violation_date + INTERVAL '30 days'
                                     AND fv.first_violation_date + INTERVAL '60 days'
        ) THEN 1 ELSE 0 END AS active_30d,
        CASE WHEN EXISTS (
            SELECT 1 FROM content_items ci2
            WHERE ci2.creator_id = fv.creator_id
              AND ci2.created_at BETWEEN fv.first_violation_date + INTERVAL '60 days'
                                     AND fv.first_violation_date + INTERVAL '90 days'
        ) THEN 1 ELSE 0 END AS active_60d,
        CASE WHEN EXISTS (
            SELECT 1 FROM content_items ci2
            WHERE ci2.creator_id = fv.creator_id
              AND ci2.created_at BETWEEN fv.first_violation_date + INTERVAL '90 days'
                                     AND fv.first_violation_date + INTERVAL '120 days'
        ) THEN 1 ELSE 0 END AS active_90d
    FROM first_violation fv
)
SELECT
    cohort_month,
    COUNT(*) AS cohort_size,
    ROUND(AVG(active_30d), 3) AS retention_30d,
    ROUND(AVG(active_60d), 3) AS retention_60d,
    ROUND(AVG(active_90d), 3) AS retention_90d
FROM post_activity
GROUP BY cohort_month
ORDER BY cohort_month;
```

**Explanation**: Cohort analysis applied to content safety. The first CTE identifies each creator's first violation date and assigns a cohort month. The second CTE checks whether the creator posted any new content in the 30-60, 60-90, and 90-120 day windows after that first violation. If retention_30d drops sharply for a specific cohort month, it may correlate with a policy change that was too punitive — valuable for calibrating enforcement severity.

#### Problem 14 (PG): Recursive CTE — Escalation Chains

**Problem**: Some moderation actions reference prior actions (e.g., an appeal references the original removal). Given a parent_action_id column in moderation_actions, find the full chain of actions for a given content item, from initial flag through final resolution.

```sql
-- Assume moderation_actions has an additional column: parent_action_id BIGINT (nullable)

WITH RECURSIVE action_chain AS (
    -- Base case: root actions (no parent)
    SELECT
        action_id,
        content_id,
        moderator_type,
        action,
        actioned_at,
        parent_action_id,
        1 AS chain_depth,
        ARRAY[action_id] AS chain_path
    FROM moderation_actions
    WHERE content_id = 12345  -- parameterize
      AND parent_action_id IS NULL

    UNION ALL

    -- Recursive step: follow the chain
    SELECT
        ma.action_id,
        ma.content_id,
        ma.moderator_type,
        ma.action,
        ma.actioned_at,
        ma.parent_action_id,
        ac.chain_depth + 1,
        ac.chain_path || ma.action_id
    FROM moderation_actions ma
    JOIN action_chain ac ON ma.parent_action_id = ac.action_id
    WHERE ac.chain_depth < 10  -- safety limit
)
SELECT
    chain_depth,
    action_id,
    moderator_type,
    action,
    actioned_at,
    chain_path
FROM action_chain
ORDER BY chain_depth;
```

**Explanation**: Recursive CTE traverses the action tree. The base case selects root actions (no parent). The recursive step joins back to find child actions. The `chain_depth < 10` guard prevents infinite loops. The `chain_path` array tracks the full sequence. In practice, this reveals patterns like: auto-flag -> human-remove -> creator-appeal -> appeal-reinstate -> re-flag -> re-remove, indicating a back-and-forth that suggests either an edge-case violation or an inconsistent policy.

#### Problem 15 (PG): Complex Analytics — Comparing Enforcement Consistency Across Regions

**Problem**: For each violation_category, compute the removal rate by region, then flag regions where the removal rate deviates more than 2 standard deviations from the global mean for that category.

```sql
WITH regional_rates AS (
    SELECT
        ma.violation_category,
        ci.region,
        COUNT(*) FILTER (WHERE ma.action = 'remove') AS removals,
        COUNT(*) AS total_actions,
        COUNT(*) FILTER (WHERE ma.action = 'remove')::DECIMAL / NULLIF(COUNT(*), 0) AS removal_rate
    FROM moderation_actions ma
    JOIN content_items ci ON ma.content_id = ci.content_id
    WHERE ma.actioned_at >= CURRENT_DATE - INTERVAL '90 days'
    GROUP BY ma.violation_category, ci.region
    HAVING COUNT(*) >= 100  -- minimum sample size
),
category_stats AS (
    SELECT
        violation_category,
        AVG(removal_rate) AS mean_rate,
        STDDEV(removal_rate) AS std_rate
    FROM regional_rates
    GROUP BY violation_category
)
SELECT
    rr.violation_category,
    rr.region,
    rr.removal_rate,
    cs.mean_rate,
    cs.std_rate,
    (rr.removal_rate - cs.mean_rate) / NULLIF(cs.std_rate, 0) AS z_score,
    CASE
        WHEN ABS((rr.removal_rate - cs.mean_rate) / NULLIF(cs.std_rate, 0)) > 2 THEN 'OUTLIER'
        ELSE 'NORMAL'
    END AS flag
FROM regional_rates rr
JOIN category_stats cs ON rr.violation_category = cs.violation_category
ORDER BY ABS((rr.removal_rate - cs.mean_rate) / NULLIF(cs.std_rate, 0)) DESC;
```

**Explanation**: Two-pass analysis: first compute regional removal rates, then compute cross-region statistics per category. Z-score identifies outlier regions. A region with z > 2 (much higher removal rate) may have stricter enforcement or a genuine abuse hotspot. A region with z < -2 (much lower removal rate) may have under-enforcement — possibly due to language coverage gaps in the classifier. This query directly supports the team's goal of global enforcement consistency.

### 2.2 MySQL 8+

The same 15 problems rewritten for MySQL 8.0+. Problem statements and explanations are identical to 2.1 — only the SQL dialect changes. Each block shows the full MySQL query followed by **Conversion Notes** listing every syntactic change from the PostgreSQL version.

#### Problem 1 (MySQL): Basic Ratio — Content Removal Rate by Type

```sql
SELECT
    content_type,
    COUNT(*) AS total_content,
    SUM(CASE WHEN is_removed THEN 1 ELSE 0 END) AS removed_count,
    ROUND(CAST(SUM(CASE WHEN is_removed THEN 1 ELSE 0 END) AS DECIMAL(10,4)) / COUNT(*), 4) AS removal_rate
FROM content_items
GROUP BY content_type
ORDER BY removal_rate DESC;
```

**Conversion Notes**:
- `::DECIMAL` → `CAST(... AS DECIMAL(10,4))`

#### Problem 2 (MySQL): Date Filtering — Reports in the Last 30 Days by Reason

```sql
SELECT
    report_reason,
    COUNT(*) AS report_count
FROM content_reports
WHERE reported_at >= CURRENT_DATE - INTERVAL 30 DAY
GROUP BY report_reason
HAVING COUNT(*) > 100
ORDER BY report_count DESC;
```

**Conversion Notes**:
- `INTERVAL '30 days'` → `INTERVAL 30 DAY` (no quotes, singular unit keyword)

#### Problem 3 (MySQL): JOIN — False Positive Rate by Moderator Type

```sql
SELECT
    r.moderator_type,
    COUNT(DISTINCT r.content_id) AS total_removals,
    COUNT(DISTINCT a.content_id) AS reinstated_count,
    ROUND(CAST(COUNT(DISTINCT a.content_id) AS DECIMAL(10,4)) / NULLIF(COUNT(DISTINCT r.content_id), 0), 4) AS fpr
FROM moderation_actions r
LEFT JOIN moderation_actions a
    ON r.content_id = a.content_id
    AND a.action = 'reinstate'
    AND a.moderator_type = 'appeal_review'
WHERE r.action = 'remove'
GROUP BY r.moderator_type
ORDER BY fpr DESC;
```

**Conversion Notes**:
- `::DECIMAL` → `CAST(... AS DECIMAL(10,4))`

#### Problem 4 (MySQL): GROUP BY with Multiple Dimensions — Violation Heatmap

```sql
SELECT
    region,
    violation_category,
    COUNT(*) AS violation_count
FROM content_items ci
JOIN moderation_actions ma ON ci.content_id = ma.content_id
WHERE ma.action = 'remove'
  AND ma.actioned_at >= '2025-01-01'
  AND ma.actioned_at < '2026-01-01'
GROUP BY region, violation_category
HAVING COUNT(*) > 50
ORDER BY violation_count DESC;
```

**Conversion Notes**:
- No changes — already standard SQL.

#### Problem 5 (MySQL): Subquery — Repeat Offenders

```sql
SELECT
    ci.creator_id,
    COUNT(DISTINCT ma.content_id) AS removal_count,
    uv.account_status
FROM content_items ci
JOIN moderation_actions ma
    ON ci.content_id = ma.content_id
    AND ma.action = 'remove'
    AND ma.actioned_at >= CURRENT_DATE - INTERVAL 90 DAY
JOIN (
    SELECT user_id, account_status
    FROM user_violations
    WHERE violation_date = (SELECT MAX(violation_date) FROM user_violations uv2 WHERE uv2.user_id = user_violations.user_id)
) uv ON ci.creator_id = uv.user_id
WHERE uv.account_status = 'active'
GROUP BY ci.creator_id, uv.account_status
HAVING COUNT(DISTINCT ma.content_id) > 5
ORDER BY removal_count DESC;
```

**Conversion Notes**:
- `INTERVAL '90 days'` → `INTERVAL 90 DAY`

#### Problem 6 (MySQL): Window Function — Running Violation Count per User

```sql
SELECT
    user_id,
    violation_date,
    violation_category,
    strike_count,
    SUM(strike_count) OVER (
        PARTITION BY user_id
        ORDER BY violation_date
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS cumulative_strikes
FROM user_violations
ORDER BY user_id, violation_date;
```

**Conversion Notes**:
- No changes — window function syntax is identical in MySQL 8+.

#### Problem 7 (MySQL): 7-Day Rolling Average — Daily Removal Rate

```sql
WITH daily_stats AS (
    SELECT
        DATE(created_at) AS dt,
        COUNT(*) AS total_content,
        SUM(CASE WHEN is_removed THEN 1 ELSE 0 END) AS removed_count,
        CAST(SUM(CASE WHEN is_removed THEN 1 ELSE 0 END) AS DECIMAL(10,4)) / COUNT(*) AS daily_removal_rate
    FROM content_items
    GROUP BY DATE(created_at)
)
SELECT
    dt,
    daily_removal_rate,
    AVG(daily_removal_rate) OVER (
        ORDER BY dt
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ) AS rolling_7d_avg
FROM daily_stats
ORDER BY dt;
```

**Conversion Notes**:
- `::DECIMAL` → `CAST(... AS DECIMAL(10,4))`

#### Problem 8 (MySQL): CTE + Self-Join — Time Between Report and Action

```sql
WITH first_report AS (
    SELECT
        content_id,
        MIN(reported_at) AS first_reported_at
    FROM content_reports
    GROUP BY content_id
),
first_action AS (
    SELECT
        content_id,
        violation_category,
        MIN(actioned_at) AS first_actioned_at
    FROM moderation_actions
    WHERE action IN ('remove', 'restrict', 'warn', 'no_action')
    GROUP BY content_id, violation_category
),
time_deltas AS (
    SELECT
        fa.violation_category,
        TIMESTAMPDIFF(SECOND, fr.first_reported_at, fa.first_actioned_at) / 3600.0 AS hours_to_action
    FROM first_report fr
    JOIN first_action fa ON fr.content_id = fa.content_id
    WHERE fa.first_actioned_at >= fr.first_reported_at
),
ranked AS (
    SELECT
        violation_category,
        hours_to_action,
        ROW_NUMBER() OVER (PARTITION BY violation_category ORDER BY hours_to_action) AS rn,
        COUNT(*) OVER (PARTITION BY violation_category) AS cnt
    FROM time_deltas
)
SELECT
    violation_category,
    AVG(hours_to_action) AS median_hours_to_action,
    cnt AS sample_size
FROM ranked
WHERE rn IN (FLOOR((cnt + 1) / 2), CEIL((cnt + 1) / 2))
GROUP BY violation_category, cnt
ORDER BY median_hours_to_action DESC;
```

**Conversion Notes**:
- `PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY ...)` → ROW_NUMBER-based median pattern (MySQL has no built-in percentile aggregate)
- `EXTRACT(EPOCH FROM (ts1 - ts2))` → `TIMESTAMPDIFF(SECOND, ts2, ts1)` (note: start comes before end in TIMESTAMPDIFF)
- Median logic: FLOOR and CEIL of `(cnt+1)/2` selects one row for odd counts, averages two middle rows for even counts

#### Problem 9 (MySQL): LAG/LEAD — Detecting Escalation Patterns

```sql
WITH status_changes AS (
    SELECT
        user_id,
        violation_date,
        account_status,
        LAG(account_status) OVER (PARTITION BY user_id ORDER BY violation_date) AS prev_status,
        LAG(violation_date) OVER (PARTITION BY user_id ORDER BY violation_date) AS prev_date
    FROM user_violations
)
SELECT
    user_id,
    prev_date AS warned_date,
    violation_date AS restricted_date,
    DATEDIFF(violation_date, prev_date) AS days_between
FROM status_changes
WHERE prev_status = 'warned'
  AND account_status = 'restricted'
ORDER BY days_between ASC;
```

**Conversion Notes**:
- `violation_date - prev_date` (PG returns integer days) → `DATEDIFF(violation_date, prev_date)` (MySQL: end date first, start date second)

#### Problem 10 (MySQL): RANK + Filtering — Top Reported Creators per Region

```sql
WITH creator_reports AS (
    SELECT
        ci.region,
        ci.creator_id,
        COUNT(DISTINCT cr.report_id) AS report_count,
        RANK() OVER (
            PARTITION BY ci.region
            ORDER BY COUNT(DISTINCT cr.report_id) DESC
        ) AS rnk
    FROM content_reports cr
    JOIN content_items ci ON cr.content_id = ci.content_id
    WHERE cr.reported_at >= CURRENT_DATE - INTERVAL 30 DAY
    GROUP BY ci.region, ci.creator_id
)
SELECT region, creator_id, report_count, rnk
FROM creator_reports
WHERE rnk <= 3
ORDER BY region, rnk;
```

**Conversion Notes**:
- `INTERVAL '30 days'` → `INTERVAL 30 DAY`

#### Problem 11 (MySQL): Sessionization — User Report Sessions

```sql
WITH ordered_reports AS (
    SELECT
        reporter_id,
        reported_at,
        CASE
            WHEN TIMESTAMPDIFF(SECOND, LAG(reported_at) OVER (PARTITION BY reporter_id ORDER BY reported_at), reported_at) > 300
            THEN 1
            ELSE 0
        END AS new_session_flag
    FROM content_reports
),
sessioned AS (
    SELECT
        reporter_id,
        reported_at,
        SUM(new_session_flag) OVER (
            PARTITION BY reporter_id
            ORDER BY reported_at
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS session_id
    FROM ordered_reports
)
SELECT
    reporter_id,
    COUNT(DISTINCT session_id) AS num_sessions,
    CAST(COUNT(*) AS DECIMAL(10,4)) / NULLIF(COUNT(DISTINCT session_id), 0) AS avg_reports_per_session
FROM sessioned
GROUP BY reporter_id
HAVING COUNT(*) >= 10
ORDER BY avg_reports_per_session DESC;
```

**Conversion Notes**:
- `reported_at - LAG(reported_at) ... > INTERVAL '5 minutes'` → `TIMESTAMPDIFF(SECOND, LAG(reported_at) OVER (...), reported_at) > 300` (PG timestamp subtraction returns INTERVAL; MySQL requires TIMESTAMPDIFF)
- `::DECIMAL` → `CAST(... AS DECIMAL(10,4))`

#### Problem 12 (MySQL): Funnel Analysis — Content Moderation Pipeline

```sql
WITH funnel AS (
    SELECT
        DATE(ci.created_at - INTERVAL WEEKDAY(ci.created_at) DAY) AS week,
        COUNT(DISTINCT ci.content_id) AS total_posted,
        COUNT(DISTINCT CASE WHEN auto.content_id IS NOT NULL THEN ci.content_id END) AS auto_flagged,
        COUNT(DISTINCT CASE WHEN human.content_id IS NOT NULL THEN ci.content_id END) AS human_reviewed,
        COUNT(DISTINCT CASE WHEN removed.content_id IS NOT NULL THEN ci.content_id END) AS removed
    FROM content_items ci
    LEFT JOIN moderation_actions auto
        ON ci.content_id = auto.content_id AND auto.moderator_type = 'automated'
    LEFT JOIN moderation_actions human
        ON ci.content_id = human.content_id AND human.moderator_type = 'human'
    LEFT JOIN moderation_actions removed
        ON ci.content_id = removed.content_id AND removed.action = 'remove'
    GROUP BY DATE(ci.created_at - INTERVAL WEEKDAY(ci.created_at) DAY)
)
SELECT
    week,
    total_posted,
    auto_flagged,
    ROUND(CAST(auto_flagged AS DECIMAL(10,4)) / NULLIF(total_posted, 0), 4) AS flag_rate,
    human_reviewed,
    ROUND(CAST(human_reviewed AS DECIMAL(10,4)) / NULLIF(auto_flagged, 0), 4) AS human_review_rate,
    removed,
    ROUND(CAST(removed AS DECIMAL(10,4)) / NULLIF(human_reviewed, 0), 4) AS removal_rate_given_review
FROM funnel
ORDER BY week;
```

**Conversion Notes**:
- `DATE_TRUNC('week', ts)` → `DATE(ts - INTERVAL WEEKDAY(ts) DAY)` (WEEKDAY returns 0=Monday, aligning with PG ISO week start)
- `::DECIMAL` (3 occurrences) → `CAST(... AS DECIMAL(10,4))`

#### Problem 13 (MySQL): Cohort Retention — Creator Retention After First Violation

```sql
WITH first_violation AS (
    SELECT
        ci.creator_id,
        DATE(DATE_FORMAT(MIN(ma.actioned_at), '%Y-%m-01')) AS cohort_month,
        MIN(ma.actioned_at) AS first_violation_date
    FROM moderation_actions ma
    JOIN content_items ci ON ma.content_id = ci.content_id
    WHERE ma.action = 'remove'
    GROUP BY ci.creator_id
),
post_activity AS (
    SELECT DISTINCT
        fv.creator_id,
        fv.cohort_month,
        CASE WHEN EXISTS (
            SELECT 1 FROM content_items ci2
            WHERE ci2.creator_id = fv.creator_id
              AND ci2.created_at BETWEEN fv.first_violation_date + INTERVAL 30 DAY
                                     AND fv.first_violation_date + INTERVAL 60 DAY
        ) THEN 1 ELSE 0 END AS active_30d,
        CASE WHEN EXISTS (
            SELECT 1 FROM content_items ci2
            WHERE ci2.creator_id = fv.creator_id
              AND ci2.created_at BETWEEN fv.first_violation_date + INTERVAL 60 DAY
                                     AND fv.first_violation_date + INTERVAL 90 DAY
        ) THEN 1 ELSE 0 END AS active_60d,
        CASE WHEN EXISTS (
            SELECT 1 FROM content_items ci2
            WHERE ci2.creator_id = fv.creator_id
              AND ci2.created_at BETWEEN fv.first_violation_date + INTERVAL 90 DAY
                                     AND fv.first_violation_date + INTERVAL 120 DAY
        ) THEN 1 ELSE 0 END AS active_90d
    FROM first_violation fv
)
SELECT
    cohort_month,
    COUNT(*) AS cohort_size,
    ROUND(AVG(active_30d), 3) AS retention_30d,
    ROUND(AVG(active_60d), 3) AS retention_60d,
    ROUND(AVG(active_90d), 3) AS retention_90d
FROM post_activity
GROUP BY cohort_month
ORDER BY cohort_month;
```

**Conversion Notes**:
- `DATE_TRUNC('month', ts)` → `DATE(DATE_FORMAT(ts, '%Y-%m-01'))` (truncate to first of month)
- `INTERVAL '30 days'` → `INTERVAL 30 DAY` (6 occurrences across the BETWEEN clauses)

#### Problem 14 (MySQL): Recursive CTE — Escalation Chains

```sql
-- Assume moderation_actions has an additional column: parent_action_id BIGINT (nullable)

WITH RECURSIVE action_chain AS (
    -- Base case: root actions (no parent)
    SELECT
        action_id,
        content_id,
        moderator_type,
        action,
        actioned_at,
        parent_action_id,
        1 AS chain_depth,
        CAST(action_id AS CHAR(200)) AS chain_path
    FROM moderation_actions
    WHERE content_id = 12345  -- parameterize
      AND parent_action_id IS NULL

    UNION ALL

    -- Recursive step: follow the chain
    SELECT
        ma.action_id,
        ma.content_id,
        ma.moderator_type,
        ma.action,
        ma.actioned_at,
        ma.parent_action_id,
        ac.chain_depth + 1,
        CONCAT(ac.chain_path, ',', ma.action_id)
    FROM moderation_actions ma
    JOIN action_chain ac ON ma.parent_action_id = ac.action_id
    WHERE ac.chain_depth < 10  -- safety limit
)
SELECT
    chain_depth,
    action_id,
    moderator_type,
    action,
    actioned_at,
    chain_path
FROM action_chain
ORDER BY chain_depth;
```

**Conversion Notes**:
- `ARRAY[action_id]` → `CAST(action_id AS CHAR(200))` (MySQL has no array type; use comma-separated string instead)
- `chain_path || ma.action_id` (PG array append) → `CONCAT(chain_path, ',', ma.action_id)` (string concatenation)
- Output format difference: PG shows `{1,2,3}` (array literal), MySQL shows `1,2,3` (plain string)

#### Problem 15 (MySQL): Complex Analytics — Comparing Enforcement Consistency Across Regions

```sql
WITH regional_rates AS (
    SELECT
        ma.violation_category,
        ci.region,
        SUM(CASE WHEN ma.action = 'remove' THEN 1 ELSE 0 END) AS removals,
        COUNT(*) AS total_actions,
        CAST(SUM(CASE WHEN ma.action = 'remove' THEN 1 ELSE 0 END) AS DECIMAL(10,4)) / NULLIF(COUNT(*), 0) AS removal_rate
    FROM moderation_actions ma
    JOIN content_items ci ON ma.content_id = ci.content_id
    WHERE ma.actioned_at >= CURRENT_DATE - INTERVAL 90 DAY
    GROUP BY ma.violation_category, ci.region
    HAVING COUNT(*) >= 100  -- minimum sample size
),
category_stats AS (
    SELECT
        violation_category,
        AVG(removal_rate) AS mean_rate,
        STDDEV(removal_rate) AS std_rate
    FROM regional_rates
    GROUP BY violation_category
)
SELECT
    rr.violation_category,
    rr.region,
    rr.removal_rate,
    cs.mean_rate,
    cs.std_rate,
    (rr.removal_rate - cs.mean_rate) / NULLIF(cs.std_rate, 0) AS z_score,
    CASE
        WHEN ABS((rr.removal_rate - cs.mean_rate) / NULLIF(cs.std_rate, 0)) > 2 THEN 'OUTLIER'
        ELSE 'NORMAL'
    END AS flag
FROM regional_rates rr
JOIN category_stats cs ON rr.violation_category = cs.violation_category
ORDER BY ABS((rr.removal_rate - cs.mean_rate) / NULLIF(cs.std_rate, 0)) DESC;
```

**Conversion Notes**:
- `COUNT(*) FILTER (WHERE ma.action = 'remove')` → `SUM(CASE WHEN ma.action = 'remove' THEN 1 ELSE 0 END)` (MySQL has no FILTER clause; 2 occurrences)
- `::DECIMAL` → `CAST(... AS DECIMAL(10,4))`
- `INTERVAL '90 days'` → `INTERVAL 90 DAY`
- `STDDEV()` works identically in both dialects

### 2.3 PostgreSQL vs MySQL — DQL Quick Reference

| Category | Construct | PostgreSQL | MySQL 8+ | Notes |
|----------|-----------|-----------|----------|-------|
| **Type Casting** | Cast operator | `expr::TYPE` | `CAST(expr AS TYPE)` | PG also supports `CAST()` |
| **Type Casting** | Decimal cast | `expr::DECIMAL` | `CAST(expr AS DECIMAL(M,D))` | MySQL requires precision/scale |
| **Date/Time** | Interval literal | `INTERVAL '30 days'` | `INTERVAL 30 DAY` | MySQL: no quotes, singular unit |
| **Date/Time** | Truncate to week | `DATE_TRUNC('week', ts)` | `DATE(ts - INTERVAL WEEKDAY(ts) DAY)` | Both use ISO Monday-start weeks |
| **Date/Time** | Truncate to month | `DATE_TRUNC('month', ts)` | `DATE(DATE_FORMAT(ts, '%Y-%m-01'))` | — |
| **Date/Time** | Extract epoch | `EXTRACT(EPOCH FROM interval)` | `TIMESTAMPDIFF(SECOND, start, end)` | TIMESTAMPDIFF returns integer |
| **Date/Time** | Date difference | `date1 - date2` (returns int) | `DATEDIFF(date1, date2)` | Same arg order |
| **Date/Time** | Timestamp difference | `ts1 - ts2` (returns interval) | `TIMESTAMPDIFF(unit, ts2, ts1)` | MySQL arg order: unit, start, end |
| **Date/Time** | Current date | `CURRENT_DATE` | `CURRENT_DATE` | Identical |
| **Aggregation** | Filtered aggregate | `COUNT(*) FILTER (WHERE cond)` | `SUM(CASE WHEN cond THEN 1 ELSE 0 END)` | MySQL has no FILTER clause |
| **Aggregation** | Median | `PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY x)` | ROW_NUMBER-based CTE (see below) | MySQL has no built-in percentile |
| **Aggregation** | Std deviation (sample) | `STDDEV(x)` | `STDDEV(x)` | Identical |
| **Aggregation** | Std deviation (pop) | `STDDEV_POP(x)` | `STDDEV_POP(x)` | Identical |
| **Arrays** | Array literal | `ARRAY[val]` | No native equivalent | Use JSON_ARRAY(val) or CAST to CHAR |
| **Arrays** | Array append | `arr \|\| val` | `CONCAT(str, ',', val)` | MySQL: simulate with strings |
| **Window Functions** | ROW_NUMBER, RANK, etc. | Supported | Supported (MySQL 8+) | Identical syntax |
| **Window Functions** | Frame specification | `ROWS BETWEEN ...` | `ROWS BETWEEN ...` | Identical syntax |
| **Recursive CTEs** | Syntax | `WITH RECURSIVE ...` | `WITH RECURSIVE ...` | Identical base syntax |
| **Recursive CTEs** | Path tracking | Array column | String concatenation via CONCAT | Different output format |
| **Other** | Division guard | `NULLIF(expr, 0)` | `NULLIF(expr, 0)` | Identical |
| **Other** | Boolean handling | `CASE WHEN is_removed THEN 1` | `CASE WHEN is_removed THEN 1` | Both support boolean in CASE |
| **Other** | ROUND | `ROUND(expr, n)` | `ROUND(expr, n)` | Identical |

**MySQL Median Pattern (reusable template)**:

```sql
-- Computes exact median per group using ROW_NUMBER.
-- Handles both odd (one middle row) and even (average of two middle rows) counts.
WITH ranked AS (
    SELECT
        group_col,
        value_col,
        ROW_NUMBER() OVER (PARTITION BY group_col ORDER BY value_col) AS rn,
        COUNT(*) OVER (PARTITION BY group_col) AS cnt
    FROM source_table
)
SELECT
    group_col,
    AVG(value_col) AS median_value
FROM ranked
WHERE rn IN (FLOOR((cnt + 1) / 2), CEIL((cnt + 1) / 2))
GROUP BY group_col;
```

---

## Section 3: Statistics & ML

### 3.1 Hypothesis Testing

**Z-test vs. T-test: When to Use Each**

The z-test assumes the population variance is known or the sample is large (n > 30 by convention, though this is approximate). The test statistic is:

z = (x_bar - mu_0) / (sigma / sqrt(n))

The t-test is used when the population variance is unknown and must be estimated from the sample. The test statistic is:

t = (x_bar - mu_0) / (s / sqrt(n))

where s is the sample standard deviation. The t-distribution has heavier tails than the standard normal, with the degrees of freedom = n - 1. As n grows, the t-distribution converges to the normal distribution.

**When each matters in content safety:**
- **Z-test**: When comparing daily violation rates across millions of content items. With n in the millions, the distinction between z and t is negligible. Use z for operational dashboards.
- **T-test**: When comparing moderation accuracy across a small panel of human moderators (e.g., n = 15 moderators). The sample size is small enough that the t-distribution's heavier tails matter for proper coverage of confidence intervals.
- **Two-sample t-test**: When comparing removal rates between treatment and control in an A/B test. The Welch's t-test (unequal variances) is generally preferred because treatment and control may have different variance if the treatment changes the distribution of outcomes.

**Assumptions and violations:**
- **Independence**: Violated when content items from the same creator are correlated. Solution: cluster the analysis at the creator level, using cluster-robust standard errors.
- **Normality**: For proportions with small base rates (e.g., violation rate = 0.01%), the normal approximation may be poor. Solution: use exact binomial tests or the Wilson score interval.
- **Homoscedasticity** (equal variances): Welch's t-test does not assume this — always prefer Welch's over Student's t.

### 3.2 Power Analysis

Power = P(reject H0 | H1 is true) = 1 - beta. The four quantities are linked:

1. **alpha** (significance level): typically 0.05. Lowering alpha requires larger n.
2. **beta** (Type II error rate): typically 0.20, so power = 0.80. Lowering beta (increasing power) requires larger n.
3. **Effect size** (delta): the minimum detectable effect. Smaller effects require larger n.
4. **n** (sample size): what we solve for.

For a two-sample z-test on proportions (control rate p_c, treatment rate p_t):

n_per_arm = (z_{alpha/2} + z_{beta})^2 * (p_c(1-p_c) + p_t(1-p_t)) / (p_t - p_c)^2

**Content safety example**: Suppose the current false positive rate is p_c = 5%. A new classifier claims to reduce it to p_t = 4% (delta = 1pp). With alpha = 0.05 and power = 0.80:

n_per_arm = (1.96 + 0.84)^2 * (0.05*0.95 + 0.04*0.96) / (0.01)^2 = 7.84 * (0.0475 + 0.0384) / 0.0001 = 7.84 * 0.0859 / 0.0001 = 6,735 per arm

So approximately 6,735 content items per arm. If the platform processes millions daily, this experiment can reach significance in hours. But if we need to detect a 0.1pp change, n scales up 100x to ~673,500 per arm.

**Variance reduction techniques**: CUPED (Controlled-experiment Using Pre-Experiment Data) uses pre-experiment covariates to reduce variance. If the pre-experiment covariate explains R^2 of the variance, the effective sample size increases by a factor of 1/(1-R^2). For safety metrics with strong temporal autocorrelation, CUPED can reduce required sample size by 30-50%.

### 3.3 Type I and Type II Errors in Content Safety

| Error Type | Statistical Definition | Content Safety Meaning | Cost |
|---|---|---|---|
| Type I (FP) | Reject H0 when H0 is true | Remove content that doesn't violate policy | Creator frustration, appeal burden, chilling effect on speech, potential revenue loss |
| Type II (FN) | Fail to reject H0 when H1 is true | Fail to remove content that violates policy | User harm (exposure to hate speech, violence, CSAM), regulatory risk, advertiser brand safety concerns |

The cost asymmetry is domain-dependent. For CSAM, the cost of a Type II error (missed content) is astronomical — legal liability, child safety. For borderline humor/satire, the cost of a Type I error (wrongful removal) is high — creator backlash, accusations of censorship. The optimal threshold for the classifier depends on this cost ratio. If C_FN / C_FP = 50 (missing a violation is 50x worse than wrongful removal), the optimal decision threshold shifts left, increasing recall at the expense of precision.

### 3.4 Confidence Intervals

A 95% confidence interval means: if we repeated this experiment many times, 95% of the computed intervals would contain the true parameter. It does NOT mean there is a 95% probability that the true parameter lies in this specific interval.

**Bootstrap confidence intervals** are useful when the sampling distribution is unknown or the statistic is complex (e.g., median time-to-action, ratio of two correlated proportions). Algorithm:
1. Resample with replacement from the observed data, B times (B = 10,000).
2. Compute the statistic on each bootstrap sample.
3. Take the 2.5th and 97.5th percentiles of the bootstrap distribution.

The BCa (bias-corrected and accelerated) bootstrap adjusts for bias and skewness in the bootstrap distribution and generally outperforms the percentile method.

### 3.5 Multiple Testing Corrections

When running multiple hypothesis tests simultaneously, the probability of at least one false positive inflates. With m independent tests at alpha = 0.05, P(at least one FP) = 1 - (1-0.05)^m. For m = 20 tests, this is 64%.

**Bonferroni correction**: Divide alpha by m. Test each hypothesis at alpha/m. Controls the Family-Wise Error Rate (FWER) — the probability of making even one Type I error. Very conservative. Use when each false positive has severe consequences (e.g., concluding a classifier is better when it isn't, leading to a full production rollout of an inferior model).

**Benjamini-Hochberg (BH) procedure**: Controls the False Discovery Rate (FDR) — the expected fraction of rejected hypotheses that are false positives. Less conservative than Bonferroni. Algorithm:
1. Sort p-values: p_(1) <= p_(2) <= ... <= p_(m).
2. Find the largest k such that p_(k) <= k * alpha / m.
3. Reject all hypotheses with p-values <= p_(k).

Use BH when running exploratory analyses (e.g., testing for metric differences across 50 violation categories) where some false positives are tolerable but you want to control their proportion.

**Content safety application**: In an A/B test of a new moderation policy, you might look at 15 metrics (violation rate, FPR, appeal rate, creator churn, time-to-action, etc.). Bonferroni at alpha = 0.05/15 = 0.0033 per test is appropriate for the primary metric (violation rate). BH at FDR = 0.10 is appropriate for the secondary/exploratory metrics.

### 3.6 XGBoost Deep-Dive

**How Gradient Boosting Works:**

Gradient boosting builds an additive model F(x) = sum_{t=1}^{T} eta * h_t(x), where each h_t is a weak learner (decision tree) fit to the negative gradient of the loss function evaluated at the current prediction. For squared error loss, the negative gradient is the residual. For log-loss (binary classification), the negative gradient is y - p, where p is the current predicted probability. Each tree corrects the errors of the ensemble so far.

XGBoost specifically uses a second-order Taylor expansion of the loss function, fitting each tree to both the first-order gradient g_i and the second-order gradient (Hessian) h_i. The optimal leaf weight for a leaf containing sample set I is:

w* = -sum(g_i) / (sum(h_i) + lambda)

and the corresponding reduction in objective is:

Gain = 0.5 * [sum(g_i)]^2 / (sum(h_i) + lambda)

This second-order information allows XGBoost to compute optimal splits more efficiently and incorporate regularization naturally.

**Key Hyperparameters and Their Overfitting Effects:**

| Parameter | Typical Range | Effect on Overfitting | Mechanism |
|---|---|---|---|
| `max_depth` | 3-10 | Lower = less overfitting | Limits tree complexity. Depth 6 = max 64 leaves. In safety (many features, sparse signals), depth 4-6 usually works. |
| `learning_rate` (eta) | 0.01-0.3 | Lower = less overfitting | Shrinks each tree's contribution. Must increase `n_estimators` proportionally. |
| `n_estimators` | 100-10000 | Higher = more overfitting (without early stopping) | More trees = more capacity. Always use early stopping with validation set. |
| `subsample` | 0.5-1.0 | Lower = less overfitting | Row subsampling per tree (stochastic gradient boosting). 0.8 = each tree sees 80% of data. |
| `colsample_bytree` | 0.5-1.0 | Lower = less overfitting | Column subsampling per tree (like random forest feature bagging). Reduces feature co-adaptation. |
| `min_child_weight` | 1-10 | Higher = less overfitting | Minimum sum of Hessian in a leaf. For binary classification with balanced classes, Hessian ~ p(1-p) ~ 0.25 per sample, so min_child_weight=5 means minimum ~20 samples per leaf. |
| `gamma` | 0-5 | Higher = less overfitting | Minimum loss reduction for a split. Post-pruning: if a split doesn't improve the objective by at least gamma, it's removed. |
| `reg_alpha` (L1) | 0-1 | Higher = less overfitting | L1 regularization on leaf weights. Encourages sparsity (some leaves get weight 0). |
| `reg_lambda` (L2) | 1-10 | Higher = less overfitting | L2 regularization on leaf weights. Shrinks all leaf weights toward 0. Default is 1. |

**Safety-specific tuning**: For fraud/violation detection with 0.1% positive rate, set `scale_pos_weight = 999` (ratio of negative to positive), use `eval_metric = 'aucpr'` (area under precision-recall curve, more informative than AUC-ROC for imbalanced data), and monitor both precision and recall on the validation set separately.

### 3.7 Overfitting Prevention (Comprehensive)

1. **Regularization**: L1 (feature selection effect), L2 (weight shrinkage), and tree-specific regularization (max_depth, min_child_weight, gamma).
2. **Early stopping**: Monitor validation loss; stop training when validation loss hasn't improved for `early_stopping_rounds` iterations. This automatically selects `n_estimators`.
3. **Cross-validation**: k-fold (typically k=5) for hyperparameter selection. Stratified k-fold for imbalanced data to preserve class ratios in each fold.
4. **Feature selection**: Remove features with low importance (XGBoost's `feature_importances_`), high correlation with other features (VIF > 10), or data leakage (features that wouldn't be available at prediction time).
5. **Data augmentation**: For text-based content safety features, augment training data with paraphrases.
6. **Ensemble**: Blend XGBoost with logistic regression and neural network — the combined model is typically more robust than any individual.

### 3.8 Class Imbalance

Content moderation inherently involves extreme class imbalance. Violation rates of 0.01-1% are common.

**SMOTE (Synthetic Minority Over-sampling Technique)**: Generates synthetic positive samples by interpolating between existing positives and their k-nearest neighbors in feature space. Effective when: feature space is continuous, decision boundary is smooth, and positives aren't too sparse. Can create noisy samples if positives overlap with negatives. Variants: Borderline-SMOTE (only oversample near the decision boundary), ADASYN (focus oversampling on harder examples).

**Class weights**: Set the loss function to weight positive samples higher. In XGBoost, `scale_pos_weight = N_neg / N_pos`. In logistic regression, `class_weight='balanced'`. This doesn't change the data, only the optimization objective. Equivalent to upsampling positives by the weight ratio.

**Threshold tuning**: Train the model on the original (imbalanced) data, then select the classification threshold that optimizes the target metric. Default threshold = 0.5 is rarely optimal for imbalanced data. Plot precision vs. recall at various thresholds; select the threshold where the cost-weighted sum of FP and FN is minimized. For content moderation:

Optimal threshold = argmin_t [ C_FP * FP(t) + C_FN * FN(t) ]

**Precision-recall tradeoffs in content moderation**: The operating point depends on the violation category. For CSAM: optimize for recall (miss nothing), accept lower precision (more false removals reviewed by humans). For borderline humor: optimize for precision (don't wrongly remove), accept lower recall (some violations slip through but get caught by reports).

### 3.9 Bias-Variance Tradeoff

Total error = Bias^2 + Variance + Irreducible noise.

- **High bias** (underfitting): Model is too simple. Logistic regression on a nonlinear decision boundary. Training and validation errors are both high.
- **High variance** (overfitting): Model is too complex. Deep decision tree on small data. Training error is low but validation error is high.

**Practical example**: A content safety classifier using 3 hand-crafted features (text length, number of flagged words, account age) has high bias — it cannot capture complex abuse patterns. Adding 200 behavioral features (posting frequency, network structure, temporal patterns) and using a deep XGBoost model reduces bias but may increase variance. The sweet spot is found via cross-validation: track both training and validation error as model complexity increases.

**Regularization paths**: Plot validation error as a function of regularization strength (e.g., lambda in L2). At lambda = 0, maximum variance. As lambda increases, variance decreases but bias increases. The optimal lambda minimizes validation error.

### 3.10 Simpson's Paradox in Content Moderation

**Example**: Automated moderation has a 90% accuracy overall, while human moderation has 85% accuracy overall. Should we replace humans with automation?

Disaggregate by violation category:
- **Spam**: Automated = 95% accuracy on 90K cases. Human = 92% accuracy on 10K cases.
- **Hate speech**: Automated = 60% accuracy on 10K cases. Human = 78% accuracy on 40K cases.

Overall automated accuracy: (0.95*90K + 0.60*10K) / 100K = 91.5%. Overall human accuracy: (0.92*10K + 0.78*40K) / 50K = 80.8%.

But within each category, human accuracy on hate speech (78%) dominates automated (60%). The paradox arises because automated systems handle a much higher volume of the easy category (spam), inflating their overall accuracy. The correct conclusion: automated excels at spam, humans excel at hate speech. The optimal system routes each category to the better moderator.

This paradox is directly relevant when evaluating aggregate safety metrics — always disaggregate by content type, language, and region.

---

## Section 4: Product Cases

### Case 1: "Should We Launch a Stricter Content Moderation Policy?"

**Context**: The Trust & Safety team proposes lowering the classifier confidence threshold from 0.85 to 0.70, which would flag and remove more content. The question is whether this improves platform safety enough to justify the increased false positive rate.

**Step 1: Define Metrics**

*North Star*: **Violating View Rate (VVR)** = total views on content that violates policy / total content views. This captures both the volume of violating content and its reach. If VVR decreases, the policy is working.

*Guardrail Metrics* (must not degrade beyond acceptable thresholds):
- **False Positive Rate (FPR)**: Fraction of removed content that was actually compliant. Measured through a random sample of removals audited by human reviewers. Threshold: FPR must not increase by more than 2pp (e.g., from 5% to 7%).
- **Creator Appeal Rate**: Fraction of affected creators who file appeals. Threshold: appeal rate must not increase by more than 50% relative.
- **Creator Churn (7-day)**: Fraction of active creators who stop posting within 7 days of content removal. Threshold: churn must not increase by more than 1pp.
- **DAU/MAU**: Overall engagement must not drop by more than 0.5%.

*Diagnostic Metrics*: Per-category violation rates, per-region enforcement rates, time-to-action, classifier precision and recall at the new threshold, human review queue depth.

**Step 2: Experiment Design**

*Unit of randomization*: Creator-level. All content from a creator is moderated under the same threshold. This avoids the inconsistency of a single creator having some content moderated at 0.85 and other content at 0.70.

*Sample size*: With baseline VVR = 0.3%, we want to detect a 10% relative reduction (to 0.27%). MDE = 0.03pp. Using the formula for proportions: n = (1.96 + 0.84)^2 * 2 * 0.003 * 0.997 / (0.0003)^2 = approximately 1.65M creators per arm. At 1B monthly active creators, this is feasible — allocate 1% to treatment.

*Duration*: Minimum 2 weeks to capture weekly cyclicity. 4 weeks preferred to observe lagged effects on creator behavior. Pre-register the analysis plan before launch.

*Interference considerations*: If a bad actor's content is removed in treatment but their alt accounts are in control, the control group's metrics improve (spillover). To mitigate: randomize at the community/cluster level using network clustering algorithms to minimize cross-cluster edges.

**Step 3: Analyze Tradeoffs**

The fundamental tradeoff is safety vs. creator freedom. Lowering the threshold from 0.85 to 0.70 will:
- Increase recall (catch more violations): estimate from offline evaluation using held-out labeled data. If recall goes from 78% to 89%, that's 11pp more violations caught — significant safety improvement.
- Increase false positives: precision may drop from 92% to 83%. The absolute number of wrongful removals increases substantially because the base volume is so large.
- Increase human review burden: more appeals means more work for the moderation team.

**Step 4: Decision Framework**

Define a net benefit function:

Net Benefit = C_safety * delta_VVR - C_FP * delta_FPR - C_churn * delta_churn

Assign weights based on company priorities (these should be elicited from leadership, but propose defaults):
- C_safety = 100 (per pp reduction in VVR)
- C_FP = 20 (per pp increase in FPR)
- C_churn = 50 (per pp increase in creator churn)

If Net Benefit > 0 AND all guardrails are within thresholds, **launch**. If Net Benefit > 0 BUT a guardrail is breached, **iterate** (investigate the guardrail failure, propose mitigation). If Net Benefit <= 0, **do not launch**.

Present results with confidence intervals: "VVR decreased by 0.04pp (95% CI: [0.02, 0.06]), FPR increased by 1.8pp (95% CI: [1.2, 2.4]). Net Benefit = 100*0.04 - 20*1.8 = 4.0 - 36.0 = -32.0. The false positive cost outweighs the safety benefit. Recommend not launching at threshold 0.70. Investigate intermediate threshold (e.g., 0.78) that may have a better tradeoff."

### Case 2: "How Would You Detect Coordinated Inauthentic Behavior?"

**Context**: Coordinated inauthentic behavior (CIB) includes bot farms amplifying content, astroturfing campaigns creating fake grassroots support, and coordinated harassment campaigns targeting specific creators. CIB is distinct from individual bad actors because the threat comes from the coordination, not any single account.

**Step 1: Define the Problem Precisely**

CIB has three hallmarks: (1) multiple accounts acting in concert, (2) behavior that appears organic individually but is statistically improbable in aggregate, (3) a shared objective (amplify/suppress content, harass, spread misinformation). The detection problem is: given the set of all accounts, identify clusters of accounts that exhibit coordinated behavior.

**Step 2: Feature Engineering**

*Temporal features*: Compute inter-event time distributions per account. Coordinated accounts often act in synchronized bursts — compute the cross-correlation of posting times between account pairs. Normal users have diverse, irregular posting patterns; coordinated accounts have suspiciously similar temporal signatures. Specifically, compute: (a) time-of-day entropy (bots have low entropy — they post at uniform times), (b) burst score (fraction of activity in the top 5% of active windows), (c) pairwise posting time correlation with other accounts in the same content niche.

*Network features*: Build the interaction graph (follows, likes, comments, shares). Compute: (a) bipartite clustering coefficient (fraction of account's targets that also target each other — high for coordinated groups), (b) shared audience overlap (Jaccard similarity of follower sets), (c) graph embeddings via node2vec or GNN — accounts in the same CIB campaign will cluster in embedding space.

*Content similarity features*: Compute pairwise content similarity (TF-IDF cosine similarity, or sentence embeddings via SBERT). Coordinated accounts often post identical or near-identical content. Also: (a) hashtag co-usage (Jaccard index of hashtag sets), (b) URL sharing patterns (same URLs shared by many accounts in a short window).

*Account metadata features*: Account age, profile completeness, avatar uniqueness (perceptual hash), username entropy, follower/following ratio. Bot accounts often have: new creation dates clustered in time, default/stock avatars, low-entropy usernames (sequential patterns), and abnormal follower/following ratios.

**Step 3: Modeling Approach**

*Stage 1 — Pairwise similarity*: Compute pairwise similarity scores using the features above. For computational efficiency (O(n^2) is infeasible for billions of accounts), use locality-sensitive hashing (LSH) to generate candidate pairs, then compute exact similarity only for candidates.

*Stage 2 — Graph-based clustering*: Build a similarity graph where edges connect accounts with similarity above a threshold. Apply community detection (Louvain, Leiden) to find clusters. Each cluster is a CIB candidate.

*Stage 3 — Cluster-level scoring*: For each detected cluster, compute an anomaly score based on: cluster size, internal cohesion (average pairwise similarity), temporal synchrony, content uniformity, and deviation from organic baseline. Use an isolation forest or a trained XGBoost classifier on labeled cluster data (past confirmed CIB campaigns as positives, organic communities as negatives).

*Stage 4 — Human review*: Clusters above the anomaly threshold are routed to human investigators who confirm or reject the CIB designation.

**Step 4: Evaluation**

*Precision-recall tradeoffs*: Precision matters because a false positive means disrupting a genuine community. Recall matters because a missed CIB campaign can cause large-scale harm. Set the operating point to achieve high precision (>90%) at the cluster level — it's better to miss some campaigns than to falsely accuse legitimate groups. Supplement recall by continuously improving features and adding new signal sources.

*Adversarial robustness*: CIB operators adapt. They introduce randomized delays to break temporal synchrony, use paraphrasing to reduce content similarity, and create heterogeneous account profiles. Countermeasures: (a) use higher-order features that are harder to fake (network topology is expensive to fake convincingly), (b) adversarial training — include known evasion tactics in the training data, (c) continuous model retraining with a human-in-the-loop labeling pipeline.

*Metrics for the detection system*: Track: number of CIB campaigns detected per month, median detection latency (time from campaign start to detection), precision@90recall on a held-out evaluation set, and the reduction in amplified violating content attributable to CIB takedowns.

---

## Section 5: HM Round — STAR Stories

Each story follows Situation-Task-Action-Result with deep technical substance. Numbers, methods, and tools are emphasized throughout.

### Story 1: Improved 3D Medical Image Reconstruction Quality with Conditional GAN

**Situation**: The existing 3D PET image reconstruction pipeline used OSEM (Ordered Subset Expectation Maximization) with 4 iterations and 8 subsets, producing images with SSIM = 0.82 against the ground truth and visibly noisy textures that degraded diagnostic confidence. **Task**: Improve reconstruction quality without additional scan time or dose. **Action**: Implemented a conditional GAN (pix2pix3D architecture) with a U-Net generator (encoder: 5 downsampling blocks with 3D convolutions, bottleneck: 512 channels, decoder: 5 upsampling blocks with skip connections) and a PatchGAN discriminator operating on 70x70x70 voxel patches. The loss function combined adversarial loss (lambda_adv = 1), L1 reconstruction loss (lambda_L1 = 100), and a perceptual loss using VGG features (lambda_perceptual = 10). Trained on 1,200 paired OSEM/ground-truth volumes with Adam optimizer (lr=2e-4, beta1=0.5, beta2=0.999), batch size 4, for 200 epochs with learning rate decay starting at epoch 100. **Result**: SSIM improved from 0.82 to 0.91 (+11%), PSNR improved by 3.2dB (from 28.1 to 31.3), and RMSE decreased by 34%. Qualitative evaluation by two radiologists confirmed improved diagnostic confidence on a 5-point Likert scale (3.2 to 4.4).

### Story 2: Solved Data Loading Bottleneck for 3D DICOM Volumes

**Situation**: Training the 3D reconstruction model was bottlenecked by data I/O — loading DICOM volumes from disk took 4.2 seconds per volume, while the forward+backward pass took only 0.8 seconds. GPU utilization was 16%. **Task**: Reduce data loading time to achieve >80% GPU utilization. **Action**: Profiled the pipeline and identified three bottlenecks: (1) pydicom parsed each DICOM slice independently (~200 slices per volume), (2) no caching of preprocessed volumes, (3) single-threaded loading. Implemented three optimizations: (a) converted DICOM to memory-mapped NumPy arrays (.npy format) in a one-time preprocessing step, reducing per-volume load time from 4.2s to 0.3s, (b) implemented a multi-worker DataLoader with 8 workers and prefetch_factor=4, (c) added on-the-fly data augmentation (random 3D rotations, elastic deformations) on CPU while GPU processes the previous batch. **Result**: Effective per-volume load time dropped to 0.05s with pipelining. GPU utilization increased from 16% to 89%. Total training time for 200 epochs decreased from 72 hours to 18 hours — a 4x throughput improvement. The memory-mapped approach also reduced peak RAM usage from 48GB to 12GB because volumes were read lazily.

### Story 3: Designed Validation Framework for OSEM vs. ML Reconstruction

**Situation**: Multiple ML reconstruction methods (conditional GAN, diffusion model, supervised CNN) were being compared against OSEM, but evaluations used different metrics, different test sets, and different statistical methods — making comparisons unreliable. **Task**: Design a rigorous validation framework that produces statistically valid comparisons. **Action**: Defined a standard evaluation protocol: (1) fixed test set of 300 volumes (stratified by scanner type, dose level, and pathology), (2) seven metrics (SSIM, PSNR, RMSE, LPIPS, FID on axial slices, diagnostic task accuracy via a pre-trained lesion detector, and radiologist preference in paired comparison), (3) statistical testing using paired t-tests with Bonferroni correction across 7 metrics * 4 pairwise comparisons = 28 tests (alpha = 0.05/28 = 0.0018). Also computed 95% bootstrap confidence intervals for each metric (B=10,000 resamples) and effect sizes (Cohen's d). **Result**: The framework revealed that the conditional GAN was statistically superior to OSEM on SSIM (p < 0.001, d = 0.94) and PSNR (p < 0.001, d = 0.87) but NOT on diagnostic task accuracy (p = 0.12, d = 0.21). This changed the team's recommendation from "deploy the GAN" to "deploy the GAN with a diagnostic quality monitoring pipeline."

### Story 4: Built A/B Testing Framework for LLM Prompt Variants

**Situation**: The team was evaluating LLM prompt variants for content classification by manually comparing output quality on 50 examples — no statistical rigor, no power analysis, no control for evaluator bias. **Task**: Build a proper A/B testing framework for prompt evaluation. **Action**: Designed a framework with: (1) automated evaluation using a rubric-based scoring system (relevance, accuracy, completeness on 1-5 scales), (2) power analysis to determine sample size — with baseline accuracy 72% and MDE of 5pp, required n = 1,024 samples per variant (computed using McNemar's test power formula since samples are paired), (3) multiple comparison correction using the Holm-Bonferroni step-down procedure for testing k=6 prompt variants against the baseline (less conservative than Bonferroni, still controls FWER), (4) stratified evaluation by content difficulty (easy/medium/hard) to detect heterogeneous treatment effects. **Result**: The framework identified that Prompt Variant C improved accuracy to 79% overall (p = 0.0003 after Holm correction), but the improvement was concentrated in "easy" content (68% to 82%) with no significant improvement on "hard" content (41% to 43%, p = 0.62). This led to a hybrid approach: Variant C for easy content, specialized Variant E for hard content, yielding 81% overall accuracy.

### Story 5: Detected Prompt Injection Vulnerability Through Systematic Red-Teaming

**Situation**: The LLM-based content classifier was deployed for detecting hate speech. There was no systematic evaluation of adversarial robustness. **Task**: Quantify the classifier's vulnerability to prompt injection and adversarial inputs. **Action**: Designed a red-teaming protocol with three attack categories: (1) direct prompt injection (inserting "ignore previous instructions" style payloads) — tested 200 templates, (2) encoding attacks (base64, Unicode homoglyphs, zero-width characters) — tested 150 variants, (3) semantic evasion (paraphrasing hate speech to use euphemisms, coded language) — tested 300 adversarial examples crafted with another LLM. For each category, measured the bypass rate (fraction of adversarial inputs that evaded detection) and the false negative confidence (the classifier's confidence score on bypassed examples). **Result**: Direct injection bypassed 34% of cases (classifier confidence dropped to 0.3 on average). Encoding attacks bypassed 52% — the most vulnerable category. Semantic evasion bypassed 18%. Quantified the risk: at current traffic volumes, encoding attacks alone could allow approximately 12,000 violating items per day to evade detection. Proposed mitigations: input sanitization layer (reduced encoding bypass to 3%), and adversarial training with the red-team dataset (reduced semantic bypass to 9%).

### Story 6: Diagnosed and Fixed Popularity Bias in Recommendation Model

**Situation**: The recommendation model was serving 78% of impressions from the top 1% of content — a severe popularity bias that suppressed niche creators and created a monoculture risk. **Task**: Reduce popularity bias while maintaining engagement metrics. **Action**: Diagnosed the root cause: the model used raw interaction counts as features, which created a feedback loop (popular items get more impressions, get more interactions, become more popular). Implemented three debiasing techniques: (1) inverse propensity scoring (IPS) — weighted training examples by 1/P(impression), estimated from the logging policy using a separate propensity model, (2) causal embedding — trained item embeddings that factored out the position bias using a two-tower model with a position-bias tower and a relevance tower, (3) exploration bonus — added an epsilon-greedy exploration layer that served 10% of traffic to uniformly random items for data collection. Evaluated using the Gini coefficient of impression distribution (0 = perfectly equal, 1 = all to one item). **Result**: Gini coefficient decreased from 0.89 to 0.64. Tail content (bottom 50% by popularity) impression share increased from 3% to 18%. Critically, engagement (CTR) decreased only 2.3% relative — within the acceptable guardrail threshold of 5%. User diversity metrics (number of unique creators consumed per user per week) increased by 31%.

### Story 7: Designed Online A/B Test for Recommendation Algorithm Change

**Situation**: A new recommendation algorithm was ready for online testing. Previous tests had been run with fixed-horizon analyses, leading to inflated false positive rates from repeated significance testing. **Task**: Design a statistically rigorous online A/B test with proper sequential testing. **Action**: Implemented a testing protocol with: (1) pre-registered primary metric (engagement-adjusted diversity score), guardrails (CTR, daily active creators, revenue per session), and analysis plan, (2) sequential testing using the mSPRT (mixture Sequential Probability Ratio Test) to allow continuous monitoring without inflating Type I error — the spending function allocated alpha = 0.05 across weekly interim analyses, (3) variance reduction via CUPED using 14-day pre-experiment engagement as the covariate (R^2 = 0.42, effective sample size increase of 72%), (4) cluster-based randomization at the user-community level (Louvain communities with avg size 50 users) to mitigate network effects. Duration: 4 weeks with weekly interim analyses. **Result**: The mSPRT rejected the null hypothesis at week 3 (z = 2.87, always-valid p-value = 0.003). The engagement-adjusted diversity score improved by 8.2% (95% CI: [4.1%, 12.3%]). All guardrails were within bounds. Shipped to 100% of traffic. The sequential testing approach saved one week compared to the fixed-horizon design.

### Story 8: Engineered Temporal Features for Fraud Detection

**Situation**: The fraud detection model used static features (account age, transaction amount, merchant category) and achieved AUC = 0.89. Fraud patterns were evolving faster than the model was being retrained, causing AUC to degrade by ~0.02 per month in production. **Task**: Improve detection accuracy by incorporating temporal behavioral patterns. **Action**: Engineered 12 temporal features: (1) transaction velocity — count and amount in 1h/6h/24h/7d sliding windows, (2) hour-of-day and day-of-week interaction terms, (3) time since last transaction (raw and z-scored against user's historical distribution), (4) coefficient of variation of inter-transaction times over 30-day window, (5) entropy of merchant category distribution over 7d/30d windows, (6) deviation of current transaction amount from user's 30-day rolling mean (z-score), (7) geographic velocity — distance between consecutive transactions divided by time gap. Used the Featuretools library for automated feature engineering of aggregation primitives, then manually curated to 12 features based on permutation importance (removed 8 auto-generated features with importance < 0.01). **Result**: AUC improved from 0.89 to 0.94. More importantly, the temporal features made the model adaptive: month-over-month AUC decay reduced from 0.02 to 0.005 because temporal features capture behavioral shifts in real time. False positive rate at 80% recall threshold decreased from 2.1% to 0.9%, saving an estimated 1,200 manual review hours per month.

### Story 9: Handled Extreme Class Imbalance in Fraud Detection

**Situation**: The fraud dataset had a 0.1% positive rate (1 fraud per 1,000 legitimate transactions). A baseline logistic regression model predicted all-negative and achieved 99.9% accuracy but 0% recall. **Task**: Build a model that achieves meaningful fraud detection despite the extreme imbalance. **Action**: Implemented a multi-pronged approach: (1) Undersampling — created 10 balanced subsets using random undersampling of negatives, trained an XGBoost model on each, and ensembled predictions via soft averaging (BaggingClassifier with 10 estimators), (2) SMOTE with k=5 on the minority class within each cross-validation fold (never on the full dataset to prevent data leakage — SMOTE applied inside the CV loop only), (3) threshold optimization — swept thresholds from 0.01 to 0.99, plotted precision-recall curve, selected threshold at the point maximizing F2-score (weighting recall 2x over precision, appropriate for fraud where missing fraud is costlier than false alarms), (4) cost-sensitive learning — set `scale_pos_weight = 999` in XGBoost, effectively weighting positive samples 999x. Compared all four approaches using 5-fold stratified cross-validation, evaluating on AUC-PR (area under precision-recall curve). **Result**: The ensemble undersampling approach performed best: AUC-PR = 0.72 (vs. 0.38 for vanilla XGBoost, 0.61 for SMOTE alone, 0.67 for cost-sensitive alone). At the F2-optimal threshold (0.12), achieved precision = 31%, recall = 84%. This translates to: for every 100 fraud cases, we catch 84 with 187 false alarms — a manageable manual review load.

### Story 10: Built Real-Time Anomaly Detection Pipeline

**Situation**: Fraud detection ran as a daily batch job — predictions generated overnight, reviewed next morning. A coordinated fraud attack could cause 18+ hours of damage before detection. **Task**: Reduce detection latency from hours to minutes. **Action**: Redesigned the pipeline: (1) deployed the XGBoost model as a real-time scoring service using ONNX Runtime (model serialized to ONNX format for 10x inference speedup — 0.3ms per prediction vs. 3ms with native XGBoost), (2) built a streaming feature computation layer using Apache Flink — sliding window aggregations (1h, 6h, 24h transaction counts and amounts) computed incrementally on the transaction event stream, (3) implemented a two-stage detection: Stage 1 scored every transaction in real-time (latency: p99 = 50ms), Stage 2 ran a graph-based community detection on accounts flagged in Stage 1 every 5 minutes to identify coordinated attacks. Set alerting thresholds using EWMA (Exponentially Weighted Moving Average) on the hourly fraud score distribution — alert when the mean shifts by > 3 sigma. **Result**: Detection latency decreased from 18+ hours to 4.2 minutes median. During the first month, the real-time system caught 3 coordinated attacks that the daily batch would have missed until the next morning. Estimated prevented losses: $340K across the three incidents. The ONNX Runtime deployment reduced infrastructure cost by 60% (fewer GPU-hours needed) while improving throughput from 500 to 5,000 predictions/second.

### Story 11: Resolved Conflicting Metrics Between Teams

**Situation**: The product team measured success by engagement (time spent, DAU), while the safety team measured success by violation rate reduction. A new content policy reduced violation rate by 15% but also reduced time spent by 3%. The teams disagreed on whether the policy was a net positive. **Task**: Propose a resolution framework that integrates both perspectives. **Action**: Built a composite metric: Safety-Adjusted Engagement (SAE) = time_spent * (1 - k * violating_view_rate), where k is a penalty coefficient representing the negative value of violating content exposure. Estimated k empirically: analyzed the relationship between violating content exposure and long-term user retention using a DID framework (users who were randomly exposed to more violating content due to a classifier degradation incident showed 2.1% higher 30-day churn, p = 0.003). This implied k = 2.1/0.3 * (mean_time_spent) = ~7. With this k, the SAE change was: SAE_new = 0.97 * (1 - 7*0.0025) / (1 - 7*0.003) = positive 0.4%. The policy was a net positive. **Result**: SAE was adopted as a joint north star metric for all policy launches. The composite metric framework was documented and reused for 4 subsequent policy evaluations, reducing cross-team disagreements from 2+ weeks of debate per launch to a single meeting.

### Story 12: Navigated Ambiguous Requirements with Phased Rollout

**Situation**: Leadership wanted to "reduce bad content on the platform" but provided no specific target, no metric definition, and no success criteria. The timeline was "as soon as possible." **Task**: Transform the ambiguous directive into a measurable, phased plan. **Action**: Quantified the problem first: computed baseline metrics across 6 violation categories (hate speech, spam, violence, nudity, misinformation, harassment) using 30 days of moderation data. Identified that 68% of violating impressions came from spam and hate speech. Proposed a phased rollout with explicit success criteria: Phase 1 (weeks 1-4) — reduce spam VVR by 20% relative (from 0.15% to 0.12%) by tuning the spam classifier threshold. Phase 2 (weeks 5-8) — reduce hate speech VVR by 15% relative by deploying a new multilingual classifier. Phase 3 (weeks 9-12) — address remaining categories. Each phase had pre-registered metrics, guardrails, and a ship/no-ship decision gate. Used a decision matrix with expected impact (VVR reduction), effort (engineering days), and risk (FPR increase) to prioritize. **Result**: Phase 1 achieved 23% VVR reduction for spam (exceeded target) with FPR increase of only 0.8pp. Phase 2 achieved 18% VVR reduction for hate speech. Total VVR reduction after 12 weeks: 14.2% across all categories. The phased approach delivered measurable results within 4 weeks rather than spending 12 weeks on an unscoped initiative.

### Story 13: Prioritized Competing Analysis Requests Using Impact-Effort Framework

**Situation**: Received 7 simultaneous analysis requests from 4 different teams: classifier evaluation (ML team), policy impact assessment (T&S team), quarterly safety report (leadership), creator churn analysis (product team), regional enforcement audit (legal team), experimental design for new feature (product team), and ad-hoc investigation of a metric anomaly (ops team). **Task**: Prioritize these requests to maximize total value delivered within a 2-week sprint. **Action**: Scored each request on three dimensions: (1) Impact — estimated revenue/safety impact if the analysis informed a decision, using order-of-magnitude estimates (e.g., classifier evaluation impacts $2M monthly ad revenue at risk; regional audit has legal compliance deadline), (2) Effort — estimated analyst-days (1, 2, 3, 5, 8), calibrated against similar past requests, (3) Urgency — days until the decision this analysis informs is made (shorter = more urgent because a late analysis has zero value). Computed priority score = (Impact * Time_Discount) / Effort, where Time_Discount = 1 if deadline > 2 weeks, 2 if deadline is this week, 0 if deadline already passed. **Result**: Final priority order: (1) metric anomaly investigation (Impact: high, Effort: 0.5 days, Urgency: today), (2) classifier evaluation (Impact: $2M, Effort: 3 days, Urgency: this week), (3) regional audit (Impact: compliance, Effort: 2 days, Urgency: end of month), (4-7) queued for next sprint. Completed top 3 in 6 days, freeing 4 days for the quarterly report. All decisions were informed on time.

### Story 14: Optimized Slow SQL Query from 45min to 2min

**Situation**: A daily dashboard query computing 7-day rolling violation rates by region and category was taking 45 minutes, causing the dashboard to be stale during the morning standup. The query joined a 500M-row content table with a 50M-row moderation_actions table. **Task**: Reduce query time to under 5 minutes. **Action**: Analyzed the query plan using EXPLAIN ANALYZE. Identified three issues: (1) the join was doing a full table scan on content_items (no index on created_at), (2) the window function was computing over the full partition rather than using a sliding window, (3) the query was materializing an intermediate CTE of 200M rows. Optimizations: (a) added a composite index on content_items(created_at, region, content_type) — reduced the initial scan from 500M rows to 35M (last 30 days only), (b) replaced the CTE with a materialized view refreshed daily, reducing recomputation, (c) rewrote the 7-day rolling window to use a date-spine approach with a pre-aggregated daily summary table (aggregating 35M rows to 30 * 50 regions * 20 categories = 30K rows, then computing the window function on the small table). Also partitioned the content_items table by month. **Result**: Query time dropped from 45 minutes to 1 minute 48 seconds. Dashboard was fresh by 8:45 AM instead of 9:30 AM. The materialized view approach also reduced the load on the production database by eliminating redundant full table scans across multiple dashboard queries.

### Story 15: Debugged Subtle Data Pipeline Bug Causing Metric Inflation

**Situation**: The weekly safety report showed a 12% decrease in violating view rate (VVR) — a suspiciously large improvement that no policy change could explain. Leadership was about to present this as a success to the board. **Task**: Verify the metric change and identify the root cause. **Action**: Applied a systematic root cause analysis: (1) Checked metric definition — VVR = violating views / total views. Computed numerator and denominator separately. Numerator decreased 8%, denominator increased 5%. Both contributed to VVR decrease. (2) Decomposed numerator by data source — discovered that one of three content classification pipelines (handling live stream content) had stopped writing to the violations table 10 days ago. A schema migration on the upstream table added a NOT NULL constraint to a column that the pipeline was leaving NULL, causing silent row drops. (3) Quantified the impact: live stream content represented 23% of total violating views. The pipeline failure was artificially deflating the numerator by ~23%, accounting for 8-9pp of the 12% VVR decrease. (4) Verified by reprocessing the dropped data through a backfill job. After backfill, actual VVR change was -3.4% — consistent with an incremental classifier improvement deployed that week. **Result**: Prevented leadership from presenting inflated results to the board. The pipeline bug was fixed within 24 hours (added explicit NULL handling and a data quality check that alerts if row counts from any pipeline drop by >10% day-over-day). Implemented a metric integrity monitoring system that compares component-level metric contributions against historical baselines and fires an alert when any component deviates by >2 standard deviations.

---

## Section 6: Open-Ended Questions

### Question 1: "Evaluate the Effectiveness of Content Throttling on Combating Bad Sellers"

This is the actual interview question (adapted for content safety). Framework:

**Define the intervention precisely.** Content throttling (降流) reduces the distribution/reach of content from suspected bad sellers without fully removing it. It's a soft enforcement action — the seller's content still exists but reaches fewer users through reduced algorithmic amplification.

**Causal identification strategy.** If throttling was applied universally at a specific time, use a DID (difference-in-differences) design: compare the change in bad seller activity (treatment) vs. legitimate seller activity (control) before vs. after throttling. The identifying assumption is parallel trends — absent throttling, bad and legitimate seller metrics would have moved in parallel. Test this by checking pre-period parallel trends (regress metric on time * group interaction using pre-period data; the interaction coefficient should be non-significant).

If throttling was applied selectively (based on a risk score threshold), use regression discontinuity: compare sellers just above vs. just below the throttling threshold. Sellers near the threshold are quasi-randomly assigned. Validate by checking covariate balance at the threshold (McCrary density test for manipulation, covariate smoothness tests).

**Metrics to evaluate.** Primary: (1) Violating content view rate among throttled sellers' content (should decrease), (2) Bad seller GMV (gross merchandise value) — proxy for their commercial success (should decrease), (3) New bad seller account creation rate (deterrence effect). Guardrails: (4) False positive rate — fraction of throttled sellers who are actually legitimate (audit a random sample), (5) Legitimate seller engagement (spillover — if legitimate sellers are intimidated, that's a negative externality).

**Threats to validity.** (1) Selection bias — sellers who get throttled are systematically different from those who don't. Control for observables using propensity score matching or inverse propensity weighting. (2) Spillover — throttled bad sellers may migrate to other platforms or create new accounts (measure new account creation rate with device fingerprinting). (3) Displacement effect — throttling in one category may push bad sellers to other categories (check cross-category violation rates). (4) Hawthorne effect — the announcement of throttling may change behavior temporarily regardless of actual enforcement.

**Expected findings and interpretation.** If DID estimate shows a 25% reduction in bad seller GMV (95% CI: [18%, 32%]) with no significant effect on legitimate seller engagement (point estimate: -0.5%, 95% CI: [-1.8%, 0.8%]), conclude that throttling is effective at reducing bad seller commercial activity without collateral damage. If the false positive rate exceeds 10%, recommend improving the risk scoring model before expanding throttling.

### Question 2: "How Would You Measure Whether a New Hate Speech Classifier Is Better Than the Existing One?"

**Offline evaluation.** Hold out a labeled test set (n >= 5,000, stratified by hate speech subcategory and language). Compute precision, recall, F1, and AUC-PR for both classifiers on the same test set. Use the McNemar test (paired, exact) to test whether the new classifier has significantly different error rates. McNemar's test statistic = (b-c)^2 / (b+c), where b = items correctly classified by new but missed by old, c = vice versa. If the new classifier improves recall from 72% to 81% with a p-value < 0.001 (McNemar), the improvement is statistically significant.

**Online evaluation.** Run an A/B test: 50% of content is scored by the old classifier, 50% by the new. Primary metric: VVR in each arm. Guardrail: FPR (estimated by human audit of a random sample of removals). Duration: 2 weeks minimum, sample size determined by power analysis targeting a 5% relative MDE on VVR.

**Beyond accuracy.** Also evaluate: (a) latency (p99 inference time — safety classifiers must be fast), (b) calibration (are predicted probabilities accurate? Use reliability diagrams), (c) fairness (disaggregate performance by demographic group, dialect, language — a classifier that catches AAE-dialect hate speech but misses standard English hate speech is biased), (d) adversarial robustness (test against the red-team dataset from Story 5).

### Question 3: "If Violation Reports Spike 3x Overnight, How Do You Diagnose the Root Cause?"

**Immediate triage (first 30 minutes).** Check the decomposition: is the spike in report volume, report rate, or both? Volume = total reports. Rate = reports per content item. If volume spiked but rate is stable, the platform may simply have more content (viral event, holiday). If rate spiked, there's a genuine increase in reported behavior.

**Segmentation (next 1-2 hours).** Disaggregate the spike by: (a) report_reason — is it concentrated in one violation type (e.g., spam spike vs. broad increase)? (b) region — localized to one geography (may correlate with a political event or a new abuse campaign), (c) content_type — limited to comments, live, or video? (d) reporter behavior — is the spike driven by a small number of reporters (possible report bombing) or broadly distributed?

**Pipeline check.** Rule out data issues: (a) was there a logging change that double-counted reports? Check raw event counts upstream of aggregation. (b) Was there a UI change that made reporting easier (new report button)? Check product release notes. (c) Was there a classifier change that increased the "suggest report" prompt?

**Root cause determination.** If the spike is real and concentrated (e.g., spam in region X), investigate the content: sample 100 reported items, identify common patterns (same URL, similar text, account creation time clusters). If it's a coordinated campaign, escalate to the CIB detection pipeline (Section 4, Case 2). Quantify: "The spike is attributable to approximately 2,000 new accounts created in the last 48 hours in region X, posting content with 94% textual similarity, linking to 3 external domains."

### Question 4: "How Would You Design a Metric to Measure 'Platform Health'?"

**Decomposition framework.** Platform health is multidimensional. Propose a composite index with weighted components:

(1) **Safety score** (weight 0.30): 1 - VVR (violating view rate). Higher is safer. (2) **Creator ecosystem health** (weight 0.25): active creator retention (30-day) * creator diversity Gini complement (1 - Gini). (3) **User trust** (weight 0.25): 1 - (report_rate + appeal_overturn_rate) / 2. Low report rates and low appeal overturn rates indicate users trust the platform and enforcement is accurate. (4) **Enforcement quality** (weight 0.20): (precision + recall) / 2 for content moderation, weighted by violation category severity.

**Operationalization.** Each component is normalized to [0, 1] using min-max scaling against historical baselines. The composite index is a weighted sum. Track weekly with 95% bootstrap confidence intervals. Set alert thresholds: if the composite index drops by more than 5% week-over-week, trigger an investigation.

**Validation.** Correlate the platform health index with long-term outcomes: 90-day user retention, advertiser spend, and regulatory incident frequency. If the index is predictive (Granger causality test, p < 0.05, lag 4-8 weeks), it is a useful leading indicator. Iterate on weights and components based on this validation.

### Question 5: "Should We Optimize for Precision or Recall in Content Moderation? How Do You Decide?"

**Framework: cost-asymmetry analysis.** Compute the cost of each error type:

- C_FP (false positive — wrongful removal): estimated from creator churn uplift. If 5% of wrongfully removed creators churn and each creator generates $X in platform value, C_FP = 0.05 * X + appeal_processing_cost.
- C_FN (false negative — missed violation): estimated from user harm. Exposure to violating content increases 30-day churn by Y%, plus regulatory fine risk. C_FN = Y * user_value + regulatory_risk.

The optimal operating point minimizes total cost: argmin_t [C_FP * FP(t) + C_FN * FN(t)].

**Category-dependent answer.** The answer varies by violation type. CSAM: recall priority (C_FN is essentially infinite — legal mandate, child safety). Set recall > 99%, accept precision as low as 50% (human review filters false positives). Borderline humor/satire: precision priority (wrongful removal causes creator backlash, media scrutiny). Set precision > 95%, accept lower recall, supplement with user reports. Spam: balanced (high-volume, lower stakes per item). Optimize F1 or use cost-weighted F-beta where beta = C_FN/C_FP.

Present the tradeoff curve (precision vs. recall at various thresholds) to stakeholders with cost estimates at each operating point. Let the business make the final call on risk tolerance, but ensure the decision is informed by quantitative analysis.

### Question 6: "How Would You Evaluate the ROI of Investing in More Human Moderators vs. Improving Automated Classifiers?"

**Marginal analysis framework.** For each option, compute the marginal cost and marginal benefit of the next unit of investment:

*Human moderators*: Cost = salary + training + tooling (~$60K/year per moderator). Benefit = marginal reduction in FNR. Estimate: each human moderator reviews ~500 items/day, with 95% accuracy. If the current queue backlog is 10K items/day, adding 5 moderators clears 2,500 more items, reducing FNR by approximately 2,500 * current_miss_rate / total_violations.

*Automated classifiers*: Cost = ML engineering time + compute (~$200K for a model improvement cycle). Benefit = recall improvement that applies to ALL content (billions of items). Even a 1pp recall improvement on the automated classifier catches far more violations than adding moderators.

**Break-even analysis.** The classifier investment dominates when: delta_recall_auto * total_content_volume > n_moderators * items_per_day * accuracy. For a platform with 1B daily content items and a 1pp recall improvement, the classifier catches 10M more items per day. To achieve the same coverage with human moderators at 500 items/day/moderator, you'd need 20,000 additional moderators — clearly infeasible.

**Nuance.** Humans are better for novel/ambiguous violation types where classifiers lack training data. The optimal strategy is hybrid: invest in classifiers for high-volume, well-defined categories (spam, nudity) and invest in human moderators for emerging threats and edge cases that require contextual judgment.

### Question 7: "If an Experiment Shows Treatment Increases Engagement but Also Increases Violations, What Do You Recommend?"

**Quantify both effects with confidence intervals.** Report: "Engagement (DAU) increased by 2.3% (95% CI: [1.8%, 2.8%]). VVR increased by 0.05pp (95% CI: [0.02pp, 0.08pp])."

**Decompose the violation increase.** Is it driven by: (a) more content overall (the engagement increase leads to more content, including violating content — the rate is unchanged), or (b) a genuine rate increase (the treatment changes behavior or exposure patterns in a way that increases the fraction of violating content)? If (a), the VVR (rate metric) should be stable even though absolute violations increased. If (b), the treatment is genuinely degrading safety.

**Apply the composite metric.** Use the Safety-Adjusted Engagement metric from Story 11: SAE = engagement * (1 - k * VVR). If SAE_treatment > SAE_control, the engagement gain outweighs the safety cost. If SAE_treatment < SAE_control, the safety cost dominates.

**Recommendation.** If VVR rate is stable (only absolute increase), ship with enhanced monitoring. If VVR rate increased, investigate whether targeted mitigations (e.g., stricter enforcement on the marginal content surfaced by the treatment) can preserve the engagement gain while neutralizing the safety cost. If no mitigation works, do not ship — safety guardrails are non-negotiable.

### Question 8: "How Do You Handle the Cold-Start Problem for New Users in Risk Scoring?"

**Progressive feature enrichment.** At account creation (t=0): only registration features are available (IP geolocation, device fingerprint, email domain, registration time-of-day, referral source). Train a separate "new user" risk model using these features. Known signal: accounts created from datacenter IPs, using disposable email domains, at unusual hours, from previously flagged device fingerprints are higher risk. This model achieves moderate AUC (~0.75) but catches the most obvious bad actors.

**Behavioral ramp-up.** After first 10 actions (t=hours): compute early behavioral features — time between first actions (bots are faster), content diversity (bots post similar content), interaction patterns (bots follow many accounts immediately). Update the risk score using an online learning model or a time-aware feature set.

**Transfer learning.** Use graph embeddings from the social network: even if a new account has no behavioral history, its connections to existing accounts provide signal. An account that immediately connects to a cluster of known bad actors inherits risk through guilt-by-association (graph neural network with neighborhood aggregation).

**Calibration.** The cold-start model must be separately calibrated because the feature space and base rates differ from the general population model. Evaluate calibration using reliability diagrams on a holdout set of new accounts that have matured (now have 30+ days of behavioral data to determine ground truth).

### Question 9: "Design a System to Detect if Your A/B Test Is Contaminated by Network Effects"

**Pre-experiment: design for detection.** Use a two-level randomization design: first randomize clusters (communities, geographic regions) to treatment/control, then within each cluster, randomize individual users. Network effects manifest as within-cluster spillover: treated users in a cluster influence control users in the same cluster.

**Detection method 1: interference test.** Compare outcomes for control users who have many treated neighbors vs. control users who have few treated neighbors. If these differ significantly, network effects are present. Formally: regress Y_control on fraction_treated_neighbors, controlling for covariates. A significant positive coefficient indicates positive spillover.

**Detection method 2: ego-cluster analysis.** For each user, define their ego network (direct connections). Compute the treatment exposure = fraction of ego network in treatment. If the dose-response relationship between treatment exposure and outcome is non-zero for control users, interference exists.

**Detection method 3: variance inflation.** Under no interference, variance of the treatment effect estimator depends only on individual-level variance. Under interference, the effective sample size is reduced by the design effect (1 + rho * (m-1)), where rho is the intra-cluster correlation and m is average cluster size. If the observed variance of the treatment effect is significantly larger than the no-interference prediction (test using a chi-squared test on the variance ratio), interference is likely.

**Mitigation if detected.** Switch to a cluster-randomized analysis: estimate the ATE using cluster-level means (each cluster = one observation), which is valid under interference as long as cross-cluster spillover is negligible. Report: "Individual-level analysis estimates ATE = 5.2%, but interference test is significant (p = 0.01). Cluster-level analysis estimates ATE = 3.8% (95% CI: [1.2%, 6.4%]). The cluster-level estimate is preferred."

### Question 10: "How Would You Quantify the Long-Term Impact of Over-Moderation on Creator Ecosystem Health?"

**Define over-moderation.** Over-moderation = high false positive rate in content removal. Operationalize as: FPR > X% (where X is determined by audit), or appeal overturn rate > Y%, or wrongful removal rate (estimated from random audit) > Z%.

**Causal identification.** Exploit variation in over-moderation exposure. Use an instrumental variable: a classifier model update that exogenously increased FPR in certain content categories but not others. The IV approach: first stage regresses a creator's wrongful removal count on classifier FPR for their content category (strong first stage, F-statistic > 10). Second stage regresses long-term outcome (6-month creator retention, posting frequency, content quality score) on instrumented wrongful removal count.

**Metrics.** (1) Creator retention: survival analysis (Kaplan-Meier curves) stratified by wrongful removal count. Cox proportional hazards model with wrongful_removals as a covariate, controlling for account age, follower count, content category. Hazard ratio > 1 means each wrongful removal increases churn risk. (2) Posting frequency: interrupted time series analysis — does a wrongful removal cause a drop in posting frequency in the following 30 days? Use a pre-post design with the creator as their own control. (3) Content quality: does over-moderation cause creators to self-censor (post blander content)? Measure content diversity (entropy of topics/styles) before and after wrongful removal events.

**Long-term ecosystem effects.** Beyond individual creators: (a) does over-moderation reduce new creator onboarding (potential creators see removals and choose not to join)? Measure new creator registration rate as a function of platform-wide FPR using a time series regression. (b) Does over-moderation cause a chilling effect (surviving creators shift to "safe" content, reducing platform diversity)? Measure the Gini coefficient of content topic distribution over time.

**Quantification example.** "Each wrongful removal increases 6-month churn hazard by 1.4x (95% CI: [1.2, 1.6], p < 0.001). At current FPR = 6%, an estimated 12,000 creators per month experience wrongful removal, of whom approximately 4,800 churn within 6 months who would not have otherwise. At an estimated $200 lifetime value per creator, this represents $960K/month in lost creator value."

---

## Section 7: Behavioral Questions

Every behavioral answer is framed with technical substance, not soft skills. The HM rejected collaboration framing — these answers lead with methodology.

### Question 1: "Tell Me About a Time You Had a Conflict with a Colleague. How Did You Resolve It?"

**Technical framing: competing statistical methodologies.**

**Situation**: A colleague on the ML team proposed evaluating a new hate speech classifier using AUC-ROC on a balanced test set. I argued this was misleading because production data has 0.3% positive rate, and AUC-ROC is insensitive to performance on the minority class under extreme imbalance. **Task**: Reach a methodologically sound evaluation consensus. **Action**: I prepared a quantitative comparison: generated predictions from both the old and new classifiers on (a) the balanced test set and (b) a production-representative test set with 0.3% positive rate. On the balanced set, the new classifier's AUC-ROC was 0.95 vs. 0.92 — looks like a clear win. On the production-representative set, AUC-PR was 0.41 vs. 0.38 — the improvement was much smaller and not statistically significant (DeLong test p = 0.18). More critically, at the operating threshold used in production, precision dropped from 82% to 74% while recall improved from 71% to 76%. I presented both analyses side by side, showing that the evaluation metric choice materially changed the conclusion. **Result**: We adopted a standard evaluation protocol: AUC-PR as the primary offline metric, plus precision and recall at the production threshold, evaluated on a production-representative test set. The new classifier was not deployed until recall improvements were achieved without precision degradation. The protocol was documented and adopted for all subsequent classifier evaluations.

### Question 2: "Describe a Time You Had to Make a Decision with Incomplete Information."

**Technical framing: insufficient data to make a decision.**

**Situation**: A new type of fraud emerged (account takeover via SIM swap), but we had only 23 confirmed cases in our training data — far too few to train or evaluate a dedicated model. Leadership wanted a detector deployed within 2 weeks. **Task**: Build a detection system with extremely limited labeled data. **Action**: Calculated that with n=23 positives, a classifier evaluation would have a 95% CI on recall of +/- 20pp (Wilson score interval) — unacceptably wide. Instead of training a supervised model, I took a semi-supervised approach: (1) used the 23 confirmed cases to characterize the behavioral signature (SIM swap fraud has a distinctive temporal pattern: password reset -> device change -> rapid outbound transfers, all within 30 minutes), (2) wrote rule-based detectors for this pattern (3 sequential events within 30-minute window from a new device), (3) computed the expected false positive rate analytically using independence assumptions and historical event frequencies (~0.001% = ~100 false alarms/day, manageable for human review), (4) deployed with a 2-week evaluation period where all flagged cases were human-reviewed to generate labels for future model training. **Result**: The rule-based detector caught 8 of the next 11 SIM swap cases in the first 2 weeks (72% recall, bootstrapped 95% CI: [39%, 93%]). The 100 human-reviewed false positives + 8 true positives provided enough data to train a proper ML model by week 4, which achieved 89% recall with 0.5% FPR. The phased approach (rules -> labels -> ML) turned insufficient data into a learning system.

### Question 3: "How Do You Prioritize When Everything Is Urgent?"

**Technical framing: expected value calculation.**

**Situation**: Three "urgent" requests arrived simultaneously: (1) CEO dashboard showing a metric anomaly (potential data bug), (2) A/B test results due for a ship/no-ship decision on Friday, (3) quarterly compliance report due to legal. **Task**: Allocate 3 analyst-days optimally. **Action**: Computed expected value of timely completion for each: (1) Metric anomaly: if it's a real issue, early detection prevents ~$500K in wrong decisions; if it's a data bug, early detection prevents embarrassment. P(real issue) = 0.3 based on base rate of past anomaly alerts. EV = 0.3 * $500K + 0.7 * $50K = $185K. Effort: 0.5 days. EV/effort = $370K/day. (2) A/B test: ship decision affects ~$2M annual revenue. But the experiment data is already collected — analysis can be done Monday with only a 3-day delay. EV of on-time delivery = $2M * P(decision changes if delayed) * cost_of_delay. P(decision changes) ~ 0 (data is deterministic). Cost_of_delay = 3 days of lost feature value = $16K. EV/effort = $16K/1.5 days = $11K/day. (3) Compliance report: legal deadline is hard. Fine for late submission = $100K. EV/effort = $100K/1 day = $100K/day. **Result**: Priority order: (1) metric anomaly (0.5 days), (2) compliance report (1 day), (3) A/B test (1.5 days). The metric anomaly turned out to be a data pipeline bug (saved $50K in wrong decisions). Compliance report was on time. A/B test results were delivered Monday — the 3-day delay had zero impact on the decision.

### Question 4: "Tell Me About a Time You Failed. What Did You Learn?"

**Technical framing: model deployed with bug, quantified impact, built monitoring.**

**Situation**: Deployed a fraud detection model update that had a feature engineering bug — a look-ahead bias where one feature used future transaction data that wouldn't be available at prediction time. The feature had high importance (0.14 out of 1.0), inflating offline AUC from the true 0.91 to 0.95. In production, the feature was replaced by a stale value (last available), causing precision to drop from 85% to 62% — a 23pp degradation that went undetected for 5 days. **Task**: Quantify the damage, fix the bug, and prevent recurrence. **Action**: Quantified impact: during the 5 days, the degraded precision generated approximately 3,400 additional false positive alerts, each requiring 15 minutes of manual review = 850 analyst-hours wasted ($42K in labor cost). Additionally, the model's recall also dropped by 4pp because the corrupted feature introduced noise, resulting in approximately 12 missed fraud cases worth an estimated $180K. Total impact: $222K. Root cause: the feature engineering pipeline used a pandas merge that didn't properly enforce temporal ordering, allowing future data to leak. Fix: added explicit temporal validation checks (assert all feature timestamps < prediction timestamp) and deployed a production monitoring system that tracks 4 key model health metrics hourly: (a) feature distribution divergence (KL divergence vs. training distribution), (b) prediction score distribution shift, (c) precision/recall estimated from a fast-feedback human review sample, (d) feature staleness (time since feature was last refreshed). **Result**: Set alerts at KL divergence > 0.1 and prediction distribution KS-statistic p < 0.01. Backtested: the monitoring system would have caught the bug within 6 hours instead of 5 days, reducing total impact from $222K to ~$18K. The temporal validation check has caught 2 additional look-ahead bugs in subsequent model updates before they reached production.

### Question 5: "Describe a Time You Showed Leadership."

**Technical framing: designed experiment framework adopted by 3 teams.**

**Situation**: Three teams (safety, growth, and monetization) were running A/B tests independently with inconsistent methodology — different significance levels (0.05, 0.10, 0.01), different multiple testing corrections (none, Bonferroni, Sidak), no pre-registration, and no guardrail metrics. Two experiments shipped changes that later showed negative safety impact that wasn't detected because the safety team's metrics weren't included as guardrails. **Task**: Standardize experimentation methodology across teams. **Action**: Built an experiment design template that codified: (1) mandatory pre-registration (hypothesis, primary metric, guardrail metrics, sample size calculation, analysis plan), (2) standard significance level (alpha = 0.05, two-sided) with mandatory safety guardrails for all experiments regardless of team, (3) sequential testing protocol (mSPRT with alpha-spending function for interim analyses — eliminates p-hacking from repeated peeking), (4) CUPED variance reduction as default (reduced required experiment duration by 30-40% across the board), (5) automated post-experiment analysis pipeline that generates a standardized report (point estimates, CIs, guardrail checks, novelty effect detection via time-series decomposition of daily treatment effects). Validated the framework by re-analyzing the two experiments that shipped negative safety impact — both would have been caught by the mandatory safety guardrails. Presented a quantitative retrospective: "Framework would have prevented $1.2M in safety-related costs across these two incidents." **Result**: Framework was adopted by all three teams within 6 weeks. In the following quarter, 47 experiments used the framework. Three experiments were blocked by safety guardrails that would not have been checked under the old process. Experiment velocity increased 25% because CUPED reduced required duration, and the automated analysis pipeline reduced analyst time from 2 days to 3 hours per experiment.

---

## Section 8: Questions to Ask

### Team & Metrics

**1. "What does the metrics stack look like? What are the top 3-5 KPIs the team reviews weekly, and how are they computed?"**
This demonstrates metric fluency and signals you want to understand the existing infrastructure before proposing changes. Listen for: data freshness (real-time vs. daily batch), metric definitions (are they well-documented or tribal knowledge), and known measurement gaps.

**2. "How do you balance automated moderation with human review? What fraction of moderation decisions are fully automated vs. human-in-the-loop?"**
This reveals the team's operational model. A heavily automated team needs DS to focus on classifier evaluation and threshold tuning. A heavily human team needs DS to focus on prioritization algorithms and queue optimization.

**3. "What does the feedback loop from enforcement to model improvement look like? How frequently are classifiers retrained, and how is training data generated?"**
This probes the ML operations maturity. Active learning pipelines, regular retraining schedules, and well-maintained label taxonomy suggest a mature team. Ad-hoc retraining and inconsistent labeling suggest high-impact DS opportunities.

### Technical

**4. "What experimentation platform do you use, and does it support cluster randomization and sequential testing?"**
This signals that you understand the unique challenges of safety experimentation (network effects, continuous monitoring). If the platform doesn't support these, you're identifying a concrete contribution you can make.

**5. "How do you handle interference in experiments on a social network? Do you use cluster-based randomization, and if so, how are clusters defined?"**
This is a deeply technical question that demonstrates you understand why naive user-level randomization fails in social network settings. The answer reveals the team's methodological sophistication.

**6. "What's the current approach to measuring long-term effects of safety interventions? Do you track creator ecosystem health metrics, and how?"**
Safety interventions can have delayed effects (creator churn takes weeks to manifest). This question shows you think beyond short-term experiment results.

### Growth

**7. "What does the DS career ladder look like here? What distinguishes a senior DS from a staff DS in terms of scope and impact?"**
Understanding the growth path helps you calibrate your contributions. Staff-level typically means cross-team influence and methodology ownership (which aligns with what you'd bring from the experimentation framework experience).

**8. "How do individual contributors influence policy decisions? Is there a formal process for DS-recommended policy changes?"**
This probes whether DS has genuine decision-making influence or is purely a support function. A strong team has DS presenting analysis that directly changes enforcement thresholds, classifier deployment decisions, and policy definitions.

### Culture & Operations

**9. "How do you handle false positive escalations from high-profile creators? Is there a separate review process, and how do you ensure consistency?"**
This tests whether the team has a mature escalation process. High-profile false positives are the most politically charged events in content moderation. The answer reveals how the team balances consistency (same rules for everyone) with pragmatism (some mistakes are more costly than others).

**10. "What's the on-call structure for safety incidents? How quickly does the team need to respond to metric anomalies, and what does the incident response process look like?"**
This shows you understand that safety DS is not a 9-to-5 role. The answer sets expectations for the operational load and reveals whether the team has mature incident response (runbooks, escalation paths, post-mortems) or is in a more reactive mode.

---

## Appendix: Key Formulas Quick Reference

| Concept | Formula | When to Use |
|---|---|---|
| Z-test statistic | z = (x_bar - mu_0) / (sigma / sqrt(n)) | Known variance, large n |
| T-test statistic | t = (x_bar - mu_0) / (s / sqrt(n)), df = n-1 | Unknown variance, small n |
| Welch's t-test df | df = (s1^2/n1 + s2^2/n2)^2 / ((s1^2/n1)^2/(n1-1) + (s2^2/n2)^2/(n2-1)) | Two-sample, unequal variances |
| Sample size (proportions) | n = (z_alpha/2 + z_beta)^2 * (p1(1-p1) + p2(1-p2)) / (p1-p2)^2 | Power analysis for A/B test on proportions |
| Confidence interval | x_bar +/- z_alpha/2 * (s / sqrt(n)) | Estimating population parameter |
| Bonferroni threshold | alpha_adj = alpha / m | FWER control, m tests |
| BH procedure | Reject if p_(k) <= k * alpha / m, largest such k | FDR control, exploratory analysis |
| XGBoost optimal leaf weight | w* = -sum(g_i) / (sum(h_i) + lambda) | Understanding regularization |
| XGBoost split gain | Gain = 0.5 * [G_L^2/(H_L+lambda) + G_R^2/(H_R+lambda) - (G_L+G_R)^2/(H_L+H_R+lambda)] - gamma | Feature split decision |
| CUPED variance reduction | Var_cuped = Var_Y * (1 - R^2) | Variance reduction in experiments |
| DID estimator | tau = (Y_treat_post - Y_treat_pre) - (Y_ctrl_post - Y_ctrl_pre) | Causal effect without randomization |
| Cost-optimal threshold | t* = argmin_t [C_FP * FP(t) + C_FN * FN(t)] | Setting classifier threshold |
| Design effect (clustering) | DEFF = 1 + rho * (m - 1) | Adjusting sample size for clustered designs |

---

*Last updated: 2026-02-28. Prepared for TikTok Integrity & Safety DS interview. Every answer optimized for technical depth per HM feedback.*
