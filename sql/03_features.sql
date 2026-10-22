-- One feature row per labelled (msno, cohort_month), built strictly from data before the
-- cohort's cutoff (cohort_month, exclusive).
--
-- Leakage checklist:
--   * every transactions / user_logs row is filtered with `< b.cohort_month`
--   * days_to_expiry uses the expiry as known at the cutoff (labels.expire_date)
--   * no column of labels other than the cohort key and that expiry is read here
--   * members is a static snapshot (city, age, gender, registration channel). It is not
--     time-stamped, so registration dates on/after the cutoff are nulled out.
--   tests/test_features.py proves the first point: adding or changing any row dated on or
--   after the cutoff leaves every feature unchanged.
WITH base AS (
    SELECT msno, cohort_month, expire_date FROM labels
),
tx AS (
    SELECT
        b.msno,
        b.cohort_month,
        t.transaction_date,
        t.is_cancel,
        t.is_auto_renew,
        t.payment_plan_days,
        t.plan_list_price,
        t.actual_amount_paid,
        t.payment_method_id,
        -- same-day ties: cancellation after subscription, as in 02_labels.sql
        row_number() OVER w AS seq,
        lag(t.is_auto_renew) OVER w AS prev_auto_renew
    FROM base b
    JOIN transactions t ON t.msno = b.msno AND t.transaction_date < b.cohort_month
    WINDOW w AS (
        PARTITION BY b.msno, b.cohort_month
        ORDER BY t.transaction_date, t.is_cancel, t.membership_expire_date
    )
),
tx_feat AS (
    SELECT
        msno,
        cohort_month,
        count(*)                                                        AS n_tx,
        count(*) FILTER (WHERE transaction_date >= cohort_month - 90)   AS n_tx_90d,
        count(*) FILTER (WHERE is_cancel)                               AS n_cancels,
        count(*) FILTER (WHERE is_cancel AND transaction_date >= cohort_month - 90)
                                                                        AS n_cancels_90d,
        count(*) FILTER (WHERE is_auto_renew <> prev_auto_renew)        AS n_auto_renew_flips,
        count(DISTINCT payment_plan_days)                               AS n_distinct_plans,
        count(DISTINCT payment_method_id)                               AS n_payment_methods,
        avg((actual_amount_paid < plan_list_price)::INT)                AS share_discounted,
        sum(actual_amount_paid)                                         AS total_paid,
        date_diff('day', min(transaction_date), cohort_month)           AS tenure_days,
        date_diff('day', max(transaction_date), cohort_month)           AS days_since_last_tx,
        arg_max(is_auto_renew, seq)::INT                                AS last_auto_renew,
        arg_max(is_cancel, seq)::INT                                    AS last_is_cancel,
        arg_max(payment_plan_days, seq)                                 AS last_plan_days,
        arg_max(plan_list_price, seq)                                   AS last_list_price,
        arg_max(actual_amount_paid, seq)                                AS last_amount_paid,
        arg_max(payment_method_id, seq)                                 AS last_payment_method
    FROM tx
    GROUP BY msno, cohort_month
),
lg AS (
    SELECT
        b.msno,
        b.cohort_month,
        date_diff('day', l.date, b.cohort_month) AS age,  -- 1 = the day before the cutoff
        least(greatest(l.total_secs, 0), 86400) AS secs,  -- a few values are corrupt
        l.num_25,
        l.num_50 + l.num_75 + l.num_985 + l.num_100 + l.num_25 AS num_plays,
        l.num_100,
        l.num_unq
    FROM base b
    JOIN user_logs l
      ON l.msno = b.msno AND l.date < b.cohort_month AND l.date >= b.cohort_month - 90
),
gaps AS (
    SELECT msno, cohort_month, age - lag(age) OVER (
        PARTITION BY msno, cohort_month ORDER BY age
    ) AS gap
    FROM (SELECT DISTINCT msno, cohort_month, age FROM lg)
),
log_feat AS (
    SELECT
        msno,
        cohort_month,
        sum(secs) FILTER (WHERE age <= 7) / 60                  AS minutes_7d,
        sum(secs) FILTER (WHERE age <= 30) / 60                 AS minutes_30d,
        sum(secs) FILTER (WHERE age BETWEEN 31 AND 60) / 60     AS minutes_31_60d,
        sum(secs) / 60                                          AS minutes_90d,
        count(DISTINCT age) FILTER (WHERE age <= 7)             AS active_days_7d,
        count(DISTINCT age) FILTER (WHERE age <= 30)            AS active_days_30d,
        count(DISTINCT age) FILTER (WHERE age BETWEEN 31 AND 60) AS active_days_31_60d,
        count(DISTINCT age)                                     AS active_days_90d,
        sum(num_unq) FILTER (WHERE age <= 30)                   AS unique_songs_30d,
        sum(num_25) FILTER (WHERE age <= 30)
            / nullif(sum(num_plays) FILTER (WHERE age <= 30), 0) AS skip_ratio_30d,
        sum(num_100) FILTER (WHERE age <= 30)
            / nullif(sum(num_plays) FILTER (WHERE age <= 30), 0) AS completion_ratio_30d,
        min(age)                                                AS days_since_last_listen
    FROM lg
    GROUP BY msno, cohort_month
),
gap_feat AS (
    SELECT msno, cohort_month, max(gap) AS max_gap_days_90d
    FROM gaps
    GROUP BY msno, cohort_month
)
SELECT
    b.msno,
    b.cohort_month,
    date_diff('day', b.cohort_month, b.expire_date) AS days_to_expiry,
    -- transactions (every labelled user has at least one, by cohort construction)
    t.n_tx, t.n_tx_90d, t.n_cancels, t.n_cancels_90d, t.n_auto_renew_flips,
    t.n_distinct_plans, t.n_payment_methods, t.share_discounted, t.total_paid,
    t.tenure_days, t.days_since_last_tx, t.last_auto_renew, t.last_is_cancel,
    t.last_plan_days, t.last_list_price, t.last_amount_paid,
    t.last_list_price - t.last_amount_paid AS last_discount,
    t.last_payment_method,
    -- listening: no log rows means no listening, so sums and counts are 0, ratios stay NULL
    (lf.msno IS NOT NULL)::INT AS has_logs,
    coalesce(lf.minutes_7d, 0) AS minutes_7d,
    coalesce(lf.minutes_30d, 0) AS minutes_30d,
    coalesce(lf.minutes_31_60d, 0) AS minutes_31_60d,
    coalesce(lf.minutes_90d, 0) AS minutes_90d,
    coalesce(lf.active_days_7d, 0) AS active_days_7d,
    coalesce(lf.active_days_30d, 0) AS active_days_30d,
    coalesce(lf.active_days_31_60d, 0) AS active_days_31_60d,
    coalesce(lf.active_days_90d, 0) AS active_days_90d,
    coalesce(lf.unique_songs_30d, 0) AS unique_songs_30d,
    lf.skip_ratio_30d,
    lf.completion_ratio_30d,
    lf.days_since_last_listen,
    g.max_gap_days_90d,
    -- trend: last 30 days vs the 30 before
    coalesce(lf.minutes_30d, 0) - coalesce(lf.minutes_31_60d, 0) AS minutes_delta_30d,
    lf.minutes_30d / nullif(lf.minutes_31_60d, 0) AS minutes_ratio_30d,
    coalesce(lf.active_days_30d, 0) - coalesce(lf.active_days_31_60d, 0) AS active_days_delta_30d,
    -- members snapshot
    m.city,
    CASE WHEN m.bd BETWEEN 10 AND 80 THEN m.bd END AS age,
    m.gender,
    m.registered_via,
    CASE WHEN m.registration_date < b.cohort_month
         THEN date_diff('day', m.registration_date, b.cohort_month) END AS days_since_registration
FROM base b
JOIN tx_feat t USING (msno, cohort_month)
LEFT JOIN log_feat lf USING (msno, cohort_month)
LEFT JOIN gap_feat g USING (msno, cohort_month)
LEFT JOIN members m USING (msno)
