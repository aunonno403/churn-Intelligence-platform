-- Churn label for ONE monthly cohort. Template vars: ${cutoff}, ${month_end}, ${window_days}
-- Mirrors KKBox's official WSDMChurnLabeller.scala (shipped with the competition data).
--
-- Transaction order (as in the official labeller): by transaction_date; same-day ties by plan
-- signature (price||days||method as strings, descending); within the same plan, subscriptions
-- precede cancellations, renewals sort by expiry ascending, cancellations by expiry descending.
--
-- Cohort: users whose last transaction before ${cutoff}, in that order, expires within
--         [${cutoff}, ${month_end}]. Uses pre-cutoff data only.
--         Deliberate deviation: we use all pre-cutoff history; the official labeller only
--         looks at the month before the cutoff.
-- Label:  walk the user's transactions from the cutoff on. Cancellations before the first
--         renewal can only pull the expiry earlier. Not churned iff that first non-cancel
--         transaction is dated < ${window_days} days after the (possibly earlier) expiry.
--         Nothing after expire_date + ${window_days} can change the outcome, so it is not read.
WITH ordered AS (
    SELECT
        *,
        plan_list_price::VARCHAR || payment_plan_days::VARCHAR || payment_method_id::VARCHAR
            AS plan_sig,
        CASE WHEN is_cancel THEN -1 ELSE 1 END
            * date_diff('day', DATE '1970-01-01', membership_expire_date) AS expire_key
    FROM transactions
),
state_at_cutoff AS (
    SELECT msno, membership_expire_date AS expire_date
    FROM ordered
    WHERE transaction_date < DATE '${cutoff}'
    QUALIFY row_number() OVER (
        PARTITION BY msno
        ORDER BY transaction_date DESC, plan_sig ASC, is_cancel DESC, expire_key DESC
    ) = 1
),
cohort AS (
    SELECT *
    FROM state_at_cutoff
    WHERE expire_date BETWEEN DATE '${cutoff}' AND DATE '${month_end}'
),
future AS (
    SELECT
        c.msno,
        o.transaction_date,
        o.membership_expire_date,
        o.is_cancel,
        row_number() OVER (
            PARTITION BY c.msno
            ORDER BY o.transaction_date, o.plan_sig DESC, o.is_cancel, o.expire_key
        ) AS seq
    FROM cohort c
    JOIN ordered o
      ON o.msno = c.msno
     AND o.transaction_date >= DATE '${cutoff}'
     AND o.transaction_date <= c.expire_date + ${window_days}
),
walk AS (
    SELECT
        *,
        min(CASE WHEN NOT is_cancel THEN seq END) OVER (PARTITION BY msno) AS renewal_seq
    FROM future
),
outcome AS (
    SELECT
        msno,
        max(transaction_date) FILTER (WHERE seq = renewal_seq) AS renewal_date,
        min(membership_expire_date) FILTER (
            WHERE is_cancel AND (renewal_seq IS NULL OR seq < renewal_seq)
        ) AS cancel_expire
    FROM walk
    GROUP BY msno
)
SELECT
    c.msno,
    DATE '${cutoff}' AS cohort_month,
    c.expire_date,
    coalesce(
        date_diff(
            'day',
            least(c.expire_date, coalesce(o.cancel_expire, c.expire_date)),
            o.renewal_date
        ) >= ${window_days},
        true  -- no renewal at all
    ) AS is_churn
FROM cohort c
LEFT JOIN outcome o USING (msno)
