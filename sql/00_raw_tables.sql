-- Load a deterministic user sample of the raw KKBox CSVs into typed DuckDB tables.
-- Template vars: ${raw_dir}, ${sample_mod}, ${sample_keep}
-- Sampling is md5-based on msno, so the same users are kept in every table and every run.

CREATE OR REPLACE MACRO in_sample(id) AS md5_number(id) % ${sample_mod} < ${sample_keep};

-- transactions.csv + transactions_v2.csv overlap in March 2017 -> DISTINCT removes duplicates.
CREATE OR REPLACE TABLE transactions AS
SELECT DISTINCT
    msno,
    payment_method_id::INTEGER                            AS payment_method_id,
    payment_plan_days::INTEGER                            AS payment_plan_days,
    plan_list_price::INTEGER                              AS plan_list_price,
    actual_amount_paid::INTEGER                           AS actual_amount_paid,
    is_auto_renew::BOOLEAN                                AS is_auto_renew,
    strptime(transaction_date, '%Y%m%d')::DATE            AS transaction_date,
    strptime(membership_expire_date, '%Y%m%d')::DATE      AS membership_expire_date,
    is_cancel::BOOLEAN                                    AS is_cancel
FROM read_csv(
    '${raw_dir}/**/transactions*.csv',
    header = true, union_by_name = true,
    types = {'transaction_date': 'VARCHAR', 'membership_expire_date': 'VARCHAR'}
)
WHERE in_sample(msno);

CREATE OR REPLACE TABLE sampled_users AS
SELECT DISTINCT msno FROM transactions;

-- user_logs is loaded separately (01_user_logs.sql) from a pre-filtered stream; see logs_filter.py.

CREATE OR REPLACE TABLE members AS
SELECT
    msno,
    city::INTEGER           AS city,
    bd::INTEGER             AS bd,
    NULLIF(gender, '')      AS gender,
    registered_via::INTEGER AS registered_via,
    try_strptime(registration_init_time, '%Y%m%d')::DATE AS registration_date
FROM read_csv(
    '${raw_dir}/**/members_v3.csv',
    header = true,
    types = {'registration_init_time': 'VARCHAR', 'gender': 'VARCHAR'}
)
WHERE in_sample(msno);

-- Official WSDM labels for the Feb-2017 expiry cohort; used only to validate our label logic.
CREATE OR REPLACE TABLE official_labels AS
SELECT msno, is_churn::BOOLEAN AS is_churn
FROM read_csv('${raw_dir}/**/train.csv', header = true)
WHERE in_sample(msno);
