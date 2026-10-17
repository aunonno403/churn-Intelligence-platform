-- Load the daily listening logs. Template vars: ${logs_dir}
-- Input is already restricted to sampled users and the feature lookback window by
-- logs_filter.py (the raw file is ~30 GB, so it is filtered while decompressing).
CREATE OR REPLACE TABLE user_logs AS
SELECT
    msno,
    strptime(date, '%Y%m%d')::DATE AS date,
    num_25::INTEGER    AS num_25,
    num_50::INTEGER    AS num_50,
    num_75::INTEGER    AS num_75,
    num_985::INTEGER   AS num_985,
    num_100::INTEGER   AS num_100,
    num_unq::INTEGER   AS num_unq,
    total_secs::DOUBLE AS total_secs
FROM read_csv(
    '${logs_dir}/*.csv',
    header = true, union_by_name = true,
    types = {'date': 'VARCHAR'}
);
