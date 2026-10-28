-- AWS Region: ap-southeast-2 (Sydney)
-- Query results:
-- s3://wajeeh-customer-churn-20261025/athena-results/
-- Run each statement separately in the Athena query editor.

CREATE DATABASE IF NOT EXISTS customer_churn;

CREATE EXTERNAL TABLE IF NOT EXISTS customer_churn.customers (
    customer_id STRING,
    gender STRING,
    senior_citizen INT,
    partner STRING,
    dependents STRING,
    tenure INT,
    phone_service STRING,
    multiple_lines STRING,
    internet_service STRING,
    online_security STRING,
    online_backup STRING,
    device_protection STRING,
    tech_support STRING,
    streaming_tv STRING,
    streaming_movies STRING,
    contract STRING,
    paperless_billing STRING,
    payment_method STRING,
    monthly_charges DOUBLE,
    total_charges DOUBLE,
    churn STRING
)
ROW FORMAT SERDE 'org.apache.hadoop.hive.serde2.OpenCSVSerde'
WITH SERDEPROPERTIES (
    'separatorChar' = ',',
    'quoteChar' = '"'
)
STORED AS TEXTFILE
LOCATION 's3://wajeeh-customer-churn-20261025/processed/clean/'
TBLPROPERTIES ('skip.header.line.count' = '1');

CREATE EXTERNAL TABLE IF NOT EXISTS customer_churn.predictions_demo (
    customer_id STRING,
    monthly_charges DOUBLE,
    churn_probability DOUBLE,
    predicted_churn INT,
    risk_flag STRING,
    revenue_exposure_score DOUBLE
)
ROW FORMAT SERDE 'org.apache.hadoop.hive.serde2.OpenCSVSerde'
WITH SERDEPROPERTIES (
    'separatorChar' = ',',
    'quoteChar' = '"'
)
STORED AS TEXTFILE
LOCATION 's3://wajeeh-customer-churn-20261025/predictions/demo/'
TBLPROPERTIES ('skip.header.line.count' = '1');