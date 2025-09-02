-- Improve read/write performance for scoring/day aggregates

-- Analytics: typical lookup by (reference_dt, model_version, feature_version, horizon, valid_until)
CREATE INDEX IF NOT EXISTS ix_pred_ref_ver_feat_hz_valid
ON analytics.prediction_user_churn (reference_dt, model_version, feature_version, churn_horizon_days, valid_until);

-- Also helpful for date filtering by KST report day if you use it later
-- CREATE INDEX IF NOT EXISTS ix_pred_report_kst ON analytics.prediction_user_churn (report_dt_kst);

-- Mart: ensure 1 row per day
CREATE UNIQUE INDEX IF NOT EXISTS ux_mart_daily_churn ON mart.daily_churn_prediction_aggr(date);
