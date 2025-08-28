-- =====================================================================
-- Idempotent UPSERT for analytics.prediction_user_churn
-- Conflict target = table PK: (user_id, scored_at, model_version, churn_horizon_days)
-- Policy:
--   - OVERWRITE (latest): core prediction fields + operational counters
--   - PRESERVE (first write): data_cutoff_at, reference_dt, churn_threshold_dt
-- =====================================================================

INSERT INTO analytics.prediction_user_churn (
    user_id, scored_at, model_name, model_version, feature_version,
    data_cutoff_at, reference_dt, churn_horizon_days, churn_threshold_dt,
    churn_probability_raw, risk_band, score_percentile,
    top1_feature, top1_shap, top2_feature, top2_shap, top3_feature, top3_shap,
    order_count, total_spend, avg_order_value, avg_days_between_orders,
    login_count, cart_count, recency_days, rfm_sum, age, gender, age_group,
    used_coupon, avg_cart_per_login, category_diversity, rfm_bucket, kmeans_cluster,
    action_code_suggested,
    imputations_count, anomalies_count, pipeline_run_id, valid_from, valid_until
)
VALUES
    -- bind one row per record via client (e.g., psycopg2 executemany) or use COPY to temp + SELECT
    (
      /* $1..$N: replace with your driver placeholders or use named params */
      :user_id, :scored_at, :model_name, :model_version, :feature_version,
      :data_cutoff_at, :reference_dt, :churn_horizon_days, :churn_threshold_dt,
      :churn_probability_raw, :risk_band, :score_percentile,
      :top1_feature, :top1_shap, :top2_feature, :top2_shap, :top3_feature, :top3_shap,
      :order_count, :total_spend, :avg_order_value, :avg_days_between_orders,
      :login_count, :cart_count, :recency_days, :rfm_sum, :age, :gender, :age_group,
      :used_coupon, :avg_cart_per_login, :category_diversity, :rfm_bucket, :kmeans_cluster,
      :action_code_suggested,
      :imputations_count, :anomalies_count, :pipeline_run_id, :valid_from, :valid_until
    )
ON CONFLICT (user_id, scored_at, model_version, churn_horizon_days)
DO UPDATE SET
    -- OVERWRITE on rerun
    churn_probability_raw = EXCLUDED.churn_probability_raw,
    risk_band             = EXCLUDED.risk_band,
    score_percentile      = EXCLUDED.score_percentile,

    top1_feature          = EXCLUDED.top1_feature,
    top1_shap             = EXCLUDED.top1_shap,
    top2_feature          = EXCLUDED.top2_feature,
    top2_shap             = EXCLUDED.top2_shap,
    top3_feature          = EXCLUDED.top3_feature,
    top3_shap             = EXCLUDED.top3_shap,

    order_count           = EXCLUDED.order_count,
    total_spend           = EXCLUDED.total_spend,
    avg_order_value       = EXCLUDED.avg_order_value,
    avg_days_between_orders = EXCLUDED.avg_days_between_orders,
    login_count           = EXCLUDED.login_count,
    cart_count            = EXCLUDED.cart_count,
    recency_days          = EXCLUDED.recency_days,
    rfm_sum               = EXCLUDED.rfm_sum,
    age                   = EXCLUDED.age,
    gender                = EXCLUDED.gender,
    age_group             = EXCLUDED.age_group,
    used_coupon           = EXCLUDED.used_coupon,
    avg_cart_per_login    = EXCLUDED.avg_cart_per_login,
    category_diversity    = EXCLUDED.category_diversity,
    rfm_bucket            = EXCLUDED.rfm_bucket,
    kmeans_cluster        = EXCLUDED.kmeans_cluster,
    action_code_suggested = EXCLUDED.action_code_suggested,

    imputations_count     = COALESCE(EXCLUDED.imputations_count, analytics.prediction_user_churn.imputations_count),
    anomalies_count       = COALESCE(EXCLUDED.anomalies_count, analytics.prediction_user_churn.anomalies_count),
    pipeline_run_id       = EXCLUDED.pipeline_run_id,
    valid_from            = EXCLUDED.valid_from,
    valid_until           = EXCLUDED.valid_until

    -- PRESERVE first-write timing refs for same PK
    -- (comment next three lines if you prefer overwrite instead)
    /* data_cutoff_at        = analytics.prediction_user_churn.data_cutoff_at, */
    /* reference_dt          = analytics.prediction_user_churn.reference_dt,   */
    /* churn_threshold_dt    = analytics.prediction_user_churn.churn_threshold_dt */
;
