-- =====================================================================
-- ECPS-CHURN / analytics & mart DDL (Schema Contract - FINAL)
-- Source of truth for prediction storage & downstream mart/view
-- NOTE: All timestamps are stored in UTC (timestamptz).
-- =====================================================================

-- ----- SCHEMAS --------------------------------------------------------
CREATE SCHEMA IF NOT EXISTS analytics AUTHORIZATION postgres;
CREATE SCHEMA IF NOT EXISTS mart       AUTHORIZATION postgres;

-- ----- TABLE: analytics.dim_customers -------------------------------
CREATE TABLE IF NOT EXISTS analytics.dim_customers (
    user_id      BIGINT       PRIMARY KEY,
    user_name    TEXT         NULL,
    tenure_days  INTEGER      NULL,
    updated_at   TIMESTAMPTZ  NOT NULL DEFAULT now()
);

-- ----- TABLE: analytics.prediction_user_churn ------------------------
-- PK = (user_id, scored_at, model_version, churn_horizon_days)
-- Constraints match risk band, percentile, gender, age_group domains.
CREATE TABLE IF NOT EXISTS analytics.prediction_user_churn (
    user_id                 BIGINT       NOT NULL,
    scored_at               TIMESTAMPTZ  NOT NULL,     -- scoring job wall-clock (UTC)
    model_name              TEXT         NOT NULL,      -- 'lgbm' only (constraint below)
    model_version           TEXT         NOT NULL,      -- e.g., lgbm_v20250828_ab12c34
    feature_version         TEXT         NOT NULL,      -- e.g., fe_20250828_ab12c34
    data_cutoff_at          TIMESTAMPTZ  NOT NULL,      -- latest event timestamp included in features
    reference_dt            TIMESTAMPTZ  NOT NULL,      -- scoring reference ts (usually = scored_at)
    churn_horizon_days      INTEGER      NOT NULL DEFAULT 90,
    churn_threshold_dt      TIMESTAMPTZ  NOT NULL,      -- reference_dt - interval 'horizon days'
    churn_probability_raw   DOUBLE PRECISION NOT NULL,  -- 0..1
    risk_band               TEXT         NOT NULL,      -- {'VH','H','M','L'}
    score_percentile        DOUBLE PRECISION NOT NULL,  -- 0..100 (batch-global)
    top1_feature            TEXT         NULL,
    top1_shap               DOUBLE PRECISION NULL,
    top2_feature            TEXT         NULL,
    top2_shap               DOUBLE PRECISION NULL,
    top3_feature            TEXT         NULL,
    top3_shap               DOUBLE PRECISION NULL,

    -- Optional profile/aggregates (nullable)
    order_count             INTEGER      NULL,
    total_spend             BIGINT       NULL,
    avg_order_value         DOUBLE PRECISION NULL,
    avg_days_between_orders DOUBLE PRECISION NULL,
    login_count             INTEGER      NULL,
    cart_count              INTEGER      NULL,
    recency_days            INTEGER      NULL,
    rfm_sum                 INTEGER      NULL,
    age                     INTEGER      NULL,
    gender                  TEXT         NULL,
    age_group               TEXT         NULL,
    used_coupon             BOOLEAN      NULL,
    avg_cart_per_login      DOUBLE PRECISION NULL,
    category_diversity      DOUBLE PRECISION NULL,
    rfm_bucket              TEXT         NULL,
    kmeans_cluster          TEXT         NULL,
    action_code_suggested   TEXT         NULL,

    -- Operational
    imputations_count       INTEGER      NOT NULL DEFAULT 0,
    anomalies_count         INTEGER      NOT NULL DEFAULT 0,
    pipeline_run_id         TEXT         NOT NULL,
    valid_from              TIMESTAMPTZ  NOT NULL,
    valid_until             TIMESTAMPTZ  NOT NULL,

    CONSTRAINT prediction_user_churn_pk
        PRIMARY KEY (user_id, scored_at, model_version, churn_horizon_days),

    -- Domain checks
    CONSTRAINT chk_model_name
        CHECK (model_name = 'lgbm'),
    CONSTRAINT chk_prob_range
        CHECK (churn_probability_raw >= 0.0 AND churn_probability_raw <= 1.0),
    CONSTRAINT chk_percentile_range
        CHECK (score_percentile >= 0.0 AND score_percentile <= 100.0),
    CONSTRAINT chk_risk_band
        CHECK (risk_band IN ('VH','H','M','L')),
    CONSTRAINT chk_gender
        CHECK (gender IS NULL OR gender IN ('MALE','FEMALE')),
    CONSTRAINT chk_age_group
        CHECK (age_group IS NULL OR age_group IN ('<25','25-34','35-44','45+'))
);

-- Helpful indexes
CREATE INDEX IF NOT EXISTS idx_puc_user_id
    ON analytics.prediction_user_churn (user_id);
CREATE INDEX IF NOT EXISTS idx_puc_scored_at
    ON analytics.prediction_user_churn (scored_at DESC);
CREATE INDEX IF NOT EXISTS idx_puc_risk_band
    ON analytics.prediction_user_churn (risk_band);

-- ----- TABLE: analytics.prediction_logs ------------------------------
CREATE TABLE IF NOT EXISTS analytics.prediction_logs (
    prediction_log_id     BIGSERIAL    PRIMARY KEY,
    user_id               BIGINT       NOT NULL,
    scored_at             TIMESTAMPTZ  NOT NULL,
    model_version         TEXT         NOT NULL,
    churn_horizon_days    INTEGER      NOT NULL DEFAULT 90,
    prediction_timestamp  TIMESTAMPTZ  NOT NULL DEFAULT now(),
    group_code            TEXT         NULL,           -- {A,B,C,D} etc
    extra_meta            JSONB        NULL,
    CONSTRAINT chk_group_code
        CHECK (group_code IS NULL OR group_code IN ('A','B','C','D')),
    CONSTRAINT fk_logs_to_prediction
        FOREIGN KEY (user_id, scored_at, model_version, churn_horizon_days)
        REFERENCES analytics.prediction_user_churn(user_id, scored_at, model_version, churn_horizon_days)
        ON UPDATE CASCADE ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_prediction_logs_scored_at
    ON analytics.prediction_logs (scored_at DESC);

-- ----- TABLE: analytics.action_recommendations -----------------------
CREATE TABLE IF NOT EXISTS analytics.action_recommendations (
    recommendation_id   BIGSERIAL     PRIMARY KEY,
    prediction_log_id   BIGINT        NOT NULL,
    operator_id         BIGINT        NULL,
    details             TEXT          NULL,
    status              TEXT          NOT NULL DEFAULT 'PENDING',
    created_at          TIMESTAMPTZ   NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ   NOT NULL DEFAULT now(),
    CONSTRAINT chk_action_status
        CHECK (status IN ('PENDING','SENT','SCHEDULED','DONE','DISMISSED')),
    CONSTRAINT action_recommendations_prediction_log_id_fkey
        FOREIGN KEY (prediction_log_id)
        REFERENCES analytics.prediction_logs(prediction_log_id)
        ON UPDATE CASCADE ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_action_reco_status
    ON analytics.action_recommendations (status, updated_at DESC);

-- ----- MART: dashboards ---------------------------------------------
CREATE TABLE IF NOT EXISTS mart.dashboard_metrics (
    metric_id     BIGSERIAL     PRIMARY KEY,
    metric_date   DATE          NOT NULL,
    metric_name   TEXT          NOT NULL,
    metric_value  NUMERIC       NOT NULL,
    segment       TEXT          NULL,
    created_at    TIMESTAMPTZ   NOT NULL DEFAULT now(),
    CONSTRAINT uq_dashboard_metrics UNIQUE (metric_date, metric_name, segment)
);

CREATE INDEX IF NOT EXISTS idx_dashboard_metrics_date
    ON mart.dashboard_metrics (metric_date DESC);

-- Latest prediction per user (by scored_at)
CREATE OR REPLACE VIEW mart.v_latest_prediction_per_user AS
SELECT DISTINCT ON (user_id)
    user_id, scored_at, model_name, model_version, churn_horizon_days,
    churn_probability_raw, risk_band, score_percentile,
    reference_dt, churn_threshold_dt,
    order_count, total_spend, avg_order_value, avg_days_between_orders,
    login_count, cart_count, recency_days, rfm_sum, age, gender, age_group,
    used_coupon, avg_cart_per_login, category_diversity, rfm_bucket,
    kmeans_cluster, action_code_suggested,
    valid_from, valid_until
FROM analytics.prediction_user_churn
ORDER BY user_id, scored_at DESC;
