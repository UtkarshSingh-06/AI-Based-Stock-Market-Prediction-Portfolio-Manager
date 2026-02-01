-- ==========================================
-- PostgreSQL Initialization Script
-- Stock Predictor Platform
-- ==========================================

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";           -- Text search
CREATE EXTENSION IF NOT EXISTS "btree_gin";         -- Indexing
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements"; -- Query stats

-- Create schemas
CREATE SCHEMA IF NOT EXISTS analytics;
CREATE SCHEMA IF NOT EXISTS ml_models;
CREATE SCHEMA IF NOT EXISTS audit;

-- Grant permissions to application user
GRANT ALL PRIVILEGES ON SCHEMA public TO stockuser;
GRANT ALL PRIVILEGES ON SCHEMA analytics TO stockuser;
GRANT ALL PRIVILEGES ON SCHEMA ml_models TO stockuser;
GRANT ALL PRIVILEGES ON SCHEMA audit TO stockuser;

-- ==========================================
-- ANALYTICS SCHEMA - Materialized Views
-- ==========================================

-- Daily prediction statistics
CREATE MATERIALIZED VIEW IF NOT EXISTS analytics.daily_predictions AS
SELECT 
    symbol,
    DATE(prediction_date) as date,
    COUNT(*) as prediction_count,
    AVG(predicted_price) as avg_predicted,
    AVG(actual_price) as avg_actual,
    AVG(CASE 
        WHEN actual_price IS NOT NULL 
        THEN ABS(predicted_price - actual_price) / NULLIF(actual_price, 0) * 100 
        ELSE NULL 
    END) as avg_error_pct,
    MIN(predicted_price) as min_predicted,
    MAX(predicted_price) as max_predicted
FROM predictions
WHERE actual_price IS NOT NULL
GROUP BY symbol, DATE(prediction_date);

-- Create index on materialized view
CREATE UNIQUE INDEX IF NOT EXISTS idx_daily_predictions_symbol_date 
ON analytics.daily_predictions(symbol, date);

-- User activity summary
CREATE MATERIALIZED VIEW IF NOT EXISTS analytics.user_activity AS
SELECT 
    u.id as user_id,
    u.email,
    u.subscription_tier,
    COUNT(DISTINCT p.id) as total_predictions,
    COUNT(DISTINCT DATE(p.prediction_date)) as active_days,
    AVG(p.accuracy_score) as avg_accuracy,
    MAX(p.prediction_date) as last_prediction_date
FROM users u
LEFT JOIN predictions p ON u.id = p.user_id
GROUP BY u.id, u.email, u.subscription_tier;

-- Model performance summary
CREATE MATERIALIZED VIEW IF NOT EXISTS analytics.model_performance AS
SELECT 
    symbol,
    model_type,
    model_version,
    COUNT(*) as prediction_count,
    AVG(val_rmse) as avg_rmse,
    AVG(val_mape) as avg_mape,
    AVG(sharpe_ratio) as avg_sharpe,
    MAX(trained_at) as last_trained
FROM model_metrics
GROUP BY symbol, model_type, model_version;

-- ==========================================
-- FUNCTIONS
-- ==========================================

-- Function to refresh all materialized views
CREATE OR REPLACE FUNCTION analytics.refresh_all_views()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY analytics.daily_predictions;
    REFRESH MATERIALIZED VIEW CONCURRENTLY analytics.user_activity;
    REFRESH MATERIALIZED VIEW CONCURRENTLY analytics.model_performance;
    
    RAISE NOTICE 'All materialized views refreshed at %', NOW();
END;
$$ LANGUAGE plpgsql;

-- Function to calculate prediction accuracy
CREATE OR REPLACE FUNCTION calculate_prediction_accuracy(
    p_prediction_id INTEGER
)
RETURNS FLOAT AS $$
DECLARE
    v_predicted FLOAT;
    v_actual FLOAT;
    v_accuracy FLOAT;
BEGIN
    SELECT predicted_price, actual_price 
    INTO v_predicted, v_actual
    FROM predictions 
    WHERE id = p_prediction_id;
    
    IF v_actual IS NULL THEN
        RETURN NULL;
    END IF;
    
    v_accuracy := 100 - (ABS(v_predicted - v_actual) / v_actual * 100);
    
    UPDATE predictions 
    SET accuracy_score = v_accuracy,
        absolute_error = ABS(v_predicted - v_actual),
        percentage_error = ABS(v_predicted - v_actual) / v_actual * 100
    WHERE id = p_prediction_id;
    
    RETURN v_accuracy;
END;
$$ LANGUAGE plpgsql;

-- Function to get top performing models
CREATE OR REPLACE FUNCTION ml_models.get_top_models(
    p_symbol VARCHAR DEFAULT NULL,
    p_limit INTEGER DEFAULT 10
)
RETURNS TABLE (
    symbol VARCHAR,
    model_type VARCHAR,
    model_version VARCHAR,
    avg_accuracy FLOAT,
    prediction_count BIGINT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        p.symbol,
        p.model_type,
        p.model_version,
        AVG(p.accuracy_score) as avg_accuracy,
        COUNT(*) as prediction_count
    FROM predictions p
    WHERE 
        p.actual_price IS NOT NULL
        AND (p_symbol IS NULL OR p.symbol = p_symbol)
    GROUP BY p.symbol, p.model_type, p.model_version
    HAVING COUNT(*) >= 10
    ORDER BY avg_accuracy DESC
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql;

-- Function to clean up old audit logs
CREATE OR REPLACE FUNCTION audit.cleanup_old_logs(
    p_days INTEGER DEFAULT 90
)
RETURNS INTEGER AS $$
DECLARE
    v_deleted INTEGER;
BEGIN
    DELETE FROM audit_logs 
    WHERE timestamp < NOW() - INTERVAL '1 day' * p_days;
    
    GET DIAGNOSTICS v_deleted = ROW_COUNT;
    
    RAISE NOTICE 'Deleted % audit log records older than % days', v_deleted, p_days;
    
    RETURN v_deleted;
END;
$$ LANGUAGE plpgsql;

-- ==========================================
-- TRIGGERS
-- ==========================================

-- Trigger to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply trigger to relevant tables (will be created after tables exist)
-- CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users
--     FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ==========================================
-- INDEXES FOR PERFORMANCE
-- ==========================================

-- Predictions indexes
-- CREATE INDEX IF NOT EXISTS idx_predictions_symbol ON predictions(symbol);
-- CREATE INDEX IF NOT EXISTS idx_predictions_user_date ON predictions(user_id, prediction_date DESC);
-- CREATE INDEX IF NOT EXISTS idx_predictions_accuracy ON predictions(accuracy_score DESC) WHERE actual_price IS NOT NULL;

-- Users indexes  
-- CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
-- CREATE INDEX IF NOT EXISTS idx_users_subscription ON users(subscription_tier);
-- CREATE INDEX IF NOT EXISTS idx_users_api_key ON users(api_key);

-- Audit logs indexes
-- CREATE INDEX IF NOT EXISTS idx_audit_logs_user ON audit_logs(user_id);
-- CREATE INDEX IF NOT EXISTS idx_audit_logs_timestamp ON audit_logs(timestamp DESC);
-- CREATE INDEX IF NOT EXISTS idx_audit_logs_action ON audit_logs(action);

-- ==========================================
-- SCHEDULED JOBS (requires pg_cron extension)
-- ==========================================

-- Uncomment if pg_cron is installed:
-- SELECT cron.schedule(
--     'refresh-analytics',
--     '0 2 * * *',  -- Daily at 2 AM
--     'SELECT analytics.refresh_all_views()'
-- );

-- SELECT cron.schedule(
--     'cleanup-audit-logs',
--     '0 3 * * 0',  -- Weekly on Sunday at 3 AM
--     'SELECT audit.cleanup_old_logs(90)'
-- );

-- ==========================================
-- COMMENTS
-- ==========================================

COMMENT ON SCHEMA analytics IS 'Analytics and reporting views';
COMMENT ON SCHEMA ml_models IS 'Machine learning model metadata';
COMMENT ON SCHEMA audit IS 'Audit trail and logging';

COMMENT ON MATERIALIZED VIEW analytics.daily_predictions IS 'Daily aggregated prediction statistics per symbol';
COMMENT ON MATERIALIZED VIEW analytics.user_activity IS 'User activity and engagement metrics';
COMMENT ON MATERIALIZED VIEW analytics.model_performance IS 'Model performance metrics by symbol and type';

COMMENT ON FUNCTION analytics.refresh_all_views() IS 'Refresh all materialized views for analytics';
COMMENT ON FUNCTION calculate_prediction_accuracy(INTEGER) IS 'Calculate and update prediction accuracy score';
COMMENT ON FUNCTION ml_models.get_top_models(VARCHAR, INTEGER) IS 'Get top performing models by accuracy';
COMMENT ON FUNCTION audit.cleanup_old_logs(INTEGER) IS 'Delete audit logs older than specified days';

-- ==========================================
-- COMPLETION MESSAGE
-- ==========================================

DO $$
BEGIN
    RAISE NOTICE '========================================';
    RAISE NOTICE 'Database initialization completed!';
    RAISE NOTICE '========================================';
    RAISE NOTICE 'Extensions created: uuid-ossp, pg_trgm, btree_gin, pg_stat_statements';
    RAISE NOTICE 'Schemas created: analytics, ml_models, audit';
    RAISE NOTICE 'Functions created: 4';
    RAISE NOTICE 'Materialized views created: 3';
    RAISE NOTICE '========================================';
END $$;