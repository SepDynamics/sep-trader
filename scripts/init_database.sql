-- SEP Professional Trader-Bot Database Schema
-- PostgreSQL + TimescaleDB optimized for trading data

-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Market candles table (time-series data)
CREATE TABLE IF NOT EXISTS market_candles (
    timestamp    TIMESTAMPTZ NOT NULL,
    pair         VARCHAR(10) NOT NULL,
    timeframe    VARCHAR(5) NOT NULL,   -- M1, M5, M15, H1, H4, D1
    open         DECIMAL(12,6) NOT NULL,
    high         DECIMAL(12,6) NOT NULL,
    low          DECIMAL(12,6) NOT NULL,
    close        DECIMAL(12,6) NOT NULL,
    volume       BIGINT DEFAULT 0,
    spread       DECIMAL(8,5) DEFAULT 0,
    tick_count   INTEGER DEFAULT 0
);

-- Convert to TimescaleDB hypertable for time-series optimization
SELECT create_hypertable('market_candles', 'timestamp', 
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

-- Create indexes for fast queries
CREATE INDEX IF NOT EXISTS idx_candles_pair_timeframe_time 
    ON market_candles (pair, timeframe, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_candles_pair_time 
    ON market_candles (pair, timestamp DESC);

-- Trading signals table
CREATE TABLE IF NOT EXISTS trading_signals (
    id           SERIAL PRIMARY KEY,
    timestamp    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    pair         VARCHAR(10) NOT NULL,
    direction    VARCHAR(4) NOT NULL CHECK (direction IN ('BUY', 'SELL')),
    confidence   DECIMAL(5,4) NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    qfh_metrics  JSONB,
    pattern_data JSONB,
    executed     BOOLEAN DEFAULT FALSE,
    execution_price DECIMAL(12,6),
    result       JSONB,
    created_by   VARCHAR(50) DEFAULT 'sep_engine'
);

-- Index for signal queries
CREATE INDEX IF NOT EXISTS idx_signals_timestamp ON trading_signals (timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_signals_pair_executed ON trading_signals (pair, executed);

-- Trade executions table
CREATE TABLE IF NOT EXISTS trade_executions (
    id              SERIAL PRIMARY KEY,
    signal_id       INTEGER REFERENCES trading_signals(id),
    oanda_order_id  VARCHAR(50),
    pair            VARCHAR(10) NOT NULL,
    direction       VARCHAR(4) NOT NULL,
    units           INTEGER NOT NULL,
    entry_price     DECIMAL(12,6) NOT NULL,
    exit_price      DECIMAL(12,6),
    stop_loss       DECIMAL(12,6),
    take_profit     DECIMAL(12,6),
    opened_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    closed_at       TIMESTAMPTZ,
    pnl             DECIMAL(12,2),
    status          VARCHAR(20) DEFAULT 'OPEN' 
                   CHECK (status IN ('OPEN', 'CLOSED', 'CANCELLED', 'FILLED'))
);

-- Index for trade queries
CREATE INDEX IF NOT EXISTS idx_trades_opened_at ON trade_executions (opened_at DESC);
CREATE INDEX IF NOT EXISTS idx_trades_pair_status ON trade_executions (pair, status);

-- System configuration table
CREATE TABLE IF NOT EXISTS system_config (
    key          VARCHAR(100) PRIMARY KEY,
    value        JSONB NOT NULL,
    description  TEXT,
    updated_at   TIMESTAMPTZ DEFAULT NOW(),
    updated_by   VARCHAR(50) DEFAULT 'system'
);

-- Trading pairs configuration
CREATE TABLE IF NOT EXISTS trading_pairs (
    pair            VARCHAR(10) PRIMARY KEY,
    enabled         BOOLEAN DEFAULT FALSE,
    risk_level      DECIMAL(3,2) DEFAULT 0.02, -- 2% risk per trade
    max_position    INTEGER DEFAULT 10000,      -- Max units
    spread_filter   DECIMAL(6,3) DEFAULT 2.0,   -- Max spread in pips
    active_hours    JSONB,                      -- Trading hours config
    last_signal     TIMESTAMPTZ,
    total_trades    INTEGER DEFAULT 0,
    win_rate        DECIMAL(5,2) DEFAULT 0,
    total_pnl       DECIMAL(12,2) DEFAULT 0,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

-- Performance metrics table
CREATE TABLE IF NOT EXISTS performance_metrics (
    id              SERIAL PRIMARY KEY,
    date            DATE NOT NULL,
    pair            VARCHAR(10),
    total_signals   INTEGER DEFAULT 0,
    executed_trades INTEGER DEFAULT 0,
    win_rate        DECIMAL(5,2) DEFAULT 0,
    daily_pnl       DECIMAL(12,2) DEFAULT 0,
    max_drawdown    DECIMAL(12,2) DEFAULT 0,
    accuracy        DECIMAL(5,2) DEFAULT 0,
    confidence_avg  DECIMAL(5,4) DEFAULT 0,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- Index for performance queries
CREATE INDEX IF NOT EXISTS idx_performance_date_pair ON performance_metrics (date DESC, pair);

-- System logs table
CREATE TABLE IF NOT EXISTS system_logs (
    id          SERIAL PRIMARY KEY,
    timestamp   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    level       VARCHAR(10) NOT NULL CHECK (level IN ('DEBUG', 'INFO', 'WARN', 'ERROR')),
    component   VARCHAR(50) NOT NULL,
    message     TEXT NOT NULL,
    metadata    JSONB,
    session_id  VARCHAR(50)
);

-- Convert logs to hypertable for efficient log storage
SELECT create_hypertable('system_logs', 'timestamp', 
    chunk_time_interval => INTERVAL '1 week',
    if_not_exists => TRUE
);

-- Insert default configuration
INSERT INTO system_config (key, value, description) VALUES
    ('trading_enabled', 'false', 'Master trading enable/disable switch'),
    ('max_concurrent_trades', '5', 'Maximum number of concurrent open trades'),
    ('global_risk_limit', '0.10', 'Maximum total portfolio risk (10%)'),
    ('market_hours', '{"start": "22:00", "end": "22:00", "timezone": "UTC", "days": [0,1,2,3,4]}', 'Forex market hours'),
    ('system_version', '"1.0.0"', 'SEP Trading System version'),
    ('last_sync', 'null', 'Last data synchronization timestamp')
ON CONFLICT (key) DO NOTHING;

-- Insert default trading pairs
INSERT INTO trading_pairs (pair, enabled, risk_level) VALUES
    ('EUR_USD', false, 0.02),
    ('GBP_USD', false, 0.02),
    ('USD_JPY', false, 0.02),
    ('USD_CHF', false, 0.02),
    ('AUD_USD', false, 0.02),
    ('USD_CAD', false, 0.02),
    ('NZD_USD', false, 0.02),
    ('EUR_GBP', false, 0.015),
    ('EUR_JPY', false, 0.015),
    ('GBP_JPY', false, 0.015)
ON CONFLICT (pair) DO NOTHING;

-- Create views for common queries
CREATE OR REPLACE VIEW v_active_pairs AS
SELECT * FROM trading_pairs WHERE enabled = true;

CREATE OR REPLACE VIEW v_recent_signals AS
SELECT s.*, tp.enabled as pair_enabled
FROM trading_signals s
JOIN trading_pairs tp ON s.pair = tp.pair
WHERE s.timestamp >= NOW() - INTERVAL '24 hours'
ORDER BY s.timestamp DESC;

CREATE OR REPLACE VIEW v_open_trades AS
SELECT * FROM trade_executions 
WHERE status = 'OPEN'
ORDER BY opened_at DESC;

CREATE OR REPLACE VIEW v_daily_performance AS
SELECT 
    date,
    COUNT(*) as total_signals,
    SUM(CASE WHEN executed THEN 1 ELSE 0 END) as executed_trades,
    AVG(confidence) as avg_confidence,
    COUNT(DISTINCT pair) as active_pairs
FROM trading_signals 
WHERE timestamp >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY date
ORDER BY date DESC;

-- Set up automatic data retention (keep 1 year of minute data, 5 years of daily)
SELECT add_retention_policy('market_candles', INTERVAL '1 year');
SELECT add_retention_policy('system_logs', INTERVAL '3 months');

-- Create function to update trading pair statistics
CREATE OR REPLACE FUNCTION update_pair_stats(pair_symbol VARCHAR(10)) 
RETURNS VOID AS $$
BEGIN
    UPDATE trading_pairs SET
        total_trades = (SELECT COUNT(*) FROM trade_executions WHERE pair = pair_symbol),
        win_rate = (
            SELECT COALESCE(
                100.0 * COUNT(CASE WHEN pnl > 0 THEN 1 END) / NULLIF(COUNT(*), 0), 
                0
            )
            FROM trade_executions 
            WHERE pair = pair_symbol AND status = 'CLOSED'
        ),
        total_pnl = (
            SELECT COALESCE(SUM(pnl), 0) 
            FROM trade_executions 
            WHERE pair = pair_symbol AND status = 'CLOSED'
        ),
        last_signal = (
            SELECT MAX(timestamp) 
            FROM trading_signals 
            WHERE pair = pair_symbol
        ),
        updated_at = NOW()
    WHERE pair = pair_symbol;
END;
$$ LANGUAGE plpgsql;

-- Grant permissions to trading user
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO sep_trader;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO sep_trader;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO sep_trader;

-- Create database info view
CREATE OR REPLACE VIEW v_database_info AS
SELECT 
    'SEP Professional Trader-Bot Database' as system_name,
    current_database() as database_name,
    version() as postgresql_version,
    (SELECT extversion FROM pg_extension WHERE extname = 'timescaledb') as timescaledb_version,
    pg_size_pretty(pg_database_size(current_database())) as database_size,
    NOW() as initialized_at;

-- Log successful initialization
INSERT INTO system_logs (level, component, message, metadata) VALUES
    ('INFO', 'database', 'SEP Trading Database initialized successfully', 
     '{"tables_created": 8, "views_created": 5, "functions_created": 1}');

-- Display setup summary
SELECT 'SEP Professional Trader-Bot Database Setup Complete!' as status;
SELECT * FROM v_database_info;
