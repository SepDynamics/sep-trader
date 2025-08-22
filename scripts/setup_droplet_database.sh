#!/bin/bash

# SEP Droplet Database Setup Script
# Configures PostgreSQL + TimescaleDB + Redis for SEP trading system

set -e

echo "🗄️ Setting up SEP database infrastructure..."

# Install PostgreSQL 15 + TimescaleDB
apt-get update
apt-get install -y postgresql-15 postgresql-client-15 postgresql-contrib-15

# Add TimescaleDB repository and install
echo "deb https://packagecloud.io/timescale/timescaledb/ubuntu/ $(lsb_release -c -s) main" | tee /etc/apt/sources.list.d/timescaledb.list
wget --quiet -O - https://packagecloud.io/timescale/timescaledb/gpgkey | apt-key add -
apt-get update
apt-get install -y timescaledb-2-postgresql-15

# Install Redis
apt-get install -y redis-server

# Configure PostgreSQL data directory on volume
systemctl stop postgresql
mkdir -p /mnt/volume_nyc3_01/sep-data/postgres/data
chown -R postgres:postgres /mnt/volume_nyc3_01/sep-data/postgres

# Update PostgreSQL configuration
sed -i "s|#data_directory = 'ConfigDir'|data_directory = '/mnt/volume_nyc3_01/sep-data/postgres/data'|g" /etc/postgresql/15/main/postgresql.conf
sed -i "s|#listen_addresses = 'localhost'|listen_addresses = '*'|g" /etc/postgresql/15/main/postgresql.conf
sed -i "s|#shared_preload_libraries = ''|shared_preload_libraries = 'timescaledb'|g" /etc/postgresql/15/main/postgresql.conf

# Configure PostgreSQL authentication
cat >> /etc/postgresql/15/main/pg_hba.conf << EOF

# SEP Trading System
host    sep_trading     sep_user        0.0.0.0/0               md5
host    all             sep_admin       0.0.0.0/0               md5
EOF

# Initialize PostgreSQL data directory
sudo -u postgres /usr/lib/postgresql/15/bin/initdb -D /mnt/volume_nyc3_01/sep-data/postgres/data

# Start PostgreSQL
systemctl start postgresql
systemctl enable postgresql

# Create SEP database and users
sudo -u postgres psql << EOF
CREATE DATABASE sep_trading;
CREATE USER sep_user WITH PASSWORD 'sep_password';
CREATE USER sep_admin WITH PASSWORD 'sep_admin_password';

-- Grant permissions
GRANT ALL PRIVILEGES ON DATABASE sep_trading TO sep_user;
GRANT ALL PRIVILEGES ON DATABASE sep_trading TO sep_admin;

-- Connect to sep_trading database
\c sep_trading

-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Create training data table
CREATE TABLE training_data (
    pair VARCHAR(10) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    features JSONB NOT NULL,
    target DOUBLE PRECISION NOT NULL,
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (pair, timestamp)
);

-- Convert to hypertable for time-series optimization
SELECT create_hypertable('training_data', 'timestamp', 'pair', 4);

-- Create models table
CREATE TABLE models (
    model_id VARCHAR(64) PRIMARY KEY,
    pair VARCHAR(10) NOT NULL,
    accuracy DOUBLE PRECISION NOT NULL,
    trained_at TIMESTAMPTZ NOT NULL,
    hyperparameters JSONB,
    redis_key VARCHAR(128),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create market data table
CREATE TABLE market_data (
    pair VARCHAR(10) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    open DOUBLE PRECISION NOT NULL,
    high DOUBLE PRECISION NOT NULL,
    low DOUBLE PRECISION NOT NULL,
    close DOUBLE PRECISION NOT NULL,
    volume BIGINT DEFAULT 0,
    timeframe VARCHAR(10) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (pair, timestamp, timeframe)
);

-- Convert market data to hypertable
SELECT create_hypertable('market_data', 'timestamp', 'pair', 4);

-- Create indices for better performance
CREATE INDEX idx_training_data_pair_time ON training_data (pair, timestamp DESC);
CREATE INDEX idx_models_pair_accuracy ON models (pair, accuracy DESC);
CREATE INDEX idx_market_data_pair_timeframe ON market_data (pair, timeframe, timestamp DESC);

-- Set up data retention policies (keep 2 years of training data, 1 year of market data)
SELECT add_retention_policy('training_data', INTERVAL '2 years');
SELECT add_retention_policy('market_data', INTERVAL '1 year');

-- Grant table permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO sep_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO sep_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO sep_admin;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO sep_admin;
EOF

# Configure Redis
sed -i 's/^bind 127.0.0.1/bind 0.0.0.0/' /etc/redis/redis.conf
sed -i 's/^# requirepass foobared/requirepass sep_redis_password/' /etc/redis/redis.conf
sed -i 's|^dir /var/lib/redis|dir /mnt/volume_nyc3_01/sep-data/redis|' /etc/redis/redis.conf

# Create Redis data directory
mkdir -p /mnt/volume_nyc3_01/sep-data/redis
chown redis:redis /mnt/volume_nyc3_01/sep-data/redis

# Start Redis
systemctl restart redis-server
systemctl enable redis-server

# Configure firewall
ufw allow 5432/tcp  # PostgreSQL
ufw allow 6380/tcp  # Redis

echo "✅ Database setup complete!"
echo "🐘 PostgreSQL: sep_trading database with TimescaleDB"
echo "🔴 Redis: Model cache and real-time data"
echo "📊 Tables: training_data, models, market_data"
echo "🔐 Users: sep_user, sep_admin"
