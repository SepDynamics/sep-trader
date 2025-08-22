#!/bin/bash

# SEP Droplet Database Setup Script - Ubuntu 24.04 Compatible
# Configures PostgreSQL + TimescaleDB + Redis for SEP trading system

set -e

echo "🗄️ Setting up SEP database infrastructure on Ubuntu 24.04..."

# Install PostgreSQL (default version for Ubuntu 24.04)
apt-get update
apt-get install -y postgresql postgresql-client postgresql-contrib

# Install Redis
apt-get install -y redis-server

# Check if external volume is mounted
if [ ! -d "/mnt/volume_nyc3_01" ]; then
    echo "❌ External volume not found at /mnt/volume_nyc3_01"
    echo "Creating local data directories instead..."
    mkdir -p /opt/sep-data/postgres/data
    mkdir -p /opt/sep-data/redis
    DATA_ROOT="/opt/sep-data"
else
    echo "✅ External volume found"
    mkdir -p /mnt/volume_nyc3_01/sep-data/postgres/data
    mkdir -p /mnt/volume_nyc3_01/sep-data/redis
    DATA_ROOT="/mnt/volume_nyc3_01/sep-data"
fi

# Configure PostgreSQL data directory
systemctl stop postgresql
chown -R postgres:postgres $DATA_ROOT/postgres

# Get PostgreSQL version
PG_VERSION=$(sudo -u postgres psql -c "SHOW server_version;" | grep -o '[0-9]\+' | head -1)
echo "📊 PostgreSQL version: $PG_VERSION"

# Update PostgreSQL configuration
PG_CONFIG="/etc/postgresql/$PG_VERSION/main/postgresql.conf"
sed -i "s|#data_directory = .*|data_directory = '$DATA_ROOT/postgres/data'|g" $PG_CONFIG
sed -i "s|#listen_addresses = 'localhost'|listen_addresses = '*'|g" $PG_CONFIG

# Configure PostgreSQL authentication
cat >> /etc/postgresql/$PG_VERSION/main/pg_hba.conf << EOF

# SEP Trading System
host    sep_trading     sep_user        0.0.0.0/0               md5
host    all             sep_admin       0.0.0.0/0               md5
EOF

# Initialize PostgreSQL data directory
sudo -u postgres /usr/lib/postgresql/$PG_VERSION/bin/initdb -D $DATA_ROOT/postgres/data

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

-- Create basic tables for pattern storage
CREATE TABLE patterns (
    pattern_id VARCHAR(64) PRIMARY KEY,
    pair VARCHAR(10) NOT NULL,
    pattern_data TEXT NOT NULL,
    accuracy DOUBLE PRECISION NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Create models table
CREATE TABLE models (
    model_id VARCHAR(64) PRIMARY KEY,
    pair VARCHAR(10) NOT NULL,
    accuracy DOUBLE PRECISION NOT NULL,
    trained_at TIMESTAMP NOT NULL,
    hyperparameters TEXT,
    redis_key VARCHAR(128),
    created_at TIMESTAMP DEFAULT NOW()
);

-- Grant table permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO sep_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO sep_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO sep_admin;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO sep_admin;
EOF

# Configure Redis
sed -i 's/^bind 127.0.0.1/bind 0.0.0.0/' /etc/redis/redis.conf
sed -i 's/^# requirepass foobared/requirepass sep_redis_password/' /etc/redis/redis.conf
sed -i "s|^dir /var/lib/redis|dir $DATA_ROOT/redis|" /etc/redis/redis.conf

# Set Redis data directory ownership
chown redis:redis $DATA_ROOT/redis

# Start Redis
systemctl restart redis-server
systemctl enable redis-server

# Configure firewall
ufw allow 5432/tcp  # PostgreSQL
ufw allow 6380/tcp  # Redis

echo "✅ Database setup complete!"
echo "🐘 PostgreSQL: sep_trading database"
echo "🔴 Redis: Model cache and real-time data"
echo "📂 Data directory: $DATA_ROOT"
echo "🔐 Users: sep_user, sep_admin"
echo "🔑 Redis password: sep_redis_password"
