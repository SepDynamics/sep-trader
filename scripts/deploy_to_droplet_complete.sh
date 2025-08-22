#!/bin/bash

# Complete SEP Droplet Deployment Script
# Handles volume setup, database configuration, and service deployment

set -e

DROPLET_IP="165.227.109.187"
DROPLET_USER="root"

echo "🚀 Starting complete SEP droplet deployment..."

# Function to run commands on droplet
run_remote() {
    ssh -o StrictHostKeyChecking=no ${DROPLET_USER}@${DROPLET_IP} "$1"
}

# Function to copy files to droplet
copy_to_droplet() {
    scp -o StrictHostKeyChecking=no "$1" ${DROPLET_USER}@${DROPLET_IP}:"$2"
}

echo "📋 Step 1: Testing connection to droplet..."
if ! run_remote "echo 'Connection successful'"; then
    echo "❌ Failed to connect to droplet. Check SSH access."
    exit 1
fi

echo "💾 Step 2: Setting up volume and data directories..."
copy_to_droplet "./scripts/setup_droplet_volume.sh" "/tmp/"
run_remote "chmod +x /tmp/setup_droplet_volume.sh && /tmp/setup_droplet_volume.sh"

echo "🗄️ Step 3: Setting up database infrastructure..."
copy_to_droplet "./scripts/setup_droplet_database.sh" "/tmp/"
run_remote "chmod +x /tmp/setup_droplet_database.sh && /tmp/setup_droplet_database.sh"

echo "📦 Step 4: Installing system dependencies..."
run_remote "apt-get update && apt-get install -y curl wget git build-essential cmake ninja-build pkg-config libssl-dev libcurl4-openssl-dev libpq-dev libhwloc-dev crow-dev"

echo "🐳 Step 5: Installing Docker..."
run_remote "
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh
systemctl enable docker
systemctl start docker
"

echo "🔧 Step 6: Installing NVIDIA Container Toolkit..."
run_remote "
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
apt-get update
apt-get install -y nvidia-container-toolkit
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker
"

echo "📁 Step 7: Creating SEP application directory..."
run_remote "mkdir -p /opt/sep-trader && chown -R ${DROPLET_USER}:${DROPLET_USER} /opt/sep-trader"

echo "🔄 Step 8: Syncing SEP source code..."
rsync -avz --exclude='.git' --exclude='build' --exclude='node_modules' --exclude='.cache' \
    ./ ${DROPLET_USER}@${DROPLET_IP}:/opt/sep-trader/

echo "⚙️ Step 9: Creating environment configuration..."
cat > /tmp/sep_production.env << EOF
# SEP Production Environment Configuration
ENVIRONMENT=production
LOG_LEVEL=info

# Database Configuration
DB_HOST=localhost
DB_PORT=5432
DB_NAME=sep_trading
DB_USER=sep_user
DB_PASSWORD=sep_password

# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6380
REDIS_PASSWORD=sep_redis_password

# OANDA Configuration (replace with your credentials)
OANDA_API_URL=https://api-fxpractice.oanda.com
OANDA_ACCOUNT_ID=your_account_id
OANDA_ACCESS_TOKEN=your_access_token

# Data Paths
DATA_ROOT=/opt/sep-data
MODELS_PATH=/opt/sep-data/models
TRAINING_DATA_PATH=/opt/sep-data/training
LOGS_PATH=/opt/sep-data/logs

# Performance Configuration
CUDA_VISIBLE_DEVICES=0
OMP_NUM_THREADS=4
TBB_NUM_THREADS=4
EOF

copy_to_droplet "/tmp/sep_production.env" "/opt/sep-trader/"

echo "🔄 Step 10: Building SEP on droplet..."
run_remote "
cd /opt/sep-trader
chmod +x build.sh
./build.sh
"

echo "🌐 Step 11: Creating systemd service..."
cat > /tmp/sep-trader.service << EOF
[Unit]
Description=SEP Professional Trader Bot
After=network.target postgresql.service redis-server.service
Requires=postgresql.service redis-server.service

[Service]
Type=simple
User=root
WorkingDirectory=/opt/sep-trader
EnvironmentFile=/opt/sep-trader/sep_production.env
ExecStart=/opt/sep-trader/build/src/apps/oanda_trader/quantum_tracker
Restart=always
RestartSec=10
StandardOutput=append:/opt/sep-data/logs/trader.log
StandardError=append:/opt/sep-data/logs/trader-error.log

# Security settings
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ReadWritePaths=/opt/sep-data

[Install]
WantedBy=multi-user.target
EOF

copy_to_droplet "/tmp/sep-trader.service" "/etc/systemd/system/"
run_remote "systemctl daemon-reload && systemctl enable sep-trader"

echo "📊 Step 12: Creating management scripts..."
cat > /tmp/sep_management.sh << EOF
#!/bin/bash

# SEP Management Commands
case "\$1" in
    start)
        systemctl start sep-trader
        echo "✅ SEP Trader started"
        ;;
    stop)
        systemctl stop sep-trader
        echo "🛑 SEP Trader stopped"
        ;;
    restart)
        systemctl restart sep-trader
        echo "🔄 SEP Trader restarted"
        ;;
    status)
        systemctl status sep-trader
        ;;
    logs)
        tail -f /opt/sep-data/logs/trader.log
        ;;
    errors)
        tail -f /opt/sep-data/logs/trader-error.log
        ;;
    test)
        cd /opt/sep-trader
        source sep_production.env
        ./build/src/trading/quantum_pair_trainer test
        ;;
    update)
        cd /opt/sep-trader
        git pull
        ./build.sh
        systemctl restart sep-trader
        echo "📦 SEP updated and restarted"
        ;;
    backup)
        pg_dump sep_trading > /opt/sep-data/backups/database_\$(date +%Y%m%d_%H%M%S).sql
        echo "💾 Database backup created"
        ;;
    *)
        echo "Usage: \$0 {start|stop|restart|status|logs|errors|test|update|backup}"
        exit 1
        ;;
esac
EOF

copy_to_droplet "/tmp/sep_management.sh" "/usr/local/bin/sep"
run_remote "chmod +x /usr/local/bin/sep"

echo "🔍 Step 13: Running system validation..."
run_remote "
cd /opt/sep-trader
source sep_production.env
echo 'Testing database connection...'
./build/src/trading/quantum_pair_trainer test || echo 'Warning: Training CLI test failed'
echo 'Checking services...'
systemctl is-active postgresql redis-server
"

echo "✅ Deployment complete!"
echo ""
echo "🎯 Next steps:"
echo "1. Configure OANDA credentials:"
echo "   ssh root@${DROPLET_IP}"
echo "   nano /opt/sep-trader/sep_production.env"
echo ""
echo "2. Start the trading system:"
echo "   ssh root@${DROPLET_IP}"
echo "   sep start"
echo ""
echo "3. Monitor the system:"
echo "   sep logs     # View trading logs"
echo "   sep status   # Check service status"
echo "   sep test     # Run system tests"
echo ""
echo "📚 Management commands available via 'sep' command on droplet"
