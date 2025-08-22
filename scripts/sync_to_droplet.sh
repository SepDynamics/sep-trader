#!/bin/bash
# SEP Professional Trader-Bot - Data Synchronization Script

set -e

# Configuration
DROPLET_IP="159.203.131.149"
DEPLOY_USER="root"
APP_DIR="/opt/sep-trader"
REMOTE_PATH="$APP_DIR"

echo "🔄 SEP Data Sync to Droplet"
echo "=========================="

# Check connection
if ! ssh -o ConnectTimeout=10 $DEPLOY_USER@$DROPLET_IP "echo 'Connected'" > /dev/null 2>&1; then
    echo "❌ Cannot connect to droplet"
    exit 1
fi

echo "✅ Connected to droplet"

# Sync trading signals and metrics
echo "📊 Syncing trading signals..."
if [ -d "output/" ]; then
    rsync -avz --progress output/ $DEPLOY_USER@$DROPLET_IP:$APP_DIR/data/
    echo "✅ Output data synced"
fi

# Sync configuration updates
echo "⚙️ Syncing configuration..."
if [ -d "config/" ]; then
    rsync -avz --progress --exclude="*.env" config/ $DEPLOY_USER@$DROPLET_IP:$APP_DIR/config/
    echo "✅ Configuration synced"
fi

# Sync trained models
echo "🧠 Syncing trained models..."
if [ -d "models/" ]; then
    rsync -avz --progress models/ $DEPLOY_USER@$DROPLET_IP:$APP_DIR/data/models/
    echo "✅ Models synced"
fi

# Export latest metrics from Redis
PAIR="${1:-${PAIR}}"
if [ -n "$PAIR" ]; then
    echo "📡 Exporting latest metrics for $PAIR..."
    LATEST_KEY=$(redis-cli --scan --pattern "pattern:${PAIR}:*" | sort | tail -n 1)
    if [ -n "$LATEST_KEY" ]; then
        mkdir -p output
        redis-cli --raw DUMP "$LATEST_KEY" > "output/latest_metrics_${PAIR}.rdb"
        rsync -avz --progress "output/latest_metrics_${PAIR}.rdb" $DEPLOY_USER@$DROPLET_IP:$REMOTE_PATH/data/
        echo "✅ Latest metrics synced"
    else
        echo "⚠️ No matching Redis keys found for pattern:${PAIR}:*"
    fi
else
    echo "⚠️ PAIR not specified; skipping metrics export"
fi

# Check if market is open for trading
echo "📈 Checking market status..."
MARKET_STATUS=$(ssh $DEPLOY_USER@$DROPLET_IP "curl -s http://localhost:8080/api/market/status 2>/dev/null || echo 'unknown'")
echo "Market status: $MARKET_STATUS"

# Trigger data reload on droplet
echo "🔄 Triggering data reload..."
ssh $DEPLOY_USER@$DROPLET_IP << 'EOF'
cd /opt/sep-trader/sep-trader
if docker-compose ps | grep -q "sep-trader.*Up"; then
    echo "📡 Sending reload signal to trading service..."
    curl -X POST http://localhost:8080/api/data/reload || echo "⚠️ Reload signal failed"
else
    echo "⚠️ Trading service not running"
fi
EOF

# Display sync summary
echo ""
echo "✅ Sync completed successfully!"
echo ""
echo "📋 Summary:"
echo "   Data synced to: $DROPLET_IP:$APP_DIR/data/"
echo "   Config synced to: $DROPLET_IP:$APP_DIR/config/"
echo "   Service status: $(ssh $DEPLOY_USER@$DROPLET_IP 'cd /opt/sep-trader/sep-trader && docker-compose ps --services' 2>/dev/null || echo 'unknown')"
echo ""
echo "🔗 Check status: curl http://$DROPLET_IP/api/status"
echo "📊 View logs: ssh $DEPLOY_USER@$DROPLET_IP 'cd /opt/sep-trader/sep-trader && docker-compose logs -f'"
