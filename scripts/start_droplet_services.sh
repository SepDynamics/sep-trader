#!/bin/bash
# SEP Professional Trader-Bot - Start Droplet Services

set -e

# Configuration
DROPLET_IP="165.227.109.187"
DEPLOY_USER="root"
APP_DIR="/opt/sep-trader"

echo "🚀 Starting SEP Trading Services on Droplet"
echo "========================================="

echo "🔄 Syncing project files to droplet..."
rsync -avz --exclude='.git' --exclude='node_modules' --exclude='build' --exclude='build_minimal' --exclude='docs_archive' --exclude='pitch' --exclude='qdrant_storage' ./ "$DEPLOY_USER@$DROPLET_IP:$APP_DIR/sep-trader/"

# Start Docker services
echo "🐳 Starting Docker services..."
ssh $DEPLOY_USER@$DROPLET_IP << EOF
if sudo lsof -t -i:80; then
    sudo lsof -t -i:80 | sudo xargs kill -9
fi
cd $APP_DIR/sep-trader
echo "Stopping existing services..."
docker-compose down
echo "Building and starting services..."
docker-compose up -d --build --remove-orphans

echo "Waiting for services to start..."
sleep 10

echo "Service status:"
docker-compose ps

echo ""
echo "Container logs (last 20 lines):"
docker-compose logs --tail=20
EOF

# Test connectivity
echo ""
echo "🔍 Testing service connectivity..."

echo "Testing health endpoint..."
if curl -s --connect-timeout 10 http://$DROPLET_IP/health > /dev/null; then
    echo "✅ Health endpoint responding"
    curl -s http://$DROPLET_IP/health | jq . || curl -s http://$DROPLET_IP/health
else
    echo "❌ Health endpoint not responding"
fi

echo ""
echo "Testing API status endpoint..."
if curl -s --connect-timeout 10 http://$DROPLET_IP/api/status > /dev/null; then
    echo "✅ API status endpoint responding"
    curl -s http://$DROPLET_IP/api/status | jq . || curl -s http://$DROPLET_IP/api/status
else
    echo "❌ API status endpoint not responding"
fi

echo ""
echo "✅ Service startup completed!"
echo ""
echo "🔗 Service URLs:"
echo "   Health: http://$DROPLET_IP/health"
echo "   API Status: http://$DROPLET_IP/api/status"
echo "   SSH: ssh $DEPLOY_USER@$DROPLET_IP"
echo ""
echo "📊 To monitor logs: ssh $DEPLOY_USER@$DROPLET_IP 'cd $APP_DIR/sep-trader && docker-compose logs -f'"