#!/bin/bash
# SEP Professional Trader-Bot - Droplet Diagnostics Script

set -e

# Configuration
DROPLET_IP="165.227.109.187"
DEPLOY_USER="root"
APP_DIR="/opt/sep-trader"

echo "🔍 SEP Droplet Diagnostics"
echo "========================="
echo "Target: $DROPLET_IP"
echo ""

# Test basic connectivity
echo "📡 Testing SSH connectivity..."
if ! ssh -o ConnectTimeout=10 $DEPLOY_USER@$DROPLET_IP "echo 'SSH connection successful'" 2>/dev/null; then
    echo "❌ Cannot connect via SSH"
    exit 1
fi
echo "✅ SSH connectivity confirmed"

# Check system status
echo ""
echo "🖥️ System Status:"
ssh $DEPLOY_USER@$DROPLET_IP << 'EOF'
echo "   OS: $(lsb_release -d 2>/dev/null | cut -f2 || echo 'Unknown')"
echo "   Uptime: $(uptime -p 2>/dev/null || uptime)"
echo "   Memory: $(free -h | awk '/^Mem:/ {print $3 "/" $2}')"
echo "   Disk: $(df -h / | awk 'NR==2 {print $3 "/" $2 " (" $5 " used)"}')"
EOF

# Check Docker status
echo ""
echo "🐳 Docker Status:"
ssh $DEPLOY_USER@$DROPLET_IP << 'EOF'
if command -v docker >/dev/null 2>&1; then
    echo "   Docker version: $(docker --version)"
    echo "   Docker status: $(systemctl is-active docker)"
    
    if command -v docker-compose >/dev/null 2>&1; then
        echo "   Docker Compose: $(docker-compose --version)"
    else
        echo "   Docker Compose: Not installed"
    fi
else
    echo "   Docker: Not installed"
fi
EOF

# Check directory structure
echo ""
echo "📁 Directory Structure:"
ssh $DEPLOY_USER@$DROPLET_IP << EOF
if [ -d "$APP_DIR" ]; then
    echo "   App directory: ✅ $APP_DIR exists"
    ls -la $APP_DIR/ | sed 's/^/   /'
    
    if [ -d "$APP_DIR/sep-trader" ]; then
        echo "   Repository: ✅ sep-trader directory exists"
        if [ -f "$APP_DIR/sep-trader/docker-compose.yml" ]; then
            echo "   Docker Compose: ✅ Configuration exists"
        else
            echo "   Docker Compose: ❌ Configuration missing"
        fi
    else
        echo "   Repository: ❌ sep-trader directory missing"
    fi
else
    echo "   App directory: ❌ $APP_DIR does not exist"
fi
EOF

# Check running services
echo ""
echo "🚀 Service Status:"
ssh $DEPLOY_USER@$DROPLET_IP << EOF
cd $APP_DIR/sep-trader 2>/dev/null || { echo "   ❌ Cannot access sep-trader directory"; exit 0; }

if [ -f "docker-compose.yml" ]; then
    echo "   Docker Compose Services:"
    docker-compose ps 2>/dev/null | sed 's/^/   /' || echo "   ❌ Failed to get service status"
else
    echo "   ❌ No docker-compose.yml found"
fi

echo ""
echo "   System Services:"
echo "   - Docker: \$(systemctl is-active docker 2>/dev/null || echo 'inactive')"
echo "   - Nginx: \$(systemctl is-active nginx 2>/dev/null || echo 'inactive')"
echo "   - PostgreSQL: \$(systemctl is-active postgresql 2>/dev/null || echo 'inactive')"
EOF

# Check network connectivity
echo ""
echo "🌐 Network Status:"
ssh $DEPLOY_USER@$DROPLET_IP << 'EOF'
echo "   Port 8080 (Trading Service):"
if netstat -tuln 2>/dev/null | grep -q ":8080 "; then
    echo "     ✅ Port 8080 is listening"
else
    echo "     ❌ Port 8080 not listening"
fi

echo "   Port 80 (HTTP):"
if netstat -tuln 2>/dev/null | grep -q ":80 "; then
    echo "     ✅ Port 80 is listening"
else
    echo "     ❌ Port 80 not listening"
fi

# Test local endpoints
echo "   Local endpoint tests:"
if curl -s --connect-timeout 5 http://localhost:8080/health >/dev/null 2>&1; then
    echo "     ✅ localhost:8080/health responds"
else
    echo "     ❌ localhost:8080/health not responding"
fi

if curl -s --connect-timeout 5 http://localhost/health >/dev/null 2>&1; then
    echo "     ✅ localhost:80/health responds (nginx)"
else
    echo "     ❌ localhost:80/health not responding (nginx)"
fi
EOF

# Check logs
echo ""
echo "📝 Recent Logs:"
ssh $DEPLOY_USER@$DROPLET_IP << EOF
cd $APP_DIR/sep-trader 2>/dev/null || exit 0

echo "   Docker Compose Logs (last 10 lines):"
docker-compose logs --tail=10 2>/dev/null | sed 's/^/   /' || echo "   ❌ No docker logs available"

echo ""
echo "   System Logs (docker service):"
journalctl -u docker --no-pager -n 5 2>/dev/null | sed 's/^/   /' || echo "   ❌ No docker service logs"
EOF

echo ""
echo "🔧 Quick Fix Suggestions:"
echo ""

# Provide fix suggestions based on findings
ssh $DEPLOY_USER@$DROPLET_IP << EOF
cd $APP_DIR/sep-trader 2>/dev/null || {
    echo "1. 📥 Re-run deployment: ./scripts/deploy_to_droplet.sh"
    exit 0
}

if ! docker-compose ps 2>/dev/null | grep -q "Up"; then
    echo "1. 🚀 Start services: cd $APP_DIR/sep-trader && docker-compose up -d"
fi

if [ ! -f "../config/OANDA.env" ]; then
    echo "2. ⚙️ Configure OANDA credentials: nano $APP_DIR/config/OANDA.env"
fi

if ! systemctl is-active --quiet docker; then
    echo "3. 🐳 Start Docker: systemctl start docker"
fi

if ! systemctl is-active --quiet nginx; then
    echo "4. 🌐 Start Nginx: systemctl start nginx"
fi
EOF

echo ""
echo "✅ Diagnostics completed"
echo "💡 Run suggested fixes and test with: curl http://$DROPLET_IP/health"