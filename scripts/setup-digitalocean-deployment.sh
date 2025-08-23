#!/bin/bash

# SEP Trading System - DigitalOcean Droplet Setup Script
# =====================================================
# 
# This script configures the DigitalOcean droplet for the new PostgreSQL setup
# Usage: ./scripts/setup-digitalocean-deployment.sh
#
# Requirements:
# - SSH access to the droplet (129.212.145.195)
# - Root privileges on the droplet

set -euo pipefail

# Configuration
DROPLET_IP="129.212.145.195"
SSH_USER="root"
REPO_PATH="/sep"
FRONTEND_PATH="/sep/frontend"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if SSH key exists
check_ssh_access() {
    log_info "Checking SSH access to droplet..."
    if ssh -o ConnectTimeout=10 -o BatchMode=yes ${SSH_USER}@${DROPLET_IP} 'exit' 2>/dev/null; then
        log_success "SSH access confirmed"
    else
        log_error "Cannot connect to droplet. Please check SSH key setup."
        exit 1
    fi
}

# Deploy configuration and install dependencies on droplet
deploy_to_droplet() {
    log_info "Deploying to DigitalOcean droplet..."
    
    # Create deployment script for remote execution
    cat << 'REMOTE_SCRIPT' > /tmp/droplet_setup.sh
#!/bin/bash
set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Update system packages
log_info "Updating system packages..."
apt-get update -qq
apt-get upgrade -y -qq

# Install required system dependencies
log_info "Installing system dependencies..."
apt-get install -y -qq \
    curl \
    wget \
    git \
    build-essential \
    cmake \
    pkg-config \
    libssl-dev \
    libpq-dev \
    postgresql-client \
    nginx \
    supervisor \
    htop \
    tree \
    jq \
    unzip

# Install Docker if not present
if ! command -v docker &> /dev/null; then
    log_info "Installing Docker..."
    curl -fsSL https://get.docker.com -o get-docker.sh
    sh get-docker.sh
    rm get-docker.sh
    usermod -aG docker root
    systemctl enable docker
    systemctl start docker
else
    log_success "Docker already installed"
fi

# Install Docker Compose if not present
if ! command -v docker-compose &> /dev/null; then
    log_info "Installing Docker Compose..."
    curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    chmod +x /usr/local/bin/docker-compose
    ln -sf /usr/local/bin/docker-compose /usr/bin/docker-compose
else
    log_success "Docker Compose already installed"
fi

# Install Node.js 18.x for frontend builds
if ! command -v node &> /dev/null; then
    log_info "Installing Node.js 18.x..."
    curl -fsSL https://deb.nodesource.com/setup_18.x | bash -
    apt-get install -y -qq nodejs
else
    log_success "Node.js already installed"
fi

# Install Python 3.11 and pip if not present
if ! python3.11 --version &> /dev/null; then
    log_info "Installing Python 3.11..."
    apt-get install -y -qq software-properties-common
    add-apt-repository -y ppa:deadsnakes/ppa
    apt-get update -qq
    apt-get install -y -qq python3.11 python3.11-dev python3.11-venv python3-pip
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1
else
    log_success "Python 3.11 already installed"
fi

# Create application directories
log_info "Creating application directories..."
mkdir -p /app/logs
mkdir -p /app/data
mkdir -p /app/backups
mkdir -p /app/ssl
chown -R root:root /app

# Configure Nginx for SSL termination and reverse proxy
log_info "Configuring Nginx..."
cat > /etc/nginx/sites-available/sep-trading << 'NGINX_CONFIG'
server {
    listen 80;
    listen [::]:80;
    server_name 129.212.145.195 _;
    
    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Referrer-Policy "strict-origin-when-cross-origin" always;
    
    # Frontend static files
    root /sep/frontend/build;
    index index.html;
    
    # Frontend SPA routing
    location / {
        try_files $uri $uri/ /index.html;
        expires 1d;
        add_header Cache-Control "public, immutable";
    }
    
    # API proxy to backend
    location /api/ {
        proxy_pass http://127.0.0.1:5000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_cache_bypass $http_upgrade;
        proxy_read_timeout 86400;
    }
    
    # WebSocket proxy
    location /ws/ {
        proxy_pass http://127.0.0.1:8765;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
    
    # Health check
    location /health {
        access_log off;
        return 200 "healthy\n";
        add_header Content-Type text/plain;
    }
    
    # Security: deny access to sensitive files
    location ~ /\. {
        deny all;
        access_log off;
        log_not_found off;
    }
    
    location ~ \.(env|config|conf)$ {
        deny all;
        access_log off;
        log_not_found off;
    }
}
NGINX_CONFIG

# Enable the site
ln -sf /etc/nginx/sites-available/sep-trading /etc/nginx/sites-enabled/
rm -f /etc/nginx/sites-enabled/default
nginx -t
systemctl enable nginx
systemctl restart nginx

# Configure Supervisor for process management
log_info "Configuring Supervisor..."
cat > /etc/supervisor/conf.d/sep-trading.conf << 'SUPERVISOR_CONFIG'
[group:sep-trading]
programs=sep-backend,sep-websocket

[program:sep-backend]
command=/sep/venv/bin/python /sep/scripts/trading_service.py
directory=/sep
user=root
autostart=true
autorestart=true
redirect_stderr=true
stdout_logfile=/app/logs/backend.log
stdout_logfile_maxbytes=10MB
stdout_logfile_backups=5
environment=
    PATH="/sep/venv/bin:/usr/local/bin:/usr/bin:/bin",
    PYTHONPATH="/sep",
    FLASK_ENV="production"

[program:sep-websocket]
command=/sep/venv/bin/python /sep/scripts/websocket_service.py
directory=/sep
user=root
autostart=true
autorestart=true
redirect_stderr=true
stdout_logfile=/app/logs/websocket.log
stdout_logfile_maxbytes=10MB
stdout_logfile_backups=5
environment=
    PATH="/sep/venv/bin:/usr/local/bin:/usr/bin:/bin",
    PYTHONPATH="/sep"
SUPERVISOR_CONFIG

systemctl enable supervisor
systemctl restart supervisor

# Set up firewall rules
log_info "Configuring UFW firewall..."
ufw --force enable
ufw default deny incoming
ufw default allow outgoing
ufw allow ssh
ufw allow 80/tcp
ufw allow 443/tcp

# Create log rotation configuration
log_info "Setting up log rotation..."
cat > /etc/logrotate.d/sep-trading << 'LOGROTATE_CONFIG'
/app/logs/*.log {
    daily
    missingok
    rotate 14
    compress
    delaycompress
    notifempty
    create 644 root root
    postrotate
        supervisorctl reread
        supervisorctl update
    endscript
}
LOGROTATE_CONFIG

# Create system monitoring script
log_info "Creating system monitoring script..."
cat > /usr/local/bin/sep-status << 'MONITORING_SCRIPT'
#!/bin/bash
echo "=== SEP Trading System Status ==="
echo "Timestamp: $(date)"
echo ""

echo "=== System Resources ==="
free -h
df -h /
echo ""

echo "=== Services Status ==="
systemctl is-active nginx
systemctl is-active supervisor
systemctl is-active docker
echo ""

echo "=== Application Processes ==="
supervisorctl status
echo ""

echo "=== Recent Logs (last 10 lines) ==="
echo "Backend:"
tail -n 5 /app/logs/backend.log 2>/dev/null || echo "No backend logs found"
echo ""
echo "WebSocket:"
tail -n 5 /app/logs/websocket.log 2>/dev/null || echo "No websocket logs found"
echo ""

echo "=== Network Connections ==="
netstat -tlnp | grep -E ':(80|443|5000|8765) '
echo ""

echo "=== Docker Status ==="
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
MONITORING_SCRIPT

chmod +x /usr/local/bin/sep-status

log_success "Droplet setup completed successfully!"
log_info "Run 'sep-status' to check system status"
REMOTE_SCRIPT

    # Copy setup script to droplet and execute
    log_info "Copying setup script to droplet..."
    scp /tmp/droplet_setup.sh ${SSH_USER}@${DROPLET_IP}:/tmp/
    
    log_info "Executing setup script on droplet..."
    ssh ${SSH_USER}@${DROPLET_IP} 'bash /tmp/droplet_setup.sh'
    
    # Clean up
    rm /tmp/droplet_setup.sh
    ssh ${SSH_USER}@${DROPLET_IP} 'rm /tmp/droplet_setup.sh'
    
    log_success "Droplet setup completed!"
}

# Deploy application code and configurations
deploy_application() {
    log_info "Deploying application code..."
    
    # Sync the entire repository to the droplet
    log_info "Syncing repository to droplet..."
    rsync -avz --delete \
        --exclude '.git/' \
        --exclude 'node_modules/' \
        --exclude '__pycache__/' \
        --exclude '*.pyc' \
        --exclude '.env' \
        --exclude 'venv/' \
        --exclude 'build/' \
        . ${SSH_USER}@${DROPLET_IP}:/sep/
    
    # Set up Python virtual environment and install dependencies
    log_info "Setting up Python environment..."
    ssh ${SSH_USER}@${DROPLET_IP} << 'PYTHON_SETUP'
cd /sep

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install Python dependencies
if [ -f requirements.txt ]; then
    pip install -r requirements.txt
fi

# Install additional dependencies for PostgreSQL
pip install psycopg2-binary python-dotenv flask flask-cors websockets asyncio redis

# Make scripts executable
chmod +x scripts/*.sh
chmod +x scripts/*.py

# Create necessary directories
mkdir -p logs data config bin

# Set ownership
chown -R root:root /sep
PYTHON_SETUP

    # Build and deploy frontend
    log_info "Building and deploying frontend..."
    ssh ${SSH_USER}@${DROPLET_IP} << 'FRONTEND_SETUP'
cd /sep/frontend

# Install Node dependencies
npm install

# Build production frontend
npm run build

# Ensure Nginx can serve the files
chown -R www-data:www-data build/
chmod -R 755 build/
FRONTEND_SETUP

    log_success "Application deployment completed!"
}

# Configure environment variables
configure_environment() {
    log_info "Configuring environment variables..."
    
    # Deploy the configuration template
    log_info "Creating production environment file..."
    ssh ${SSH_USER}@${DROPLET_IP} << 'ENV_SETUP'
cd /sep

# Create production environment file
cat > .sep-config.env << 'PROD_ENV'
# SEP Trading System - Production Configuration
# ===========================================

# Flask Configuration
FLASK_ENV=production
FLASK_DEBUG=false
SECRET_KEY=your-production-secret-key-change-this

# Database Configuration (DigitalOcean PostgreSQL)
DB_HOST=your-db-cluster-host.db.ondigitalocean.com
DB_PORT=25060
DB_USER=doadmin
DB_PASSWORD=your-secure-db-password
DB_NAME=defaultdb
DB_SSL_MODE=require
DB_SSL_CERT=/sep/config/ca-certificate.crt

# Redis Configuration
REDIS_URL=redis://127.0.0.1:6379/0

# API Configuration
API_HOST=0.0.0.0
API_PORT=5000
CORS_ORIGINS=http://129.212.145.195,https://129.212.145.195

# WebSocket Configuration
WS_HOST=0.0.0.0
WS_PORT=8765

# OANDA Configuration
OANDA_API_KEY=your-oanda-api-key
OANDA_ACCOUNT_ID=your-oanda-account-id
OANDA_ENVIRONMENT=practice

# Security Configuration
API_KEY_HEADER=X-SEP-API-KEY
ENABLE_SSL=false
SSL_CERT_PATH=
SSL_KEY_PATH=

# Logging Configuration
LOG_LEVEL=INFO
LOG_DIR=/app/logs
ENABLE_FILE_LOGGING=true

# Performance Configuration
MAX_CONNECTIONS=100
REQUEST_TIMEOUT=30
WORKER_PROCESSES=2

# Feature Flags
ENABLE_TRADING=false
ENABLE_BACKTESTING=true
ENABLE_ANALYTICS=true
PROD_ENV

# Set proper permissions
chmod 600 .sep-config.env

echo "Created production environment template at /sep/.sep-config.env"
echo "Please edit this file with your actual database credentials and API keys"
ENV_SETUP

    log_warning "IMPORTANT: You must manually edit /sep/.sep-config.env on the droplet with:"
    log_warning "  - Your DigitalOcean PostgreSQL connection details"
    log_warning "  - Your OANDA API credentials"
    log_warning "  - A secure SECRET_KEY for Flask"
    log_warning "  - SSL certificate paths if using SSL"
}

# Test the deployment
test_deployment() {
    log_info "Testing deployment..."
    
    # Test SSH connectivity
    log_info "Testing SSH connectivity..."
    ssh ${SSH_USER}@${DROPLET_IP} 'echo "SSH connection successful"'
    
    # Test system services
    log_info "Testing system services..."
    ssh ${SSH_USER}@${DROPLET_IP} << 'SERVICE_TEST'
echo "=== Service Status ==="
systemctl is-active nginx && echo "✓ Nginx is running" || echo "✗ Nginx is not running"
systemctl is-active supervisor && echo "✓ Supervisor is running" || echo "✗ Supervisor is not running"
systemctl is-active docker && echo "✓ Docker is running" || echo "✗ Docker is not running"

echo ""
echo "=== Application Status ==="
supervisorctl status

echo ""
echo "=== Network Listening Ports ==="
netstat -tlnp | grep -E ':(80|443|5000|8765) ' || echo "No services listening on expected ports"

echo ""
echo "=== Disk Usage ==="
df -h /

echo ""
echo "=== Memory Usage ==="
free -h
SERVICE_TEST

    # Test HTTP endpoint
    log_info "Testing HTTP endpoint..."
    if curl -f -s "http://${DROPLET_IP}/health" >/dev/null; then
        log_success "HTTP endpoint is responding"
    else
        log_warning "HTTP endpoint is not responding - this is expected if services are not started yet"
    fi
    
    log_success "Deployment test completed!"
}

# Main execution
main() {
    echo -e "${BLUE}"
    echo "========================================"
    echo "  SEP Trading System Droplet Setup"
    echo "========================================"
    echo -e "${NC}"
    
    log_info "Starting DigitalOcean droplet deployment..."
    log_info "Target: ${SSH_USER}@${DROPLET_IP}"
    
    check_ssh_access
    deploy_to_droplet
    deploy_application
    configure_environment
    test_deployment
    
    echo -e "${GREEN}"
    echo "========================================"
    echo "       DEPLOYMENT COMPLETED"
    echo "========================================"
    echo -e "${NC}"
    
    log_success "SEP Trading System has been deployed to the droplet!"
    log_info "Next steps:"
    echo "  1. SSH to the droplet: ssh ${SSH_USER}@${DROPLET_IP}"
    echo "  2. Edit the configuration: nano /sep/.sep-config.env"
    echo "  3. Update database credentials and API keys"
    echo "  4. Restart services: supervisorctl restart all"
    echo "  5. Check status: sep-status"
    echo "  6. Test the application: http://${DROPLET_IP}"
    echo ""
    log_info "For database setup, run: /sep/scripts/test-db-connection.sh"
    log_info "For system monitoring: sep-status"
}

# Execute main function
main "$@"