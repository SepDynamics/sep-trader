#!/bin/bash
# SEP Professional Trader-Bot - Droplet Deployment Script

set -e

# Configuration
DROPLET_IP="165.227.109.187"
DEPLOY_USER="root"
APP_DIR="/opt/sep-trader"

echo "🚀 SEP Professional Trader-Bot - Cloud Deployment"
echo "=================================================="

# Check if we can connect
echo "📡 Testing connection to droplet..."
if ! ssh -o ConnectTimeout=10 $DEPLOY_USER@$DROPLET_IP "echo 'Connection successful'"; then
    echo "❌ Cannot connect to droplet at $DROPLET_IP"
    echo "💡 Try: ssh $DEPLOY_USER@$DROPLET_IP"
    exit 1
fi

echo "✅ Connected to droplet successfully"

# Create directory structure
echo "📁 Setting up directory structure..."
ssh $DEPLOY_USER@$DROPLET_IP << 'EOF'
mkdir -p /opt/sep-trader/{data,cache,logs,config,scripts}
mkdir -p /opt/sep-trader/services/{market-data,trading,api}
chown -R root:root /opt/sep-trader
chmod -R 755 /opt/sep-trader
EOF

# Install dependencies
echo "📦 Installing system dependencies..."
ssh $DEPLOY_USER@$DROPLET_IP << 'EOF'
apt update
apt install -y \
    docker.io \
    docker-compose \
    nginx \
    postgresql-14 \
    postgresql-contrib \
    postgresql-14-timescaledb \
    curl \
    rsync \
    python3 \
    python3-pip \
    git \
    htop \
    tmux \
    ufw

systemctl enable docker nginx postgresql
systemctl start docker nginx postgresql

# Configure firewall
ufw --force enable
ufw allow ssh
ufw allow 80
ufw allow 443
ufw allow 8080
EOF

# Setup volume storage
echo "💾 Configuring volume storage..."
ssh $DEPLOY_USER@$DROPLET_IP << 'EOF'
# Check if volume is already mounted
if ! mountpoint -q /mnt/volume_fra1_01; then
    echo "Setting up 50GB volume storage..."
    mkdir -p /mnt/sep_data
    
    # Find the volume device (usually /dev/sda or /dev/disk/by-id/...)
    VOLUME_DEVICE=$(lsblk -f | grep -E '^sd[b-z]|nvme[0-9]n[2-9]' | head -1 | awk '{print "/dev/"$1}')
    
    if [ -n "$VOLUME_DEVICE" ]; then
        # Format if not already formatted
        if ! blkid $VOLUME_DEVICE; then
            mkfs.ext4 $VOLUME_DEVICE
        fi
        
        # Mount volume
        mount $VOLUME_DEVICE /mnt/sep_data
        
        # Add to fstab for persistent mounting
        echo "$VOLUME_DEVICE /mnt/sep_data ext4 defaults 0 0" >> /etc/fstab
        
        echo "✅ Volume mounted at /mnt/sep_data"
    else
        echo "⚠️ No additional volume found - using onboard storage"
        mkdir -p /mnt/sep_data
    fi
fi

# Create database directory on volume
mkdir -p /mnt/sep_data/postgresql
mkdir -p /mnt/sep_data/trading_data
mkdir -p /mnt/sep_data/backups
chown -R postgres:postgres /mnt/sep_data/postgresql
EOF

# Configure PostgreSQL
echo "🗄️ Configuring PostgreSQL with TimescaleDB..."
ssh $DEPLOY_USER@$DROPLET_IP << 'EOF'
# Stop PostgreSQL to move data directory
systemctl stop postgresql

# Move PostgreSQL data to volume storage
if [ ! -d "/mnt/sep_data/postgresql/14" ]; then
    rsync -av /var/lib/postgresql/ /mnt/sep_data/postgresql/
fi

# Update PostgreSQL configuration
echo "data_directory = '/mnt/sep_data/postgresql/14/main'" >> /etc/postgresql/14/main/postgresql.conf
echo "shared_preload_libraries = 'timescaledb'" >> /etc/postgresql/14/main/postgresql.conf

# Configure for trading workload
cat >> /etc/postgresql/14/main/postgresql.conf << 'PGCONF'
# Trading optimization settings
max_connections = 100
shared_buffers = 2GB
effective_cache_size = 6GB
work_mem = 64MB
maintenance_work_mem = 512MB
checkpoint_completion_target = 0.9
wal_buffers = 16MB
default_statistics_target = 100
random_page_cost = 1.1
effective_io_concurrency = 200
PGCONF

# Allow connections
echo "host all all 127.0.0.1/32 md5" >> /etc/postgresql/14/main/pg_hba.conf
echo "listen_addresses = 'localhost'" >> /etc/postgresql/14/main/postgresql.conf

# Start PostgreSQL
systemctl start postgresql

# Create trading database and user
sudo -u postgres psql << 'PSQL'
CREATE DATABASE sep_trading;
CREATE USER sep_trader WITH PASSWORD 'sep_secure_password_2025';
GRANT ALL PRIVILEGES ON DATABASE sep_trading TO sep_trader;
\c sep_trading
CREATE EXTENSION IF NOT EXISTS timescaledb;
PSQL

echo "✅ PostgreSQL with TimescaleDB configured"
EOF

# Transfer configuration files
echo "⚙️ Transferring configuration..."
if [ -f "OANDA.env" ]; then
    scp OANDA.env $DEPLOY_USER@$DROPLET_IP:$APP_DIR/config/
    echo "✅ OANDA configuration uploaded"
else
    echo "⚠️ OANDA.env not found - you'll need to configure this manually"
fi

# Transfer deployment scripts
echo "📜 Transferring deployment scripts..."
scp scripts/droplet_*.sh $DEPLOY_USER@$DROPLET_IP:$APP_DIR/scripts/ 2>/dev/null || echo "⚠️ No droplet scripts found"

# Clone repository on droplet
echo "📥 Cloning repository on droplet..."
ssh $DEPLOY_USER@$DROPLET_IP << EOF
cd $APP_DIR
if [ -d "sep-trader" ]; then
    cd sep-trader
    git pull
else
    git clone https://github.com/SepDynamics/sep-trader.git
    cd sep-trader
fi
EOF

echo "🐳 Setting up Docker environment..."
ssh $DEPLOY_USER@$DROPLET_IP << 'EOF'
cd /opt/sep-trader/sep-trader

# Create lightweight Dockerfile for droplet
cat > Dockerfile.droplet << 'DOCKER_EOF'
FROM ubuntu:22.04

# Install system dependencies
RUN apt update && apt install -y \
    build-essential \
    cmake \
    python3 \
    python3-pip \
    curl \
    libssl-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy source code
COPY . .

# Install Python dependencies
RUN pip3 install requests pandas numpy python-dotenv

# Create entrypoint
RUN echo '#!/bin/bash\necho "SEP Trader Droplet Service Started"\nexec "$@"' > /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8080
ENTRYPOINT ["/entrypoint.sh"]
CMD ["python3", "scripts/trading_service.py"]
DOCKER_EOF

# Create docker-compose for services
cat > docker-compose.yml << 'COMPOSE_EOF'
version: '3.8'

services:
  sep-trader:
    build:
      context: .
      dockerfile: Dockerfile.droplet
    container_name: sep-trader
    restart: unless-stopped
    ports:
      - "8080:8080"
    volumes:
      - ../data:/app/data
      - ../cache:/app/cache
      - ../logs:/app/logs
      - ../config:/app/config
    environment:
      - ENVIRONMENT=production
      - LOG_LEVEL=info
    env_file:
      - ../config/OANDA.env

  nginx:
    image: nginx:alpine
    container_name: sep-nginx
    restart: unless-stopped
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
    depends_on:
      - sep-trader
COMPOSE_EOF

EOF

# Create basic nginx configuration
echo "🌐 Setting up nginx reverse proxy..."
ssh $DEPLOY_USER@$DROPLET_IP << 'EOF'
cat > /opt/sep-trader/sep-trader/nginx.conf << 'NGINX_EOF'
events {
    worker_connections 1024;
}

http {
    upstream sep_trader {
        server sep-trader:8080;
    }

    server {
        listen 80;
        server_name _;

        location /api/ {
            proxy_pass http://sep_trader;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        }

        location /health {
            proxy_pass http://sep_trader/health;
        }

        location / {
            return 200 'SEP Professional Trader-Bot - Cloud Service\n';
            add_header Content-Type text/plain;
        }
    }
}
NGINX_EOF
EOF

# Test deployment
echo "🧪 Testing deployment..."
ssh $DEPLOY_USER@$DROPLET_IP << 'EOF'
cd /opt/sep-trader/sep-trader
docker-compose config
EOF

echo ""
echo "✅ Deployment complete!"
echo ""
echo "📋 Next steps:"
echo "1. SSH to droplet: ssh $DEPLOY_USER@$DROPLET_IP"
echo "2. Configure OANDA credentials in /opt/sep-trader/config/OANDA.env"
echo "3. Start services: cd /opt/sep-trader/sep-trader && docker-compose up -d"
echo "4. Check status: curl http://$DROPLET_IP/health"
echo ""
echo "🔗 Access URLs:"
echo "   Public API: http://$DROPLET_IP/api/"
echo "   Health check: http://$DROPLET_IP/health"
echo "   SSH access: ssh $DEPLOY_USER@$DROPLET_IP"
echo ""
echo "💡 Use ./scripts/sync_to_droplet.sh to push trading data"
