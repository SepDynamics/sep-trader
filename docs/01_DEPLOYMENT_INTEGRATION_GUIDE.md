# SEP Professional Trading System: Deployment & Integration Guide

**Document Version:** 2.0  
**Last Updated:** August 21, 2025  
**Target Audience:** DevOps Engineers, System Integrators, Developers  

## Overview

This guide provides detailed instructions for deploying and integrating the SEP Professional Trading System across both local development and remote production environments. The system supports multiple deployment paradigms to accommodate different operational requirements.

## Deployment Architecture Summary

### Local Development Environment
- **Purpose**: Development, testing, model training, backtesting
- **Infrastructure**: Docker Compose with local storage
- **Network**: Local container networking (172.25.0.0/16)
- **GPU Access**: Full CUDA acceleration available
- **Storage**: Local bind mounts for hot-reload development

### Remote Production Environment  
- **Purpose**: Live trading, 24/7 operations, production monitoring
- **Infrastructure**: DigitalOcean Droplet with persistent storage
- **Network**: Public internet with container isolation
- **Storage**: Persistent volume storage (`/mnt/volume_nyc3_01`)
- **Scaling**: Horizontal scaling support via container orchestration

## Prerequisites

### System Requirements

#### Local Development
```bash
# Minimum requirements
- Docker Engine 24.0+
- Docker Compose 2.0+  
- 8GB RAM minimum, 16GB recommended
- 20GB available disk space
- (Optional) NVIDIA GPU with CUDA 11.8+ for training

# Recommended development setup
- Ubuntu 22.04 LTS / macOS 13+ / Windows 11 WSL2
- 32GB RAM for large dataset processing
- NVMe SSD for optimal performance
```

#### Remote Production
```bash
# DigitalOcean Droplet specifications
- 8GB RAM, 2 vCPUs minimum
- Ubuntu 24.04 LTS
- 25GB boot disk + 50GB attached volume
- Public IPv4 address
```

### Software Dependencies
```bash
# Core dependencies (handled by Docker)
- Valkey (Redis-compatible)
- Python 3.11+
- Node.js 18+ (for frontend builds)
- Nginx (for frontend serving)

# Development dependencies
- Git 2.30+
- SSH client for remote deployment
- curl for health checks
```

## Local Development Deployment

### Quick Start Deployment

```bash
# 1. Repository Setup
git clone [repository-url]
cd sep-trader

# 2. Core System Build
./install.sh --minimal --no-docker
./build.sh --no-docker

# 3. Environment Configuration
cp config/.sep-config.env.template .sep-config.env
# Edit configuration as needed

# 4. Service Deployment
./deploy.sh start

# 5. Validation
./deploy.sh health
```

### Manual Step-by-Step Deployment

#### Step 1: Core Engine Build
```bash
# Build SEP quantum engine and CLI tools
export LD_LIBRARY_PATH=./build/src/core:./build/src/config:./build/src/c_api
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -GNinja ..
ninja -j$(nproc)
cd ..
```

#### Step 2: Service Configuration
```bash
# Create required directories
mkdir -p {data,logs,config}

# Configure environment
cat > .sep-config.env << EOF
FLASK_ENV=development
VALKEY_URL=redis://valkey:6380/0
SEP_CONFIG_PATH=/app/config
PYTHONPATH=/app
PORT=5000
REACT_APP_API_URL=http://localhost:5000
REACT_APP_WS_URL=<websocket-url>
REACT_APP_ENVIRONMENT=development
EOF
```

#### Step 3: Container Deployment
```bash
# Build and start all services
docker-compose -f docker-compose.yml build --no-cache
docker-compose -f docker-compose.yml up -d

# Monitor service startup
docker-compose -f docker-compose.yml logs -f
```

#### Step 4: Service Validation
```bash
# Health check endpoints
curl -f http://localhost:5000/api/health  # Backend API
curl -f http://localhost/health           # Frontend
nc -z localhost 8765                      # WebSocket

# Full system health check
./deploy.sh health
```

### Development Workflow

#### Hot Reload Development
```bash
# Frontend development (auto-reload)
cd frontend
npm run dev  # Runs on port 3000 with hot reload

# Backend development (manual restart)
# Edit files in scripts/
docker-compose restart trading-backend

# Core engine development (rebuild required)
./build.sh --no-docker
docker-compose restart trading-backend
```

#### Debugging and Logs
```bash
# Service-specific logs
docker-compose logs -f trading-backend
docker-compose logs -f websocket-service
docker-compose logs -f frontend

# Log file access
tail -f logs/trading-service.log
tail -f logs/websocket-service.log
```

## Remote Production Deployment

### Automated Droplet Deployment

The complete droplet deployment process is automated through `deploy_to_droplet_complete.sh`:

```bash
# Configure droplet details
export DROPLET_IP="<your-droplet-ip>"
export DROPLET_USER="root"

# Execute complete deployment
./scripts/deploy_to_droplet_complete.sh
```

This script performs:
1. **Connection Testing**: Validates SSH access to droplet
2. **Volume Setup**: Configures persistent storage volumes  
3. **Database Setup**: Installs and configures PostgreSQL + TimescaleDB
4. **Docker Installation**: Installs Docker Engine and Compose
5. **NVIDIA Toolkit**: Installs container GPU support (if available)
6. **Service Deployment**: Deploys production containers
7. **Health Validation**: Verifies all services are operational

### Manual Production Deployment

#### Step 1: Droplet Preparation
```bash
# SSH to droplet
ssh root@$DROPLET_IP

# System updates
apt-get update && apt-get upgrade -y

# Essential packages
apt-get install -y curl wget git build-essential cmake \
  ninja-build pkg-config libssl-dev libcurl4-openssl-dev \
  libpq-dev libhwloc-dev crow-dev
```

#### Step 2: Volume Configuration
```bash
# Mount persistent volume
mkdir -p /mnt/volume_nyc3_01
mount /dev/disk/by-id/scsi-0DO_Volume_sep-data /mnt/volume_nyc3_01

# Create data directories
mkdir -p /mnt/volume_nyc3_01/{sep-data,sep-data/logs,config}

# Configure automatic mounting
echo '/dev/disk/by-id/scsi-0DO_Volume_sep-data /mnt/volume_nyc3_01 ext4 defaults,nofail,discard 0 2' >> /etc/fstab
```

#### Step 3: Database Setup
```bash
# Install PostgreSQL 14 with TimescaleDB
apt-get install -y postgresql-14 postgresql-client-14
apt-get install -y postgresql-14-timescaledb

# Configure database
systemctl enable postgresql
systemctl start postgresql

# Create trading database
sudo -u postgres createdb sep_trading
sudo -u postgres psql -d sep_trading -c "CREATE EXTENSION IF NOT EXISTS timescaledb;"
```

#### Step 4: Docker Installation
```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh

# Enable Docker service
systemctl enable docker
systemctl start docker
```

#### Step 5: Application Deployment
```bash
# Clone repository
git clone [repository-url] /opt/sep-trader
cd /opt/sep-trader

# Deploy production services
docker-compose -f docker-compose.production.yml build --no-cache
docker-compose -f docker-compose.production.yml up -d
```

#### Step 6: Production Validation
```bash
# Service health checks
curl -f http://$DROPLET_IP:5000/api/health
curl -f http://$DROPLET_IP/health
nc -z $DROPLET_IP 8765

# Container status
docker-compose -f docker-compose.production.yml ps

# Log monitoring
docker-compose -f docker-compose.production.yml logs -f
```

## Service Configuration

### Docker Compose Configurations

#### Local Development (`docker-compose.yml`)
```yaml
Key Features:
- Local bind mounts for development
- Hot-reload support
- Development environment variables
- Optional Valkey without persistence requirement
- Direct port mapping for debugging
```

#### Production (`docker-compose.production.yml`)  
```yaml
Key Features:
- Persistent volume storage
- Production environment variables
- External IP address configuration
- Enhanced health checks
- Production-ready restart policies
```

### Environment Configuration

#### Backend Service Environment
```bash
# Core configuration
FLASK_ENV=production
VALKEY_URL=redis://valkey:6380/0
SEP_CONFIG_PATH=/app/config
PYTHONPATH=/app
PORT=5000

# Database connection
DATABASE_URL=postgresql://user:pass@localhost:5432/sep_trading

# API configuration
API_KEY_HEADER=X-SEP-API-KEY
CORS_ORIGINS=http://localhost,http://$DROPLET_IP
```

#### Frontend Environment
```bash
# Local development
REACT_APP_API_URL=http://localhost:5000
REACT_APP_WS_URL=<websocket-url>
REACT_APP_ENVIRONMENT=development

# Production
REACT_APP_API_URL=http://$DROPLET_IP:5000
REACT_APP_WS_URL=ws://$DROPLET_IP:8765
REACT_APP_ENVIRONMENT=production
```

#### WebSocket Service Environment
```bash
VALKEY_URL=redis://valkey:6380/0
WS_HOST=0.0.0.0
WS_PORT=8765
LOG_LEVEL=INFO
```

## Integration Architecture

### API Integration

#### RESTful API Endpoints
```bash
# System status and health
GET  /api/health          # Health check
GET  /api/status          # System status
GET  /api/system/info     # System information

# Performance analytics
GET  /api/performance/metrics     # Performance metrics
GET  /api/performance/current    # Current performance
GET  /api/performance/history    # Historical performance

# Configuration management
GET  /api/config/get      # Retrieve configuration
POST /api/config/set      # Update configuration
GET  /api/config/schema   # Configuration schema
```

#### WebSocket Integration
```bash
# Real-time data streams
ws://[host]:8765/market-data    # Market data stream
ws://[host]:8765/trade-signals  # Trading signals
ws://[host]:8765/system-status  # System status updates
ws://[host]:8765/performance    # Performance metrics
```

### CLI-Web Integration Bridge

#### Command Translation Layer
```python
# CLI Bridge Architecture
Web Request → API Endpoint → CLI Bridge → SEP Engine → Response

# Key integration points:
- Command sanitization and validation
- Asynchronous command execution  
- Real-time progress updates via WebSocket
- Error handling and logging
- Authentication and authorization
```

#### Integration Examples
```bash
# Web interface actions mapped to CLI commands
Start Trading   → ./bin/trader-cli --start --config trading.json
Stop Trading    → ./bin/trader-cli --stop
Get Status      → ./bin/trader-cli --status --json
Update Config   → ./bin/trader-cli --config-update config.json
Run Backtest    → ./bin/trader-cli --backtest --from 2025-01-01
```

## Networking and Security

### Container Networking
```yaml
# Network configuration
networks:
  sep-network:
    driver: bridge
    ipam:
      config:
        - subnet: 172.25.0.0/16

# Service discovery (internal)
trading-backend:5000 # API service
websocket-service:8765 # WebSocket service
```

### Security Configuration
```bash
# API Authentication
- X-SEP-API-KEY header required for sensitive endpoints
- CORS policy restricting origins
- Rate limiting on API endpoints

# Container Security
- Non-root user execution where possible
- Read-only filesystems for static content
- Network isolation between services
- Secret management via environment variables
```

### Firewall Configuration (Production)
```bash
# DigitalOcean Firewall Rules
- Port 22: SSH (restricted to admin IPs)
- Port 80: HTTP (public access)
- Port 443: HTTPS (public access) 
- Port 5000: API (restricted or VPN only)
- Port 8765: WebSocket (restricted or VPN only)
```

## Monitoring and Observability

### Health Check Endpoints
```bash
# Service health monitoring
curl -f http://[host]:5000/api/health    # Backend health
curl -f http://[host]/health             # Frontend health  
nc -z [host] 8765                        # WebSocket connectivity
valkey-cli -h [host] -p 6380 ping       # Valkey health
```

### Log Management
```bash
# Production log locations
/mnt/volume_nyc3_01/sep-data/logs/trading-service.log
/mnt/volume_nyc3_01/sep-data/logs/websocket-service.log
/mnt/volume_nyc3_01/sep-data/logs/nginx-access.log
/mnt/volume_nyc3_01/sep-data/logs/nginx-error.log

# Container logs
docker logs sep-trading-backend
docker logs sep-websocket
docker logs sep-frontend
```

### Performance Monitoring
```bash
# System metrics
htop                              # CPU/Memory usage
iotop                             # Disk I/O monitoring  
netstat -tulpn                    # Network connections
docker stats                      # Container resource usage

# Application metrics (via API)
curl http://[host]:5000/api/performance/metrics
```

## Troubleshooting Guide

### Common Issues and Solutions

#### Service Startup Failures
```bash
# Check container logs
docker-compose logs -f [service-name]

# Verify network connectivity
docker network ls
docker network inspect sep_sep-network

# Check port conflicts
netstat -tulpn | grep -E "(5000|8765|80|443)"
```

#### Database Connection Issues
```bash
# Verify PostgreSQL service
systemctl status postgresql
sudo -u postgres psql -c "SELECT version();"

# Test database connectivity
psql -h localhost -U postgres -d sep_trading -c "SELECT NOW();"
```

#### WebSocket Connection Problems
```bash
# Test WebSocket connectivity
curl -i -N -H "Connection: Upgrade" -H "Upgrade: websocket" \
  -H "Sec-WebSocket-Version: 13" -H "Sec-WebSocket-Key: test" \
  http://[host]:8765/

# Check WebSocket service logs
docker logs sep-websocket
```

#### Frontend Build Issues
```bash
# Clear Node.js cache
cd frontend
rm -rf node_modules package-lock.json
npm cache clean --force
npm install

# Rebuild frontend container
docker-compose build --no-cache frontend
```

### Recovery Procedures

#### System Recovery
```bash
# Complete system reset (development)
docker-compose down -v --remove-orphans
docker system prune -af
./deploy.sh start

# Production recovery
docker-compose -f docker-compose.production.yml down
docker-compose -f docker-compose.production.yml up -d
```

#### Data Recovery
```bash
# Backup important data
tar -czf backup-$(date +%Y%m%d).tar.gz data/ logs/ config/

# Restore from backup
tar -xzf backup-YYYYMMDD.tar.gz
docker-compose restart
```

## Future Integration Points

### Extensibility Framework
```bash
# Plugin architecture for custom strategies
src/core/strategies/custom/     # Custom C++ strategies
scripts/strategies/            # Python strategy bridges
frontend/src/plugins/          # UI plugins

# API extension points
scripts/api/extensions/        # Custom API endpoints
frontend/src/api/custom/       # Custom API clients
```

### Scaling Considerations
```bash
# Horizontal scaling preparation
- Container orchestration (Kubernetes/Swarm)
- Load balancer configuration
- Database sharding strategies
- Valkey cluster setup
- CDN integration for frontend assets
```

---

**Document Control**  
*This deployment guide should be consulted for all system deployments and integrations. Regular updates should reflect changes in deployment procedures and infrastructure requirements.*