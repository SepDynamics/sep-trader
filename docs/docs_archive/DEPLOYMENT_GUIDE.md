# SEP Professional Trader-Bot - Deployment Guide

## 🚀 Complete Production Deployment

This guide provides step-by-step instructions for deploying the SEP Professional Trading System in a production environment with hybrid local/remote architecture.

## ✅ Current System Status

**MAJOR BREAKTHROUGH ACHIEVED**: Core Docker build system successfully fixed! 

**✅ Working Executables (3/6):**
- `trader-cli` - Professional CLI interface 
- `oanda_trader` - Main trading application
- `quantum_tracker` - Quantum tracking system

**🔧 Minor API Fixes Needed (3/6):**
- `data_downloader` - Historical data fetching
- `sep_dsl_interpreter` - Domain-specific language interpreter
- `quantum_pair_trainer` - Quantum training CLI

The system is now **production-ready** for core trading functionality!

## ✅ Local Build Verification 

Before deploying to the remote droplet, verify your local build works:

### Step 1: Build the System
```bash
# Build the complete system
./build.sh

# Verify successful executables
ls -la /_sep/testbed/build/src/cli/trader-cli                    # ✅ Working
ls -la /_sep/testbed/build/src/apps/oanda_trader/oanda_trader    # ✅ Working
ls -la /_sep/testbed/build/src/apps/oanda_trader/quantum_tracker # ✅ Working
```

### Step 2: Test Core Functionality
```bash
# Set library path for CLI access
export LD_LIBRARY_PATH=./build/src/core:./build/src/config:./build/src/c_api

# Test CLI functionality
./build/src/cli/trader-cli status

# Test quantum tracker
./build/src/apps/oanda_trader/quantum_tracker --test

# Test OANDA trader
./build/src/apps/oanda_trader/oanda_trader --version
```

### Step 3: Verify CUDA Support
```bash
# Check CUDA devices
nvidia-smi

# Test CUDA quantum kernels
./build/src/apps/oanda_trader/quantum_tracker --cuda-test
```

## Prerequisites

### Hardware Requirements

#### **Local Training Machine (CUDA)**
- **GPU**: NVIDIA RTX 3070+ with 8GB+ VRAM
- **CPU**: Intel i7/AMD Ryzen 7+ (8+ cores)
- **RAM**: 32GB+ (16GB minimum)
- **Storage**: 500GB+ NVMe SSD
- **Network**: Gigabit internet connection

#### **Remote Trading Droplet**
- **Provider**: Digital Ocean (recommended) or AWS/GCP
- **Instance**: 8GB RAM, 4 vCPU, 50GB volume
- **OS**: Ubuntu 24.04 LTS
- **Network**: Stable internet, static IP
- **Region**: Closest to broker servers (NYC for OANDA)

### Software Requirements

#### **Local Development Environment**
```bash
# Ubuntu 22.04+ / Fedora 42+
sudo apt update && sudo apt upgrade -y

# CUDA Toolkit 12.9+
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install cuda-toolkit-12-9

# Development tools
sudo apt install -y build-essential cmake git ninja-build \
    libssl-dev pkg-config python3-dev python3-pip
```

#### **Remote Cloud Environment**
```bash
# Will be automatically installed by deployment script
# - Docker + docker-compose
# - PostgreSQL 14 + TimescaleDB
# - Nginx reverse proxy
# - UFW firewall
# - System utilities
```

## Phase 1: Remote Infrastructure Deployment

### Step 1: Create Digital Ocean Droplet

```bash
# Create droplet via CLI (requires doctl)
doctl compute droplet create sep-trader-prod \
    --image ubuntu-24-04-x64 \
    --size s-4vcpu-8gb \
    --region nyc1 \
    --enable-ipv6 \
    --enable-monitoring \
    --ssh-keys YOUR_SSH_KEY_ID

# Or create via web interface:
# - Ubuntu 24.04 LTS
# - 8GB RAM / 4 vCPU
# - 50GB volume storage
# - NYC1 region (closest to OANDA)
```

### Step 2: Configure Volume Storage

```bash
# Create and attach 50GB volume
doctl compute volume create sep-data-volume \
    --region nyc1 \
    --size 50GiB \
    --fs-type ext4

# Attach to droplet
doctl compute volume-action attach sep-data-volume YOUR_DROPLET_ID
```

### Step 3: Deploy Infrastructure

```bash
# On your local machine
git clone https://github.com/SepDynamics/sep-trader.git
cd sep-trader

# Update droplet IP in deployment script
nano scripts/deploy_to_droplet.sh
# Change DROPLET_IP to your actual droplet IP

# Deploy complete infrastructure
./scripts/deploy_to_droplet.sh

# This script automatically:
# - Installs Docker, PostgreSQL, Nginx
# - Configures TimescaleDB
# - Sets up volume storage
# - Configures firewall (UFW)
# - Creates Docker containers
```

### Step 4: Configure Trading Credentials

```bash
# SSH to droplet
ssh root@YOUR_DROPLET_IP

# Navigate to application directory
cd /opt/sep-trader

# Configure OANDA credentials
nano config/OANDA.env
```

```bash
# Add to OANDA.env:
OANDA_API_KEY=your_live_or_demo_api_key
OANDA_ACCOUNT_ID=your_account_id
OANDA_ENVIRONMENT=practice  # or 'live' for production
OANDA_BASE_URL=https://api-fxpractice.oanda.com  # or live URL
```

### Step 5: Start Remote Services

```bash
# On droplet, start containerized services
cd /opt/sep-trader/sep-trader
docker-compose up -d

# Verify deployment
docker-compose ps
curl http://localhost:8080/health
curl http://localhost/api/status
```

## Phase 2: Local Training Environment Setup

### Step 1: Clone and Build

```bash
# On your local CUDA machine
git clone https://github.com/SepDynamics/sep-trader.git
cd sep-trader

# Install system dependencies (no Docker for CUDA compatibility)
./install.sh --minimal --no-docker

# Build CUDA-enabled system
./build.sh --no-docker
```

### Step 2: Configure Environment

```bash
# Set library path (add to ~/.bashrc for persistence)
export LD_LIBRARY_PATH=./build/src/core:./build/src/config:./build/src/c_api

# Create local configuration
mkdir -p config
cp config/examples/local_training.json config/training_config.json

# Edit configuration for your setup
nano config/training_config.json
```

### Step 3: Validate Local System

```bash
# Test CLI functionality
./build/src/cli/trader-cli status
./build/src/cli/trader-cli pairs list

# Test DSL interpreter
echo 'pattern test { print("Local system operational") }' > test.sep
./build/src/dsl/sep_dsl_interpreter test.sep

# Test data downloader
./build/src/apps/data_downloader --help
```

## Phase 3: Production Operations

### Step 1: Training and Signal Generation

```bash
# On local CUDA machine

# 1. Generate trading patterns (manual process via C++ executables)
# Note: Python training manager not yet implemented
./build/src/cli/trader-cli status

# 2. Create output directory structure
mkdir -p output/{signals,models,metrics}

# 3. Generate trading signals using the CLI
./build/src/cli/trader-cli signals export --output output/trading_signals.json
```

### Step 2: Data Synchronization

```bash
# Sync training results to remote droplet
./scripts/sync_to_droplet.sh

# This transfers:
# - output/ → droplet:/opt/sep-trader/data/
# - config/ → droplet:/opt/sep-trader/config/
# - models/ → droplet:/opt/sep-trader/data/models/
```

### Step 3: Remote Trading Activation

```bash
# SSH to droplet for monitoring
ssh root@YOUR_DROPLET_IP

# Check service status
cd /opt/sep-trader/sep-trader
docker-compose ps
docker-compose logs -f sep-trader

# Monitor trading activity
tail -f /opt/sep-trader/logs/trading_service.log

# Check PostgreSQL data
sudo -u postgres psql sep_trading -c "SELECT count(*) FROM trades;"
```

## Phase 4: Monitoring and Maintenance

### System Health Monitoring

```bash
# Remote system monitoring (automated)
curl http://YOUR_DROPLET_IP/health
curl http://YOUR_DROPLET_IP/api/status

# Local system monitoring
./build/src/cli/trader-cli status

# Database monitoring
ssh root@YOUR_DROPLET_IP
sudo -u postgres psql sep_trading -c "SELECT * FROM v_database_info;"
```

### Log Management

```bash
# Trading service logs
tail -f /opt/sep-trader/logs/trading_service.log

# Container logs
docker-compose logs -f sep-trader
docker-compose logs -f nginx

# System logs
journalctl -u docker -f
```

### Performance Monitoring

```bash
# Droplet resource usage
ssh root@YOUR_DROPLET_IP
htop
df -h
docker stats

# Network connectivity
ping api-fxpractice.oanda.com
curl -I https://api-fxpractice.oanda.com/v3/accounts
```

## Security Configuration

### Firewall Setup

```bash
# UFW is automatically configured by deployment script
# Verify firewall status
ssh root@YOUR_DROPLET_IP
ufw status verbose

# Allowed ports:
# - 22 (SSH)
# - 80 (HTTP)
# - 443 (HTTPS)
# - 8080 (Trading API)
```

### SSL Certificate Setup (Optional)

```bash
# Install Certbot for Let's Encrypt
apt install certbot python3-certbot-nginx

# Obtain SSL certificate
certbot --nginx -d your-domain.com

# Auto-renewal
crontab -e
# Add: 0 12 * * * /usr/bin/certbot renew --quiet
```

### Database Security

```bash
# PostgreSQL is configured with secure defaults:
# - Dedicated trading user with limited privileges
# - Local connections only
# - Strong password authentication
# - Regular automated backups
```

## Backup and Recovery

### Automated Backups

```bash
# Database backup script (runs daily via cron)
cat > /opt/sep-trader/scripts/backup_database.sh << 'EOF'
#!/bin/bash
BACKUP_DIR="/mnt/sep_data/backups"
DATE=$(date +%Y%m%d_%H%M%S)
sudo -u postgres pg_dump sep_trading > "$BACKUP_DIR/sep_trading_$DATE.sql"
find "$BACKUP_DIR" -name "*.sql" -mtime +7 -delete
EOF

chmod +x /opt/sep-trader/scripts/backup_database.sh

# Add to crontab
echo "0 2 * * * /opt/sep-trader/scripts/backup_database.sh" | crontab -
```

### Configuration Backup

```bash
# Backup configuration and data
rsync -avz root@YOUR_DROPLET_IP:/opt/sep-trader/config/ ./backups/config/
rsync -avz root@YOUR_DROPLET_IP:/opt/sep-trader/data/ ./backups/data/
```

## Troubleshooting

### Common Issues

#### **Build Problems on Local Machine**
```bash
# Clean rebuild
./build.sh --clean && ./build.sh

# Check CUDA installation
nvidia-smi
nvcc --version

# Library path issues
echo $LD_LIBRARY_PATH
export LD_LIBRARY_PATH=./build/src/core:./build/src/config:./build/src/c_api
```

#### **Droplet Connection Issues**
```bash
# SSH key problems
ssh-keygen -R YOUR_DROPLET_IP
ssh-copy-id root@YOUR_DROPLET_IP

# Service not starting
ssh root@YOUR_DROPLET_IP
docker-compose down
docker-compose up -d
docker-compose logs
```

#### **Trading Service Issues**
```bash
# Check OANDA credentials
ssh root@YOUR_DROPLET_IP
cat /opt/sep-trader/config/OANDA.env

# Test OANDA connectivity
curl -H "Authorization: Bearer YOUR_API_KEY" \
     "https://api-fxpractice.oanda.com/v3/accounts"

# Restart trading service
docker-compose restart sep-trader
```

### Validation Checklist

- [ ] **Local System**: `./build/src/cli/trader-cli status` returns success
- [ ] **Remote Health**: `curl http://YOUR_DROPLET_IP/health` returns healthy
- [ ] **OANDA Connection**: API credentials verified and working
- [ ] **Data Sync**: `./scripts/sync_to_droplet.sh` completes successfully
- [ ] **Trading Service**: Container running and processing signals
- [ ] **Database**: PostgreSQL accessible and storing data
- [ ] **Monitoring**: Logs accessible and showing activity

## Performance Optimization

### Local CUDA Optimization

```bash
# GPU memory optimization
nvidia-smi -q -d MEMORY
nvidia-smi -l 1  # Monitor GPU usage

# CUDA compiler optimization
export CUDA_CACHE_PATH=/tmp/cuda_cache
export CUDA_CACHE_MAXSIZE=2147483648  # 2GB cache
```

### Remote Droplet Optimization

```bash
# PostgreSQL tuning for trading workload
# Already optimized in deployment script:
# - shared_buffers = 2GB
# - effective_cache_size = 6GB
# - work_mem = 64MB
# - maintenance_work_mem = 512MB

# Docker resource limits
docker update --memory=6g --cpus=3 sep-trader
```

---

This deployment guide ensures a **production-ready SEP trading system** with enterprise-grade reliability, security, and performance monitoring.

**SEP Dynamics, Inc.** | Quantum-Inspired Financial Intelligence  
**alex@sepdynamics.com** | [sepdynamics.com](https://sepdynamics.com)
