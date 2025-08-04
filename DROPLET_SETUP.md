# SEP Droplet Setup - Immediate Next Steps

## 🚀 **Your New Droplet: 165.227.109.187**

**Specs Confirmed:**
- 8GB RAM, 2 vCPUs, Ubuntu 24.04 LTS
- 25GB onboard + 50GB volume storage
- PostgreSQL + TimescaleDB optimized setup

## 📋 **Step-by-Step Setup**

### 1. Test Initial Connection
```bash
# Test SSH connection with default key
ssh root@165.227.109.187

# If connection successful, you should see:
# Welcome to Ubuntu 24.04 LTS
# root@your-droplet-name:~#
```

### 2. Run Automated Deployment
```bash
# From your local /sep directory
./scripts/deploy_to_droplet.sh

# This will automatically:
# ✅ Install PostgreSQL 14 + TimescaleDB
# ✅ Configure 50GB volume storage 
# ✅ Set up optimized database config
# ✅ Create trading database schema
# ✅ Install Docker, nginx, firewall
# ✅ Clone SEP repository
```

### 3. Initialize Database
```bash
# SSH to droplet after deployment
ssh root@165.227.109.187

# Initialize trading database
cd /opt/sep-trader/sep-trader
sudo -u postgres psql sep_trading < scripts/init_database.sql

# Verify database setup
sudo -u postgres psql sep_trading -c "SELECT * FROM v_database_info;"
```

### 4. Configure OANDA Credentials
```bash
# On droplet, edit OANDA configuration
nano /opt/sep-trader/config/OANDA.env

# Add your credentials:
OANDA_API_KEY=your_api_key_here
OANDA_ACCOUNT_ID=your_account_id_here
OANDA_ENVIRONMENT=practice  # or 'live' for real trading
```

### 5. Test Services
```bash
# Start services
cd /opt/sep-trader/sep-trader
docker-compose up -d

# Check health
curl http://165.227.109.187/health

# Check API status
curl http://165.227.109.187/api/status
```

## 🔧 **Quick Commands for Setup**

### All-in-One Setup (from your local machine):
```bash
# 1. Deploy infrastructure
./scripts/deploy_to_droplet.sh

# 2. SSH and configure
ssh root@165.227.109.187 << 'EOF'
cd /opt/sep-trader/sep-trader
sudo -u postgres psql sep_trading < scripts/init_database.sql
echo "OANDA_API_KEY=your_key" > /opt/sep-trader/config/OANDA.env
echo "OANDA_ACCOUNT_ID=your_account" >> /opt/sep-trader/config/OANDA.env
echo "OANDA_ENVIRONMENT=practice" >> /opt/sep-trader/config/OANDA.env
docker-compose up -d
EOF

# 3. Test connection
curl http://165.227.109.187/health
```

### Troubleshooting Commands:
```bash
# Check droplet status
ssh root@165.227.109.187 'systemctl status postgresql docker nginx'

# Check volume mount
ssh root@165.227.109.187 'df -h | grep mnt'

# Check database
ssh root@165.227.109.187 'sudo -u postgres psql -l'

# Check logs
ssh root@165.227.109.187 'cd /opt/sep-trader/sep-trader && docker-compose logs'
```

## 🌐 **Service URLs After Setup**

- **Health Check:** http://165.227.109.187/health
- **API Status:** http://165.227.109.187/api/status  
- **SSH Access:** ssh root@165.227.109.187
- **Database:** PostgreSQL on port 5432 (internal only)

## 📊 **Expected Database Size**

After initialization, your database will contain:
- **8 tables** for trading data
- **5 views** for common queries  
- **1 function** for statistics updates
- **TimescaleDB extensions** for time-series optimization
- **Default configuration** for 10 major currency pairs

## 🎯 **Next Phase: Data Sync**

Once setup is complete:
```bash
# Generate some test signals locally
python train_manager.py status

# Sync to droplet
./scripts/sync_to_droplet.sh

# Check if data arrived
ssh root@165.227.109.187 'ls -la /opt/sep-trader/data/'
```

---

**Ready to deploy your production trading system!** 🚀

**Run:** `./scripts/deploy_to_droplet.sh` to get started.
