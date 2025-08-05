# SEP Professional Trader-Bot - Remote Droplet Setup

## 🌐 Distributed Trading Architecture

**This guide sets up the REMOTE TRADING EXECUTION system on a droplet. Training happens on your local CUDA machine.**

## Droplet Requirements (Trading Execution)
- **Ubuntu 24.04 LTS** - Stable deployment environment
- **8GB RAM, 2 vCPUs** - Sufficient for trading operations
- **OANDA Account** - Live/demo trading API access
- **No CUDA required** - CPU-only execution
- **🆕 Database Support** - PostgreSQL, Redis, and HWLOC automatically installed

## Local PC Requirements (Training & Analysis)  
- **CUDA 12.9+** - GPU acceleration for quantum processing
- **16GB+ RAM** - Multi-pair model training
- **Linux/Ubuntu** - Development environment

## Droplet Installation & Setup

### 1. Clone and Build (CPU-Only with Enterprise Dependencies)
```bash
git clone https://github.com/SepDynamics/sep-trader.git
cd sep-trader
./install.sh --minimal --no-docker
./build.sh --no-docker  # Automatically installs PostgreSQL, Redis, HWLOC
```

**Note**: The build system now automatically installs enterprise dependencies:
- `libpqxx-dev` - PostgreSQL C++ client library
- `libhiredis-dev` - Redis C client library  
- `libhwloc-dev` - Hardware locality optimization

### 2. Configure OANDA Credentials
```bash
# Edit OANDA.env with your credentials
nano OANDA.env

# Add these lines:
OANDA_API_KEY=your_api_key_here
OANDA_ACCOUNT_ID=your_account_id_here
OANDA_ENVIRONMENT=practice
```

### 3. Set Library Path and Test System
```bash
# Set library path for CLI access
export LD_LIBRARY_PATH=./build/src/core:./build/src/config:./build/src/c_api

# Test system status
./build/src/cli/trader-cli status           # ✅ Overall system status
./build/src/cli/trader-cli pairs list       # ✅ List all trading pairs
./build/src/cli/trader-cli config show      # ✅ View configuration

# 🆕 Test enterprise data layer
./build/src/cli/trader-cli data status      # ✅ Database connectivity status
./build/src/cli/trader-cli cache stats      # ✅ Redis cache performance

# Test DSL engine
echo 'pattern test { print("Droplet ready!") }' > test.sep
./build/src/dsl/sep_dsl_interpreter test.sep  # ✅ DSL execution test
```

## Professional Trading Operations

### Training Management
```bash
# Check all pair status (ready/training/failed)
python train_manager.py status

# Train specific pairs
python train_manager.py train EUR_USD
python train_manager.py train GBP_USD --quick

# Train all pairs that need training
python train_manager.py train-all --quick

# Enable/disable pairs for trading
python train_manager.py enable EUR_USD
python train_manager.py disable EUR_USD
```

### Live Trading
```bash
# Start autonomous multi-pair trading
./run_trader.sh

# The system will:
# 1. Check all pairs have valid training (60%+ accuracy)
# 2. Validate cache data (last week requirement)
# 3. Start live trading only with ready pairs
```

## Core System Architecture

```
SEP Professional Trader-Bot
├── Quantum Pattern Engine (src/quantum/)
├── OANDA Trading Integration (src/apps/oanda_trader/)
├── Training Manager (train_manager.py)
├── Configuration Management (config/)
├── Cache System (automatic)
└── Documentation (docs/)
```

## Key Features

- **Multi-Pair Autonomous Trading**: Handle 50+ currency pairs simultaneously
- **Hot-Swappable Configuration**: Add/remove pairs without system restart (roadmap)
- **Professional State Management**: Enable/disable pairs with persistent state
- **Comprehensive Cache System**: Automated weekly data retention and validation
- **Patent-Pending QFH Technology**: 60.73% prediction accuracy in live trading

## Next Steps

1. **Complete Training**: Ensure all desired pairs are trained and ready
2. **Start Live Trading**: Use `./run_trader.sh` for autonomous operation
3. **Monitor Performance**: Watch system logs and trading results
4. **Scale Operations**: Add more pairs as needed

For detailed implementation roadmap, see [PROFESSIONAL_TRADER_BOT_ROADMAP.md](PROFESSIONAL_TRADER_BOT_ROADMAP.md).

For complete system architecture, see [SYSTEM_OVERVIEW.md](SYSTEM_OVERVIEW.md).
