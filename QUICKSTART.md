# SEP Professional Trader-Bot - Quick Start Guide

## System Requirements
- **CUDA 12.9+** - GPU acceleration for quantum processing
- **OANDA Account** - Live/demo trading API access
- **Linux/Ubuntu** - Production deployment environment
- **16GB+ RAM** - Multi-pair processing requirements

## Installation & Setup

### 1. Clone and Build
```bash
git clone https://github.com/SepDynamics/sep-trader.git
cd sep-trader
./install.sh
./build.sh
```

### 2. Configure OANDA Credentials
```bash
# Edit OANDA.env with your credentials
nano OANDA.env

# Set required environment variables
export OANDA_API_KEY='your_api_key'
export OANDA_ACCOUNT_ID='your_account_id'
```

### 3. Check System Status
```bash
# Professional CLI interface (NEW)
./build/src/cli/trader-cli status           # Overall system status
./build/src/cli/trader-cli pairs list       # List all trading pairs
./build/src/cli/trader-cli config reload    # Hot reload configuration

# Legacy python interface  
python train_manager.py status             # View all pair status

# Test individual pair training
python train_currency_pair.py EUR_USD --quick

# Check if system is ready
./build/src/apps/oanda_trader/quantum_tracker --test
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
