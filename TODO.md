# SEP Professional Trader-Bot - Development Path
## Production Deployment Roadmap (August 2025)

### 🎯 IMMEDIATE PRIORITY: Local CUDA Training → Remote Trading Bot Path

#### **Phase 1: Resolve Critical Build Issues** (Priority: HIGH)
- [ ] **Fix critical std::string type conflicts and missing header paths** documented in `/sep/src/memory/dothis.md`
  - Comment out includes in `/sep/include/core/types.h` one by one to isolate problematic include
  - Focus on `compat/shim.h` for macros/typedefs conflicting with `std::string`
  - Replace `#include "engine/internal/cuda.h"` with correct public header
  - Ensure CMakeLists.txt has proper include directories for memory and core modules

- [ ] **Make CUDA optional with graceful fallback** in CMakeLists.txt
  - Remove FATAL_ERROR requirement for missing CUDA_HOME
  - Add conditional CUDA compilation with CPU fallback
  - Allow builds to succeed on systems without CUDA 12.9

- [ ] **Standardize package management** 
  - Fix mixed dnf/apt issues in `/sep/scripts/setup_training_env.sh` and `/sep/install.sh`
  - Choose consistent package manager based on OS detection

#### **Phase 2: Local CUDA Training Setup** (Priority: HIGH)
- [ ] **Enable all 6 major currency pairs** in `/sep/config/pair_registry.json`
  - Currently only EUR_USD through USD_CHF enabled (priority 1-6)
  - Set `"enabled": true` for all major pairs for initial training

- [ ] **Configure and test local CUDA training** using `/sep/config/training_config.json`
  - Verify CUDA optimization settings: device_id=0, memory_pool_size_mb=2048
  - Test "quick" training mode (100 iterations, batch_size=512) on first 2-3 currency pairs
  - Validate pattern quality thresholds: high_quality=70.0, minimum_acceptable=50.0

- [ ] **Build and test training system**
  - Run `./build.sh` and resolve any remaining build issues
  - Execute training coordinator: `./build/src/training/training_coordinator`
  - Verify CUDA acceleration and pattern generation

#### **Phase 3: Remote Sync & Redis Storage** (Priority: MEDIUM)
- [ ] **Verify external volume for droplet Redis pattern storage**
  - Check droplet at `165.227.109.187` has mounted external volume
  - Verify `/opt/sep-trader/data/` has sufficient space for pattern storage
  - Test Redis connection and pattern persistence

- [ ] **Test sync system** using `/sep/scripts/sync_to_droplet.sh`
  - Verify SSH connection to droplet `165.227.109.187`
  - Test rsync of output/, config/, and models/ directories
  - Validate pattern reload API endpoint: `http://localhost:8080/api/data/reload`

#### **Phase 4: Remote Trading Bot Activation** (Priority: HIGH)
- [ ] **Enable trading bot remotely on demo account**
  - Use `/sep/config/demo_trading.json` configuration
  - Verify OANDA demo API credentials and sandbox mode
  - Test risk management: max_position_size=1000, stop_loss_pips=20

- [ ] **Validate remote trading execution**
  - Start remote trading service on droplet
  - Monitor trade execution via API: `http://165.227.109.187/api/status`
  - Verify signal generation and order placement

#### **Phase 5: Scale Additional Tickers** (Priority: LOW)
- [ ] **Bring additional currency pair tickers online**
  - Enable cross pairs (EUR_GBP, EUR_JPY, EUR_AUD, GBP_JPY, AUD_JPY) after first pairs validated
  - Enable minor pairs (NZD_USD) as system proves stable
  - Monitor performance with `max_concurrent_pairs: 4` limit

### 🔧 CRITICAL DEPENDENCY FIXES NEEDED

1. **Header Path Issues** (BLOCKING):
   - `std::string` type conflicts in `/sep/include/core/types.h`
   - Missing `'compat/cuda.h'` and `'core/common.h'` files
   - CMakeLists.txt include directory configuration

2. **CUDA Configuration** (BLOCKING):
   - Strict CUDA 12.9 requirement prevents builds on other systems
   - No graceful CPU fallback for development

3. **Package Management** (MEDIUM):
   - Mixed dnf/apt commands causing environment-specific failures
   - Docker dependency installation during build rather than base image

### 📊 CURRENT SYSTEM STATE

**Working Components:**
- ✅ Complete todo documentation in `/sep/docs/TODO.md` (Phase 1-2 complete)
- ✅ Training configuration in `/sep/config/training_config.json`
- ✅ Currency pair registry with 12 pairs defined
- ✅ Demo trading configuration with OANDA API setup
- ✅ Sync scripts for remote droplet deployment
- ✅ Professional baseline: 60.73% high-confidence accuracy achieved

**Deployment Targets:**
- **Local**: CUDA training and pattern generation
- **Remote**: Droplet at `165.227.109.187` for live trading
- **Demo Account**: OANDA fxpractice API (account: 101-001-31229774-001)
- **Data Storage**: Redis pattern storage with external volume

### 🚀 EXECUTION SEQUENCE

1. **Fix build issues** → Run `./build.sh` successfully
2. **Train locally** → Generate patterns for first 3 currency pairs
3. **Sync to droplet** → Upload patterns and configuration
4. **Enable remote bot** → Start demo trading on droplet
5. **Scale gradually** → Add more currency pairs as system proves stable

This roadmap prioritizes getting the core system working end-to-end before scaling to additional currency pairs.
