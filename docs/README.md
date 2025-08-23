# SEP Trading Engine - Technical Documentation

## System Architecture Overview

The SEP Trading Engine is a CUDA-accelerated quantum-inspired trading system that integrates patent-pending Quantum Field Harmonics (QFH) algorithms with real-time OANDA market data. The system has achieved 60.73% high-confidence accuracy in production testing through multi-timeframe analysis and cross-asset correlation.

## Current Implementation State

### ✅ Implemented Components

1. **Quantum Signal Processing**
   - QFH-based pattern analysis (`src/core/qfh.cpp`)
   - Quantum Bit State Analysis (QBSA) (`src/quantum/`)
   - Multi-timeframe confirmation (M1/M5/M15)
   - Trajectory damping with entropy-based decay

2. **Market Data Infrastructure**
   - OANDA connector with real API integration
   - Market model cache with signal persistence
   - Enhanced cache with correlation analysis
   - Real-time data aggregation pipeline

3. **Trading Framework**
   - QuantumPairTrainer with deterministic training
   - Dynamic pair management system
   - CLI-based administration interface
   - Redis-based state persistence

### ⚠️ Stub Components Requiring De-stubbing

Based on the epics provided, the following components contain stubs or placeholder logic:

1. **MarketModelCache Demo Data Generation**
   - Location: `src/app/market_model_cache.cpp::ensureCacheForLastWeek()`
   - Issue: Falls back to generating 1,000 demo candles with random movements
   - Current behavior: Non-deterministic pipeline hiding real data problems

2. **MultiAssetSignalFusion Default Correlations**
   - Location: `src/app/multi_asset_signal_fusion.cpp::calculateDynamicCorrelation()`
   - Issue: Returns default (0.0, 0ms, 0.0) correlation triple
   - Hardcoded quantum identifiers: confidence=0.7, coherence=0.4, stability=0.5

3. **CLI Commands Stub Implementation**
   - Location: `src/core/cli_commands.cpp`
   - Issue: Methods only print messages without executing actual logic
   - Affected commands: trainPair, cleanupCache, runBenchmark

4. **Testbed/Backtesting Infrastructure**
   - Location: `_sep/testbed/` directory, `SEP_BACKTESTING` macro
   - Issue: Conditional compilation creates divergent code paths
   - CMakeLists.txt excludes real OANDA sources in some builds

5. **Environment Configuration**
   - Issue: OANDA credentials loaded ad-hoc across modules
   - Missing: Single source of truth for configuration
   - Path issues with hardcoded `_sep/` references

## Build System

### Container-Based Build
```bash
# Primary build command (Docker-based)
./build.sh

# Build artifacts location
build/src/trader_cli           # CLI interface
build/src/oanda_trader          # OANDA trading interface  
build/src/sep_app               # Main application
build/src/quantum_pair_trainer  # Training executable
```

### Dependencies
- CUDA Toolkit 12.9+
- Intel TBB (Threading Building Blocks)
- PostgreSQL client libraries
- Redis/Hiredis
- yaml-cpp
- spdlog
- fmt

### CMake Configuration
The build system uses modern CMake with CUDA support:
- C++17 standard required
- CUDA 17 standard for GPU code
- Precompiled headers for performance
- Dynamic library architecture

## OANDA Integration

### Environment Setup
```bash
# Required environment variables
export OANDA_API_KEY="your-api-key"
export OANDA_ACCOUNT_ID="your-account-id"
export OANDA_BASE_URL="https://api-fxtrade.oanda.com"  # or practice URL
```

### Data Flow
1. OANDA Connector fetches real-time/historical data
2. MarketModelCache stores and manages candle data
3. QuantumSignalBridge processes through QFH analysis
4. MultiAssetSignalFusion correlates cross-asset signals
5. Trading decisions execute through OANDA API

## Quantum Processing Pipeline

### Core Algorithm Flow
```
Market Data → Bitstream Conversion → QFH Analysis → Pattern Discovery
     ↓              ↓                      ↓              ↓
  Candles      Binary States         Harmonics      Collapse Points
     ↓              ↓                      ↓              ↓
  Cache       Entropy Calc          Coherence      Signal Generation
```

### Key Metrics
- **Confidence Threshold**: 0.65 (adaptive ±15%)
- **Coherence Threshold**: 0.30 (adaptive ±20%)
- **Signal Rate**: ~19% (high-confidence signals only)
- **Accuracy**: 60.73% on high-confidence predictions

## Testing Infrastructure

### Unit Tests
```bash
# Run all tests
./build/tests/unit_tests

# Specific test suites
./build/tests/data_integrity_test
./build/tests/cache_validator_test
./build/tests/quantum_signal_test
```

### Integration Tests
- OANDA connection validation
- Cache persistence verification
- Multi-asset correlation testing
- CLI command execution

## Performance Optimization

### CUDA Acceleration
- Supported architectures: 61, 75, 86, 89 (GTX 1070 through RTX 4090)
- Parallel bitstream processing
- GPU-accelerated correlation calculations
- Optimized memory transfers

### Cache Strategy
- Hierarchical caching (M1 → M5 → M15 → H1)
- Correlation-aware eviction
- Compressed storage format
- Memory-mapped file support

## Deployment Configuration

### Local Development
```bash
# Development mode with verbose logging
SEP_LOG_LEVEL=DEBUG ./build/src/trader_cli

# Training mode
./build/src/quantum_pair_trainer train EUR_USD
```

### Production Deployment
```bash
# Service mode with monitoring
systemctl start sep-trader
systemctl status sep-trader

# Remote synchronization
./scripts/sync_to_droplet.sh
```

## Monitoring & Observability

### Real-time Metrics
- Signal generation rate
- Prediction accuracy
- Cache hit ratio
- CUDA utilization
- Network latency to OANDA

### Logging
- Structured JSON logs via spdlog
- Rotating file output
- Configurable verbosity levels
- Performance profiling markers

## Security Considerations

### API Key Management
- Environment variable isolation
- No hardcoded credentials
- Secure storage in production
- Key rotation support

### Network Security
- TLS encryption for OANDA API
- Local Redis authentication
- Firewall rules for services
- VPN/Tailscale for remote access

## Known Limitations

1. **Data Quality**: Demo candle fallback affects determinism
2. **Correlation Cache**: Not persisted between restarts
3. **Backtesting**: Separate code paths reduce confidence
4. **CLI Integration**: Commands not fully wired to backend
5. **Path Management**: Hardcoded paths break in different environments

## Next Steps

See TODO.md for detailed implementation tasks to complete the de-stubbing process and achieve full production readiness.

## Support & Documentation

- Internal docs: `/docs/` directory
- Build logs: `/output/build_log.txt`
- Configuration: `/config/` directory
- Training results: `/live_results/` directory

## License & Patents

This system incorporates patent-pending Quantum Field Harmonics (QFH) technology. All rights reserved.