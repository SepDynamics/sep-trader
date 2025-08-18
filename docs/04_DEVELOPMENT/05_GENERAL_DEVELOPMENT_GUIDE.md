# General Development Guide

This document provides a technical overview of the SEP Engine's implementation, build system, testing framework, and deployment procedures.

## 1. Architecture & Core Components

The engine is a modular, CUDA-accelerated C++17 system.

### 1.1. Quantum Signal Bridge
- **Location:** `src/app/quantum_signal_bridge.hpp/.cpp`
- **Purpose:** Core signal generation using QFH/QBSA algorithms.
- **Key Logic:** Multi-timeframe (M1, M5, M15) confirmation and trajectory damping based on entropy and coherence.

### 1.2. Multi-Asset Signal Fusion
- **Location:** `src/app/multi_asset_signal_fusion.hpp/.cpp`
- **Purpose:** Cross-asset correlation analysis to enhance signal confidence.
- **Key Logic:** Dynamic Pearson correlation, weighted voting, and confidence boosting.

### 1.3. Market Regime Adaptive Intelligence
- **Location:** `src/app/market_regime_adaptive.hpp/.cpp`
- **Purpose:** Dynamically adapts trading thresholds based on market volatility, trend, and liquidity.

### 1.4. Market Model Cache
- **Location:** `src/app/enhanced_market_model_cache.hpp/.cpp`
- **Purpose:** An intelligent, correlation-aware cache for market data to improve performance.

## 2. Build System

- **Primary Script:** `build.sh` (for Linux) and `build.bat` (for Windows).
- **Containerization:** The build system can use Docker for hermetic, dependency-free builds.
- **CMake:** The project uses CMake for build configuration. Key files are `CMakeLists.txt` in the root and subdirectories.
- **Static Analysis:** Use `./run_codechecker_filtered.sh` for a focused static analysis.

## 3. Testing Framework

The project has a comprehensive testing suite.

### 3.1. Mathematical & Core Logic Tests
- **Pattern Classification:** `test_forward_window_metrics`
- **CUDA/CPU Parity:** `trajectory_metrics_test`
- **Core Algorithms:** `pattern_metrics_test`
- **Signal Generation Pipeline:** `quantum_signal_bridge_test`

### 3.2. Integration & Performance Tests
- **Headless System Validation:** `quantum_tracker --test`
- **Backtesting:** `pme_testbed_phase2`

## 4. Deployment

### 4.1. Production Trading
- **Executable:** `quantum_tracker` in `build/src/app/oanda_trader/`
- **Configuration:** Set OANDA credentials in an `OANDA.env` file.
- **Features:** Dynamic data bootstrapping, live trade execution via OANDA API, risk management, and market schedule awareness.

### 4.2. Optimal Configuration
- **Stability Weight:** 0.40
- **Coherence Weight:** 0.10
- **Entropy Weight:** 0.50
- **Confidence Threshold:** 0.65 (adaptive)
- **Coherence Threshold:** 0.30 (adaptive)

## 5. Development Workflow

1.  **Code:** Make changes to source files in the `src/` directory.
2.  **Build:** Run `./build.sh` or `build.bat`.
3.  **Test:** Run the relevant test suites to validate changes.
4.  **Deploy:** If all tests pass, deploy the updated `quantum_tracker` executable.