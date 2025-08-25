# SEP Professional Trading System - System Architecture

## Overview

The SEP Professional Trading System is a sophisticated quantum-enhanced trading platform that combines advanced C++/CUDA computational engines with a modern web-based interface. The system utilizes patent-pending Quantum Field Harmonics (QFH) technology for market analysis and features a comprehensive web dashboard for monitoring, configuration, and real-time trading operations.

## System Architecture

### Three-Tier Professional Platform

1. **Core Engine Layer** (C++/CUDA): Quantum processing algorithms and pattern analysis
2. **Service Layer** (Python/Flask): REST API, WebSocket services, and integration bridges
3. **Presentation Layer** (React/TypeScript): Professional web interface for system management

### Key Components

#### Core Engine Layer
- **SEP Quantum Engine**: C++/CUDA engine implementing Quantum Binary State Analysis (QBSA) and Quantum Field Harmonics (QFH)
- **GPU Acceleration**: CUDA-enabled processing for quantum algorithms and pattern analysis
- **Pattern Processing**: Memory tiering system with coherence/stability calculations
- **Quantum Metrics Library**: Shared library (libquantum_metrics.so) exposing quantum processing metrics

#### Service Layer
- **Trading Service API** (`scripts/trading_service.py`): REST API for system integration
- **WebSocket Service** (`scripts/websocket_service.py`): Real-time data streaming
- **CLI Bridge** (`scripts/cli_bridge.py`): Secure bridge between web API and CLI commands
- **Database Connection** (`scripts/database_connection.py`): Valkey (Redis-compatible) persistence layer
- **OANDA Connector** (`scripts/oanda_connector.py`): Market data ingestion from OANDA API

#### Presentation Layer
- **Web Dashboard** (`frontend/`): React/TypeScript interface for system management
- **Real-time Monitoring**: Live trading dashboard with performance analytics

## Data Flow

### Market Data Ingestion
1. OANDA API → OANDA Connector → Trading Service → Valkey Database
2. Data stored with time-series keys (e.g., `candle:{instrument}:{granularity}:{timestamp}`)
3. Sorted sets maintain chronological ordering (`candles:series:{instrument}:{granularity}`)

### Quantum Processing Pipeline
1. Market data retrieved from Valkey
2. Quantum algorithms (QBSA/QFH) process data on GPU
3. Coherence/stability/entropy metrics calculated
4. Trading signals generated and stored
5. Results exposed via API and WebSocket

### Trading Execution
1. Trading signals trigger market orders via OANDA API
2. Trade execution data stored in Valkey
3. Performance metrics calculated and displayed

## Deployment Architecture

### Local Development Environment
- **Purpose**: Development, backtesting, model training
- **Infrastructure**: Docker Compose orchestration
- **GPU Support**: Full CUDA acceleration for quantum algorithms
- **Storage**: Local volumes for data persistence

### Remote Production Environment
- **Purpose**: Live trading operations, 24/7 monitoring
- **Infrastructure**: DigitalOcean Droplet (8GB RAM, 2 vCPUs)
- **Storage**: Persistent volume storage (`/mnt/volume_nyc3_01`)
- **Network**: Professional-grade container networking

## Key Technologies

### Core Engine Technologies
- **C++20/CUDA**: High-performance quantum processing engine
- **GLM**: Mathematics library for vector operations
- **nlohmann/json**: JSON processing
- **spdlog**: High-performance logging
- **TBB**: Thread building blocks for parallel processing

### Service Layer Technologies
- **Python 3.9+**: Trading service implementation
- **Flask**: REST API framework
- **Redis-py**: Valkey database client
- **Requests**: HTTP client for OANDA API

### Presentation Layer Technologies
- **React 18**: Frontend framework
- **TypeScript**: Type-safe JavaScript
- **WebSocket**: Real-time data streaming

## Build System

### CMake Configuration
- **CUDA Support**: Enabled with GCC-14 compatibility
- **Dependencies**: Managed via CMake FetchContent and system packages
- **Targets**:
  - `sep_lib`: Core C++ library with CUDA support
  - `quantum_metrics`: Shared library exposing quantum metrics
  - `data_downloader`: Utility for market data ingestion

### Build Requirements
- **Compiler**: GCC-14 or compatible
- **CUDA**: CUDA 12.9 toolkit
- **Dependencies**: GLM, nlohmann/json, spdlog, TBB, PostgreSQL client

## Current Issues

### Build Failures
The build log shows critical linker errors affecting the `libquantum_metrics.so` shared library:
```
FAILED: src/libquantum_metrics.so 
/usr/bin/ld: failed to set dynamic section sizes: bad value
collect2: error: ld returned 1 exit status
```

This indicates issues with the CUDA toolkit configuration or compiler toolchain that prevent successful linking of the quantum metrics library, which is essential for GPU-accelerated functionality.

## Configuration

### Environment Variables
- `VALKEY_URL`: Valkey database connection URL
- `OANDA_API_KEY`: OANDA API authentication key
- `OANDA_ACCOUNT_ID`: OANDA trading account ID
- `CUDA_HOME`: CUDA toolkit installation path

### Configuration Files
- `config/.sep-config.env`: Database configuration
- `config/OANDA.env`: OANDA API credentials
- `config/pair_registry.json`: Enabled trading pairs

## Integration Points

### CLI-Web Integration Bridge
The CLI Bridge (`scripts/cli_bridge.py`) enables seamless communication between the web interface and the core SEP engine:
- Command translation: Web interface commands → CLI operations
- Status synchronization: Real-time status updates from CLI → Web
- Error handling: Professional error propagation and logging
- Authentication: Secure API key-based authentication

### Real-Time Data Flow
```
Market Data → SEP Engine → Trading Decisions
     ↓              ↓            ↓
WebSocket ← Valkey Store ← Trading Service API
     ↓
Web Dashboard (Real-time Updates)
```

## Security Features

- **API Key Authentication**: Secure service-to-service communication
- **CORS Configuration**: Professional cross-origin resource sharing
- **Rate Limiting**: API endpoint protection
- **Session Management**: Valkey-based session handling

## Monitoring and Observability

- **Health Checks**: Comprehensive service health monitoring
- **Logging**: Structured logging across all services
- **Performance Metrics**: Real-time performance tracking
- **Error Handling**: Professional error management and reporting