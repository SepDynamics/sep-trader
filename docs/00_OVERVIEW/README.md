# SEP Professional Trader-Bot Overview

## Project Overview

This project, SEP Professional Trader-Bot, is a sophisticated, production-ready autonomous trading platform. It leverages a patent-pending technology called Quantum Field Harmonics (QFH) for financial modeling and prediction. The system is designed with a hybrid architecture, separating the computationally intensive model training (local, CUDA-accelerated) from the lightweight trade execution (remote, CPU-only droplet).

The core technology boasts a 60.73% prediction accuracy in live trading scenarios. The system is built with C++ for performance-critical components, Python for higher-level orchestration, and a suite of shell scripts for deployment and management.

### Core Innovation: Quantum Field Harmonics (QFH) Engine

**Patent Application #584961162ABX** - The system uses quantum-inspired financial modeling with:
- **Bit-level Pattern Analysis** - Proprietary pattern collapse prediction
- **Real-time Pattern Degradation Detection** - Eliminates traditional lag
- **Multi-timeframe Quantum Analysis** - M1/M5/M15 synchronized processing
- **Riemannian Evolutionary Optimization** - Advanced pattern adaptation

### System Architecture: Hybrid Local/Remote Design

```
LOCAL CUDA MACHINE (Training)          REMOTE DROPLET (Execution)
├── Quantum Pattern Analysis      →    ├── Trading Execution  
├── CUDA-Accelerated Training     →    ├── Signal Processing
├── Model Generation              →    ├── Market Monitoring
└── Signal Synchronization        →    └── Performance Logging
```

**Key Design Principles:**
- **Local Training**: GPU-accelerated quantum analysis requiring CUDA
- **Remote Execution**: CPU-only cloud deployment on Digital Ocean  
- **Data Pipeline**: Automated model/signal synchronization
- **Enterprise Architecture**: PostgreSQL, Redis, Docker containerization

## Key Technologies

- **Backend:** C++, Python
- **GPU Acceleration:** CUDA
- **Databases:** PostgreSQL (with TimescaleDB), Redis
- **Deployment:** Docker, Nginx
- **Cloud Provider:** Digital Ocean
- **CI/CD:** GitHub Actions (presumably, based on `.github` directory)

## Core Commands

### Building the Project

The primary build script is `build.sh`. It should be run from the root directory.

- **Local (with CUDA):** `./build.sh`
- **Remote (CPU-only):** `./build.sh --no-docker`

### Running Tests

While a specific top-level test command isn't explicitly mentioned, the project contains numerous test files. The `trader-cli` has a test mode that can be invoked to check for basic functionality and CUDA device recognition:

```bash
export LD_LIBRARY_PATH=./build/src/core:./build/src/config:./build/src/c_api && ./build/src/apps/oanda_trader/quantum_tracker --test
```

### Running the Application

The main application is run via the `trader-cli` and various scripts.

- **Starting the remote trading service:** On the droplet, `docker-compose up -d`
- **Local CLI interaction:**
  ```bash
  export LD_LIBRARY_PATH=./build/src/core:./build/src/config:./build/src/c_api
  ./build/src/cli/trader-cli <command>
  ```
- **Training a model:**
  ```bash
  ./build/src/trading/quantum_pair_trainer train EUR_USD
  ```

## Project Structure

- `src/`: Contains the core C++ source code.
  - `src/quantum/`: The QFH engine.
  - `src/trading/`: Trading logic, data management, and training.
  - `src/cli/`: The command-line interface.
  - `src/apps/`: Standalone applications like data downloaders and traders.
- `scripts/`: Deployment and utility scripts.
- `config/`: Configuration files for the application.
- `docs/`: Project documentation.
- `build/`: Build output directory.
