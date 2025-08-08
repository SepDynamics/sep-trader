
# Gemini Code Assistant Configuration

## Project Overview

This project, SEP Professional Trader-Bot, is a sophisticated, production-ready autonomous trading platform. It leverages a patent-pending technology called Quantum Field Harmonics (QFH) for financial modeling and prediction. The system is designed with a hybrid architecture, separating the computationally intensive model training (local, CUDA-accelerated) from the lightweight trade execution (remote, CPU-only droplet).

The core technology boasts a 60.73% prediction accuracy in live trading scenarios. The system is built with C++ for performance-critical components, Python for higher-level orchestration, and a suite of shell scripts for deployment and management.

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

## How to Help

1.  **Understand the Codebase:** Before making changes, familiarize yourself with the relevant parts of the C++ and Python code. The documentation in the `docs/` directory is a good starting point.
2.  **Follow Existing Patterns:** The project uses a specific coding style. Adhere to it.
3.  **Focus on Integration:** Many of the core components are already built in standalone executables. The main challenge is integrating them correctly.
4.  **Prioritize Build Fixes:** The build system is complex and can be fragile. Ensure any changes you make don't break the build.
5.  **Use the CLI:** The `trader-cli` is the primary way to interact with the system. Use it to test your changes.
