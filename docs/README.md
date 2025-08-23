# SEP Trading Engine - Technical Documentation

## Overview
The SEP Trading Engine is a CUDA-accelerated trading platform that applies Quantum Field Harmonics (QFH) analytics to real-time OANDA data. The system produces high-confidence trading signals and supports remote execution through a lightweight web interface.

## Build and Binaries
- `./build.sh` compiles all targets.
- Compiled executables are stored in `bin/`:
  - `sep` – main engine service
  - `oanda_trader` – broker integration
  - `quantum_pair_trainer` – model training
  - `quantum_tracker` – live signal monitor
  - `data_downloader` – historical fetch utility

## Configuration
Runtime configuration lives under `config/`. `default.json` provides base settings and can be overridden by other files in the directory.
OANDA credentials are loaded through the `EnvLoader` utility and can be supplied via environment variables:

```
export OANDA_API_KEY=...
export OANDA_ACCOUNT_ID=...
export OANDA_BASE_URL=https://api-fxtrade.oanda.com
```

Paths such as cache, config, and logs are resolved relative to `PROJECT_ROOT`, allowing binaries in `bin/` to run from the repository root.

## Frontend and Remote Deployment
The `frontend/` directory hosts a React/TypeScript UI served by Nginx. In production the UI and backend run on a DigitalOcean droplet. Deployment helpers:

```
./scripts/deploy_to_droplet.sh    # initial droplet setup
./scripts/sync_to_droplet.sh      # push configs and models
```

The web interface communicates with the backend API on port `5000` and the WebSocket service on `8765`.

## Testing
Unit and integration tests live under `tests/` and are driven by CMake. When source code or tests change, run:

```
ctest
```

Documentation-only updates do not require running the test suite.

## Next Steps
Ongoing development tasks are tracked in `docs/TODO.md`. Review that document for remaining work items.

## Support & License
- Additional documentation lives in the `/docs/` directory.
- Patent-pending Quantum Field Harmonics (QFH) technology. All rights reserved.
