# Kernel and Architecture Feature Overview

This guide outlines core processing kernels, memory buffers, and deployment components within the SEP trading system. Each section links directly to the source implementation for quick reference.

## Tick-level Rolling Statistics
- `calculateRollingWindowsKernel` scans all ticks for each window to compute mean price, volatility, and price changes.
- `calculateRollingWindowsOptimized` stages tick data in shared memory to cut global reads.
- Source: [`src/apps/oanda_trader/tick_cuda_kernels.cu`](../src/apps/oanda_trader/tick_cuda_kernels.cu)

## Shared-memory Optimization
- `calculateRollingWindowsOptimized` loads tick chunks into shared memory and processes them on thread 0, reducing redundant reads.
- Suggested improvements include struct-of-arrays layouts and warp-level reductions to avoid bank conflicts.
- Source: [`src/apps/oanda_trader/tick_cuda_kernels.cu`](../src/apps/oanda_trader/tick_cuda_kernels.cu)

## Multi-timeframe Parallelism
- `calculateMultiTimeframeWindows` assigns each thread an index and reuses `calculateWindowStats` for hourly and daily windows.
- Cooperative groups could allow threads to share intermediate sums and window boundaries.
- Source: [`src/apps/oanda_trader/tick_cuda_kernels.cu`](../src/apps/oanda_trader/tick_cuda_kernels.cu)

## Quantum-processing Kernels
- `qbsa_kernel` performs bit-field corrections on probe indices.
- `qsh_kernel` conducts symmetry checks and rupture detection.
- Launch wrappers are in `kernels.cu`.
- Sources: [`src/cuda/kernels/quantum/qbsa_kernel.cu`](../src/cuda/kernels/quantum/qbsa_kernel.cu), [`src/cuda/kernels/quantum/qsh_kernel.cu`](../src/cuda/kernels/quantum/qsh_kernel.cu)

## Trading-computation Kernels
- Tick-window kernels above provide volatility metrics.
- `MultiAssetSignalFusion` on the CPU side updates a correlation cache and computes Pearson correlations per pair.
- Source: [`src/apps/oanda_trader/multi_asset_signal_fusion.cpp`](../src/apps/oanda_trader/multi_asset_signal_fusion.cpp)

## GPU Memory Buffers
- `PinnedBuffer` uses `cudaHostAlloc` for fast host–device transfers.
- `DeviceBuffer` wraps `cudaMalloc` and provides RAII semantics.
- `UnifiedBuffer` leverages `cudaMallocManaged` with optional prefetching.
- Sources: [`src/engine/internal/cuda/memory/pinned_buffer.cuh`](../src/engine/internal/cuda/memory/pinned_buffer.cuh), [`src/engine/internal/cuda/memory/device_buffer.cuh`](../src/engine/internal/cuda/memory/device_buffer.cuh), [`src/engine/internal/cuda/memory/unified_buffer.cuh`](../src/engine/internal/cuda/memory/unified_buffer.cuh)

## Testbed Benchmark Insights
- `cuda_marketdata_harness.cu` compares a CPU loop with a GPU kernel that doubles market mid-prices, establishing latency and throughput baselines.
- Source: [`_sep/testbed/cuda_marketdata_harness.cu`](../_sep/testbed/cuda_marketdata_harness.cu)

## Hybrid Architecture and Data Flow
- `sync_to_droplet.sh` ships outputs, configs, and models to the remote droplet while excluding `.env` files and optionally exporting Redis snapshots.
- Source: [`scripts/sync_to_droplet.sh`](../scripts/sync_to_droplet.sh)

## Deployment and Hardening
- `deploy_to_droplet.sh` installs Docker, PostgreSQL/TimescaleDB, Nginx, and enables UFW with allowances for SSH, HTTP/S, and the API port.
- Source: [`scripts/deploy_to_droplet.sh`](../scripts/deploy_to_droplet.sh)

## Remote Trading Service
- `trading_service.py` loads enabled pairs, polls for signals, applies risk checks, and exposes simple HTTP endpoints.
- Source: [`scripts/trading_service.py`](../scripts/trading_service.py)

## Monitoring Endpoints
- `/health` reports market status and active pairs.
- `/api/status` returns service metadata and last synchronization time.
- Source: [`scripts/trading_service.py`](../scripts/trading_service.py)

## Firewall and Network Stance
- UFW is enabled with allow rules for ports 22, 80, 443, and 8080.
- Source: [`scripts/deploy_to_droplet.sh`](../scripts/deploy_to_droplet.sh)

## Credential Management
- OANDA credentials reside in `config/OANDA.env` and are manually configured on the droplet.
- Sync scripts skip any `.env` files.
- Source: [`scripts/sync_to_droplet.sh`](../scripts/sync_to_droplet.sh)

## PostgreSQL Safeguards
- Deployment mounts database storage on a volume and creates a dedicated backups directory.
- Source: [`scripts/deploy_to_droplet.sh`](../scripts/deploy_to_droplet.sh)

## Networking and Logging
- Trading activity and executed trades are logged to `/app/logs/trading_service.log` and a JSON ledger.
- Source: [`scripts/trading_service.py`](../scripts/trading_service.py)

## "Quantum-Inspired" Messaging
- Documentation clarifies that algorithms are quantum-inspired and run on classical hardware, avoiding claims of quantum advantage.
- Source: [`docs/arch/POSITIONING_CLARIFICATION.md`](../docs/arch/POSITIONING_CLARIFICATION.md)

