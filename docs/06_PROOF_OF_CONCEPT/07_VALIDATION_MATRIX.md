# Pitch Claim Validation Matrix

This matrix links key claims from the investor pitch deck to concrete evidence within the repository.

| Pitch Deck Claim | Supporting Evidence |
| --- | --- |
| Patent Application Filed (August 3, 2025) | `docs/patent/PATENT_OVERVIEW.md` |
| Live Trading System with 60.73% accuracy | `docs/results/PERFORMANCE_METRICS.md` |
| Enterprise Architecture with professional CLI | `src/cli/trader_cli.cpp` |
| CUDA acceleration enables sub-ms processing | `src/apps/oanda_trader/tick_cuda_kernels.cu` |
| Multi-timeframe analysis optimized for performance | `src/apps/oanda_trader/quantum_signal_bridge.cpp` |

To add a new claim, append a row using the same format with the evidence file paths enclosed in backticks.
