# Duplicate Implementations and Code Quality Issues Report

## Overview
This document tracks outstanding code quality concerns in the SEP Engine codebase and documents recent cleanup efforts.

## Code Quality Issues

### Hardcoded Values
- `src/cuda/kernels.cu` - Hardcoded block and grid sizes.

## Recent Cleanup
- Legacy DSL bytecode and primitive modules removed (`src/util/compiler.*`,
  `src/util/core_primitives.*`, `src/util/stdlib.*`, `src/util/time_series.*`).
- Mock health monitor implementation removed (`src/app/health_monitor_c_impl.c`).
- Stub data provider enum removed (`src/core/cache_metadata.hpp`).
- Placeholder training CLI and duplicate MemoryTierService implementation removed (`src/app/cli_main.cpp`, `src/app/MemoryTierService.*`).
- Duplicate CLI removed (`src/app/trader_cli_simple.cpp`, `src/app/trader_cli_simple.hpp`).
- Unused frontend testing component removed (`frontend/src/components/TestingSuite.jsx`).
- Unused CUDA placeholder removed (`src/core/quantum_pattern_cuda.cu`).
- DSL builtin now uses `data_downloader` for real OANDA data (`src/util/interpreter.cpp`).
- Default API base URL removed to enforce explicit configuration (`frontend/src/services/api.ts`).
- Redis stub context eliminated to ensure real integration (`src/util/redis_manager.*`).
- Stub CLI commands and duplicate kernel implementations removed (`src/core/cli_commands.*`, `src/core/kernel_implementations.cu`, `tests/unit/core/cli_commands_test.cpp`).
- Sample EUR/USD data helper and duplicate dataset removed (`src/io/oanda_connector.*`, `eur_usd_m1_48h.json`).
- Redundant TraderCLI implementation and unused entry point removed (`src/app/trader_cli.*`, `src/app/app_main.cpp`).
- Legacy dashboard component removed (`frontend/src/components/Dashboard.js`).
- Redundant JavaScript SymbolContext removed (`frontend/src/context/SymbolContext.js`)
  in favor of the typed implementation.
- Placeholder quantum state replaced with real implementation
  (`src/core/pattern_types.h`, `src/core/types_serialization.cpp`).
- Unused DSL aggregation and data transformation stubs removed (`src/util/aggregation.*`, `src/util/data_transformation.*`).
- Unimplemented market data DSL builtins removed (`src/util/interpreter.cpp`).
- Testbed OANDA market data helper migrated to production with real ATR
  (`src/app/quantum_signal_bridge.cpp`).
- Unused evolutionary helper declarations and mock trade simulation removed (`src/core/evolution.h`, `src/util/interpreter.cpp`).
- Duplicate quantum coherence manager removed (`src/util/quantum_coherence_manager.*`) in favor of the core implementation.
- Magic numbers in OANDA connector replaced with constants (`src/io/oanda_connector.cpp`, `src/io/oanda_constants.h`).
- Removed duplicate market data fetch function and placeholder ATR
  (`src/app/quantum_signal_bridge.cpp`).
- Unused spdlog isolation stub removed (`src/util/spdlog_isolation.h`).
- Deprecated header shims consolidated under unified include (`src/util/cuda_safe_includes.h`, `src/util/header_fix.h`, `src/util/force_array.h`, `src/util/functional_safe.h`).
- Legacy memory tier lookup map removed (`src/util/memory_tier_manager.*`).
- Example coherence thresholds replaced with configurable values (`src/core/coherence_manager.*`).

## Recommendations
1. Remove remaining hardcoded values via configuration.
2. Standardize error handling.

