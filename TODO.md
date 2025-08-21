# SEP Testing TODO

## Test Infrastructure
- Populate `tests/` skeleton with GoogleTest targets.
- Introduce shared fixtures and utility helpers.
- Add CI step to run unit and integration tests.

## Unit Test Coverage
- **Core**: pattern processing, training coordinator, memory pool, CUDA kernels.
- **Util**: parsers, serialization, math, time series helpers.
- **IO**: OANDA connector, market data converter, C API interface.
- **App**: `trader_cli`, `data_downloader`, `sep_dsl_interpreter`, `quantum_tracker`.

## Integration Tests
- Data download → parsing → engine pipeline.
- CLI workflows and DSL script execution.
- Remote synchronization and OANDA sandbox interactions.

## Performance Benchmarks
- Kernel throughput and memory pool efficiency.
- End-to-end latency of streaming pipeline.

## Next Steps
1. Implement foundational unit tests for `core` module.
2. Create mock data sets under `tests/data/`.
3. Wire tests into `ctest` and document invocation.
4. Expand coverage iteratively across modules.
