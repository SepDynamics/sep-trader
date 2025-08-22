# SEP Test Suite

This directory defines the skeleton for a new GoogleTest based suite.

## Test Categories
### Unit Tests
* **core/** – engine algorithms, pattern processing, memory management, CUDA kernel dispatch
* **util/** – parser, serialization, mathematical helpers, memory tier management
* **io/** – OANDA connector, market data conversion, C API surface
* **app/** – `trader_cli`, `data_downloader`, `sep_dsl_interpreter`, `quantum_tracker`

### Integration Tests
Validate interactions across modules such as data download → processing → CLI
commands and live OANDA connectivity (sandbox).

### Performance Tests
Micro benchmarks for CUDA kernels, memory pool throughput, and real-time
streaming performance.

## Building
Testing is not yet implemented. Add test executables with `add_sep_test` as
fixtures are created. No CUDA runtime is required for placeholder configuration.
