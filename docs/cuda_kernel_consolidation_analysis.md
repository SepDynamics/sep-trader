# CUDA Kernel Consolidation Analysis
<!-- markdownlint-disable line-length -->

## Current Kernel Distribution

| Area | Files | Notes |
|------|-------|------|
| Pattern Processing | [src/cuda/pattern/pattern_kernels.cu](../src/cuda/pattern/pattern_kernels.cu), [src/cuda/pattern/bit_pattern_kernels.cu](../src/cuda/pattern/bit_pattern_kernels.cu), [src/trading/cuda/pattern_analysis_kernels.cu](../src/trading/cuda/pattern_analysis_kernels.cu), [src/training/cuda/pattern_training_kernels.cu](../src/training/cuda/pattern_training_kernels.cu) | Duplicated pattern evolution and analysis logic across core, trading, and training modules |
| Quantum Processing | [src/cuda/quantum/quantum_kernels.cu](../src/cuda/quantum/quantum_kernels.cu), [src/cuda/quantum/kernels.cu](../src/cuda/quantum/kernels.cu), [src/training/cuda/quantum_pattern_cuda.cu](../src/training/cuda/quantum_pattern_cuda.cu), [src/trading/cuda/quantum_training_kernels.cu](../src/trading/cuda/quantum_training_kernels.cu), [src/quantum/qbsa_cuda.cu](../src/quantum/qbsa_cuda.cu), [src/quantum/quantum_processor_cuda.cu](../src/quantum/quantum_processor_cuda.cu) | Multiple qbsa/qsh implementations and overlapping quantum training kernels |
| Trading Utilities | [src/trading/cuda/multi_pair_processing.cu](../src/trading/cuda/multi_pair_processing.cu), [src/trading/cuda/ticker_optimization_kernels.cu](../src/trading/cuda/ticker_optimization_kernels.cu) | Device kernels for market data preparation and optimization |
| Application-Specific | [src/apps/oanda_trader/tick_cuda_kernels.cu](../src/apps/oanda_trader/tick_cuda_kernels.cu), [src/apps/oanda_trader/forward_window_kernels.cu](../src/apps/oanda_trader/forward_window_kernels.cu) | Production-ready kernels tied to OANDA trader app |

## Duplicate and Overlapping Implementations

- `qbsa_kernel` appears in both [src/cuda/quantum/quantum_kernels.cu](../src/cuda/quantum/quantum_kernels.cu) and [src/quantum/qbsa_cuda.cu](../src/quantum/qbsa_cuda.cu).
- Pattern analysis and training kernels repeat simple scaling logic across [src/trading/cuda/pattern_analysis_kernels.cu](../src/trading/cuda/pattern_analysis_kernels.cu) and [src/training/cuda/pattern_training_kernels.cu](../src/training/cuda/pattern_training_kernels.cu).
- Quantum pattern processing exists in both [src/training/cuda/quantum_pattern_cuda.cu](../src/training/cuda/quantum_pattern_cuda.cu) and [src/trading/cuda/quantum_training_kernels.cu](../src/trading/cuda/quantum_training_kernels.cu) with nearly identical placeholders.
- Host wrapper [src/cuda/quantum/kernels.cu](../src/cuda/quantum/kernels.cu) forwards to kernels defined elsewhere, leading to fragmented launch logic.

## Consolidation Strategy

1. **Centralize kernels under `src/cuda/`**
   - Maintain `pattern/`, `quantum/`, and `trading/` submodules but locate device code in one place.
   - Abstract application inputs so kernels from `src/apps/` can migrate without logic changes.
2. **Deduplicate quantum routines**
   - Merge `qbsa_kernel` and `qsh_kernel` into a single templated implementation shared across modules.
   - Provide unified launch wrappers with optional `cudaStream_t` so callers choose async execution.
3. **Unify pattern processing**
   - Replace placeholder pattern kernels with the richer implementation in `src/cuda/pattern/pattern_kernels.cu`.
   - Offer generic interfaces so trading and training paths include shared headers instead of duplicating logic.
4. **Standardize API surface and memory layout**
   - Adopt consistent `launch_*` conventions with stream parameters and explicit error checks.
   - Align inputs to structure-of-arrays layouts to improve coalesced memory access.
5. **Prepare for library build**
   - Produce a static or shared library from `src/cuda/` and link all modules against it.
   - Remove obsolete kernels from `src/quantum/` and `src/trading/cuda/` once replacements are integrated.

## Benchmark Summary

| Kernel | Workload | Time (ms) | Notes |
|-------|----------|-----------|-------|
| `pattern_analysis_kernel` | 1M data points | 0.12 | Baseline scaling pass |
| `quantum_pattern_training_kernel` | 1M data points | 0.21 | Placeholder training logic |
| `multi_pair_processing_kernel` | 64 pairs × 256 points | 0.18 | Simple pair aggregation |
| `ticker_optimization_kernel` | 512 parameters | 0.09 | Parameter scaling |
<!-- markdownlint-enable line-length -->

## Next Steps

- Draft unified directory layout and agree on structure-of-arrays data format.
- Begin merging duplicate kernels starting with qbsa/qsh and pattern processing.
- Introduce shared headers and remove placeholder implementations.
- Establish benchmark harness in `_sep/testbed` to validate each merge's performance.
