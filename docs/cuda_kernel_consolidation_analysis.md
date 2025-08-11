# CUDA Kernel Consolidation Analysis

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

## Consolidation Plan

1. **Centralize kernels under `src/cuda/`**
   - Create submodules `pattern/`, `quantum/`, and `trading/` to host all device code.
   - Migrate application-specific kernels after abstracting their inputs.
2. **Deduplicate quantum routines**
   - Merge `qbsa_kernel` and `qsh_kernel` into a single implementation used by both core and trading modules.
   - Provide unified launch wrappers in `src/cuda/quantum/`.
3. **Unify pattern processing**
   - Replace placeholder pattern kernels with the richer implementation in `src/cuda/pattern/pattern_kernels.cu`.
   - Expose generic kernels that trading and training paths can call via shared headers.
4. **Standardize API surface**
   - Adopt consistent `launch_*` conventions with `cudaStream_t` parameters.
   - Consolidate error checking and device synchronization utilities.
5. **Prepare for library build**
   - Produce a static or shared library from `src/cuda/` and link all modules against it.
   - Remove obsolete kernels from `src/quantum/` and `src/trading/cuda/` once replacements are integrated.

## Next Steps

- Draft unified directory layout.
- Begin merging duplicate kernels starting with qbsa/qsh.
- Introduce shared headers and remove placeholder implementations.

