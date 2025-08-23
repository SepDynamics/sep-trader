# CUDA Kernels Overview

This directory summarizes GPU entry points used across the trading engine.
Each entry point is a host function that configures and launches a device
kernel located in the surrounding `src/cuda` sources.

## Kernel entry points

- `launchAnalyzeBitPatternsKernel` – analyzes bit windows and returns
  coherence, stability and entropy metrics. Current implementation includes
  placeholder heuristics and stubbed helpers for trend detection, coherence
  and stability calculations.
- `launchQBSAKernel` – wraps `qbsa_kernel` for quantum bit state alignment
  by comparing expected vs. observed bitfields.
- `launchQSHKernel` – dispatches `qsh_kernel` to measure symmetry heuristics
  across 64‑bit chunks.
- `launchQFHBitTransitionsKernel` – evaluates forward‑window bit transitions
  through `qfhKernel`, producing damped coherence and stability scores.
- `launchProcessPatternKernel` – runs `processPatternKernel` to evolve and
  interact pattern attributes.
- `launchMultiPairProcessingKernel` – CUDA façade that forwards work to the
  core multi‑pair processing kernel.
- `launchTickerOptimizationKernel` – front‑end for ticker optimization. The
  device `optimization_kernel` only performs a simple gradient step and is
  intended as a placeholder for a more sophisticated optimizer.

## Additional kernels

`similarity_kernel` and `blend_kernel` exist in `quantum_kernels.cu` but are
not wired to public launchers. They are kept as references for future
embedding comparison and context blending work.

