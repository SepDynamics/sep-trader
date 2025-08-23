# Core Component Overview

This document summarizes the intent of major components in `src/core` and highlights open
areas for future development. Notes on expected interactions with the `app` layer and
`cuda` utilities are included for context.

## cli_commands
- **Intent:** expose training operations through a command‑line interface.
- **Status:** current file is a stub that only logs actions.
- **TODO:** replace stub with real command dispatch and error handling.
- **Interactions:** invoked by `app` CLI utilities and expected to trigger CUDA‑backed
  training through other core managers.

## dynamic_config_manager
- **Intent:** provide a runtime configuration store sourced from files,
  environment variables, or command‑line arguments.
- **Status:** most getters/setters return defaults; load/save routines are not implemented.
- **TODO:** implement persistence, source tracking, and change callbacks.
- **Interactions:** supplies configuration to both `app` services and CUDA modules
  (e.g., GPU settings).

## kernel_implementations
- **Intent:** host‑side wrappers that launch CUDA kernels such as QBSA and QSH.
- **Status:** validates parameters then zeroes buffers; real kernel launches are missing.
- **TODO:** wire in actual CUDA kernels and comprehensive error handling.
- **Interactions:** called by higher‑level training code in `app` and core, bridging to
  low‑level `cuda` routines.

## Training Managers
### training_session_manager
- **Intent:** manage the lifecycle of a single training session for a currency pair.
- **Status:** placeholder start/end methods.
- **TODO:** initialize session state, manage coherence targets, and finalize metrics.
- **Interactions:** orchestrated by `app` services and coordinates CUDA training kernels.

### training_coordinator (Orchestrator)
- **Intent:** coordinate data fetching, feature encoding, model training/evaluation,
  registry persistence, and optional remote pushes.
- **Status:** functional sequential pipeline; limited logging and parallelism.
- **TODO:** enhance error reporting, parallel execution, and recovery logic.
- **Interactions:** invoked by CLI or service layers in `app` and delegates to trainers and
  evaluators that may leverage CUDA.

## dynamic_pair_manager
- **Intent:** manage runtime‑enabled trading pairs and enforce resource limits.
- **Status:** basic validation with placeholder resource checks.
- **TODO:** implement detailed resource requirement validation based on configuration.
- **Interactions:** used by `app` to control active pairs and to ensure resources are
  available before launching CUDA workloads.

## Interactions with `app` and `cuda`
Core modules expose APIs consumed by the `app` layer. CUDA utilities provide GPU
acceleration for compute‑intensive routines. Components such as CLI commands, dynamic
configuration, and training managers act as the glue connecting user‑facing `app`
services with CUDA implementations.

