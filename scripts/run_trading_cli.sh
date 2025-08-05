#!/bin/bash

# SEP Trading CLI Runner Script
# Sets up the proper environment for running the quantum trading CLI

# Set library paths for dynamically linked dependencies
export LD_LIBRARY_PATH="/sep/build/_deps/spdlog-build:/sep/build/_deps/tbb-build/gnu_11.4_cxx11_64_release:$LD_LIBRARY_PATH"

# Set CUDA paths if available
if [ -d "$CUDA_HOME" ]; then
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
fi

# Ensure we're in the SEP directory
cd /sep

# Set up OANDA environment if available
if [ -f "/sep/OANDA.env" ]; then
    source /sep/OANDA.env
fi

# Run the trading CLI with all arguments passed through
exec ./build/src/trading/quantum_pair_trainer "$@"
