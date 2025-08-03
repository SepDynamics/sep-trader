#!/bin/bash

# SEP DSL Fuzz Testing - Docker Runtime
# Runs fuzz tests inside Docker container like the main build system

set -e

DOCKER_BIN=${DOCKER_BIN:-docker}

echo "🧪 Running SEP DSL Fuzz Testing in Docker..."

# Check arguments
FUZZ_TARGET=${1:-"parser"}
FUZZ_TIME=${2:-60}

if [[ "$FUZZ_TARGET" != "parser" && "$FUZZ_TARGET" != "interpreter" ]]; then
    echo "Usage: $0 [parser|interpreter] [time_in_seconds]"
    echo "  parser      - Test DSL parser robustness"
    echo "  interpreter - Test DSL interpreter robustness"
    echo "  time        - Fuzzing duration in seconds (default: 60)"
    exit 1
fi

# Ensure build exists
if [ ! -d "build" ]; then
    echo "Build directory not found. Run ./build.sh first."
    exit 1
fi

# Ensure fuzz targets exist  
if [ ! -f "build/tests/fuzzing/fuzz_${FUZZ_TARGET}" ]; then
    echo "Fuzz target not found. Run ./build_fuzz.sh first."
    exit 1
fi

# Set up Docker environment like main build.sh
CUDA_PREFIX="$(dirname "$(command -v nvcc 2>/dev/null)" 2>/dev/null | sed 's#/bin$##')"
if [ -z "$CUDA_PREFIX" ] || [ ! -d "$CUDA_PREFIX" ]; then
    CUDA_PREFIX=/usr/local/cuda
fi
if [ ! -d "$CUDA_PREFIX" ]; then
    CUDA_PREFIX=/usr
fi

echo "🚀 Running ${FUZZ_TARGET} fuzzer for ${FUZZ_TIME} seconds..."

"${DOCKER_BIN}" run --gpus all --rm \
    -v $(pwd):/sep \
    -e CUDA_HOME=$CUDA_PREFIX \
    -e CUDA_TOOLKIT_ROOT_DIR=$CUDA_PREFIX \
    sep_build_env bash -c "
    cd /sep
    echo 'Starting ${FUZZ_TARGET} fuzzer with corpus...'
    
    # Set up library paths for all dependencies
    export LD_LIBRARY_PATH=/sep/build/clang_15.0_cxx17_64_debug:/usr/local/cuda-12.9/lib64:\$LD_LIBRARY_PATH
    
    # Check if fuzzer exists and is executable
    if [ ! -f ./build/tests/fuzzing/fuzz_${FUZZ_TARGET} ]; then
        echo '❌ Fuzz target not found!'
        exit 1
    fi
    
    # Verify libraries
    echo 'Checking library dependencies...'
    ldd ./build/tests/fuzzing/fuzz_${FUZZ_TARGET} | head -5
    
    ./build/tests/fuzzing/fuzz_${FUZZ_TARGET} \
        tests/fuzzing/corpus/ \
        -max_total_time=${FUZZ_TIME} \
        -print_stats=1 \
        -print_final_stats=1 \
        -max_len=1000
    
    echo '✅ Fuzzing completed!'
"

echo "📊 Fuzz testing completed successfully!"
echo ""
echo "💡 Tips:"
echo "  - Run longer tests: $0 ${FUZZ_TARGET} 3600"
echo "  - Test both targets: $0 parser 300 && $0 interpreter 300"
echo "  - Check for crashes in Docker container logs"
