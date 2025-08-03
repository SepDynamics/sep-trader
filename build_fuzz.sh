#!/bin/bash

# SEP DSL Fuzz Testing Build Script - Docker Integration
set -e

echo "🧪 Building SEP DSL Fuzz Testing Suite..."

# Use Docker like the main build system
DOCKER_BIN=${DOCKER_BIN:-docker}

# Check if we should skip docker  
SKIP_DOCKER=false
for arg in "$@"; do
    case "$arg" in
        --no-docker)
            SKIP_DOCKER=true
            ;;
    esac
done

if [ "$SKIP_DOCKER" = true ]; then
    echo "🔨 Building natively with fuzz support..."
    cd build
    cmake .. -G Ninja \
        -DCMAKE_BUILD_TYPE=Debug \
        -DCMAKE_CXX_COMPILER=clang++-15 \
        -DCMAKE_C_COMPILER=clang-15 \
        -DENABLE_FUZZING=ON \
        -DSEP_USE_CUDA=OFF \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=TRUE
    ninja fuzz_parser fuzz_interpreter 2>&1 | tee ../output/fuzz_build_log.txt
    cd ..
    echo "✅ Native fuzz build complete!"
    exit 0
fi

# Docker-based build (matches main build.sh)
CUDA_PREFIX="$(dirname "$(command -v nvcc 2>/dev/null)" 2>/dev/null | sed 's#/bin$##')"
if [ -z "$CUDA_PREFIX" ] || [ ! -d "$CUDA_PREFIX" ]; then
    CUDA_PREFIX=/usr/local/cuda
fi
if [ ! -d "$CUDA_PREFIX" ]; then
    CUDA_PREFIX=/usr
fi

echo "🔨 Building fuzz targets in Docker..."
"${DOCKER_BIN}" run --gpus all --rm \
    -v $(pwd):/sep \
    -e CUDA_HOME=$CUDA_PREFIX \
    -e CUDA_TOOLKIT_ROOT_DIR=$CUDA_PREFIX \
    sep_build_env bash -c '
    git config --global --add safe.directory "*"
    cd /sep/build
    
    # Configure with fuzzing enabled
    cmake .. -G Ninja \
        -DCMAKE_BUILD_TYPE=Debug \
        -DCMAKE_C_COMPILER=clang-15 \
        -DCMAKE_CXX_COMPILER=clang++-15 \
        -DENABLE_FUZZING=ON \
        -DSEP_USE_CUDA=ON \
        -DCUDAToolkit_ROOT=$CUDA_HOME \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=TRUE
    
    # Build fuzz targets
    ninja fuzz_parser fuzz_interpreter 2>&1 | tee /sep/output/fuzz_build_log.txt
'

echo "✅ Docker fuzz build complete!"
echo ""
echo "📋 Available fuzz targets:"
echo "   ./build/tests/fuzzing/fuzz_parser"
echo "   ./build/tests/fuzzing/fuzz_interpreter"
echo ""
echo "🚀 Run fuzzing with:"
echo "   ./build/tests/fuzzing/fuzz_parser -max_total_time=60"
echo "   ./build/tests/fuzzing/fuzz_interpreter -max_total_time=60"
