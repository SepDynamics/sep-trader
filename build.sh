#!/bin/bash
# Simplified build script for running inside a container

# Allow overriding the container runtime via DOCKER_BIN.
DOCKER_BIN=${DOCKER_BIN:-docker}

set -uo pipefail

REBUILD=false
SKIP_DOCKER=false
for arg in "$@"; do
    case "$arg" in
        --rebuild)
            REBUILD=true
            ;;
        --no-docker)
            SKIP_DOCKER=true
            ;;
    esac
done

echo "Building SEP Engine..."

if [ "$REBUILD" = true ]; then
    echo "Performing a full rebuild..."
    # Clean up previous build artifacts
    sudo rm -rf CMakeCache.txt CMakeFiles output Makefile build .cache .codechecker
    sleep 2
    clear
    sudo rm -rf /sep/.Trash-1000
    sleep 1
fi

mkdir -p output
mkdir -p build

# Ensure proper permissions for CodeChecker directories
USER_ID=$(id -u)
GROUP_ID=$(id -g)

# Use native build if --no-docker or container runtime unavailable
if [ "$SKIP_DOCKER" = true ] || ! "$DOCKER_BIN" info >/dev/null 2>&1; then
    if [ "$SKIP_DOCKER" = true ]; then
        echo "Building natively (--no-docker)..."
    else
        echo "Container runtime $DOCKER_BIN not available. Building natively..."
    fi
    cd build
    
    # Detect CUDA availability for native builds
    CUDA_FLAGS=""
    if command -v nvcc >/dev/null 2>&1; then
        echo "CUDA detected, enabling CUDA support..."
        CUDA_FLAGS="-DSEP_USE_CUDA=ON"
    else
        echo "CUDA not detected, building without CUDA support..."
        CUDA_FLAGS="-DSEP_USE_CUDA=OFF"
    fi
    
    cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release $CUDA_FLAGS \
        -DCMAKE_CXX_COMPILER=g++-10 -DCMAKE_C_COMPILER=gcc-10 \
        -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=TRUE
    ninja -k 0 2>&1 | tee ../output/build_log.txt
    
    # Copy compile_commands.json for IDE integration
    cp compile_commands.json ../ && cd ..
    exit 0
fi

# Build and setup development environment using Docker
"${DOCKER_BIN}" run --gpus all --rm \
    -v $(pwd):/sep \
    sep_build_env bash -c '
    # Add exception for dubious ownership
    git config --global --add safe.directory "*"
    
    cd /sep/build
    
    # Configure and build with Docker container paths
    cmake .. -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_COMPILER=/usr/bin/gcc-10 \
        -DCMAKE_CXX_COMPILER=/usr/bin/g++-10 \
        -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-10 \
        -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=TRUE \
        -DSEP_USE_CUDA=ON
    
    ninja -k 0 2>&1 | tee /sep/output/build_log.txt
    
    # Copy compile_commands.json for IDE
    cp compile_commands.json ..
'

# Fix ownership of all generated files
sudo chown -R $USER_ID:$GROUP_ID /sep/.cache /sep/.codechecker /sep/build /sep/output 
fix_compile_commands() {
    # Replace container paths with host paths for IDE integration
    sed -i "s|/sep/|$(pwd)/|g" compile_commands.json
}

# Extract errors from build log
if [ -f output/build_log.txt ]; then
    echo "Extracting errors to output/errors.txt..."
    grep -i "error\|failed\|fatal" output/build_log.txt > output/errors.txt 2>/dev/null || echo "No errors found" > output/errors.txt
fi


echo "Build complete!"

