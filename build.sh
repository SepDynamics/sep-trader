#!/bin/bash
# Simplified build script for running inside a container

# Source environment configuration if available
if [ -f .sep-config.env ]; then
    source .sep-config.env
fi

# Set default workspace path if not defined
SEP_WORKSPACE_PATH=${SEP_WORKSPACE_PATH:-/workspace}

# Allow overriding the container runtime via DOCKER_BIN.
DOCKER_BIN=${DOCKER_BIN:-docker}

set -uo pipefail

REBUILD=false
SKIP_DOCKER=false
NATIVE_BUILD=false

for arg in "$@"; do
    case "$arg" in
        --rebuild)
            REBUILD=true
            ;;
        --no-docker)
            SKIP_DOCKER=true
            ;;
        --native)
            NATIVE_BUILD=true
            ;;
    esac
done

# Check if we're already in a container
if [ -f /.dockerenv ] || [ -f /run/.containerenv ]; then
    NATIVE_BUILD=true
fi

echo "Building SEP Engine..."

if [ "$REBUILD" = true ]; then
    echo "Performing a full rebuild..."
    # Clean up previous build artifacts
    sudo rm -rf CMakeCache.txt CMakeFiles output Makefile build .cache .codechecker
    sleep 2
    clear
    # Clean up any temporary trash directories
    find . -name ".Trash-*" -type d -exec sudo rm -rf {} +
    sleep 1
fi

mkdir -p output
mkdir -p build

# Ensure proper permissions for CodeChecker directories
USER_ID=$(id -u)
GROUP_ID=$(id -g)

# Use native build if in container, --no-docker specified, or container runtime unavailable
if [ "$NATIVE_BUILD" = true ] || [ "$SKIP_DOCKER" = true ] || ! "$DOCKER_BIN" info >/dev/null 2>&1; then
    if [ "$NATIVE_BUILD" = true ]; then
        echo "Building natively (container environment detected)..."
    elif [ "$SKIP_DOCKER" = true ]; then
        echo "Building natively (--no-docker)..."
    else
        echo "Container runtime $DOCKER_BIN not available. Building natively..."
    fi
    cd build

    # Configure CUDA for native builds
    CUDA_FLAGS=""
    if command -v nvcc >/dev/null 2>&1; then
        echo "CUDA detected, enabling CUDA support..."
        
        # Auto-detect CUDA_HOME if not set
        if [ -z "$CUDA_HOME" ]; then
            NVCC_PATH=$(which nvcc)
            CUDA_HOME=$(dirname $(dirname "$NVCC_PATH"))
            export CUDA_HOME
            echo "Auto-detected CUDA_HOME: $CUDA_HOME"
        else
            echo "Using existing CUDA_HOME: $CUDA_HOME"
        fi
        
        CUDA_FLAGS="-DSEP_USE_CUDA=ON"
    else
        echo "CUDA not detected, building without CUDA support..."
        CUDA_FLAGS="-DSEP_USE_CUDA=OFF"
    fi
    
    

    # Temporarily use system libraries for cmake/ninja to avoid GCC-11 library conflicts
    export LD_LIBRARY_PATH="/usr/lib64:/lib64:$LD_LIBRARY_PATH"

    # Configure with cmake, using system's default compilers and a global header fix.
    # This avoids brittle LD_PRELOAD hacks and hardcoded compiler paths.
    cmake .. -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CXX_COMPILER=g++-11 \
        -DCMAKE_C_COMPILER=gcc-11 \
        -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=TRUE \
        -DSEP_USE_GUI=OFF -DCMAKE_CXX_STANDARD=20 \
        -DCMAKE_CXX_FLAGS="-std=c++17 -Wno-unknown-warning-option -Wno-invalid-source-encoding -D_GLIBCXX_USE_CXX11_ABI=0 -Wno-cpp" \
        -DCMAKE_CUDA_FLAGS="-Wno-deprecated-gpu-targets -Xcompiler -Wno-unknown-warning-option -Xcompiler -Wno-invalid-source-encoding -D_GLIBCXX_USE_CXX11_ABI=0 -Xcompiler -Wno-cpp" \
        -DCMAKE_CUDA_STANDARD=17 \
        $CUDA_FLAGS

    # Build with ninja using system libraries
    ninja -k 0 2>&1 | tee ../output/build_log.txt

    # Copy compile_commands.json for IDE integration
    cp compile_commands.json ../ && cd ..
    exit 0
fi

echo "Mounting local directory $(pwd) to ${SEP_WORKSPACE_PATH} in the container."

# Build and setup development environment using Docker
"${DOCKER_BIN}" run --gpus all --rm \
    -v $(pwd):${SEP_WORKSPACE_PATH} \
    -e SEP_WORKSPACE_PATH=${SEP_WORKSPACE_PATH} \
    sep_build_env bash -c \
    '\
    # Add exception for dubious ownership
    git config --global --add safe.directory "*"
    
    cd /workspace
    
    # Clean any existing build directory to avoid cache conflicts
    rm -rf build
    mkdir -p build output
    cd build
    
    # Configure and build with Docker container paths
    cmake .. -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_COMPILER=/usr/bin/gcc-11 \
        -DCMAKE_CXX_COMPILER=/usr/bin/g++-11 \
        -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-11 \
        -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=TRUE \
        -DSEP_USE_CUDA=ON \
        -DSEP_USE_GUI=OFF \
        -DCMAKE_CXX_FLAGS="-Wno-pedantic -Wno-unknown-warning-option -Wno-invalid-source-encoding -D_GLIBCXX_USE_CXX11_ABI=0"         -DCMAKE_CXX_STANDARD=20         -DCMAKE_CUDA_FLAGS="-Wno-deprecated-gpu-targets -Xcompiler -Wno-pedantic -Xcompiler -Wno-unknown-warning-option -Xcompiler -Wno-invalid-source-encoding -D_GLIBCXX_USE_CXX11_ABI=0"
    
    ninja -k 0 2>&1 | tee ${SEP_WORKSPACE_PATH}/output/build_log.txt
    
    # Copy compile_commands.json for IDE
    cp compile_commands.json ..
'

# Fix ownership of all generated files
sudo chown -R $USER_ID:$GROUP_ID .cache .codechecker build output 2>/dev/null || true
fix_compile_commands() {
    # Replace container paths with host paths for IDE integration
    sed -i -e "s|${SEP_WORKSPACE_PATH}/|$(pwd)/|g" compile_commands.json
}

# Extract errors from build log
if [ -f output/build_log.txt ]; then
    grep -i "error\|failed\|fatal" output/build_log.txt > output/build_log.txt.tmp && mv output/build_log.txt.tmp output/build_log.txt || echo "No errors found" > output/build_log.txt
fi


echo "Build complete!"