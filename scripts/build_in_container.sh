#!/bin/bash
set -e

# This script is run inside the build container.

# Add exception for dubious ownership
git config --global --add safe.directory "*"

cd /workspace

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
    -DCMAKE_CXX_FLAGS="-Wno-pedantic -Wno-unknown-warning-option -Wno-invalid-source-encoding -D_GLIBCXX_USE_CXX11_ABI=0" \
    -DCMAKE_CXX_STANDARD=20 \
    -DCMAKE_CUDA_FLAGS="-Wno-deprecated-gpu-targets -Xcompiler -Wno-pedantic -Xcompiler -Wno-unknown-warning-option -Xcompiler -Wno-invalid-source-encoding -D_GLIBCXX_USE_CXX11_ABI=0"

ninja -k 0 2>&1 | tee /workspace/output/build_log.txt

# Copy compile_commands.json for IDE
cp compile_commands.json ..
