#!/bin/bash
set -e

# Build with coverage flags
export CXXFLAGS="-g -O0 --coverage -fprofile-arcs -ftest-coverage"
export LDFLAGS="--coverage"

mkdir -p build_coverage
cd build_coverage

cmake .. \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_CXX_FLAGS="-g -O0 --coverage -fprofile-arcs -ftest-coverage" \
    -DCMAKE_EXE_LINKER_FLAGS="--coverage" \
    -DCMAKE_SHARED_LINKER_FLAGS="--coverage"

make -j$(nproc)
