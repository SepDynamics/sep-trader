#!/bin/bash
set -e

# This script trains all available pairs.
# It assumes the executables have already been built by build_local.sh

echo "🚀 Training all pairs..."
echo "========================================="

export LD_LIBRARY_PATH=./build/src/core:./build/src/config:./build/src/c_api
./bin/trader_cli train-all

echo "✅ All pairs trained."
