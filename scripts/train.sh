#!/bin/bash
set -e

# This script runs the training process for a given pair.
# It assumes the executables have already been built by build.sh

PAIR_TO_TRAIN=${1:-EUR_USD} # Default to EUR_USD if no argument is provided

echo "🧠 Training model for $PAIR_TO_TRAIN..."

export LD_LIBRARY_PATH=./build/src/core:./build/src/config:./build/src/c_api
./bin/quantum_pair_trainer train $PAIR_TO_TRAIN

echo "✅ Training complete."