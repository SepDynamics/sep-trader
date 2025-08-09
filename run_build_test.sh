#!/bin/bash
echo "Starting Docker build test..."
docker run --rm -v $(pwd):/workspace --gpus all sep_build_env bash -c "cd /workspace && ./build.sh" > output/build_log_final.txt 2>&1
echo "Build completed. Check output/build_log_final.txt for results."
