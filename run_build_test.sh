#!/bin/bash
echo "Starting Docker build test..."
docker run --rm -v /sep:/workspace --gpus all sep_build_env bash -c "cd /workspace && ./build.sh" > /sep/output/build_log_final.txt 2>&1
echo "Build completed. Check /sep/output/build_log_final.txt for results."
