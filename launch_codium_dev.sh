#!/bin/bash

# Launch VSCodium with the DevPod development container
set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

# Build the Docker image first
echo "Building Docker image: sep_build_env..."
docker build -t sep_build_env -f "cache/devcontainer.bak/Dockerfile" "."

# Configure DevPod to use the Docker provider
echo "Configuring DevPod provider..."
devpod provider add docker || true # Fails silently if provider already exists
devpod provider use docker


echo "🚀 Starting SEP Trading System Development Environment with DevPod..."

# Start the DevPod environment
devpod up .

# Launch Codium
echo "🎯 Launching Codium..."
codium "$PROJECT_DIR"
