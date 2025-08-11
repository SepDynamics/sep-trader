#!/bin/bash

# Qdrant Setup Script for SEP Engine
# This script installs and configures Qdrant for the SEP Engine

# Check if Docker is installed
if ! command -v docker &> /dev/null
then
    echo "Docker is not installed. Please install Docker first."
    exit 1
fi

# Stop and remove existing Qdrant container if it exists
echo "Checking for existing Qdrant container..."
if docker ps -a | grep -q "qdrant-sep"; then
    echo "Stopping and removing existing Qdrant container..."
    docker stop qdrant-sep
    docker rm qdrant-sep
fi

# Create persistent storage directory if it doesn't exist
echo "Setting up persistent storage..."
mkdir -p ~/sep_data/qdrant

# Check if Qdrant is already running on port 6333
if curl -s http://localhost:6333/healthz > /dev/null 2>&1; then
    echo "Qdrant is already running on port 6333."
    echo "Using existing Qdrant instance."
    echo "REST API endpoint: http://localhost:6333"
else
    # Run Qdrant container
    echo "Starting Qdrant container..."
    docker run -d \
        --name qdrant-sep \
        -p 6333:6333 \
        -p 6334:6334 \
        -v ~/sep_data/qdrant:/qdrant/storage \
        qdrant/qdrant

    # Wait for Qdrant to start
    echo "Waiting for Qdrant to start..."
    sleep 5

    # Check if Qdrant is running
    echo "Checking if Qdrant is running..."
    if curl -s http://localhost:6333/healthz | grep -q "ok"; then
        echo "Qdrant is running successfully!"
        echo "REST API endpoint: http://localhost:6333"
        echo "gRPC endpoint: http://localhost:6334"
    else
        echo "Failed to start Qdrant. Please check Docker logs:"
        echo "docker logs qdrant-sep"
    fi
fi

echo ""
echo "To build and run the Qdrant test program:"
echo "cd /sep"
echo "mkdir -p build && cd build"
echo "cmake .."
echo "make test_qdrant"
echo "./src/connectors/test_qdrant"