#!/bin/bash

# Start development container for SEP trading system
echo "Starting development container..."

# Build the image if it doesn't exist
if [[ "$(docker images -q sep_build_env 2> /dev/null)" == "" ]]; then
    echo "Building sep_build_env image..."
    docker build -t sep_build_env .
fi

# Start the container with proper mounts and environment
docker run -it --rm \
    --name sep_dev \
    -v "$(pwd):/workspace" \
    -w /workspace \
    --gpus=all \
    -p 8080:8080 \
    -p 5432:5432 \
    sep_build_env \
    bash

echo "Development container started. You can now run:"
echo "  ./build.sh --no-docker"
echo "  make -C build"
