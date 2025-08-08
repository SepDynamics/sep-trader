#!/bin/bash

# Launch VSCodium with development container
set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

echo "🚀 Starting SEP Trading System Development Environment..."

# Start the persistent development container
echo "📦 Starting development container..."
docker-compose -f docker-compose.dev.yml up -d

# Wait for container to be ready
echo "⏳ Waiting for container to be ready..."
sleep 3

# Verify container is running
if ! docker exec sep_dev_container echo "Container is ready" > /dev/null 2>&1; then
    echo "❌ Container failed to start properly"
    exit 1
fi

echo "✅ Development container is ready"

# Launch VSCodium with the project
echo "🎯 Launching VSCodium..."
codium "$PROJECT_DIR" &

# Optional: Open a terminal to the container
read -p "🔧 Open terminal to development container? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🖥️  Opening container terminal..."
    docker exec -it sep_dev_container bash
fi
