#!/bin/bash
# Launch VSCodium with direct container integration
set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

# Source environment configuration if available
if [ -f .sep-config.env ]; then
    source .sep-config.env
fi

# Set default workspace path if not defined
SEP_WORKSPACE_PATH=${SEP_WORKSPACE_PATH:-/workspace}
export SEP_WORKSPACE_PATH

echo "🔌 Installing remote development extension for Codium..."
codium --install-extension ms-vscode-remote.remote-containers --force

echo "🚀 Building development container image..."
docker build -t sep-engine-builder -f .devcontainer/Dockerfile .

echo "🚀 Starting SEP Trading System Development Environment..."

# Launch Codium with dev container
echo "🎯 Launching Codium with container (workspace path: ${SEP_WORKSPACE_PATH})..."
codium --folder-uri "vscode-remote://dev-container+${PROJECT_DIR}"