#!/bin/bash
set -e

# The new Docker data root
NEW_DOCKER_ROOT="/mnt/volume_nyc3_01/docker"

# Stop Docker
echo "Stopping Docker service..."
systemctl stop docker

# Create the new Docker root directory if it doesn't exist
mkdir -p $NEW_DOCKER_ROOT

# Sync the old Docker data to the new location
echo "Syncing Docker data to the new location..."
rsync -avz /var/lib/docker/ $NEW_DOCKER_ROOT/

# Create the Docker daemon configuration file
echo "Configuring Docker to use the new data root..."
cat <<EOF > /etc/docker/daemon.json
{
  "data-root": "$NEW_DOCKER_ROOT"
}
EOF

# Restart Docker
echo "Restarting Docker service..."
systemctl start docker

echo "Docker is now configured to use the new volume at $NEW_DOCKER_ROOT."
