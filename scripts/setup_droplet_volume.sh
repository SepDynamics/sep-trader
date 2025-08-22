#!/bin/bash
set -e

# Configuration
DROPLET_IP="159.203.131.149"
DEPLOY_USER="root"
SCRIPT_TO_RUN="configure_docker_volume.sh"

echo "🚀 Configuring Docker volume on droplet..."
echo "========================================="

echo "🔄 Uploading configuration script to droplet..."
rsync -avz scripts/$SCRIPT_TO_RUN "$DEPLOY_USER@$DROPLET_IP:/root/"

echo "🔧 Running configuration script on droplet..."
ssh $DEPLOY_USER@$DROPLET_IP "bash /root/$SCRIPT_TO_RUN"

echo "✅ Droplet volume configuration complete."
echo "Please run the main deployment script now:"
echo "./scripts/start_droplet_services.sh"